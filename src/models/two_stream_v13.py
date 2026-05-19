"""
Two-Stream v13: Dual-Frame Reconstruction + Motion-Routed Latent Prediction +
                DINO-style Global CLS Distillation.

v11 reconstruction + motion-routing 구조 위에 사용자 제안(post-v12 conversation):

  1. frame_t / frame_{t+k} **둘 다** student P encoder를 통과 → 각각 reconstruction (L_t, L_tk).
     v11에서 frame_{t+k}이 reconstruction target이기만 했던 한계 해소.
     P encoder가 두 frame 모두를 직접 보고 학습.

  2. Motion-routing path는 frame_t의 p_state에서 시작 → interpreter_2 → predicted_p_tk.
     이게 V-JEPA-style latent prediction의 student.

  3. Teacher (EMA copy of student P encoder ONLY) — 두 가지 input 처리:
     (a) cropped frame_{t+k} (224x224)  → encoder → patches            (V-JEPA target)
     (b) raw     frame_{t+k} (256x256)  → encoder → CLS → dino_head    (DINO target)
     **Encoder-level routing design**: motion-routing이 encoder 직후 (interpreter_1 전)에 위치.
     interpreter_1은 reconstruction decoder 전용 (frame_t/tk 동일 path).
     interpreter_2는 motion-routing의 부정확함을 다듬는 refining 역할 → student의
     "encoder representation을 motion으로 변환"을 teacher의 unmasked encoder representation과
     align (정통 BYOL/V-JEPA asymmetric pattern).

  4. Loss:
        L_total = L_t + L_tk
                + λ_patch · SmoothL1(predicted_patches_tk, target_patches_tk)
                + λ_cls   · DINO(predicted_CLS_tk,         target_CLS_tk_global)

  Anti-collapse 메커니즘 (위 대화 SSL paradigm 정리와 매핑):
    • L_t / L_tk           → MAE-style input-dependent target (구조적)
    • L_pred_patch         → V-JEPA-style EMA + per-patch unique target (구조적)
    • L_pred_cls (DINO)    → centering + cosine; CLS에 직접 압력 (명시적)
  v11 cls_p collapse 문제를 source-level + 명시적 압력 양면으로 차단.

  M stream: v11 그대로. M decoder output은 motion-routing의 K/V로만 사용 (indirect 학습).

설계 상세: post-CoRL conversation `2026-05-01 v12 sanity 결과 + v13 재설계` 세션 참조.

NOTE: TwoStreamV11Encoder downstream wrapper는 student M/P encoder만 추출 →
      v13 ckpt에 그대로 사용 가능 (state_dict prefix 일치).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .two_stream_v11 import TwoStreamV11Model


# ============================================================================
# Helpers
# ============================================================================

def interpolate_pos_embed_2d(
    pos_embed: torch.Tensor, target_grid: int
) -> torch.Tensor:
    """[1, 1+N_src, D] → [1, 1+target_grid², D] (CLS 보존, patch만 bicubic interpolate).

    Args:
        pos_embed:    [1, 1+N_src, D]   (CLS + grid_src×grid_src patches)
        target_grid:  int               (target patch grid edge length)
    """
    cls_part = pos_embed[:, :1]                    # [1, 1, D]
    patch_part = pos_embed[:, 1:]                  # [1, N_src, D]
    n_src = patch_part.shape[1]
    grid_src = int(round(n_src ** 0.5))
    if grid_src * grid_src != n_src:
        raise ValueError(f"pos_embed patch count {n_src} is not a square grid")
    if grid_src == target_grid:
        return pos_embed
    D = patch_part.shape[-1]
    # [1, N, D] → [1, D, grid_src, grid_src]
    patch_2d = patch_part.reshape(1, grid_src, grid_src, D).permute(0, 3, 1, 2)
    patch_2d = F.interpolate(
        patch_2d.float(), size=(target_grid, target_grid),
        mode="bicubic", align_corners=False,
    ).to(pos_embed.dtype)
    # [1, D, target_grid, target_grid] → [1, target_grid², D]
    patch_part_resized = patch_2d.permute(0, 2, 3, 1).reshape(
        1, target_grid * target_grid, D,
    )
    return torch.cat([cls_part, patch_part_resized], dim=1)


class DINOHead(nn.Module):
    """DINO prototype head — 표준 DINO/DINOv2 구조.

    Pipeline: x [B, D] → MLP(D→hidden→bottleneck) → L2 normalize → weight_norm Linear(bottleneck, K)
      - MLP 내부: GELU 활성, last linear는 L2 norm projection (DINO 안전장치)
      - weight_norm linear: scale=1 고정, direction만 학습 (DINO 표준)

    이전 버전은 MLP 생략 (단순 normalize+Linear) — ep10+에서 uniform collapse 관찰됨 →
    표준 DINO MLP head 복원하여 collapse 방어 강화.
    """

    def __init__(
        self,
        embed_dim: int,
        num_prototypes: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        # 새 API parametrizations.weight_norm 사용 (deepcopy 호환, legacy weight_norm은
        # deepcopy 시 RuntimeError — teacher EMA copy에 필요).
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, num_prototypes, bias=False),
            name="weight",
        )
        # DINO 표준: weight의 norm(scale)을 1로 고정, direction만 학습.
        try:
            self.last_layer.parametrizations.weight.original0.data.fill_(1.0)
            self.last_layer.parametrizations.weight.original0.requires_grad = False
        except AttributeError:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)  # [B, K]


def dino_distillation_loss(
    student_logits: torch.Tensor,    # [B, K]
    teacher_logits: torch.Tensor,    # [B, K]  detached
    center: torch.Tensor,            # [1, K]
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
) -> torch.Tensor:
    """표준 DINO self-distillation loss — centering + sharpening + cross-entropy.

    Teacher: low temp softmax (sharp distribution, target).
    Student: high temp log-softmax (softer distribution, learner).
    Centering: teacher logits에 EMA mean subtract (uniform collapse 방어).

    Loss = -Σ p_teacher · log(p_student)  (cross-entropy in K-prototype space)
    """
    teacher_centered = teacher_logits.float() - center.float()
    teacher_probs = F.softmax(teacher_centered / teacher_temp, dim=-1)        # [B, K]
    student_log_probs = F.log_softmax(student_logits.float() / student_temp, dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()


# ============================================================================
# Teacher P (v13) — cropped per-patch + global CLS forward
# ============================================================================

class TeacherPv13(nn.Module):
    """EMA copy of student P encoder. 두 input 모드:
      forward_cropped(image_224)  → [B, 1+N, D]   (per-patch + CLS, all visible)
      forward_global(image_256)   → [B, D]        (CLS only, pos_embed bicubic interpolated)

    EMA update: encoder + cls_token + pos_embed 모두 momentum update.
    """

    def __init__(self, student: "TwoStreamV13Model", patch_size: int):
        super().__init__()
        # v13 encoder-level routing: teacher는 P encoder만 거침.
        # Student의 motion-routing + interpreter_2가 encoder 직후에 있어 student-teacher
        # 모두 encoder-level representation에서 align (정통 BYOL/V-JEPA asymmetric pattern).
        self.preprocessing = copy.deepcopy(student.preprocessing)
        self.patch_embed_p = copy.deepcopy(student.patch_embed_p)
        self.cls_token_p = nn.Parameter(student.cls_token_p.data.clone())
        self.pos_embed_p = nn.Parameter(student.pos_embed_p.data.clone())  # [1, 1+196, D]
        self.blocks_p = copy.deepcopy(student.blocks_p)
        self.norm_p = copy.deepcopy(student.norm_p)
        # DINO prototype head (EMA copy of student head)
        self.dino_head = copy.deepcopy(student.dino_head)
        self.patch_size = patch_size

        for p in self.parameters():
            p.requires_grad = False

    def _encode(self, p_channel: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        """P encoder만 (encoder-level representation). [B, 5, H, W] → [B, 1+N, D]."""
        B = p_channel.shape[0]
        patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_p.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1) + pos_embed
        for block in self.blocks_p:
            x = block(x, freqs_cis=None)
        return self.norm_p(x)

    @torch.no_grad()
    def forward_cropped(self, image_cropped: torch.Tensor) -> torch.Tensor:
        """[B, 3, 224, 224] → [B, 1+N, D] (encoder output only)."""
        p_channel = self.preprocessing.compute_p_channel(image_cropped)
        return self._encode(p_channel, self.pos_embed_p)

    @torch.no_grad()
    def forward_global(self, image_global: torch.Tensor) -> torch.Tensor:
        """[B, 3, 256, 256] → [B, 1+N_global, D]. pos_embed bicubic interpolate."""
        p_channel = self.preprocessing.compute_p_channel(image_global)
        H = image_global.shape[-1]
        target_grid = H // self.patch_size
        pos_embed_global = interpolate_pos_embed_2d(self.pos_embed_p, target_grid)
        return self._encode(p_channel, pos_embed_global)

    @torch.no_grad()
    def update(self, student: "TwoStreamV13Model", momentum: float):
        """Encoder + cls/pos token + DINO head EMA update (encoder-only teacher)."""
        student_modules = {
            "patch_embed_p": student.patch_embed_p,
            "blocks_p": student.blocks_p,
            "norm_p": student.norm_p,
            "dino_head": student.dino_head,
        }
        teacher_modules = {
            "patch_embed_p": self.patch_embed_p,
            "blocks_p": self.blocks_p,
            "norm_p": self.norm_p,
            "dino_head": self.dino_head,
        }
        for key in student_modules:
            for p_t, p_s in zip(
                teacher_modules[key].parameters(), student_modules[key].parameters()
            ):
                p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        self.cls_token_p.data.mul_(momentum).add_(student.cls_token_p.data, alpha=1.0 - momentum)
        self.pos_embed_p.data.mul_(momentum).add_(student.pos_embed_p.data, alpha=1.0 - momentum)


# ============================================================================
# v13 Model
# ============================================================================

class TwoStreamV13Model(TwoStreamV11Model):
    """Two-Stream v13: dual-frame reconstruction + motion-routed latent prediction +
    DINO-style global CLS distillation.

    v11 architecture는 거의 그대로 (M stream, P encoder/decoder, motion-routing,
    interpreter_1/2, recon_head). 추가:
      - TeacherPv13 (EMA): cropped + global view 둘 다 forward 가능
      - DINO center (running mean) buffer
      - frame_t + frame_{t+k} 모두 student P encoder 통과 (각자 independent mask)

    Args (v13 specific):
        patch_pred_weight (λ_patch): per-patch SmoothL1 weight (default 1.0)
        cls_pred_weight   (λ_cls):   DINO cosine loss weight  (default 0.1)
        dino_center_momentum: EMA momentum for DINO center (default 0.9)
        그 외 args는 TwoStreamV11Model과 동일.
    """

    def __init__(
        self,
        # v11 args
        embed_dim: int = 768,
        p_depth: int = 12,
        m_depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        image_size: int = 224,
        patch_size: int = 16,
        mask_ratio_m: float = 0.3,
        mask_ratio_p: float = 0.75,
        decoder_depth_m: int = 3,
        interpreter_depth: int = 3,
        num_motion_iters: int = 2,
        rotation_aug: bool = False,
        independent_rotation_prob: float = 0.1,
        routing_mode: str = "v_from_p",
        # v13 args
        patch_pred_weight: float = 1.5,
        cls_pred_weight: float = 0.3,
        dino_center_momentum: float = 0.9,
        # v13 DINO full
        num_prototypes: int = 4096,
        dino_teacher_temp: float = 0.04,
        dino_student_temp: float = 0.1,
        mask_ratio_p_dino: float = 0.4,  # DINO path 전용 mask (recon mask와 분리)
    ):
        super().__init__(
            embed_dim=embed_dim,
            p_depth=p_depth,
            m_depth=m_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            image_size=image_size,
            patch_size=patch_size,
            mask_ratio_m=mask_ratio_m,
            mask_ratio_p=mask_ratio_p,
            decoder_depth_m=decoder_depth_m,
            interpreter_depth=interpreter_depth,
            num_motion_iters=num_motion_iters,
            rotation_aug=rotation_aug,
            independent_rotation_prob=independent_rotation_prob,
            routing_mode=routing_mode,
        )

        self.patch_pred_weight = patch_pred_weight
        self.cls_pred_weight = cls_pred_weight
        self.dino_center_momentum = dino_center_momentum
        self.dino_teacher_temp = dino_teacher_temp
        self.dino_student_temp = dino_student_temp
        self.num_prototypes = num_prototypes
        self.mask_ratio_p_dino = mask_ratio_p_dino

        # DINO prototype head (student) — student CLS → K-prototype logits
        self.dino_head = DINOHead(embed_dim, num_prototypes=num_prototypes)

        # DINO center: teacher prototype output running mean ([1, K], uniform collapse 방지)
        self.register_buffer("dino_center", torch.zeros(1, num_prototypes))

        # Teacher (EMA copy of student P encoder + DINO head)
        # NOTE: TeacherPv13가 self.dino_head를 deepcopy하므로 head 등록 후 호출.
        self.teacher_p = TeacherPv13(self, patch_size=patch_size)

    # ----------------------------------------------------------------------
    # EMA / center update
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """Training loop에서 매 optimizer step 후 호출."""
        self.teacher_p.update(self, momentum)

    @torch.no_grad()
    def update_dino_center(self, teacher_proto_logits: torch.Tensor):
        """DINO center를 teacher prototype logits의 running mean으로 갱신.

        Args:
            teacher_proto_logits: [B, K]  (teacher.dino_head(teacher_global_cls) 결과)
        """
        m = self.dino_center_momentum
        batch_mean = teacher_proto_logits.float().mean(dim=0, keepdim=True)
        self.dino_center.data.mul_(m).add_(batch_mean, alpha=1.0 - m)

    # ----------------------------------------------------------------------
    # Helpers — student single-frame forward (v11 forward의 frame_t 부분 재사용)
    # ----------------------------------------------------------------------

    def _student_p_encode(
        self, p_channel: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """[B, 5, H, W] + mask [B, N] → encoded [B, 1+N_visible, D]."""
        B = p_channel.shape[0]
        patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_p.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos_embed_p
        return self._encode_stream_visible(tokens, mask, self.blocks_p, self.norm_p)

    def _build_full_seq(
        self, p_encoded: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Encoder visible only → mask_token inject → +dec_pos_embed.
        Returns: [B, 1+N, D] (encoder-level full sequence, motion-routing/recon 공통 시작점).
        """
        p_full = self._inject_mask_tokens(p_encoded, mask, self.mask_token_p)
        return p_full + self.dec_pos_embed_p

    def _decode_for_recon(self, p_full: torch.Tensor) -> tuple:
        """Reconstruction decoder: interpreter_1 → recon_head."""
        p_decoded = self._run_interpreter(
            p_full, self.interpreter_1, self.interpreter_1_norm,
        )
        patch_pred = self.recon_head(p_decoded[:, 1:])
        return patch_pred, p_decoded

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def forward(
        self,
        image_current: torch.Tensor,
        image_future: torch.Tensor,
        image_current_global: torch.Tensor = None,
        image_future_global: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            image_current:         [B, 3, 224, 224]  frame_t  (cropped, recon+motion path)
            image_future:          [B, 3, 224, 224]  frame_tk (cropped, recon+motion path)
            image_current_global:  [B, 3, 256, 256]  frame_t  (raw 256, DINO teacher view)
            image_future_global:   [B, 3, 256, 256]  frame_tk (raw 256, DINO teacher view)
                                                     둘 중 하나라도 None이면 L_pred_cls 비활성

        Returns dict 주요 키:
            loss, loss_t, loss_tk, loss_pred_patch, loss_pred_cls
            pred_t, pred_tk     ([B, 3, H, W] visualization용)
            predicted_patches_tk, predicted_cls_tk   ([B, N, D], [B, D]) — motion path
            student_dino_cls_t, student_dino_cls_tk  (DINO student CLS, mask_ratio_p_dino)
            target_cls_t_global, target_cls_tk_global  (teacher CLS, EMA frozen)
            cls_m, cls_p (v11 호환 — frame_t 기준)
        """
        # ── Rotation aug — global view들도 함께 동일 회전 적용. 90도 단위(rot90).
        if self.rotation_aug and self.training:
            if torch.rand(1).item() < self.independent_rotation_prob:
                k_t = torch.randint(0, 4, (1,)).item()
                k_tk = torch.randint(0, 4, (1,)).item()
            else:
                k_t = torch.randint(0, 4, (1,)).item()
                k_tk = k_t
            if k_t != 0:
                image_current = torch.rot90(image_current, k_t, dims=(2, 3))
                if image_current_global is not None:
                    image_current_global = torch.rot90(image_current_global, k_t, dims=(2, 3))
            if k_tk != 0:
                image_future = torch.rot90(image_future, k_tk, dims=(2, 3))
                if image_future_global is not None:
                    image_future_global = torch.rot90(image_future_global, k_tk, dims=(2, 3))

        B = image_current.shape[0]
        device = image_current.device

        # ── Preprocessing — M (frame_t,frame_{t+k} 차이) + P_t + P_{t+k} ──
        # M channel: v11 동일 (preprocessing.forward는 m_channel + p_channel returns,
        # p_channel은 image_previous=current 기준이라 frame_t용)
        m_channel, p_channel_t = self.preprocessing(image_current, image_future)
        # Frame_{t+k}의 P channel은 추가로 직접 호출
        p_channel_tk = self.preprocessing.compute_p_channel(image_future)

        # ── Independent masks for M, P_t, P_{t+k} ──────────────────────
        mask_m = self._random_mask(B, device, self.mask_ratio_m)
        mask_p_t = self._random_mask(B, device, self.mask_ratio_p)
        mask_p_tk = self._random_mask(B, device, self.mask_ratio_p)

        # ── M stream — v11 동일 (visible-only encode + decoder full) ────
        m_patches = self.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        m_cls = self.cls_token_m.expand(B, -1, -1)
        m_tokens = torch.cat([m_cls, m_patches], dim=1) + self.pos_embed_m
        m_encoded = self._encode_stream_visible(
            m_tokens, mask_m, self.blocks_m, self.norm_m,
        )
        cls_m_repr = m_encoded[:, 0]

        m_full = self._inject_mask_tokens(m_encoded, mask_m, self.mask_token_m)
        m_full = m_full + self.dec_pos_embed_m
        for block in self.m_decoder_blocks:
            m_full = block(m_full, freqs_cis=None)
        m_completed = self.m_decoder_norm(m_full)

        # ── P stream — visible-only encoding (MAE) ──────────────────────
        p_t_visible = self._student_p_encode(p_channel_t, mask_p_t)        # [B, 1+N_v, D]
        p_tk_visible = self._student_p_encode(p_channel_tk, mask_p_tk)
        cls_p_repr = p_t_visible[:, 0]   # downstream / probing 호환 (frame_t)

        # ── 공통 시작점: mask_token inject + dec_pos_embed_p ─────────────
        # 이 단계 이후 reconstruction과 motion-routing이 분기.
        p_t_full = self._build_full_seq(p_t_visible, mask_p_t)             # [B, 1+N, D]
        p_tk_full = self._build_full_seq(p_tk_visible, mask_p_tk)

        # ── (1) Reconstruction (interpreter_1 + recon_head, frame_t/tk 동일 path) ─
        patch_pred_t, _ = self._decode_for_recon(p_t_full)
        patch_pred_tk_recon, _ = self._decode_for_recon(p_tk_full)

        # ── (2) Motion-routing on p_t_full (encoder-level prediction) ────
        # encoder output에 mask_token 채운 full sequence가 motion-routing의 입력.
        # M의 K/V를 attend → frame_t의 representation을 frame_{t+k} encoder representation으로 변환.
        p_state = p_t_full
        for routing_block in self.motion_routing:
            p_state = routing_block(p_state, m_completed)
        p_semantic_tk_pred = self._run_interpreter(
            p_state, self.interpreter_2, self.interpreter_2_norm,
        )
        predicted_cls_tk = p_semantic_tk_pred[:, 0]          # [B, D]
        predicted_patches_tk = p_semantic_tk_pred[:, 1:]     # [B, N, D]

        # ── Reconstruction loss (masked positions only, MAE 표준) ───────
        patch_target_t = self._patchify(image_current)
        patch_target_tk = self._patchify(image_future)

        err_t = (patch_pred_t - patch_target_t) ** 2
        err_tk_recon = (patch_pred_tk_recon - patch_target_tk) ** 2
        err_t = err_t.mean(dim=-1)
        err_tk_recon = err_tk_recon.mean(dim=-1)

        denom_t = mask_p_t.float().sum().clamp(min=1.0)
        denom_tk = mask_p_tk.float().sum().clamp(min=1.0)
        loss_t = (err_t * mask_p_t.float()).sum() / denom_t
        loss_tk_recon = (err_tk_recon * mask_p_tk.float()).sum() / denom_tk

        # ── Teacher target — patch-level (motion target) + DINO multi-crop CLS ──
        # Patch target: cropped frame_{t+k} (motion path align용 — 위치 매칭 필요).
        # DINO target: 256² global view CLS, frame_t/tk 각자 (multi-crop teacher).
        with torch.no_grad():
            teacher_cropped_full = self.teacher_p.forward_cropped(image_future)
            target_patches_tk = teacher_cropped_full[:, 1:].detach()   # [B, N, D]

            # DINO multi-crop targets — frame_t, frame_tk 둘 다 256² global
            if image_current_global is not None and image_future_global is not None:
                teacher_global_t_full = self.teacher_p.forward_global(image_current_global)
                teacher_global_tk_full = self.teacher_p.forward_global(image_future_global)
                target_cls_t_global = teacher_global_t_full[:, 0]            # [B, D]
                target_cls_tk_global = teacher_global_tk_full[:, 0]
                teacher_proto_logits_t = self.teacher_p.dino_head(target_cls_t_global).detach()
                teacher_proto_logits_tk = self.teacher_p.dino_head(target_cls_tk_global).detach()
            else:
                target_cls_t_global = None
                target_cls_tk_global = None
                teacher_proto_logits_t = None
                teacher_proto_logits_tk = None

        # ── Patch-level latent prediction loss (V-JEPA-style, masked-only) ─
        err_pred = F.smooth_l1_loss(
            predicted_patches_tk.float(), target_patches_tk.float(),
            reduction='none',
        ).mean(dim=-1)  # [B, N]
        mask_p_t_f = mask_p_t.float()
        denom_pred = mask_p_t_f.sum().clamp(min=1.0)
        loss_pred_patch = (err_pred * mask_p_t_f).sum() / denom_pred

        # ── DINO-style CLS distillation (DINOv2 multi-crop pattern) ──────
        # Student: P encoder CLS at mask_ratio_p_dino (= 0.4, recon mask 0.75와 분리).
        # Teacher: 256² global view CLS, frame_t/tk 각자 align (A1 self-distill).
        # P encoder가 직접 학습 신호를 받아 CLS collapse 방어 (이전 motion-routed CLS는
        # ep10+ uniform collapse 관찰됨 → root cause 해결).
        if teacher_proto_logits_t is not None:
            mask_p_t_dino = self._random_mask(B, device, self.mask_ratio_p_dino)
            mask_p_tk_dino = self._random_mask(B, device, self.mask_ratio_p_dino)
            student_dino_t = self._student_p_encode(p_channel_t, mask_p_t_dino)[:, 0]
            student_dino_tk = self._student_p_encode(p_channel_tk, mask_p_tk_dino)[:, 0]
            student_proto_logits_t = self.dino_head(student_dino_t)
            student_proto_logits_tk = self.dino_head(student_dino_tk)
            loss_pred_cls_t = dino_distillation_loss(
                student_proto_logits_t, teacher_proto_logits_t, self.dino_center,
                student_temp=self.dino_student_temp,
                teacher_temp=self.dino_teacher_temp,
            )
            loss_pred_cls_tk = dino_distillation_loss(
                student_proto_logits_tk, teacher_proto_logits_tk, self.dino_center,
                student_temp=self.dino_student_temp,
                teacher_temp=self.dino_teacher_temp,
            )
            loss_pred_cls = 0.5 * (loss_pred_cls_t + loss_pred_cls_tk)
        else:
            student_dino_t = None
            student_dino_tk = None
            student_proto_logits_t = None
            student_proto_logits_tk = None
            loss_pred_cls = torch.zeros((), device=device, dtype=loss_pred_patch.dtype)

        # ── Total ───────────────────────────────────────────────────────
        loss = (
            loss_t
            + loss_tk_recon
            + self.patch_pred_weight * loss_pred_patch
            + self.cls_pred_weight * loss_pred_cls
        )

        # ── Unpatchify (visualization) ──────────────────────────────────
        pred_t_img = self._unpatchify(patch_pred_t)
        pred_tk_img = self._unpatchify(patch_pred_tk_recon)

        return {
            "loss": loss,
            "loss_t": loss_t,
            "loss_tk": loss_tk_recon,
            "loss_pred_patch": loss_pred_patch,
            "loss_pred_cls": loss_pred_cls,
            "pred_t": pred_t_img,
            "pred_tk": pred_tk_img,
            "mask_p": mask_p_t,
            "mask_m": mask_m,
            # downstream features (encoder-level routing 디자인)
            #   m_features:    M decoder output (motion field)
            #   p_features_t:  frame_t의 encoder visible output (full sequence는 mask 위치 inferred)
            #   p_features_tk: motion-routed + interpreter_2 (predicted frame_{t+k} encoder)
            "m_features": m_completed[:, 1:],
            "p_features_t": p_t_visible[:, 1:],
            "p_features_tk": p_semantic_tk_pred[:, 1:],
            "cls_m": cls_m_repr,
            "cls_p": cls_p_repr,
            # v13 추가
            "predicted_cls_tk": predicted_cls_tk,         # motion-routed (DINO에 사용 안 함, 진단용)
            "predicted_patches_tk": predicted_patches_tk,  # motion path output (loss_pred_patch)
            "target_patches_tk": target_patches_tk,
            # DINO multi-crop CLS (frame_t/tk 각자 align)
            "student_dino_cls_t": (
                student_dino_t if student_dino_t is not None
                else torch.zeros(B, self.embed_dim, device=device)
            ),
            "student_dino_cls_tk": (
                student_dino_tk if student_dino_tk is not None
                else torch.zeros(B, self.embed_dim, device=device)
            ),
            "target_cls_t_global": (
                target_cls_t_global if target_cls_t_global is not None
                else torch.zeros(B, self.embed_dim, device=device)
            ),
            "target_cls_tk_global": (
                target_cls_tk_global if target_cls_tk_global is not None
                else torch.zeros(B, self.embed_dim, device=device)
            ),
            # Teacher prototype logits (DINO center update에 사용 — frame_t/tk 합산)
            "teacher_proto_logits": (
                torch.cat([teacher_proto_logits_t, teacher_proto_logits_tk], dim=0)
                if teacher_proto_logits_t is not None
                else torch.zeros(B, self.num_prototypes, device=device)
            ),
        }

    def compute_loss(
        self,
        image_current: torch.Tensor,
        image_future: torch.Tensor,
        image_future_global: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(image_current, image_future, image_future_global)
        return out["loss"], out["pred_tk"]
