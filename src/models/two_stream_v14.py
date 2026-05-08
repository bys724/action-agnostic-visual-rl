"""
Two-Stream v14: Stream-wise Paradigm Specialization.

v13 paradigm conflict 진단 (P encoder가 reconstruction(detail-preserving) +
DINO(detail-invariance)를 동시에 만족 못 함 → ep10+ uniform collapse) 후 도착한 redesign.

핵심 가설 — paradigm을 stream-wise로 분리하면 conflict가 구조적으로 해소된다:
  · P stream: MAE (L_t + L_tk_recon) + V-JEPA (L_pred) — 둘 다 reconstruction-anchored
  · M stream: DINO only (L_dino) — distillation only, reconstruction 압력 없음

v13의 DINO를 P stream에서 M stream으로 이동한 것이 결정적 변화. M encoder는
reconstruction loss를 받지 않으므로 DINO의 detail-invariance 요구와 충돌이 없다.

Loss:
    L_total = L_t + L_tk_recon                   ← MAE      (P stream)
            + λ_pred · L_pred                    ← V-JEPA   (P stream)
            + λ_dino · L_dino                    ← DINO     (M stream)

Anti-collapse 메커니즘:
  · L_t / L_tk_recon  → MAE는 input-dependent target (구조적 anti-collapse)
  · L_pred            → student/teacher 모두 P encoder space에서 시작·종료 (자동 정합).
                         Reconstruction이 P encoder를 collapse 못 하게 잡아주는 anchor.
  · L_dino            → centering + sharpening (uniform collapse 방어).
                         M encoder는 reconstruction 압력 없음 → DINO와 paradigm 충돌 X.

V-JEPA path 차이 (v13 → v14):
  - v13: motion routing 입력 = encoder visible + mask_token + dec_pos_embed,
         target = teacher의 cropped patches만 → mask_token이 섞이며 space 일치 모호.
  - v14: motion routing 입력 = encoder unmasked output 직접 (V-JEPA 정통 패턴),
         target = teacher의 unmasked encoder output (CLS+patches 모두) → 자동 정합.

DINO path 차이 (v13 → v14):
  - v13: student/teacher 모두 P encoder CLS. Student는 cropped(mask 0.4),
         teacher는 raw 256(no mask). P encoder가 reconstruction과 동시 학습 → conflict.
  - v14: student = M_channel(cropped pair) → M_encoder → CLS,
         teacher = M_channel(raw pair) → Teacher_M_encoder → CLS → Teacher_DINOHead.
         Student는 misaligned cropped pair (real motion + camera shift 혼합),
         teacher는 정렬된 raw pair (clean motion). 비대칭이 곧 invariance 학습.

Multi-crop strategy (Option B, 점진 도입):
  Per sample:
    cropped pair × 1 (crop_t, crop_tk): Recon, JEPA, motion routing 모두 share
    추가 cropped pair × (N-1): DINO student로만 추가 forward (raw에서 GPU random crop)
    raw pair × 1 (raw_t, raw_tk): DINO teacher (모든 student와 공유)

  L_dino = (1/N) · Σ_n CE(student_logits_n, teacher_logits)

  N=1 (sanity): 표준 학습 input pair만, 4 loss 학습 안정성 확인
  N=2 (본 학습): DINO student에 추가 random crop pair 1개. M encoder 추가 forward는
                 P encoder forward 대비 저비용 (M depth 6 vs P depth 12).

EMA teachers (3 modules):
  - TeacherPv14: V-JEPA target (P_encoder의 EMA copy)
  - TeacherMv14: DINO target (M_encoder + DINOHead의 EMA copy)
  - dino_center buffer: teacher prototype logits running mean

설계 노트: vault `Projects/Action-Agnostic Paper/v14 - Two-Stream Paradigm Specialization (Future).md`
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .two_stream_v11 import TwoStreamV11Model
from .two_stream_v13 import (
    DINOHead,
    dino_distillation_loss,
    interpolate_pos_embed_2d,
)


# ============================================================================
# Teachers
# ============================================================================

class TeacherPv14(nn.Module):
    """EMA copy of student P encoder. V-JEPA target 전용 (unmasked, cropped 224)."""

    def __init__(self, student: "TwoStreamV14Model"):
        super().__init__()
        self.preprocessing = copy.deepcopy(student.preprocessing)
        self.patch_embed_p = copy.deepcopy(student.patch_embed_p)
        self.cls_token_p = nn.Parameter(student.cls_token_p.data.clone())
        self.pos_embed_p = nn.Parameter(student.pos_embed_p.data.clone())
        self.blocks_p = copy.deepcopy(student.blocks_p)
        self.norm_p = copy.deepcopy(student.norm_p)

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward_unmasked(self, p_channel: torch.Tensor) -> torch.Tensor:
        """[B, 5, 224, 224] → [B, 1+N, D] (CLS + all patches, V-JEPA target)."""
        B = p_channel.shape[0]
        patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_p.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1) + self.pos_embed_p
        for block in self.blocks_p:
            x = block(x, freqs_cis=None)
        return self.norm_p(x)

    @torch.no_grad()
    def update(self, student: "TwoStreamV14Model", momentum: float):
        for p_t, p_s in zip(self.patch_embed_p.parameters(), student.patch_embed_p.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        for p_t, p_s in zip(self.blocks_p.parameters(), student.blocks_p.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        for p_t, p_s in zip(self.norm_p.parameters(), student.norm_p.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        self.cls_token_p.data.mul_(momentum).add_(student.cls_token_p.data, alpha=1.0 - momentum)
        self.pos_embed_p.data.mul_(momentum).add_(student.pos_embed_p.data, alpha=1.0 - momentum)


class TeacherMv14(nn.Module):
    """EMA copy of student M encoder + DINOHead. DINO target 전용 (unmasked, raw 256)."""

    def __init__(self, student: "TwoStreamV14Model", patch_size: int):
        super().__init__()
        self.patch_embed_m = copy.deepcopy(student.patch_embed_m)
        self.cls_token_m = nn.Parameter(student.cls_token_m.data.clone())
        # pos_embed는 cropped grid (14×14+1)에 맞춰져 있음. raw 256(16×16+1) 입력 시
        # forward_global_cls()에서 bicubic interpolate하여 사용.
        self.pos_embed_m = nn.Parameter(student.pos_embed_m.data.clone())
        self.blocks_m = copy.deepcopy(student.blocks_m)
        self.norm_m = copy.deepcopy(student.norm_m)
        # DINOHead도 EMA copy (DINOv2 표준)
        self.dino_head = copy.deepcopy(student.dino_head)
        self.patch_size = patch_size

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward_global_cls(self, m_channel_raw: torch.Tensor) -> torch.Tensor:
        """[B, 3, 256, 256] → CLS [B, D]. pos_embed bicubic interpolate to raw grid."""
        B = m_channel_raw.shape[0]
        H = m_channel_raw.shape[-1]
        target_grid = H // self.patch_size

        patches = self.patch_embed_m(m_channel_raw).flatten(2).transpose(1, 2)
        cls = self.cls_token_m.expand(B, -1, -1)
        pos_embed_global = interpolate_pos_embed_2d(self.pos_embed_m, target_grid)
        x = torch.cat([cls, patches], dim=1) + pos_embed_global
        for block in self.blocks_m:
            x = block(x, freqs_cis=None)
        x = self.norm_m(x)
        return x[:, 0]

    @torch.no_grad()
    def update(self, student: "TwoStreamV14Model", momentum: float):
        for p_t, p_s in zip(self.patch_embed_m.parameters(), student.patch_embed_m.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        for p_t, p_s in zip(self.blocks_m.parameters(), student.blocks_m.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        for p_t, p_s in zip(self.norm_m.parameters(), student.norm_m.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        for p_t, p_s in zip(self.dino_head.parameters(), student.dino_head.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        self.cls_token_m.data.mul_(momentum).add_(student.cls_token_m.data, alpha=1.0 - momentum)
        self.pos_embed_m.data.mul_(momentum).add_(student.pos_embed_m.data, alpha=1.0 - momentum)


# ============================================================================
# v14 Model
# ============================================================================

class TwoStreamV14Model(TwoStreamV11Model):
    """Two-Stream v14: Stream-wise Paradigm Specialization.

    v11 base 위에 추가:
      - P encoder unmasked forward (V-JEPA student) + TeacherPv14 (V-JEPA target)
      - M encoder unmasked forward (DINO student + motion routing source)
      - TeacherMv14 (DINO target, raw 256 input, pos_embed bicubic interpolate)
      - DINOHead (student) + dino_center buffer
      - Multi-crop N (DINO student에만 추가 random crop forward)

    v14에서 M stream은 unmasked로만 사용 (DINO + motion routing 둘 다 unmasked).
    `mask_ratio_m`은 v11 인터페이스 호환을 위해 받지만 실제로는 0.0으로 hardcoded.

    Args (v14 specific):
        lambda_pred: V-JEPA loss weight (default 1.0)
        lambda_dino: DINO loss weight (default 1.0)
        dino_n_crop: DINO student multi-crop count (sanity=1, 본=2)
        num_prototypes: DINO prototype K (default 1024, 데이터셋 보수적 시작값)
        dino_teacher_temp: τ_T (default 0.04, sharpening)
        dino_student_temp: τ_S (default 0.1)
        dino_center_momentum: EMA momentum for DINO center (default 0.9)
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
        mask_ratio_m: float = 0.0,
        mask_ratio_p: float = 0.75,
        decoder_depth_m: int = 3,
        interpreter_depth: int = 3,
        num_motion_iters: int = 2,
        rotation_aug: bool = False,
        independent_rotation_prob: float = 0.1,
        routing_mode: str = "v_from_p",
        # v14 args
        lambda_pred: float = 1.0,
        lambda_dino: float = 1.0,
        dino_n_crop: int = 1,
        num_prototypes: int = 1024,
        dino_teacher_temp: float = 0.04,
        dino_student_temp: float = 0.1,
        dino_center_momentum: float = 0.9,
    ):
        # v14 — M stream은 항상 unmasked. mask_ratio_m을 0으로 강제.
        super().__init__(
            embed_dim=embed_dim,
            p_depth=p_depth,
            m_depth=m_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            image_size=image_size,
            patch_size=patch_size,
            mask_ratio_m=0.0,
            mask_ratio_p=mask_ratio_p,
            decoder_depth_m=decoder_depth_m,
            interpreter_depth=interpreter_depth,
            num_motion_iters=num_motion_iters,
            rotation_aug=rotation_aug,
            independent_rotation_prob=independent_rotation_prob,
            routing_mode=routing_mode,
        )

        self.lambda_pred = lambda_pred
        self.lambda_dino = lambda_dino
        self.dino_n_crop = dino_n_crop
        self.num_prototypes = num_prototypes
        self.dino_teacher_temp = dino_teacher_temp
        self.dino_student_temp = dino_student_temp
        self.dino_center_momentum = dino_center_momentum

        # DINO Head (student) — M CLS → K prototype logits
        self.dino_head = DINOHead(embed_dim, num_prototypes=num_prototypes)

        # DINO center buffer (uniform collapse 방어)
        self.register_buffer("dino_center", torch.zeros(1, num_prototypes))

        # Teachers — dino_head 등록 후 호출 (TeacherMv14가 dino_head를 deepcopy)
        self.teacher_p = TeacherPv14(self)
        self.teacher_m = TeacherMv14(self, patch_size=patch_size)

        # v14는 M stream을 항상 unmasked로 사용 → mask_token_m forward에서 안 쓰임.
        # 그대로 두면 DDP가 두번째 step에서 unused-param 에러 발생.
        self.mask_token_m.requires_grad_(False)

    # ----------------------------------------------------------------------
    # EMA / center update
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """Training loop에서 매 optimizer step 후 호출."""
        self.teacher_p.update(self, momentum)
        self.teacher_m.update(self, momentum)

    @torch.no_grad()
    def update_dino_center(self, teacher_proto_logits: torch.Tensor):
        """DINO center를 teacher prototype logits의 running mean으로 갱신."""
        m = self.dino_center_momentum
        batch_mean = teacher_proto_logits.float().mean(dim=0, keepdim=True)
        self.dino_center.data.mul_(m).add_(batch_mean, alpha=1.0 - m)

    # ----------------------------------------------------------------------
    # Helpers — encoder forward variants
    # ----------------------------------------------------------------------

    def _encode_p_unmasked(self, p_channel: torch.Tensor) -> torch.Tensor:
        """[B, 5, 224, 224] → [B, 1+N, D] (no masking, V-JEPA student)."""
        B = p_channel.shape[0]
        patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_p.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1) + self.pos_embed_p
        for block in self.blocks_p:
            x = block(x, freqs_cis=None)
        return self.norm_p(x)

    def _encode_m_unmasked(self, m_channel: torch.Tensor) -> torch.Tensor:
        """[B, 3, 224, 224] → [B, 1+N, D] (no masking, DINO student + motion routing source)."""
        B = m_channel.shape[0]
        patches = self.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_m.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1) + self.pos_embed_m
        for block in self.blocks_m:
            x = block(x, freqs_cis=None)
        return self.norm_m(x)

    def _decode_m_unmasked(self, m_encoded: torch.Tensor) -> torch.Tensor:
        """M encoder unmasked output → m_completed (motion routing K/V).

        v11 m_decoder pattern을 따르되 mask_token inject 없음 (input이 이미 full).
        """
        x = m_encoded + self.dec_pos_embed_m
        for block in self.m_decoder_blocks:
            x = block(x, freqs_cis=None)
        return self.m_decoder_norm(x)

    def _student_p_encode_visible(
        self, p_channel: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """[B, 5, H, W] + mask [B, N] → encoded [B, 1+N_visible, D] (MAE student)."""
        B = p_channel.shape[0]
        patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_p.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos_embed_p
        return self._encode_stream_visible(tokens, mask, self.blocks_p, self.norm_p)

    def _build_full_seq(
        self, p_visible: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Encoder visible-only → mask_token inject → +dec_pos_embed_p (recon decoder 입력)."""
        p_full = self._inject_mask_tokens(p_visible, mask, self.mask_token_p)
        return p_full + self.dec_pos_embed_p

    def _decode_for_recon(self, p_full: torch.Tensor) -> torch.Tensor:
        """Reconstruction decoder: interpreter_1 → recon_head → patch pixel pred."""
        p_decoded = self._run_interpreter(
            p_full, self.interpreter_1, self.interpreter_1_norm,
        )
        return self.recon_head(p_decoded[:, 1:])

    def _gpu_random_crop_pair(
        self, raw_t: torch.Tensor, raw_tk: torch.Tensor, crop_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-crop additional view 생성. raw [B,3,H,W] pair → cropped [B,3,crop_size,crop_size] pair.

        각 sample / 각 frame 독립 random crop 좌표 (dataset의 첫 crop 패턴과 일치).
        Raw pair는 이미 GPU에 있으므로 indexing slice만으로 cost ~0.
        """
        B, _, H, W = raw_t.shape
        max_y = H - crop_size
        max_x = W - crop_size
        device = raw_t.device

        # Per-sample, per-frame 독립 좌표
        y_t = torch.randint(0, max_y + 1, (B,), device=device)
        x_t = torch.randint(0, max_x + 1, (B,), device=device)
        y_tk = torch.randint(0, max_y + 1, (B,), device=device)
        x_tk = torch.randint(0, max_x + 1, (B,), device=device)

        crop_t = torch.stack([
            raw_t[i, :, y_t[i]:y_t[i] + crop_size, x_t[i]:x_t[i] + crop_size]
            for i in range(B)
        ])
        crop_tk = torch.stack([
            raw_tk[i, :, y_tk[i]:y_tk[i] + crop_size, x_tk[i]:x_tk[i] + crop_size]
            for i in range(B)
        ])
        return crop_t, crop_tk

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
            image_current:        [B, 3, 224, 224]  crop_t (cropped pair)
            image_future:         [B, 3, 224, 224]  crop_tk (cropped pair)
            image_current_global: [B, 3, 256, 256]  raw_t (DINO teacher view + multi-crop source)
            image_future_global:  [B, 3, 256, 256]  raw_tk
                                  global이 None이면 V-JEPA / DINO loss 모두 0 (eval mode 호환).

        Returns dict 주요 키:
            loss, loss_t, loss_tk, loss_pred, loss_dino
            pred_t, pred_tk             ([B, 3, H, W] visualization)
            mask_p, mask_m              (mask_m은 zeros placeholder — v14는 M unmasked)
            m_features, p_features_t, p_features_tk, cls_m, cls_p (v11 호환)
            student_dino_cls, teacher_dino_cls, teacher_proto_logits (DINO 진단)
            predicted_tk_repr, target_tk_repr (V-JEPA 진단)
        """
        # ── Rotation aug — global view도 함께 동일 회전 ────────────────────
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

        # ── Channel preprocessing ──────────────────────────────────────────
        # Cropped pair: M_channel(crop_t, crop_tk) + P_channel(crop_t) + P_channel(crop_tk)
        m_channel_cropped, p_channel_t = self.preprocessing(image_current, image_future)
        p_channel_tk = self.preprocessing.compute_p_channel(image_future)

        # ── Track 1+2: MAE on P stream (masked, recon decoder 공유) ───────
        mask_p_t = self._random_mask(B, device, self.mask_ratio_p)
        mask_p_tk = self._random_mask(B, device, self.mask_ratio_p)

        p_t_visible = self._student_p_encode_visible(p_channel_t, mask_p_t)
        p_tk_visible = self._student_p_encode_visible(p_channel_tk, mask_p_tk)
        cls_p_repr = p_t_visible[:, 0]  # downstream / probing 호환 (v11 동일)

        p_t_full = self._build_full_seq(p_t_visible, mask_p_t)
        p_tk_full = self._build_full_seq(p_tk_visible, mask_p_tk)

        patch_pred_t = self._decode_for_recon(p_t_full)
        patch_pred_tk_recon = self._decode_for_recon(p_tk_full)

        patch_target_t = self._patchify(image_current)
        patch_target_tk = self._patchify(image_future)
        err_t = ((patch_pred_t - patch_target_t) ** 2).mean(dim=-1)
        err_tk = ((patch_pred_tk_recon - patch_target_tk) ** 2).mean(dim=-1)
        denom_t = mask_p_t.float().sum().clamp(min=1.0)
        denom_tk = mask_p_tk.float().sum().clamp(min=1.0)
        loss_t = (err_t * mask_p_t.float()).sum() / denom_t
        loss_tk_recon = (err_tk * mask_p_tk.float()).sum() / denom_tk

        # ── Track 3: V-JEPA on P stream (unmasked, encoder space alignment) ─
        # Student: P_encoder(crop_t, unmasked) → motion routing → interpreter_2 → predicted_tk_repr
        # Teacher: Teacher_P_encoder(crop_tk, unmasked) → target_tk_repr (sg)
        p_t_repr_unmasked = self._encode_p_unmasked(p_channel_t)             # [B, 1+N, D]
        m_local_unmasked = self._encode_m_unmasked(m_channel_cropped)        # [B, 1+N, D]
        cls_m_repr = m_local_unmasked[:, 0]                                  # M CLS (DINO student와 share)
        m_completed = self._decode_m_unmasked(m_local_unmasked)              # [B, 1+N, D]

        # Motion routing (Q,K from m_completed, V from p_t_repr — v11 novelty 유지)
        p_state = p_t_repr_unmasked
        for routing_block in self.motion_routing:
            p_state = routing_block(p_state, m_completed)
        predicted_tk_repr = self._run_interpreter(
            p_state, self.interpreter_2, self.interpreter_2_norm,
        )                                                                    # [B, 1+N, D]

        if image_future_global is not None:
            with torch.no_grad():
                target_tk_repr = self.teacher_p.forward_unmasked(p_channel_tk).detach()
            # SmoothL1 over CLS + patches (encoder-level representation 정합)
            loss_pred = F.smooth_l1_loss(
                predicted_tk_repr.float(), target_tk_repr.float(), reduction="mean",
            )
        else:
            target_tk_repr = predicted_tk_repr.detach()
            loss_pred = torch.zeros((), device=device, dtype=loss_t.dtype)

        # ── Track 4: DINO on M stream (multi-crop student, raw teacher) ────
        if image_current_global is not None and image_future_global is not None:
            # Teacher: M_channel(raw_t, raw_tk) → Teacher_M_encoder → CLS → Teacher_DINOHead
            with torch.no_grad():
                m_channel_raw = self.preprocessing.compute_m_channel(
                    image_current_global, image_future_global,
                )
                cls_m_global = self.teacher_m.forward_global_cls(m_channel_raw)        # [B, D]
                teacher_proto_logits = self.teacher_m.dino_head(cls_m_global).detach()  # [B, K]

            # Student multi-crop (Option B):
            # crop 0 = 표준 학습 input pair → 이미 cls_m_repr 있음 (m_local_unmasked share)
            student_logits_list = [self.dino_head(cls_m_repr)]
            for _ in range(1, self.dino_n_crop):
                # 추가 cropped pair: raw 256에서 GPU random crop (frame당 독립 좌표)
                crop_t_n, crop_tk_n = self._gpu_random_crop_pair(
                    image_current_global, image_future_global, self.image_size,
                )
                m_channel_n = self.preprocessing.compute_m_channel(crop_t_n, crop_tk_n)
                m_local_n = self._encode_m_unmasked(m_channel_n)
                student_logits_list.append(self.dino_head(m_local_n[:, 0]))

            loss_dino = sum(
                dino_distillation_loss(
                    s, teacher_proto_logits, self.dino_center,
                    student_temp=self.dino_student_temp,
                    teacher_temp=self.dino_teacher_temp,
                )
                for s in student_logits_list
            ) / len(student_logits_list)

            student_dino_cls = cls_m_repr
            teacher_dino_cls = cls_m_global
        else:
            loss_dino = torch.zeros((), device=device, dtype=loss_t.dtype)
            teacher_proto_logits = torch.zeros(B, self.num_prototypes, device=device)
            student_dino_cls = cls_m_repr
            teacher_dino_cls = torch.zeros(B, self.embed_dim, device=device)

        # ── Total ──────────────────────────────────────────────────────────
        loss = (
            loss_t
            + loss_tk_recon
            + self.lambda_pred * loss_pred
            + self.lambda_dino * loss_dino
        )

        pred_t_img = self._unpatchify(patch_pred_t)
        pred_tk_img = self._unpatchify(patch_pred_tk_recon)

        return {
            "loss": loss,
            "loss_t": loss_t,
            "loss_tk": loss_tk_recon,
            "loss_pred": loss_pred,
            "loss_dino": loss_dino,
            "pred_t": pred_t_img,
            "pred_tk": pred_tk_img,
            "mask_p": mask_p_t,
            # v14는 M unmasked. v11/v13 호환을 위한 placeholder.
            "mask_m": torch.zeros(B, self.num_patches, device=device, dtype=torch.bool),
            # Downstream features (v11 호환 키 유지)
            "m_features": m_completed[:, 1:],            # [B, N, D]
            "p_features_t": p_t_visible[:, 1:],          # [B, N_vis, D] MAE visible (v11 호환)
            "p_features_tk": predicted_tk_repr[:, 1:],   # [B, N, D] V-JEPA prediction (paired)
            "cls_m": cls_m_repr,
            "cls_p": cls_p_repr,
            # v14 진단
            "predicted_tk_repr": predicted_tk_repr,      # [B, 1+N, D]
            "target_tk_repr": target_tk_repr,            # [B, 1+N, D]
            "student_dino_cls": student_dino_cls,        # [B, D]
            "teacher_dino_cls": teacher_dino_cls,        # [B, D]
            "teacher_proto_logits": teacher_proto_logits,  # [B, K]  (DINO center update에 사용)
        }

    def compute_loss(
        self,
        image_current: torch.Tensor,
        image_future: torch.Tensor,
        image_current_global: torch.Tensor = None,
        image_future_global: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(
            image_current, image_future, image_current_global, image_future_global,
        )
        return out["loss"], out["pred_tk"]
