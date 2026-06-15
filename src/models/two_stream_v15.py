"""
Two-Stream v15 final: Layered specialization with compositional auxiliary + 3-frame triple training.

v14 학습 진단(EgoDex action probing 악화) + v15 sanity 1차/2차 (DINO collapse, L_pred 단조 증가)
이후 redesign. 본 학습 진입 전 design spec → v15 final로 통합.

핵심 변경 (v15 초안 → v15 final, 사용자 spec `docs/v15_compositional_aux_design.md`):
  1. DINO 제거 → L_compose 추가 (compositional structure on M_encoder)
     - DINO 정당화 약함 (sanity v1/v2에서 student CLS uniform collapse 확정)
     - L_compose는 v16 chunk-level inference의 prerequisite
       (M_encoder가 anchor-relative motion latent의 algebraic 가산성 학습)
  2. V-JEPA-M target = TeacherM_encoder only (Option B, V-JEPA strict)
     - 기존: encoder + decoder unmasked 통과 → 변경: encoder만
     - student M_decoder는 V-JEPA-M Predictor 단일 역할
  3. Motion routing input = M_encoder unmasked output (v15 초안 그대로)
  4. Input pipeline: cropped triple (frame_t, frame_t+n, frame_t+m)
  5. EMA copy 4개 → 2개 (TeacherP + TeacherM_encoder만)
  6. **3-frame 확장 (옵션 B)**: P MAE 3 frame + V-JEPA P 3 segment + V-JEPA M 1 segment(long) + L_compose

Loss:
    L_total = L_t + L_tn + L_tm                   ← MAE on 3 frames
            + λ_pred · mean(L_pred_short, L_pred_step, L_pred_long)   ← V-JEPA P × 3 segment
            + λ_m_jepa · L_m_jepa                                     ← V-JEPA M × 1 (long, Option B)
            + λ_compose · L_compose                                   ← compositional structure

3-frame motion segments:
    short : crop_t   → crop_t+n   (~0.5s, range 1~30 frame, triangular center=15)
    step  : crop_t+n → crop_t+m   (~0.5s)
    long  : crop_t   → crop_t+m   (1~2s, 합성 = short + step)

L_compose 핵심 가설 (motion field additivity):
    M_encoder(M_chan(t,t+m)) ≈ M_encoder(M_chan(t,t+n)) + M_encoder(M_chan(t+n,t+m))
    → 학습 목표: m_long − m_short ≈ m_target (= M_encoder(M_chan(t+n, t+m)))
    composition_head:
      Phase A.1 (sanity): "linear_residual" — m_predicted = m_long − m_short (parameter 0)
      Phase A.2 / 본 학습:  "mlp" / "linear" — capacity 추가
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .two_stream import TransformerBlock
from .two_stream_v11 import TwoStreamV11Model, MotionRoutingBlock


# ============================================================================
# Routing-Interpreter step (P motion decoder unit)
# ============================================================================

class RoutingInterpreterStep(nn.Module):
    """One step of P motion decoder.

    routing(M→P) cross-attention + interpreter self-attention. interleaved 패턴.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, routing_mode: str):
        super().__init__()
        self.routing = MotionRoutingBlock(
            embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            routing_mode=routing_mode,
        )
        self.interp = TransformerBlock(
            embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
        )

    def forward(self, p_state: torch.Tensor, m_local: torch.Tensor) -> torch.Tensor:
        p_state = self.routing(p_state, m_local)
        p_state = self.interp(p_state, freqs_cis=None)
        return p_state


# ============================================================================
# Composition head — replaces DINO
# ============================================================================

class CompositionHead(nn.Module):
    """L_compose: (m_short, m_long) → m_predicted in M_encoder space.

    3 mode 사다리:
      - "linear_residual": parameter 0. m_long − m_short. M_encoder가 직접 가산성 학습 강제.
      - "linear":          weight 1개. concat 후 linear projection.
      - "mlp":             shallow MLP (1 hidden). 본 학습용 capacity.
    """

    def __init__(self, embed_dim: int, mode: str = "linear_residual",
                 hidden_dim: Optional[int] = None):
        super().__init__()
        self.mode = mode
        if mode == "linear_residual":
            pass  # no parameters
        elif mode == "linear":
            self.proj = nn.Linear(2 * embed_dim, embed_dim)
        elif mode == "mlp":
            hidden_dim = hidden_dim or embed_dim
            self.net = nn.Sequential(
                nn.Linear(2 * embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:
            raise ValueError(f"Unknown composition mode: {mode}")

    def forward(self, m_short: torch.Tensor, m_long: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear_residual":
            return m_long - m_short
        x = torch.cat([m_short, m_long], dim=-1)
        if self.mode == "linear":
            return self.proj(x)
        return self.net(x)


# ============================================================================
# Teachers — v15 final: encoder only EMA copies (decoder / dino_head 제거)
# ============================================================================

class TeacherPv15(nn.Module):
    """EMA copy of student P encoder.
    V-JEPA P target 제공 (frame_tk 정답지, stop-grad). 2026-05-27 변경 전에는
    anchor(frame_t)도 teacher였으나(predictor-only), 이제 anchor는 student P encoder가
    인코딩 → P encoder가 V-JEPA gradient를 받음 (표준 V-JEPA)."""

    def __init__(self, student: "TwoStreamV15Model"):
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
        B = p_channel.shape[0]
        patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_p.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1) + self.pos_embed_p
        for block in self.blocks_p:
            x = block(x, freqs_cis=None)
        return self.norm_p(x)

    @torch.no_grad()
    def update(self, student: "TwoStreamV15Model", momentum: float):
        for p_t, p_s in zip(self.patch_embed_p.parameters(), student.patch_embed_p.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        for p_t, p_s in zip(self.blocks_p.parameters(), student.blocks_p.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        for p_t, p_s in zip(self.norm_p.parameters(), student.norm_p.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        self.cls_token_p.data.mul_(momentum).add_(student.cls_token_p.data, alpha=1.0 - momentum)
        self.pos_embed_p.data.mul_(momentum).add_(student.pos_embed_p.data, alpha=1.0 - momentum)


class TeacherMv15(nn.Module):
    """EMA copy of student M encoder ONLY (decoder/dino_head 없음 — Option B + L_compose target용).

    forward_unmasked_encoder_only(): V-JEPA-M target + L_compose target 둘 다 제공.
    """

    def __init__(self, student: "TwoStreamV15Model"):
        super().__init__()
        self.patch_embed_m = copy.deepcopy(student.patch_embed_m)
        self.cls_token_m = nn.Parameter(student.cls_token_m.data.clone())
        self.pos_embed_m = nn.Parameter(student.pos_embed_m.data.clone())
        self.blocks_m = copy.deepcopy(student.blocks_m)
        self.norm_m = copy.deepcopy(student.norm_m)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward_unmasked_encoder_only(self, m_channel: torch.Tensor) -> torch.Tensor:
        """[B, 3, 224, 224] → [B, 1+N, D] (M_encoder only, no decoder)."""
        B = m_channel.shape[0]
        patches = self.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_m.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1) + self.pos_embed_m
        for block in self.blocks_m:
            x = block(x, freqs_cis=None)
        return self.norm_m(x)

    @torch.no_grad()
    def update(self, student: "TwoStreamV15Model", momentum: float):
        for p_t, p_s in zip(self.patch_embed_m.parameters(), student.patch_embed_m.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        for p_t, p_s in zip(self.blocks_m.parameters(), student.blocks_m.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        for p_t, p_s in zip(self.norm_m.parameters(), student.norm_m.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        self.cls_token_m.data.mul_(momentum).add_(student.cls_token_m.data, alpha=1.0 - momentum)
        self.pos_embed_m.data.mul_(momentum).add_(student.pos_embed_m.data, alpha=1.0 - momentum)


# ============================================================================
# v15 Model
# ============================================================================

class TwoStreamV15Model(TwoStreamV11Model):
    """Two-Stream v15 final.

    Args (v15 specific):
        lambda_pred:    V-JEPA P loss weight (segment 평균에 적용)
        lambda_m_jepa:  V-JEPA M loss weight (long segment, Option B)
        lambda_compose: Composition loss weight (L_compose)
        mask_ratio_m_jepa:  M stream mask ratio for V-JEPA-M
        composition_mode:   "linear_residual" / "linear" / "mlp"
        composition_hidden_dim: mlp mode hidden dim
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
        use_sobel: bool = True,
        # v15 args
        lambda_pred: float = 1.0,
        lambda_m_jepa: float = 1.0,
        lambda_compose: float = 1.0,
        mask_ratio_m_jepa: float = 0.5,
        composition_mode: str = "linear_residual",
        composition_hidden_dim: Optional[int] = None,
        use_compose: bool = True,
        pair_mode: bool = False,
        lambda_ssim: float = 0.02,
        lambda_var: float = 0.0,
        target_ln: bool = False,
    ):
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
            use_sobel=use_sobel,
        )

        self.lambda_pred = lambda_pred
        self.lambda_m_jepa = lambda_m_jepa
        self.lambda_compose = lambda_compose
        self.mask_ratio_m_jepa = mask_ratio_m_jepa
        # P MAE 복구 loss에 SSIM term 보강 (mse + λ_ssim·(1-SSIM)).
        # MSE는 패치 평균색만 맞춰도 낮아 형상/고주파 collapse에 둔감 → 구버전 two-stream
        # (v4~v10)처럼 SSIM으로 구조 보존 압력 추가. lambda_ssim=0이면 비활성(MSE only).
        self.lambda_ssim = lambda_ssim
        # Run A anti-collapse (2026-06-15): V-JEPA P 자기참조 constant collapse 방어.
        #   target_ln=True: target(teacher) repr에 LayerNorm(affine 없음) — I-JEPA/V-JEPA 표준.
        #     predictor 출력은 이미 p_motion_decoder_norm 통과 → 양쪽 scale 정합, magnitude 붕괴 차단.
        #   lambda_var>0: VICReg식 variance reg — P encoder 출력 per-dim std<1에 hinge 패널티로
        #     std collapse(우리가 본 균일 붕괴)를 직접 금지. 둘 다 기본 off라 기존 동작 불변.
        self.lambda_var = lambda_var
        self.target_ln = target_ln
        # pair_mode=True: 2-frame (t, t+k), L_compose 제거, V-JEPA P/M 단일 segment.
        #   compose 미입증 → 배제하고 motion routing만 검증. composition_head 미생성
        #   (DDP unused-param 회피). 3-frame ckpt resume 시 strict=False로 로드.
        self.pair_mode = pair_mode
        self.use_compose = use_compose and not pair_mode

        # ── p_motion_decoder = (routing + interp) × N (interleaved) ──────
        self.p_motion_decoder = nn.ModuleList([
            RoutingInterpreterStep(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                routing_mode=routing_mode,
            )
            for _ in range(num_motion_iters)
        ])
        self.p_motion_decoder_norm = nn.LayerNorm(embed_dim)

        # ── Composition head (NEW, replaces DINO) ────────────────────────
        # pair_mode(2-frame)에선 L_compose 제거 → composition_head 미생성 (DDP unused-param 회피).
        if self.use_compose:
            self.composition_head = CompositionHead(
                embed_dim=embed_dim,
                mode=composition_mode,
                hidden_dim=composition_hidden_dim,
            )

        # Frozen v11 modules (DDP unused-param 회피)
        for p in self.motion_routing.parameters():
            p.requires_grad_(False)
        for p in self.interpreter_2.parameters():
            p.requires_grad_(False)
        for p in self.interpreter_2_norm.parameters():
            p.requires_grad_(False)

        # ── Teachers ─────────────────────────────────────────────────────
        self.teacher_p = TeacherPv15(self)
        self.teacher_m = TeacherMv15(self)

        # mask_token_m 활성화 (V-JEPA-M에서 학습)
        self.mask_token_m.requires_grad_(True)

    # ----------------------------------------------------------------------
    # EMA update
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        self.teacher_p.update(self, momentum)
        self.teacher_m.update(self, momentum)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _encode_p_unmasked(self, p_channel: torch.Tensor) -> torch.Tensor:
        B = p_channel.shape[0]
        patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_p.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1) + self.pos_embed_p
        for block in self.blocks_p:
            x = block(x, freqs_cis=None)
        return self.norm_p(x)

    def _encode_m_unmasked(self, m_channel: torch.Tensor) -> torch.Tensor:
        B = m_channel.shape[0]
        patches = self.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_m.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1) + self.pos_embed_m
        for block in self.blocks_m:
            x = block(x, freqs_cis=None)
        return self.norm_m(x)

    def _encode_m_masked(self, m_channel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = m_channel.shape[0]
        patches = self.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_m.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos_embed_m
        return self._encode_stream_visible(tokens, mask, self.blocks_m, self.norm_m)

    def _decode_m(self, m_full: torch.Tensor) -> torch.Tensor:
        x = m_full + self.dec_pos_embed_m
        for block in self.m_decoder_blocks:
            x = block(x, freqs_cis=None)
        return self.m_decoder_norm(x)

    def _student_p_encode_visible(
        self, p_channel: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        B = p_channel.shape[0]
        patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_p.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos_embed_p
        return self._encode_stream_visible(tokens, mask, self.blocks_p, self.norm_p)

    def _build_full_seq_p(self, p_visible: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p_full = self._inject_mask_tokens(p_visible, mask, self.mask_token_p)
        return p_full + self.dec_pos_embed_p

    def _build_full_seq_m(self, m_visible: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._inject_mask_tokens(m_visible, mask, self.mask_token_m)

    def _decode_for_recon(self, p_full: torch.Tensor) -> torch.Tensor:
        p_decoded = self._run_interpreter(
            p_full, self.interpreter_1, self.interpreter_1_norm,
        )
        return self.recon_head(p_decoded[:, 1:])

    def _mae_one_frame(
        self,
        image: torch.Tensor, p_channel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-frame MAE: returns (loss, patch_pred, mask, p_visible)."""
        B = image.shape[0]
        device = image.device
        mask = self._random_mask(B, device, self.mask_ratio_p)
        p_visible = self._student_p_encode_visible(p_channel, mask)
        p_full = self._build_full_seq_p(p_visible, mask)
        patch_pred = self._decode_for_recon(p_full)
        patch_target = self._patchify(image)
        err = ((patch_pred - patch_target) ** 2).mean(dim=-1)
        denom = mask.float().sum().clamp(min=1.0)
        loss = (err * mask.float()).sum() / denom
        # SSIM 보강: 형상/구조 보존 압력 (MSE는 평균색에 둔감 → collapse 미검출).
        # 지연 import로 순환 import 회피 (training.pretrain → models.two_stream).
        if self.lambda_ssim > 0:
            from src.training.pretrain import ssim_loss
            recon_img = self._unpatchify(patch_pred)
            loss = loss + self.lambda_ssim * ssim_loss(recon_img, image)
        return loss, patch_pred, mask, p_visible

    def _vjepa_p_one_segment(
        self,
        m_channel: torch.Tensor,
        p_channel_anchor: torch.Tensor,
        p_channel_target: torch.Tensor,
        m_local_routing: Optional[torch.Tensor] = None,
        anchor_repr_S: Optional[torch.Tensor] = None,
        target_repr_T: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """V-JEPA P for one motion segment.

        anchor → routing → predicted target_repr.
        m_local_routing: 미리 계산한 routing K/Q source (m_channel에서 student M_encoder unmasked).
                         None이면 함수가 직접 계산.
        anchor_repr_S / target_repr_T: forward에서 unique frame당 1회 인코딩한 결과를 전달받아
                         중복 unmasked P-encoder forward 제거 (무손실 최적화). None이면 직접 계산.
        Returns (loss, predicted_tk_repr, target_tk_repr).
        """
        if m_local_routing is None:
            m_local_routing = self._encode_m_unmasked(m_channel)

        # 변경 (2026-05-27): anchor를 teacher → STUDENT P encoder로 (표준 V-JEPA 복원).
        # 기존: anchor=teacher_p(frame_t).detach() → P encoder가 V-JEPA gradient 못 받음
        #       (predictor-only V-JEPA). P encoder = MAE only.
        # 현재: anchor=student P encoder(frame_t), grad ON → P encoder가 motion routing
        #       gradient를 받아 motion-predictable representation 학습 (catalyst 의도 복원).
        #   - anchor (context) = student P encoder(frame_t)
        #   - target (정답지)   = teacher_p(frame_tk).detach() (EMA, stop-grad)
        #   - predictor (p_motion_decoder) + M routing이 student anchor → teacher target 예측
        # dropout/droppath 없음 → unmasked forward는 deterministic. 같은 frame을 여러 segment가
        # 공유하면 forward에서 1회만 인코딩해 재사용 (gradient 합산 동치, 무손실).
        if anchor_repr_S is None:
            anchor_repr_S = self._encode_p_unmasked(p_channel_anchor)  # student frame_t, grad ON
        if target_repr_T is None:
            with torch.no_grad():
                target_repr_T = self.teacher_p.forward_unmasked(p_channel_target).detach()

        p_state = anchor_repr_S
        for step in self.p_motion_decoder:
            p_state = step(p_state, m_local_routing)
        predicted_tk_repr = self.p_motion_decoder_norm(p_state)
        # Run A: target LayerNorm(affine 없음). 정답지(EMA teacher)가 작아져 이기는 scale 붕괴 차단.
        # predicted는 이미 p_motion_decoder_norm(LayerNorm) 통과 → 양쪽 정규화로 scale 정합.
        target_for_loss = (
            F.layer_norm(target_repr_T.float(), (target_repr_T.shape[-1],))
            if self.target_ln else target_repr_T.float()
        )
        loss = F.smooth_l1_loss(
            predicted_tk_repr.float(), target_for_loss, reduction="mean",
        )
        return loss, predicted_tk_repr, target_repr_T

    @staticmethod
    def _variance_loss(z: torch.Tensor) -> torch.Tensor:
        """VICReg variance term: per-dim batch std<1에 hinge 패널티 → std collapse 직접 금지.

        z: [B, N, D] (CLS 제외 patch tokens 권장). 토큰을 배치 차원으로 펴서 차원별 std 계산.
        """
        z = z.reshape(-1, z.shape[-1]).float()
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1.0 - std))

    # ----------------------------------------------------------------------
    # Forward (pair) — 2-frame, no L_compose
    # ----------------------------------------------------------------------

    def _forward_pair(self, image_current: torch.Tensor, image_future: torch.Tensor) -> dict:
        """2-frame Parvo forward (t, t+k). L_compose 제거, V-JEPA P/M 단일 segment.

        Loss = (L_t + L_tk) + λ_pred·L_pred + λ_m_jepa·L_m_jepa   (compose 없음)
        """
        if self.rotation_aug and self.training:
            if torch.rand(1).item() < self.independent_rotation_prob:
                ks = [torch.randint(0, 4, (1,)).item() for _ in range(2)]
            else:
                k0 = torch.randint(0, 4, (1,)).item()
                ks = [k0, k0]
            if ks[0] != 0:
                image_current = torch.rot90(image_current, ks[0], dims=(2, 3))
            if ks[1] != 0:
                image_future = torch.rot90(image_future, ks[1], dims=(2, 3))

        B = image_current.shape[0]
        device = image_current.device

        p_channel_t = self.preprocessing.compute_p_channel(image_current)
        p_channel_tk = self.preprocessing.compute_p_channel(image_future)
        m_chan = self.preprocessing.compute_m_channel(image_current, image_future)  # t → t+k

        # 1. P MAE × 2 frame
        loss_t, patch_pred_t, mask_p_t, p_t_visible = self._mae_one_frame(image_current, p_channel_t)
        loss_tk, patch_pred_tk, mask_p_tk, p_tk_visible = self._mae_one_frame(image_future, p_channel_tk)
        cls_p_repr = p_t_visible[:, 0]

        # 2. M encoder unmasked (routing source)
        m_local = self._encode_m_unmasked(m_chan)

        # 3. V-JEPA P × 1 (t → t+k): student anchor + M routing → teacher target
        loss_pred, predicted_repr, target_repr = self._vjepa_p_one_segment(
            m_chan, p_channel_t, p_channel_tk, m_local_routing=m_local,
        )

        # 4. V-JEPA M × 1 (Option B): M masked + decoder vs TeacherM unmasked
        mask_m = self._random_mask(B, device, self.mask_ratio_m_jepa)
        m_visible = self._encode_m_masked(m_chan, mask_m)
        m_full = self._build_full_seq_m(m_visible, mask_m)
        m_decoded_masked = self._decode_m(m_full)
        with torch.no_grad():
            m_target_encoded = self.teacher_m.forward_unmasked_encoder_only(m_chan).detach()
        err_m = F.smooth_l1_loss(
            m_decoded_masked[:, 1:].float(), m_target_encoded[:, 1:].float(), reduction="none",
        ).mean(dim=-1)
        denom_m = mask_m.float().sum().clamp(min=1.0)
        loss_m_jepa = (err_m * mask_m.float()).sum() / denom_m

        # Run A variance reg: P encoder 출력(frame_t visible patches)의 per-dim std collapse 금지.
        zero = torch.zeros((), device=device, dtype=loss_t.dtype)
        loss_var = self._variance_loss(p_t_visible[:, 1:]) if self.lambda_var > 0 else zero

        # Total (no compose)
        loss = (loss_t + loss_tk) + self.lambda_pred * loss_pred \
            + self.lambda_m_jepa * loss_m_jepa + self.lambda_var * loss_var

        cls_m_repr = m_local[:, 0]
        return {
            "loss": loss,
            "loss_t": loss_t, "loss_tn": loss_t, "loss_tk": loss_tk,
            "loss_pred": loss_pred, "loss_pred_short": loss_pred,
            "loss_pred_step": loss_pred, "loss_pred_long": loss_pred,
            "loss_m_jepa": loss_m_jepa, "loss_compose": zero, "loss_dino": zero,
            "loss_var": loss_var,
            "pred_t": self._unpatchify(patch_pred_t), "pred_tk": self._unpatchify(patch_pred_tk),
            "mask_p": mask_p_t, "mask_m": mask_m,
            "m_features": m_local[:, 1:], "p_features_t": p_t_visible[:, 1:],
            "p_features_tk": predicted_repr[:, 1:], "cls_m": cls_m_repr, "cls_p": cls_p_repr,
            "predicted_tk_repr": predicted_repr, "target_tk_repr": target_repr,
            "m_local_short": m_local, "m_local_step": m_local, "m_local_long": m_local,
            "m_compose_target": m_local, "m_predicted": m_local,
            "student_dino_cls": cls_m_repr,
            "teacher_dino_cls": torch.zeros(B, self.embed_dim, device=device, dtype=loss.dtype),
            "teacher_proto_logits": torch.zeros(B, 1, device=device, dtype=loss.dtype),
        }

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def forward(
        self,
        image_current: torch.Tensor,   # crop_t        [B, 3, 224, 224]
        image_short:   torch.Tensor,   # crop_t+n      [B, 3, 224, 224]
        image_future:  torch.Tensor = None,  # crop_t+m [B, 3, 224, 224] (3-frame만)
    ) -> dict:
        """
        v15 final forward (3-frame triple). pair_mode면 _forward_pair(2-frame)로 분기.

        Loss tracks (옵션 B):"""
        if self.pair_mode:
            return self._forward_pair(image_current, image_short)  # image_short = t+k
        _orig_docstring = """
          1. P MAE × 3:           L_t (frame_t), L_tn (frame_t+n), L_tm (frame_t+m)
          2. V-JEPA P × 3 segment: short (t→t+n), step (t+n→t+m), long (t→t+m)
          3. V-JEPA M × 1 (long): masked + decoder vs TeacherM_encoder unmasked (Option B)
          4. L_compose:            m_long − m_short ≈ teacher(m_step)

        Returns dict with all losses + diagnostic features.
        """
        if self.rotation_aug and self.training:
            if torch.rand(1).item() < self.independent_rotation_prob:
                ks = [torch.randint(0, 4, (1,)).item() for _ in range(3)]
            else:
                k0 = torch.randint(0, 4, (1,)).item()
                ks = [k0, k0, k0]
            if ks[0] != 0:
                image_current = torch.rot90(image_current, ks[0], dims=(2, 3))
            if ks[1] != 0:
                image_short = torch.rot90(image_short, ks[1], dims=(2, 3))
            if ks[2] != 0:
                image_future = torch.rot90(image_future, ks[2], dims=(2, 3))

        B = image_current.shape[0]
        device = image_current.device

        # ── Channel preprocessing ──────────────────────────────────────────
        # P channels: 3 frames 각각
        p_channel_t = self.preprocessing.compute_p_channel(image_current)
        p_channel_tn = self.preprocessing.compute_p_channel(image_short)
        p_channel_tm = self.preprocessing.compute_p_channel(image_future)

        # M channels: 3 segments (재사용)
        m_chan_short = self.preprocessing.compute_m_channel(image_current, image_short)   # t → t+n
        m_chan_step = self.preprocessing.compute_m_channel(image_short, image_future)     # t+n → t+m
        m_chan_long = self.preprocessing.compute_m_channel(image_current, image_future)   # t → t+m

        # ── 1. P MAE × 3 frames ────────────────────────────────────────────
        loss_t, patch_pred_t, mask_p_t, p_t_visible = self._mae_one_frame(image_current, p_channel_t)
        loss_tn, _, _, p_tn_visible = self._mae_one_frame(image_short, p_channel_tn)
        loss_tm, patch_pred_tm, mask_p_tm, p_tm_visible = self._mae_one_frame(image_future, p_channel_tm)

        cls_p_repr = p_t_visible[:, 0]  # downstream/probing (anchor frame CLS)

        # ── 2. M_encoder UNMASKED forward × 3 segment (routing source + L_compose) ─
        m_local_short = self._encode_m_unmasked(m_chan_short)
        m_local_step = self._encode_m_unmasked(m_chan_step)
        m_local_long = self._encode_m_unmasked(m_chan_long)

        # ── 3. V-JEPA P × 3 segment (predictor only) ───────────────────────
        # 중복 unmasked P-encoder forward 제거 (무손실): 3 segment의 anchor/target는
        #   anchor(student): t(short,long), t+n(step)   → unique {t, t+n}
        #   target(teacher): t+n(short), t+m(step,long) → unique {t+n, t+m}
        # dropout 없어 deterministic → unique frame당 1회 인코딩해 재사용 (6→4 full P forward).
        p_anchor_t = self._encode_p_unmasked(p_channel_t)    # student, short+long anchor
        p_anchor_tn = self._encode_p_unmasked(p_channel_tn)  # student, step anchor
        with torch.no_grad():
            p_target_tn = self.teacher_p.forward_unmasked(p_channel_tn).detach()  # short target
            p_target_tm = self.teacher_p.forward_unmasked(p_channel_tm).detach()  # step+long target

        loss_pred_short, predicted_repr_short, target_repr_short = self._vjepa_p_one_segment(
            m_chan_short, p_channel_t, p_channel_tn, m_local_routing=m_local_short,
            anchor_repr_S=p_anchor_t, target_repr_T=p_target_tn,
        )
        loss_pred_step, predicted_repr_step, target_repr_step = self._vjepa_p_one_segment(
            m_chan_step, p_channel_tn, p_channel_tm, m_local_routing=m_local_step,
            anchor_repr_S=p_anchor_tn, target_repr_T=p_target_tm,
        )
        loss_pred_long, predicted_repr_long, target_repr_long = self._vjepa_p_one_segment(
            m_chan_long, p_channel_t, p_channel_tm, m_local_routing=m_local_long,
            anchor_repr_S=p_anchor_t, target_repr_T=p_target_tm,
        )
        loss_pred = (loss_pred_short + loss_pred_step + loss_pred_long) / 3.0

        # ── 4. V-JEPA M × 1 (long segment, Option B) ──────────────────────
        # Student: M_encoder masked + M_decoder + mask_token → m_decoded_masked
        # Target:  TeacherM_encoder unmasked (no decoder) → m_target_encoded
        mask_m = self._random_mask(B, device, self.mask_ratio_m_jepa)
        m_visible = self._encode_m_masked(m_chan_long, mask_m)
        m_full = self._build_full_seq_m(m_visible, mask_m)
        m_decoded_masked = self._decode_m(m_full)  # [B, 1+N, D]

        with torch.no_grad():
            m_target_encoded = self.teacher_m.forward_unmasked_encoder_only(m_chan_long).detach()

        student_patches = m_decoded_masked[:, 1:]
        target_patches = m_target_encoded[:, 1:]
        err_m = F.smooth_l1_loss(
            student_patches.float(), target_patches.float(), reduction="none",
        ).mean(dim=-1)
        denom_m = mask_m.float().sum().clamp(min=1.0)
        loss_m_jepa = (err_m * mask_m.float()).sum() / denom_m

        # ── 5. L_compose ──────────────────────────────────────────────────
        with torch.no_grad():
            m_compose_target = self.teacher_m.forward_unmasked_encoder_only(m_chan_step).detach()

        m_predicted = self.composition_head(m_local_short, m_local_long)
        loss_compose = F.smooth_l1_loss(
            m_predicted.float(), m_compose_target.float(), reduction="mean",
        )

        # ── Total loss ────────────────────────────────────────────────────
        loss = (
            loss_t + loss_tn + loss_tm
            + self.lambda_pred * loss_pred
            + self.lambda_m_jepa * loss_m_jepa
            + self.lambda_compose * loss_compose
        )

        # Diagnostic features (downstream/probing 호환 키)
        cls_m_repr = m_local_long[:, 0]  # M encoder CLS (long segment)

        pred_t_img = self._unpatchify(patch_pred_t)
        pred_tk_img = self._unpatchify(patch_pred_tm)

        return {
            "loss": loss,
            "loss_t": loss_t,
            "loss_tn": loss_tn,
            "loss_tk": loss_tm,           # v14 호환 (loss_tk = frame_tm)
            "loss_pred": loss_pred,
            "loss_pred_short": loss_pred_short,
            "loss_pred_step": loss_pred_step,
            "loss_pred_long": loss_pred_long,
            "loss_m_jepa": loss_m_jepa,
            "loss_compose": loss_compose,
            "loss_dino": torch.zeros((), device=device, dtype=loss.dtype),  # v14 호환 (제거)
            "pred_t": pred_t_img,
            "pred_tk": pred_tk_img,
            "mask_p": mask_p_t,
            "mask_m": mask_m,
            # Downstream features (v11/v14 호환)
            "m_features": m_local_long[:, 1:],
            "p_features_t": p_t_visible[:, 1:],
            "p_features_tk": predicted_repr_long[:, 1:],
            "cls_m": cls_m_repr,
            "cls_p": cls_p_repr,
            # v15 진단
            "predicted_tk_repr": predicted_repr_long,
            "target_tk_repr": target_repr_long,
            "m_local_short": m_local_short,
            "m_local_step": m_local_step,
            "m_local_long": m_local_long,
            "m_compose_target": m_compose_target,
            "m_predicted": m_predicted,
            # v14 호환 (DINO 관련 — 제로 placeholder)
            "student_dino_cls": cls_m_repr,
            "teacher_dino_cls": torch.zeros(B, self.embed_dim, device=device, dtype=loss.dtype),
            "teacher_proto_logits": torch.zeros(B, 1, device=device, dtype=loss.dtype),
        }

    def compute_loss(
        self,
        image_current: torch.Tensor,
        image_short: torch.Tensor,
        image_future: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(image_current, image_short, image_future)
        return out["loss"], out["pred_tk"]
