"""
Two-Stream v15: Layered paradigm specialization with predictor-only V-JEPA + masked V-JEPA-M.

v14 학습에서 발견된 design issue:
  · V-JEPA path의 V source가 student P_encoder unmasked → MAE로 변동하는 source
    → motion-routing이 "변동 V를 안정 target에 매핑" 하는 noise canceling 학습
  · M stream은 DINO global CLS만 받음 → patch-level motion 학습 신호 부재
  · DINO target이 M_encoder 직후 CLS → encoder가 motion routing(raw)과
    motion semantic(distillation) 둘 다 만족해야 paradigm conflict
  · routing × N → 한 번의 interpreter_2: routing 누적 후 한꺼번에 정제 →
    누적이 noise 누적이 될 위험 (V-JEPA 2 표준은 interleaved cross+self attn)

v15 변경 (5가지):
  1. L_pred V source: TeacherP unmasked (stable). predictor-only learning
     (V-JEPA 2 표준의 한 발 비켜선 형태 — context encoder도 teacher).
     Gradient flow: M_encoder(routing K/Q) + p_motion_decoder만. P_encoder는 MAE only.

  2. p_motion_decoder = (routing_block + interpreter_block) × N (interleaved)
     매 routing 후 즉시 interpreter로 정제 → 다음 routing의 V가 정제된 representation.

  3. M_encoder unmasked forward separately for routing K/Q source — clean full grid.
     (옵션 C: M stream masking은 V-JEPA-M에서만)

  4. V-JEPA-M (NEW): M_encoder masked + M_decoder + mask_token_m → m_completed_masked.
     Target = TeacherM(encoder+decoder, unmasked) → m_completed_target.
     L_m_jepa = SmoothL1 over masked patches only (CLS 제외 — DINO와 conflict 회피).

  5. DINO target/student level: M_decoder 후 CLS (motion semantic).
     M_encoder = raw motion field (routing source), M_decoder = motion semantic refinement.

Loss:
    L_total = L_t + L_tk                     ← MAE (P stream, recon decoder)
            + λ_pred  · L_pred               ← V-JEPA P (predictor only, encoder-level)
            + λ_m_jepa · L_m_jepa            ← V-JEPA M (masked patches, CLS 제외)
            + λ_dino  · L_dino               ← DINO M (M_decoder 후 CLS only)

Modules:
  - student P_encoder, student M_encoder, student M_decoder
  - student p_motion_decoder = (routing + interp) × N
  - student recon_decoder = interpreter_1 (depth=3) + recon_head
  - student dino_head, mask_token_m (활성화), mask_token_p
  - TeacherPv15 (P_encoder EMA copy)
  - TeacherMv15 (M_encoder + M_decoder + dino_head EMA copy)
  - dino_center buffer

mask_token_m: v14에서 frozen이었으나 v15는 V-JEPA-M에 사용 → 학습됨.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .two_stream import TransformerBlock
from .two_stream_v11 import TwoStreamV11Model, MotionRoutingBlock
from .two_stream_v13 import (
    DINOHead,
    dino_distillation_loss,
    interpolate_pos_embed_2d,
)


# ============================================================================
# Routing-Interpreter step (P motion decoder unit)
# ============================================================================

class RoutingInterpreterStep(nn.Module):
    """One step of P motion decoder.

    routing(M→P) + interpreter(self-attn on P state).
    routing은 cross-attention(Q,K from M, V from P) — main entity는 P state.
    interpreter는 P self-attention — 즉시 정제.

    이 unit을 N번 반복 → V-JEPA 2 표준의 interleaved cross+self attn 패턴.
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
# Teachers
# ============================================================================

class TeacherPv15(nn.Module):
    """EMA copy of student P encoder. V-JEPA P의 V source + target 모두 제공 (stable)."""

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
        """[B, 5, 224, 224] → [B, 1+N, D] (CLS + all patches)."""
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
    """EMA copy of student M_encoder + M_decoder + dino_head.
    V-JEPA-M target (encoder+decoder unmasked) + DINO target (raw 256, encoder+decoder).
    """

    def __init__(self, student: "TwoStreamV15Model", patch_size: int):
        super().__init__()
        self.patch_embed_m = copy.deepcopy(student.patch_embed_m)
        self.cls_token_m = nn.Parameter(student.cls_token_m.data.clone())
        self.pos_embed_m = nn.Parameter(student.pos_embed_m.data.clone())
        self.blocks_m = copy.deepcopy(student.blocks_m)
        self.norm_m = copy.deepcopy(student.norm_m)
        # M decoder: dec_pos_embed_m + decoder_blocks + decoder_norm
        self.dec_pos_embed_m = nn.Parameter(student.dec_pos_embed_m.data.clone())
        self.m_decoder_blocks = copy.deepcopy(student.m_decoder_blocks)
        self.m_decoder_norm = copy.deepcopy(student.m_decoder_norm)
        # DINO head
        self.dino_head = copy.deepcopy(student.dino_head)
        self.patch_size = patch_size
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _encode_unmasked(self, m_channel: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        B = m_channel.shape[0]
        patches = self.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_m.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1) + pos_embed
        for block in self.blocks_m:
            x = block(x, freqs_cis=None)
        return self.norm_m(x)

    @torch.no_grad()
    def _decode_unmasked(self, m_encoded: torch.Tensor, dec_pos_embed: torch.Tensor) -> torch.Tensor:
        x = m_encoded + dec_pos_embed
        for block in self.m_decoder_blocks:
            x = block(x, freqs_cis=None)
        return self.m_decoder_norm(x)

    @torch.no_grad()
    def forward_unmasked_full(self, m_channel: torch.Tensor) -> torch.Tensor:
        """[B, 3, 224, 224] → m_completed [B, 1+N, D] (encoder + decoder, V-JEPA-M target)."""
        m_local = self._encode_unmasked(m_channel, self.pos_embed_m)
        m_completed = self._decode_unmasked(m_local, self.dec_pos_embed_m)
        return m_completed

    @torch.no_grad()
    def forward_global_full(self, m_channel_raw: torch.Tensor) -> torch.Tensor:
        """[B, 3, 256, 256] → m_completed_global [B, 1+N_g, D] (encoder + decoder, DINO target).

        pos_embed_m + dec_pos_embed_m 모두 bicubic interpolate to raw grid.
        """
        H = m_channel_raw.shape[-1]
        target_grid = H // self.patch_size
        pos_embed_g = interpolate_pos_embed_2d(self.pos_embed_m, target_grid)
        dec_pos_embed_g = interpolate_pos_embed_2d(self.dec_pos_embed_m, target_grid)
        m_local = self._encode_unmasked(m_channel_raw, pos_embed_g)
        m_completed = self._decode_unmasked(m_local, dec_pos_embed_g)
        return m_completed

    @torch.no_grad()
    def update(self, student: "TwoStreamV15Model", momentum: float):
        def _ema(t_params, s_params):
            for p_t, p_s in zip(t_params, s_params):
                p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)

        _ema(self.patch_embed_m.parameters(), student.patch_embed_m.parameters())
        _ema(self.blocks_m.parameters(), student.blocks_m.parameters())
        _ema(self.norm_m.parameters(), student.norm_m.parameters())
        _ema(self.m_decoder_blocks.parameters(), student.m_decoder_blocks.parameters())
        _ema(self.m_decoder_norm.parameters(), student.m_decoder_norm.parameters())
        _ema(self.dino_head.parameters(), student.dino_head.parameters())
        self.cls_token_m.data.mul_(momentum).add_(student.cls_token_m.data, alpha=1.0 - momentum)
        self.pos_embed_m.data.mul_(momentum).add_(student.pos_embed_m.data, alpha=1.0 - momentum)
        self.dec_pos_embed_m.data.mul_(momentum).add_(student.dec_pos_embed_m.data, alpha=1.0 - momentum)


# ============================================================================
# v15 Model
# ============================================================================

class TwoStreamV15Model(TwoStreamV11Model):
    """Two-Stream v15: Layered paradigm specialization with V-JEPA-M + interleaved p_motion_decoder.

    v11 base 위에 추가/변경:
      - p_motion_decoder = (routing + interpreter) × N (interleaved). 기존 v11
        motion_routing + interpreter_2는 미사용 (frozen).
      - V-JEPA P: V source + target 모두 TeacherP에서 옴 (predictor only learning)
      - V-JEPA M: M_encoder masked + M_decoder + mask_token → m_completed_masked,
                   target = TeacherM unmasked encoder + decoder. patches[masked]에 SmoothL1.
      - DINO target/student: M_decoder 후 CLS (motion semantic).
      - mask_token_m 활성화 (V-JEPA-M에 사용).

    Args (v15 specific):
        lambda_pred: V-JEPA P loss weight
        lambda_m_jepa: V-JEPA M loss weight
        lambda_dino: DINO loss weight
        mask_ratio_m_jepa: V-JEPA-M에서 M stream mask ratio (default 0.5)
        dino_n_crop: DINO student multi-crop count (224² 페어 N개; sanity=1, 본 학습=3~4)
        num_prototypes / dino_*: DINO hyperparameters (v14 동일)
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
        # v15 args
        lambda_pred: float = 1.0,
        lambda_m_jepa: float = 1.0,
        lambda_dino: float = 1.0,
        mask_ratio_m_jepa: float = 0.5,
        dino_n_crop: int = 1,
        num_prototypes: int = 1024,
        dino_teacher_temp: float = 0.04,
        dino_student_temp: float = 0.1,
        dino_center_momentum: float = 0.9,
    ):
        super().__init__(
            embed_dim=embed_dim,
            p_depth=p_depth,
            m_depth=m_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            image_size=image_size,
            patch_size=patch_size,
            mask_ratio_m=0.0,  # v15는 self.mask_ratio_m_jepa로 별도 관리
            mask_ratio_p=mask_ratio_p,
            decoder_depth_m=decoder_depth_m,
            interpreter_depth=interpreter_depth,
            num_motion_iters=num_motion_iters,
            rotation_aug=rotation_aug,
            independent_rotation_prob=independent_rotation_prob,
            routing_mode=routing_mode,
        )

        self.lambda_pred = lambda_pred
        self.lambda_m_jepa = lambda_m_jepa
        self.lambda_dino = lambda_dino
        self.mask_ratio_m_jepa = mask_ratio_m_jepa
        self.dino_n_crop = dino_n_crop
        self.num_prototypes = num_prototypes
        self.dino_teacher_temp = dino_teacher_temp
        self.dino_student_temp = dino_student_temp
        self.dino_center_momentum = dino_center_momentum

        # ── p_motion_decoder = (routing + interp) × N (interleaved) ────────
        # v11의 self.motion_routing + self.interpreter_2는 v15에서 사용 안 함 (frozen).
        self.p_motion_decoder = nn.ModuleList([
            RoutingInterpreterStep(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                routing_mode=routing_mode,
            )
            for _ in range(num_motion_iters)
        ])
        self.p_motion_decoder_norm = nn.LayerNorm(embed_dim)

        # Frozen v11 modules (DDP unused-param 회피)
        for p in self.motion_routing.parameters():
            p.requires_grad_(False)
        for p in self.interpreter_2.parameters():
            p.requires_grad_(False)
        for p in self.interpreter_2_norm.parameters():
            p.requires_grad_(False)

        # ── DINO Head + center ─────────────────────────────────────────────
        self.dino_head = DINOHead(embed_dim, num_prototypes=num_prototypes)
        self.register_buffer("dino_center", torch.zeros(1, num_prototypes))

        # ── Teachers (encoder + decoder + dino_head 모두 EMA) ──────────────
        self.teacher_p = TeacherPv15(self)
        self.teacher_m = TeacherMv15(self, patch_size=patch_size)

        # mask_token_m 활성화 (V-JEPA-M에서 학습됨)
        self.mask_token_m.requires_grad_(True)

    # ----------------------------------------------------------------------
    # EMA / center update
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        self.teacher_p.update(self, momentum)
        self.teacher_m.update(self, momentum)

    @torch.no_grad()
    def update_dino_center(self, teacher_proto_logits: torch.Tensor):
        m = self.dino_center_momentum
        batch_mean = teacher_proto_logits.float().mean(dim=0, keepdim=True)
        self.dino_center.data.mul_(m).add_(batch_mean, alpha=1.0 - m)

    # ----------------------------------------------------------------------
    # Helpers — encoder/decoder forward variants
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
        """Visible-only encoder forward. Returns [B, 1+N_vis, D]."""
        B = m_channel.shape[0]
        patches = self.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_m.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos_embed_m
        return self._encode_stream_visible(tokens, mask, self.blocks_m, self.norm_m)

    def _decode_m(self, m_full: torch.Tensor) -> torch.Tensor:
        """M decoder: + dec_pos_embed_m → decoder blocks → norm. Input [B, 1+N, D]."""
        x = m_full + self.dec_pos_embed_m
        for block in self.m_decoder_blocks:
            x = block(x, freqs_cis=None)
        return self.m_decoder_norm(x)

    def _decode_m_unmasked(self, m_encoded: torch.Tensor) -> torch.Tensor:
        """Unmasked encoder output → m_completed (DINO student crop용)."""
        return self._decode_m(m_encoded)

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
        """M visible → mask_token inject (no pos_embed yet — _decode_m이 더함)."""
        return self._inject_mask_tokens(m_visible, mask, self.mask_token_m)

    def _decode_for_recon(self, p_full: torch.Tensor) -> torch.Tensor:
        """P recon decoder: interpreter_1 → recon_head → patch pixel pred."""
        p_decoded = self._run_interpreter(
            p_full, self.interpreter_1, self.interpreter_1_norm,
        )
        return self.recon_head(p_decoded[:, 1:])

    def _gpu_random_crop_pair(
        self, raw_t: torch.Tensor, raw_tk: torch.Tensor, crop_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-crop additional view (raw 256 → cropped pair). Per-sample, per-frame 독립 좌표."""
        B, _, H, W = raw_t.shape
        max_y = H - crop_size
        max_x = W - crop_size
        device = raw_t.device

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
            image_current:        [B, 3, 224, 224] crop_t
            image_future:         [B, 3, 224, 224] crop_tk
            image_current_global: [B, 3, 256, 256] raw_t (DINO teacher view + multi-crop source)
            image_future_global:  [B, 3, 256, 256] raw_tk

        global이 None이면 V-JEPA P/M, DINO loss 모두 0 (eval mode 호환).

        Returns dict:
            loss, loss_t, loss_tk, loss_pred, loss_m_jepa, loss_dino
            pred_t, pred_tk
            mask_p, mask_m
            m_features, p_features_t, p_features_tk, cls_m, cls_p (v11 호환 키)
            student_dino_cls, teacher_dino_cls, teacher_proto_logits
            predicted_tk_repr, target_tk_repr (V-JEPA P 진단)
        """
        # ── Rotation aug ───────────────────────────────────────────────────
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
        dtype_for_zero = torch.bfloat16 if image_current.dtype == torch.bfloat16 else torch.float32

        # ── Channel preprocessing ──────────────────────────────────────────
        m_channel_cropped, p_channel_t = self.preprocessing(image_current, image_future)
        p_channel_tk = self.preprocessing.compute_p_channel(image_future)

        # ── 1. P stream MAE (L_t + L_tk) ─────────────────────────────────────
        mask_p_t = self._random_mask(B, device, self.mask_ratio_p)
        mask_p_tk = self._random_mask(B, device, self.mask_ratio_p)

        p_t_visible = self._student_p_encode_visible(p_channel_t, mask_p_t)
        p_tk_visible = self._student_p_encode_visible(p_channel_tk, mask_p_tk)
        cls_p_repr = p_t_visible[:, 0]  # downstream / probing 호환

        p_t_full = self._build_full_seq_p(p_t_visible, mask_p_t)
        p_tk_full = self._build_full_seq_p(p_tk_visible, mask_p_tk)

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

        # Global 인자 미제공 시 (eval) — V-JEPA P/M, DINO 0
        if image_current_global is None or image_future_global is None:
            loss_pred = torch.zeros((), device=device, dtype=loss_t.dtype)
            loss_m_jepa = torch.zeros((), device=device, dtype=loss_t.dtype)
            loss_dino = torch.zeros((), device=device, dtype=loss_t.dtype)

            # Fallback features for downstream/probing 호환
            with torch.no_grad():
                m_local = self._encode_m_unmasked(m_channel_cropped)
            cls_m_repr = m_local[:, 0]
            predicted_tk_repr = torch.zeros(B, 1 + self.num_patches, self.embed_dim, device=device, dtype=loss_t.dtype)
            target_tk_repr = predicted_tk_repr.detach()
            mask_m_jepa = torch.zeros(B, self.num_patches, device=device, dtype=torch.bool)
            teacher_proto_logits = torch.zeros(B, self.num_prototypes, device=device, dtype=loss_t.dtype)
            teacher_dino_cls_for_diag = torch.zeros(B, self.embed_dim, device=device, dtype=loss_t.dtype)

            loss = loss_t + loss_tk_recon
            return self._build_output_dict(
                loss=loss, loss_t=loss_t, loss_tk=loss_tk_recon,
                loss_pred=loss_pred, loss_m_jepa=loss_m_jepa, loss_dino=loss_dino,
                patch_pred_t=patch_pred_t, patch_pred_tk_recon=patch_pred_tk_recon,
                mask_p_t=mask_p_t, mask_m_jepa=mask_m_jepa,
                m_features=m_local[:, 1:], p_t_visible=p_t_visible,
                predicted_tk_repr=predicted_tk_repr, target_tk_repr=target_tk_repr,
                cls_m_repr=cls_m_repr, cls_p_repr=cls_p_repr,
                teacher_dino_cls=teacher_dino_cls_for_diag,
                teacher_proto_logits=teacher_proto_logits,
            )

        # ── 2. M_encoder UNMASKED forward (routing source + DINO crop 0) ───
        m_local_unmasked = self._encode_m_unmasked(m_channel_cropped)  # [B, 1+N, D]

        # ── 3. V-JEPA P (L_pred): predictor only, V/target both from TeacherP ─
        with torch.no_grad():
            p_t_repr_T = self.teacher_p.forward_unmasked(p_channel_t).detach()
            p_tk_repr_T = self.teacher_p.forward_unmasked(p_channel_tk).detach()

        p_state = p_t_repr_T  # V source from TeacherP (stable)
        for step in self.p_motion_decoder:
            p_state = step(p_state, m_local_unmasked)
        predicted_tk_repr = self.p_motion_decoder_norm(p_state)
        target_tk_repr = p_tk_repr_T

        loss_pred = F.smooth_l1_loss(
            predicted_tk_repr.float(), target_tk_repr.float(), reduction="mean",
        )

        # ── 4. V-JEPA M (L_m_jepa): masked patches only, CLS 제외 ─────────
        # student M_encoder masked + M_decoder + mask_token → m_completed_masked
        # combine mask for crop_t and crop_tk independently (concat batch)
        # 단순화: crop_t, crop_tk 모두 forward (concat B 차원)
        m_combined = torch.cat([m_channel_cropped[:, :3], m_channel_cropped[:, :3]], dim=0)  # [2B,3,H,W] — placeholder; 실제는 crop_t/tk 별도 추출 필요
        # NOTE: m_channel_cropped는 (crop_t, crop_tk) preprocessing 결과 → 여기서 분리할 수 없음.
        # 단순화: m_channel_cropped 자체에 V-JEPA-M 적용 (학습 신호로 충분).
        # crop_t/tk 각각을 별도로 forward할 수도 있지만, m_channel은 (Δ, edge 등) 두 frame 결합 신호라
        # 단일 forward로 운영.
        del m_combined

        mask_m = self._random_mask(B, device, self.mask_ratio_m_jepa)
        m_visible = self._encode_m_masked(m_channel_cropped, mask_m)
        m_full = self._build_full_seq_m(m_visible, mask_m)
        m_completed_masked = self._decode_m(m_full)  # [B, 1+N, D]

        with torch.no_grad():
            m_completed_target = self.teacher_m.forward_unmasked_full(m_channel_cropped).detach()  # [B,1+N,D]

        student_patches = m_completed_masked[:, 1:]   # [B, N, D]
        target_patches = m_completed_target[:, 1:]
        # SmoothL1 element-wise → [B, N, D] → mean over D → [B, N]
        err_m_full = F.smooth_l1_loss(
            student_patches.float(), target_patches.float(), reduction="none",
        ).mean(dim=-1)
        denom_m = mask_m.float().sum().clamp(min=1.0)
        loss_m_jepa = (err_m_full * mask_m.float()).sum() / denom_m

        # DINO student CLS for crop 0 (v15: M_decoder 후, M stream unmasked path 사용)
        # crop 0은 m_local_unmasked → m_decoder_unmasked → cls
        m_completed_unmasked_crop0 = self._decode_m_unmasked(m_local_unmasked)
        cls_m_repr = m_completed_unmasked_crop0[:, 0]  # downstream/probing key

        # ── 5. DINO M (L_dino): M_decoder 후 CLS, raw 256 teacher target ───
        with torch.no_grad():
            m_channel_raw = self.preprocessing.compute_m_channel(
                image_current_global, image_future_global,
            )
            cls_m_global_completed = self.teacher_m.forward_global_full(m_channel_raw)  # [B, 1+N_g, D]
            cls_m_global = cls_m_global_completed[:, 0]
            teacher_proto_logits = self.teacher_m.dino_head(cls_m_global).detach()

        student_logits_list = [self.dino_head(cls_m_repr)]
        for _ in range(1, self.dino_n_crop):
            crop_t_n, crop_tk_n = self._gpu_random_crop_pair(
                image_current_global, image_future_global, self.image_size,
            )
            m_channel_n = self.preprocessing.compute_m_channel(crop_t_n, crop_tk_n)
            m_local_n = self._encode_m_unmasked(m_channel_n)
            m_completed_n = self._decode_m_unmasked(m_local_n)
            student_logits_list.append(self.dino_head(m_completed_n[:, 0]))

        loss_dino = sum(
            dino_distillation_loss(
                s, teacher_proto_logits, self.dino_center,
                student_temp=self.dino_student_temp,
                teacher_temp=self.dino_teacher_temp,
            )
            for s in student_logits_list
        ) / len(student_logits_list)

        # ── Total ──────────────────────────────────────────────────────────
        loss = (
            loss_t
            + loss_tk_recon
            + self.lambda_pred * loss_pred
            + self.lambda_m_jepa * loss_m_jepa
            + self.lambda_dino * loss_dino
        )

        return self._build_output_dict(
            loss=loss, loss_t=loss_t, loss_tk=loss_tk_recon,
            loss_pred=loss_pred, loss_m_jepa=loss_m_jepa, loss_dino=loss_dino,
            patch_pred_t=patch_pred_t, patch_pred_tk_recon=patch_pred_tk_recon,
            mask_p_t=mask_p_t, mask_m_jepa=mask_m,
            m_features=m_local_unmasked[:, 1:], p_t_visible=p_t_visible,
            predicted_tk_repr=predicted_tk_repr, target_tk_repr=target_tk_repr,
            cls_m_repr=cls_m_repr, cls_p_repr=cls_p_repr,
            teacher_dino_cls=cls_m_global,
            teacher_proto_logits=teacher_proto_logits,
        )

    def _build_output_dict(
        self, loss, loss_t, loss_tk, loss_pred, loss_m_jepa, loss_dino,
        patch_pred_t, patch_pred_tk_recon, mask_p_t, mask_m_jepa,
        m_features, p_t_visible, predicted_tk_repr, target_tk_repr,
        cls_m_repr, cls_p_repr, teacher_dino_cls, teacher_proto_logits,
    ):
        pred_t_img = self._unpatchify(patch_pred_t)
        pred_tk_img = self._unpatchify(patch_pred_tk_recon)
        return {
            "loss": loss,
            "loss_t": loss_t,
            "loss_tk": loss_tk,
            "loss_pred": loss_pred,
            "loss_m_jepa": loss_m_jepa,
            "loss_dino": loss_dino,
            "pred_t": pred_t_img,
            "pred_tk": pred_tk_img,
            "mask_p": mask_p_t,
            "mask_m": mask_m_jepa,
            "m_features": m_features,
            "p_features_t": p_t_visible[:, 1:],
            "p_features_tk": predicted_tk_repr[:, 1:],
            "cls_m": cls_m_repr,
            "cls_p": cls_p_repr,
            "predicted_tk_repr": predicted_tk_repr,
            "target_tk_repr": target_tk_repr,
            "student_dino_cls": cls_m_repr,
            "teacher_dino_cls": teacher_dino_cls,
            "teacher_proto_logits": teacher_proto_logits,
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
