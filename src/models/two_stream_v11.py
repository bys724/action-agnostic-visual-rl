"""
Two-Stream v11 Model: Motion-Guided Attention Routing + Dual-Target Reconstruction.

v7~v10의 "P stream 내부 강화" 접근이 모두 collapse로 실패 → 발상 전환:
  - P stream 강화 포기
  - M이 P를 semantic operator로 조작 (motion-guided spatial routing)
  - Dual-target (L_t + L_tk): Phase 1 self-attn reconstruction이 collapse anchor

설계 의도 (M/P 비대칭):
  - P encoder = semantic backbone (depth 12 = ViT-Base)
  - M encoder = pure motion sensor (depth 6, 작게)
       · M aux loss 없음 — P 조작 효과성으로만 학습됨
       · semantic-free spatial routing field 생성이 역할
  - Downstream에서는 m_features(decoder out) ⊕ p_features_tk concat 사용

구조:
  [Encoder]
     └─ M encoder: patch_embed(3→D) + APE + TransformerBlock × m_depth (mask 0.3)
     └─ P encoder: patch_embed(5→D) + APE + TransformerBlock × p_depth (mask 0.75)

  [M Decoder] (loss 없음, motion field 완성)
     visible + mask_token_m → APE → self-attn × 3 → m_completed

  [P Decoder] 3-phase dual-target
     ┌ Phase 1: p_full → interpreter_1 (self-attn × 3) → p_semantic_t
     │           recon_head(p_semantic_t[:, 1:]) = predicted_t
     │           L_t = MSE(predicted_t, img_t) [masked positions]
     │
     ├ Phase 2: motion routing × 2 (iteration별 독립 projections)
     │           p_state₀ = p_semantic_t  (Phase 1 output을 초기값으로 사용 —
     │                                     joint gradient: interpreter_1이 L_t + L_tk 둘 다 학습)
     │           for i in 0..1:
     │             Q, K ← m_completed,  V ← p_state_i
     │             p_state_{i+1} = p_state_i + (Q K^T softmax) @ V
     │             p_state_{i+1} = p_state_{i+1} + FFN_motion_i(p_state_{i+1})
     │
     └ Phase 3: p_state₂ → interpreter_2 (self-attn × 3, non-shared) → p_semantic_tk
                recon_head(p_semantic_tk[:, 1:]) = predicted_tk  (SAME head)
                L_tk = MSE(predicted_tk, img_{t+k}) [masked positions]

  L_total = L_t + L_tk

v11 체크포인트는 v6/v10과 호환 불가 (구조 상이).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import TwoStreamPreprocessing
from .two_stream import TransformerBlock


# ============================================================================
# Motion-Routing Block (Phase 2)
# ============================================================================

class MotionRoutingBlock(nn.Module):
    """Motion-guided attention routing.

    Q, K는 M stream의 완성된 motion field에서,
    V는 P stream의 현재 state에서 추출.

    "M의 self-attention 그래프를 P의 value에 적용" — M이 정의하는
    spatial routing pattern으로 P 내용물을 재조합.

    표준 cross-attention (V from M)과 차별화되는 핵심 novelty.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q/K from M (iteration별 독립 projections)
        self.norm_m = nn.LayerNorm(embed_dim)
        self.qk_m = nn.Linear(embed_dim, embed_dim * 2)

        # V from P
        self.norm_p = nn.LayerNorm(embed_dim)
        self.v_p = nn.Linear(embed_dim, embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

        # Post-attention FFN
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, p_state: torch.Tensor, m_completed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_state: [B, N+1, D] — P stream state (CLS + patches)
            m_completed: [B, N+1, D] — M stream completed (CLS + patches)
        Returns:
            p_state_updated: [B, N+1, D]
        """
        B, N, D = p_state.shape

        # Q, K from M
        m_normed = self.norm_m(m_completed)
        qk = self.qk_m(m_normed).reshape(B, N, 2, self.num_heads, self.head_dim)
        q, k = qk.unbind(dim=2)  # each: [B, N, H, D_head]

        # V from P
        v = self.v_p(self.norm_p(p_state)).reshape(B, N, self.num_heads, self.head_dim)

        # SDPA
        q = q.transpose(1, 2)  # [B, H, N, D_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, D)

        # Residual + projection
        p_state = p_state + self.proj_out(attn)

        # FFN + residual
        p_state = p_state + self.ffn(self.norm_ffn(p_state))

        return p_state


# ============================================================================
# V11 Main Model
# ============================================================================

class TwoStreamV11Model(nn.Module):
    """Two-Stream v11: Motion-Guided Attention Routing + Dual-Target.

    Encoder는 v11 전용으로 직접 구성 (InterleavedTwoStreamViT 미사용).
    M/P가 별도 모듈이며 depth 비대칭 (P=12 semantic, M=6 motion sensor).

    Args:
        embed_dim: Transformer embedding dim (default: 768).
            M/P 동일 — Phase 2 motion routing의 Q/K(M)·V(P) projection이
            head_dim 매칭을 요구하므로 차원 통일.
        p_depth: P encoder depth (default: 12, ViT-Base = semantic backbone)
        m_depth: M encoder depth (default: 6, motion sensor 역할이라 작게)
        num_heads: Attention heads (default: 12)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        image_size: Input resolution (default: 224)
        patch_size: ViT patch size (default: 16)
        mask_ratio_m: M-stream mask ratio (default: 0.3, sparse 유지)
        mask_ratio_p: P-stream mask ratio (default: 0.75, MAE-style 도전적)
        decoder_depth_m: M decoder depth (default: 3)
        interpreter_depth: interpreter_1/interpreter_2 depth (default: 3)
        num_motion_iters: Phase 2 motion routing iterations (default: 2)
        rotation_aug: 학습 시 rotation augmentation 적용
        independent_rotation_prob: 독립 회전 확률 (default: 0.1)
    """

    def __init__(
        self,
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
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.p_depth = p_depth
        self.m_depth = m_depth
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_patches_per_side = image_size // patch_size
        self.mask_ratio_m = mask_ratio_m
        self.mask_ratio_p = mask_ratio_p
        self.num_motion_iters = num_motion_iters
        self.rotation_aug = rotation_aug
        self.independent_rotation_prob = independent_rotation_prob

        # ── Preprocessing (RGB pair → M:3ch, P:5ch) ───────────────────────
        self.preprocessing = TwoStreamPreprocessing()

        # ── Stream-independent encoders (v11 전용 직접 구성) ─────────────
        # M: 3ch (ΔL + Sobel(ΔL)) → D, mask 0.3, depth m_depth
        # P: 5ch (Sobel + RGB) → D, mask 0.75, depth p_depth
        self.patch_embed_m = nn.Conv2d(
            in_channels=3, out_channels=embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.patch_embed_p = nn.Conv2d(
            in_channels=5, out_channels=embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # CLS tokens (M=1, P=1)
        self.cls_token_m = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_p = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # APE for encoder (CLS + N patches)
        self.pos_embed_m = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_embed_p = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Encoder transformer blocks (M=m_depth, P=p_depth)
        self.blocks_m = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(m_depth)
        ])
        self.blocks_p = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(p_depth)
        ])
        self.norm_m = nn.LayerNorm(embed_dim)
        self.norm_p = nn.LayerNorm(embed_dim)

        # Encoder weight init (CLS / APE / patch_embed)
        nn.init.trunc_normal_(self.cls_token_m, std=0.02)
        nn.init.trunc_normal_(self.cls_token_p, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_m, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_p, std=0.02)

        # ── Mask tokens ──────────────────────────────────────────────────
        self.mask_token_m = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token_p = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token_m, std=0.02)
        nn.init.trunc_normal_(self.mask_token_p, std=0.02)

        # ── Decoder APE (mask token inject 후 위치 정보 주입) ────────────
        #   M decoder: [CLS_M, N patches]
        #   P decoder: [CLS_P, N patches]
        self.dec_pos_embed_m = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dec_pos_embed_p = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.dec_pos_embed_m, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed_p, std=0.02)

        # ── M decoder: self-attn × 3 (loss 없음, motion field 완성 역할) ─
        self.m_decoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(decoder_depth_m)
        ])
        self.m_decoder_norm = nn.LayerNorm(embed_dim)

        # ── P decoder Phase 1: interpreter_1 (self-attn × 3) ─────────────
        self.interpreter_1 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(interpreter_depth)
        ])
        self.interpreter_1_norm = nn.LayerNorm(embed_dim)

        # ── P decoder Phase 2: motion-routing × N (iteration별 독립 projections) ─
        self.motion_routing = nn.ModuleList([
            MotionRoutingBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_motion_iters)
        ])

        # ── P decoder Phase 3: interpreter_2 (self-attn × 3, non-shared) ─
        self.interpreter_2 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(interpreter_depth)
        ])
        self.interpreter_2_norm = nn.LayerNorm(embed_dim)

        # ── Recon head (shared between Phase 1 & Phase 3) ────────────────
        # 1 linear layer: D → patch_size² × 3  (MAE 표준)
        self.recon_head = nn.Linear(embed_dim, patch_size * patch_size * 3)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _random_mask(self, B: int, device: torch.device, ratio: float) -> torch.Tensor:
        """Random mask: returns [B, N] bool, True=masked."""
        N = self.num_patches
        num_masked = int(ratio * N)
        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :N - num_masked], False)
        return mask

    def _encode_stream_visible(
        self,
        tokens: torch.Tensor,        # [B, 1+N, D] CLS + patches with APE already added
        mask: torch.Tensor,          # [B, N] bool, True=masked
        blocks: nn.ModuleList,       # encoder blocks for this stream
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        """Stream-independent visible-only encoding (CLS exchange 없음).

        Returns:
            encoded_visible: [B, 1 + N_vis, D] (CLS + visible patches)
        """
        cls = tokens[:, :1]
        patches = tokens[:, 1:]
        B, N, D = patches.shape

        visible = patches[~mask].reshape(B, -1, D)
        x = torch.cat([cls, visible], dim=1)

        for block in blocks:
            x = block(x, freqs_cis=None)  # APE path (freqs_cis=None)
        x = norm(x)
        return x

    def _inject_mask_tokens(
        self,
        encoded: torch.Tensor,       # [B, 1 + N_vis, D]
        mask: torch.Tensor,          # [B, N] bool
        mask_token: torch.Tensor,    # [1, 1, D]
    ) -> torch.Tensor:
        """Restore full sequence by injecting mask tokens at masked positions.

        Returns:
            full: [B, 1 + N, D] (CLS + all patches in original order)
        """
        cls = encoded[:, :1]
        visible = encoded[:, 1:]        # [B, N_vis, D]
        B, _, D = visible.shape
        N = mask.shape[1]

        full_patches = mask_token.expand(B, N, -1).clone()
        full_patches[~mask] = visible.reshape(-1, D)
        return torch.cat([cls, full_patches], dim=1)

    def _run_interpreter(
        self, x: torch.Tensor, blocks: nn.ModuleList, norm: nn.LayerNorm,
    ) -> torch.Tensor:
        for block in blocks:
            x = block(x, freqs_cis=None)
        return norm(x)

    def _unpatchify(self, patch_preds: torch.Tensor) -> torch.Tensor:
        """[B, N, ps²·3] → [B, 3, H, W]."""
        B, N, _ = patch_preds.shape
        P = self.num_patches_per_side
        ps = self.patch_size
        img = patch_preds.reshape(B, P, P, ps, ps, 3)
        img = img.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, 3, P, ps, P, ps]
        return img.reshape(B, 3, P * ps, P * ps)

    def _patchify(self, img: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] → [B, N, ps²·3]."""
        B, C, H, W = img.shape
        ps = self.patch_size
        P = H // ps
        x = img.reshape(B, C, P, ps, P, ps)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, P, P, ps, ps, C]
        return x.reshape(B, P * P, ps * ps * C)

    def _apply_rotation_aug(
        self, img_t: torch.Tensor, img_tk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """90%: 동일 회전, 10%: 독립 회전. 0/90/180/270 균등."""
        if torch.rand(1).item() < self.independent_rotation_prob:
            k_t = torch.randint(0, 4, (1,)).item()
            k_tk = torch.randint(0, 4, (1,)).item()
        else:
            k_t = torch.randint(0, 4, (1,)).item()
            k_tk = k_t
        if k_t != 0:
            img_t = torch.rot90(img_t, k_t, dims=(2, 3))
        if k_tk != 0:
            img_tk = torch.rot90(img_tk, k_tk, dims=(2, 3))
        return img_t, img_tk

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def forward(
        self,
        image_current: torch.Tensor,
        image_future: torch.Tensor,
    ) -> dict:
        """
        Args:
            image_current: [B, 3, H, W] range [0, 1]
            image_future:  [B, 3, H, W] range [0, 1]

        Returns:
            dict with keys:
                loss: scalar, L_t + L_tk
                loss_t: scalar, Phase 1 MSE (masked positions)
                loss_tk: scalar, Phase 3 MSE (masked positions)
                pred_t: [B, 3, H, W] Phase 1 predicted frame_t
                pred_tk: [B, 3, H, W] Phase 3 predicted frame_{t+k}
                mask_p, mask_m: [B, N] bool masks (viz/eval)

                m_features:    [B, N, D] M decoder output (CLS 제외) — 권장 downstream
                p_features_t:  [B, N, D] Phase 1 P patches — single-image / pre-motion
                p_features_tk: [B, N, D] Phase 3 P patches — paired / post-motion (권장)

                cls_m: [B, D] M encoder CLS (CLS-only probing 비교용)
                cls_p: [B, D] P encoder CLS (CLS-only probing 비교용)

        Downstream 사용 권장:
            paired (frame_t, frame_{t+k}) → m_features ⊕ p_features_tk concat
            single (frame_t)              → encode_single() 사용
        """
        # ── Rotation aug (학습 시에만) ──────────────────────────────────
        if self.rotation_aug and self.training:
            image_current, image_future = self._apply_rotation_aug(
                image_current, image_future
            )

        B = image_current.shape[0]
        device = image_current.device

        # ── Preprocessing: RGB pair → M (3ch) + P (5ch) ────────────────
        m_channel, p_channel = self.preprocessing(image_current, image_future)

        # ── Independent masks ──────────────────────────────────────────
        mask_m = self._random_mask(B, device, self.mask_ratio_m)  # [B, N]
        mask_p = self._random_mask(B, device, self.mask_ratio_p)

        # ── Patch embed + CLS + APE ────────────────────────────────────
        m_patches = self.patch_embed_m(m_channel).flatten(2).transpose(1, 2)  # [B,N,D]
        p_patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)

        m_cls = self.cls_token_m.expand(B, -1, -1)
        p_cls = self.cls_token_p.expand(B, -1, -1)
        m_tokens = torch.cat([m_cls, m_patches], dim=1) + self.pos_embed_m
        p_tokens = torch.cat([p_cls, p_patches], dim=1) + self.pos_embed_p

        # ── Stream-independent visible-only encoding (CLS exchange 없음) ─
        m_encoded = self._encode_stream_visible(
            m_tokens, mask_m, self.blocks_m, self.norm_m,
        )   # [B, 1+N_vis_m, D]
        p_encoded = self._encode_stream_visible(
            p_tokens, mask_p, self.blocks_p, self.norm_p,
        )   # [B, 1+N_vis_p, D]

        # Downstream representation: CLS of each stream (encoder pure output)
        cls_m_repr = m_encoded[:, 0]
        cls_p_repr = p_encoded[:, 0]

        # ── M decoder: mask token inject + APE + self-attn × 3 ─────────
        m_full = self._inject_mask_tokens(m_encoded, mask_m, self.mask_token_m)
        m_full = m_full + self.dec_pos_embed_m
        for block in self.m_decoder_blocks:
            m_full = block(m_full, freqs_cis=None)
        m_completed = self.m_decoder_norm(m_full)  # [B, 1+N, D]

        # ── P decoder: 3-phase dual-target ─────────────────────────────
        p_full = self._inject_mask_tokens(p_encoded, mask_p, self.mask_token_p)
        p_full = p_full + self.dec_pos_embed_p

        # Phase 1: pre-motion semantic interpretation
        p_semantic_t = self._run_interpreter(
            p_full, self.interpreter_1, self.interpreter_1_norm,
        )   # [B, 1+N, D]

        # Phase 2 motion-routing 초기값: p_state₀ = p_semantic_t
        # (Phase 1 output을 그대로 받음 → joint gradient. interpreter_1이 L_t + L_tk 둘 다 학습)
        p_state = p_semantic_t

        # Phase 2: motion routing × N (iter 별 독립 projections)
        # Q, K ← m_completed (M motion field), V ← p_state (현재 P 상태)
        for routing_block in self.motion_routing:
            p_state = routing_block(p_state, m_completed)

        # Phase 3: post-motion semantic re-interpretation (non-shared weights)
        p_semantic_tk = self._run_interpreter(
            p_state, self.interpreter_2, self.interpreter_2_norm,
        )   # [B, 1+N, D]

        # ── Recon head (shared): pixel-space prediction ────────────────
        patch_pred_t = self.recon_head(p_semantic_t[:, 1:])     # [B, N, ps²·3]
        patch_pred_tk = self.recon_head(p_semantic_tk[:, 1:])

        # ── Loss: masked positions only (per MAE) ──────────────────────
        patch_target_t = self._patchify(image_current)          # [B, N, ps²·3]
        patch_target_tk = self._patchify(image_future)

        # Select masked positions
        mask_p_flat = mask_p                                    # [B, N]
        # per-patch mean squared error, then mean over masked positions
        err_t = (patch_pred_t - patch_target_t) ** 2            # [B, N, ps²·3]
        err_tk = (patch_pred_tk - patch_target_tk) ** 2

        err_t = err_t.mean(dim=-1)                              # [B, N]
        err_tk = err_tk.mean(dim=-1)

        # Masked-only mean: sum(err * mask) / sum(mask)
        denom_p = mask_p_flat.float().sum().clamp(min=1.0)
        loss_t = (err_t * mask_p_flat.float()).sum() / denom_p
        loss_tk = (err_tk * mask_p_flat.float()).sum() / denom_p

        loss = loss_t + loss_tk

        # ── Unpatchify preds to images (for visualization/compat) ──────
        pred_t_img = self._unpatchify(patch_pred_t)
        pred_tk_img = self._unpatchify(patch_pred_tk)

        return {
            "loss": loss,
            "loss_t": loss_t,
            "loss_tk": loss_tk,
            "pred_t": pred_t_img,
            "pred_tk": pred_tk_img,
            "mask_p": mask_p,
            "mask_m": mask_m,
            # Downstream representation (CLS 제외, patch tokens only)
            "m_features": m_completed[:, 1:],       # [B, N, D] M decoder out
            "p_features_t": p_semantic_t[:, 1:],    # [B, N, D] Phase 1 (single-image)
            "p_features_tk": p_semantic_tk[:, 1:],  # [B, N, D] Phase 3 (paired)
            # CLS-only (probing 비교용)
            "cls_m": cls_m_repr,
            "cls_p": cls_p_repr,
        }

    @torch.no_grad()
    def encode_single(self, image: torch.Tensor) -> dict:
        """Single-image inference (motion 신호 없음).

        Single frame에서는 ΔL = 0이라 M channel이 모두 0 → motion routing
        무의미. P encoder + interpreter_1(Phase 1)까지만 forward.
        Mask는 사용하지 않음 (full sequence 인코딩).

        Args:
            image: [B, 3, H, W] range [0, 1]

        Returns:
            dict:
                p_features_t: [B, N, D] Phase 1 patches (recon_head 직전)
                cls_p:        [B, D]    P encoder CLS
        """
        B = image.shape[0]
        # P channel만 필요 (preprocessing에 (img, img) 넣으면 ΔL=0이 되므로
        # 직접 P channel 구성: Sobel(L) + RGB) — preprocessing 모듈 재사용
        _, p_channel = self.preprocessing(image, image)

        # Patch embed + CLS + APE (mask 없음, full sequence)
        p_patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)  # [B,N,D]
        p_cls = self.cls_token_p.expand(B, -1, -1)
        p_tokens = torch.cat([p_cls, p_patches], dim=1) + self.pos_embed_p

        # P encoder (full sequence, no mask)
        for block in self.blocks_p:
            p_tokens = block(p_tokens, freqs_cis=None)
        p_encoded = self.norm_p(p_tokens)  # [B, 1+N, D]
        cls_p_repr = p_encoded[:, 0]

        # P decoder Phase 1: APE는 decoder용 별도 (학습 시와 동일하게)
        # mask 없으므로 mask_token inject 불필요 — encoded 그대로 + dec_pos_embed_p
        p_full = p_encoded + self.dec_pos_embed_p
        p_semantic_t = self._run_interpreter(
            p_full, self.interpreter_1, self.interpreter_1_norm,
        )   # [B, 1+N, D]

        return {
            "p_features_t": p_semantic_t[:, 1:],
            "cls_p": cls_p_repr,
        }

    def compute_loss(
        self, image_current: torch.Tensor, image_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training loop 호환 인터페이스.

        Returns:
            loss: scalar, L_t + L_tk
            pred: [B, 3, H, W] Phase 3 predicted frame_{t+k} (for viz)
        """
        out = self.forward(image_current, image_future)
        return out["loss"], out["pred_tk"]


# ============================================================================
# V11 Encoder Wrapper (downstream VLA integration)
# ============================================================================

class TwoStreamV11Encoder(nn.Module):
    """Encoder-only wrapper for v11 checkpoints (downstream / probing).

    Stream-independent encoder. Output은 per-stream patches를 concat.
    M/P depth가 비대칭이므로 둘 다 명시적으로 받음.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        embed_dim: int = 768,
        p_depth: int = 12,
        m_depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        self.preprocessing = TwoStreamPreprocessing()

        # v11 직접 구성 — TwoStreamV11Model encoder 부분과 1:1 대응
        self.patch_embed_m = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.patch_embed_p = nn.Conv2d(5, embed_dim, patch_size, patch_size)
        self.cls_token_m = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_p = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_m = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_embed_p = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks_m = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(m_depth)
        ])
        self.blocks_p = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(p_depth)
        ])
        self.norm_m = nn.LayerNorm(embed_dim)
        self.norm_p = nn.LayerNorm(embed_dim)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    @property
    def embed_dim(self) -> int:
        # Concat M/P patch embeddings (parallels TwoStreamEncoder separate fusion)
        return self._embed_dim * 2

    def load_from_checkpoint(self, checkpoint_path: str):
        """Load encoder weights from trained v11 model.

        v11 체크포인트에서 encoder 부분만 추출. decoder/interpreter는 제외.
        체크포인트 키는 TwoStreamV11Model의 직접 attribute (patch_embed_m,
        cls_token_m, pos_embed_m, blocks_m.*, norm_m 등) 그대로.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):] if k.startswith("module.") else k: v
                          for k, v in state_dict.items()}

        # Encoder 관련 key prefix만 추출
        encoder_prefixes = (
            "patch_embed_m.", "patch_embed_p.",
            "cls_token_m", "cls_token_p",
            "pos_embed_m", "pos_embed_p",
            "blocks_m.", "blocks_p.",
            "norm_m.", "norm_p.",
            "preprocessing.",
        )
        encoder_state = {
            k: v for k, v in state_dict.items()
            if any(k.startswith(p) or k == p.rstrip(".") for p in encoder_prefixes)
        }
        result = self.load_state_dict(encoder_state, strict=False)
        if result.missing_keys:
            print(f"  WARNING: {len(result.missing_keys)} missing encoder keys "
                  f"(first 3: {result.missing_keys[:3]})")
        print(f"Loaded v11 encoder weights from: {checkpoint_path}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 6, H, W] — (image_current + image_future) stacked
        Returns:
            patch_embeddings: [B, N, 2D] (M patches ⊕ P patches)
        """
        image_current = pixel_values[:, :3]
        image_future = pixel_values[:, 3:]

        m_channel, p_channel = self.preprocessing(image_current, image_future)
        B = pixel_values.shape[0]

        m_patches = self.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        p_patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        m_cls = self.cls_token_m.expand(B, -1, -1)
        p_cls = self.cls_token_p.expand(B, -1, -1)
        m_tokens = torch.cat([m_cls, m_patches], dim=1) + self.pos_embed_m
        p_tokens = torch.cat([p_cls, p_patches], dim=1) + self.pos_embed_p

        for block in self.blocks_m:
            m_tokens = block(m_tokens, freqs_cis=None)
        for block in self.blocks_p:
            p_tokens = block(p_tokens, freqs_cis=None)
        m_tokens = self.norm_m(m_tokens)
        p_tokens = self.norm_p(p_tokens)

        # Concat patch embeddings (CLS 제외)
        return torch.cat([m_tokens[:, 1:], p_tokens[:, 1:]], dim=-1)
