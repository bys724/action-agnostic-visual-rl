"""
CoMP-MAE: symmetric cross-reconstruction two-stream masked autoencoder.

Two input streams are derived from a frame pair (t, t+k):
    P (appearance) = RGB(frame_t)          [3ch]
    M (motion)     = |ΔL| = |L(t+k) - L(t)| [1ch], L = luminance

Masked patches of both streams are reconstructed under a shared motion-routing
decoder that uses value-ownership routing (see MotionRoutingBlock).

Two training objectives are provided (see the paper, Sec. Method):

  comp_mae=True  -- CoMP-MAE (ours). Symmetric cross-reconstruction:
      * P-recon: masked RGB recovered under null/real motion routing.
      * M-recon: masked |ΔL| recovered under P-grouping routing (mirror branch)
        -> grounds the motion encoder in its own reconstruction target.

  pixel_pred=True -- plain cross-modal MAE (matched control). P-recon only,
      standard cross-attention routing (V from M), no M-recon branch.

Both objectives operate on 2-frame pairs (pair_mode=True) with no-Sobel
channels (use_sobel=False).

Reproduces the two released checkpoints:
    comp_mae_s.pt   <- comp_mae=True,  routing_mode="v_from_p"
    plain_xmae_s.pt <- pixel_pred=True, routing_mode="v_from_m", masked_anchor=True
(constructed with embed_dim=384, num_heads=6, m_depth=6, pair_mode=True, use_sobel=False)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common.blocks import TransformerBlock, MotionRoutingBlock
from .two_stream_base import TwoStreamBase


# ============================================================================
# SSIM reconstruction loss (structural term added to the pixel MSE)
# ============================================================================

def ssim_loss(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """Structural Similarity loss. Returns 1 - SSIM (lower is better).

    A Gaussian window computes local luminance/contrast/structure statistics.
    Forced FP32 to avoid BF16 precision issues (negative variance under AMP).
    """
    with torch.cuda.amp.autocast(enabled=False):
        pred = pred.float()
        target = target.float()

        coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        window = (g.unsqueeze(0) * g.unsqueeze(1))  # [K, K]
        window = window / window.sum()
        window = window.unsqueeze(0).unsqueeze(0).expand(pred.shape[1], -1, -1, -1)  # [C, 1, K, K]

        pad = window_size // 2
        mu1 = F.conv2d(pred, window, padding=pad, groups=pred.shape[1])
        mu2 = F.conv2d(target, window, padding=pad, groups=target.shape[1])

        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        # clamp: E[X^2] - E[X]^2 can go slightly negative from float error
        sigma1_sq = F.conv2d(pred ** 2, window, padding=pad, groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=pad, groups=target.shape[1]) - mu2_sq
        sigma1_sq = sigma1_sq.clamp(min=0)
        sigma2_sq = sigma2_sq.clamp(min=0)
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=pred.shape[1]) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


# ============================================================================
# Routing-Interpreter step (one motion-decoder unit)
# ============================================================================

class RoutingInterpreterStep(nn.Module):
    """One decoder step: cross-stream routing + self-attention (interleaved).

    Argument names are generic so the same class mirrors both reconstruction
    directions (a *separate* instance is used per direction = parameter symmetry):
      - P-recon: forward(owner=P_state, helper=M_routing) -> M->M attn, gather P
      - M-recon: forward(owner=M_state, helper=P_full)     -> P->P attn, gather M
    `owner`  supplies V  (the reconstruction target / residual / self-attn target).
    `helper` supplies Q/K (the routing pattern).
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, routing_mode: str,
                 decode_first: bool = False, routing_source: str = "m", v_source: str = "owner"):
        super().__init__()
        self.routing = MotionRoutingBlock(
            embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            routing_mode=routing_mode, routing_source=routing_source, v_source=v_source,
        )
        self.interp = TransformerBlock(
            embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
        )
        # decode_first=True (masked anchor): complete via self-attn from the
        # visible tokens first ("fill the hole"), then apply motion routing
        # ("then move"). decode_first=False: route first, then interpret.
        self.decode_first = decode_first

    def forward(self, owner_state: torch.Tensor, helper_state: torch.Tensor) -> torch.Tensor:
        if self.decode_first:
            owner_state = self.interp(owner_state, freqs_cis=None)    # complete first
            owner_state = self.routing(owner_state, helper_state)     # then route
        else:
            owner_state = self.routing(owner_state, helper_state)
            owner_state = self.interp(owner_state, freqs_cis=None)
        return owner_state


# ============================================================================
# CoMP-MAE model (comp_mae=True) and the plain cross-modal MAE control (pixel_pred=True)
# ============================================================================

class CoMPMAE(TwoStreamBase):
    """Two-stream symmetric cross-reconstruction MAE.

    Only the two reported pair-mode objectives are supported in this release:
      - comp_mae=True   : CoMP-MAE  (symmetric cross-reconstruction, ours)
      - pixel_pred=True : plain cross-modal MAE (matched control)

    The base class (TwoStreamBase) provides the encoders, masking utilities,
    and the shared recon head. This subclass adds the motion-routing P-recon
    decoder and, for CoMP-MAE, the mirror M-recon decoder.

    Args (most relevant):
        comp_mae:            enable the symmetric M-recon mirror branch (ours).
        pixel_pred:          plain P-recon-only control objective.
        routing_source:      "m"=motion(|ΔL|)-routed / "p"=RGB-routed control.
        m_recon_v_source:    M-recon V ownership ("m"=own V_M / "p"=V from P).
        mask_ratio_p:        P-stream mask ratio (MAE-style, default 0.75).
        mask_ratio_m_recon:  M-recon mask ratio (< P so motion patches stay visible).
        m_recon_weight_floor / _scale: per-patch |ΔL| weighting of the M-recon loss.
        caseA_weight / caseA_prob:     static-calibration (ΔL(t,t)=0) case weight/prob.
        lambda_recon / lambda_pred / lambda_m_recon: loss-term weights.
    """

    def __init__(
        self,
        # ── base architecture args ──
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
        # ── objective / loss-weight args ──
        lambda_pred: float = 1.0,
        lambda_ssim: float = 0.02,
        masked_anchor: bool = False,
        pair_mode: bool = False,
        no_motion: bool = False,
        pixel_pred: bool = False,
        lambda_recon: float = 1.0,
        routing_source: str = "m",
        # ── CoMP-MAE: symmetric cross-reconstruction (M-recon branch) ──
        comp_mae: bool = False,
        lambda_m_recon: float = 1.0,
        mask_ratio_m_recon: float = 0.5,   # guard 8: below P(0.75) so motion patches stay visible
        m_recon_iters: Optional[int] = None,  # None = same as num_motion_iters
        m_recon_weight_floor: float = 0.1,    # guard 7: static-patch calibration floor
        m_recon_weight_scale: float = 1.0,    # guard 7: |ΔL|-proportional weighting scale
        m_recon_v_source: str = "m",          # M-recon V ownership: "m"=V_M / "p"=V from P
        caseA_weight: float = 1.0,            # Case A (static calibration) relative loss weight
        caseA_prob: float = 1.0,              # Case A execution probability (<1 => per-step skip)
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

        assert pair_mode, "this release supports pair-mode (2-frame) objectives only"
        assert comp_mae or pixel_pred, (
            "this release supports comp_mae or pixel_pred objectives. When loading a "
            "checkpoint only to extract the (objective-independent) encoders for "
            "probing/BC, pass pixel_pred=not comp_mae so the module set matches the "
            "checkpoint -- the P/M encoder outputs are identical regardless."
        )
        assert not no_motion, "the no_motion ablation is not part of this release"

        self.lambda_pred = lambda_pred
        # P-recon loss = MSE + lambda_ssim * (1 - SSIM). MSE alone is insensitive
        # to shape / high-frequency collapse (a flat per-patch mean minimizes it);
        # the SSIM term adds structural pressure. lambda_ssim=0 disables it.
        self.lambda_ssim = lambda_ssim
        # CoMP-MAE forces P-recon to complete-first (decode_first=True) to mirror
        # the M-recon branch (fill-the-hole then move); the plain control passes
        # masked_anchor=True explicitly for the same behavior.
        if comp_mae:
            masked_anchor = True
        self.masked_anchor = masked_anchor
        self.pair_mode = pair_mode
        self.no_motion = no_motion

        # ── P-recon motion decoder = (routing + interp) x N (interleaved) ──
        # routing_source: "m"=motion(|ΔL|)-routed (main) / "p"=RGB-routed control.
        self.routing_source = routing_source
        self.p_motion_decoder = nn.ModuleList([
            RoutingInterpreterStep(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                routing_mode=routing_mode, decode_first=masked_anchor,
                routing_source=routing_source,
            )
            for _ in range(num_motion_iters)
        ])
        self.p_motion_decoder_norm = nn.LayerNorm(embed_dim)

        # Freeze base-class modules not used by these objectives. They remain
        # registered (and present in the checkpoint) but receive no gradient;
        # freezing avoids a DDP find_unused_parameters=False hang.
        for p in self.motion_routing.parameters():
            p.requires_grad_(False)
        for p in self.interpreter_2.parameters():
            p.requires_grad_(False)
        for p in self.interpreter_2_norm.parameters():
            p.requires_grad_(False)

        self.mask_token_m.requires_grad_(True)

        # ── plain cross-modal MAE control (pixel_pred): P-recon only ──────
        #   Unified pixel task: predict(P_visible, mask, M_routing) via p_motion_decoder.
        #     L_t/L_tk: M = M(x,x) = null (self-pair, ΔL=0) -> recover frame_t/tk
        #     L_pred:   M = M(t,tk) real                    -> motion-conditioned future
        self.pixel_pred = pixel_pred
        self.lambda_recon = lambda_recon
        if pixel_pred:
            assert not comp_mae, "pixel_pred and comp_mae are mutually exclusive"
            # Delete base modules unused by this objective (memory recovery).
            del self.interpreter_1, self.interpreter_1_norm
            del self.m_decoder_blocks, self.m_decoder_norm
            del self.dec_pos_embed_m, self.mask_token_m
            # RGB-routed control (routing_source="p"): routing uses P as Q/K, so
            # the M encoder is unused -> freeze it (DDP-safe).
            if routing_source == "p":
                for mod in (self.patch_embed_m, self.blocks_m, self.norm_m):
                    for p in mod.parameters():
                        p.requires_grad_(False)
                self.cls_token_m.requires_grad_(False)
                self.pos_embed_m.requires_grad_(False)

        # ── CoMP-MAE: symmetric cross-reconstruction (M-recon mirror) ──────
        #   Adds an M-recon branch on top of P-recon: masked |ΔL| is recovered
        #   under P-grouping routing (V=M owner, Q/K=P_full helper), grounding
        #   the motion encoder in its own reconstruction target.
        self.comp_mae = comp_mae
        self.lambda_m_recon = lambda_m_recon
        self.mask_ratio_m_recon = mask_ratio_m_recon
        self.m_recon_weight_floor = m_recon_weight_floor
        self.m_recon_weight_scale = m_recon_weight_scale
        self.m_recon_v_source = m_recon_v_source
        self.caseA_weight = caseA_weight
        self.caseA_prob = caseA_prob
        if comp_mae:
            assert not use_sobel, "comp_mae is no-Sobel only (M=|ΔL| 1ch, P=RGB 3ch)"
            assert routing_source == "m", "comp_mae requires routing_source='m'"
            assert m_recon_v_source in ("m", "p"), f"m_recon_v_source={m_recon_v_source}"
            m_in_ch = 1  # no-Sobel |ΔL|
            # Delete base modules unused by CoMP-MAE. Keep mask_token_m and
            # dec_pos_embed_m (M-recon injects mask tokens + APE). The JEPA
            # M-decoder is replaced by the dedicated m_recon_decoder below.
            del self.interpreter_1, self.interpreter_1_norm
            del self.m_decoder_blocks, self.m_decoder_norm
            # ── M-recon decoder: (M self-attn -> P-grouping routing) x N (mirror) ──
            #   routing_source="m" fixed => Q/K from helper (P_full) => P->P grouping,
            #   gather M. A separate instance from p_motion_decoder (parameter symmetry).
            #   v_source="p" (scalpel ablation) gathers V from the P helper instead
            #   of own V_M -> turns off M grounding while leaving P-recon unchanged.
            n_iters = m_recon_iters if m_recon_iters is not None else num_motion_iters
            self.m_recon_decoder = nn.ModuleList([
                RoutingInterpreterStep(
                    embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    routing_mode=routing_mode, decode_first=True, routing_source="m",
                    v_source="owner" if m_recon_v_source == "m" else "helper",
                )
                for _ in range(n_iters)
            ])
            self.m_recon_decoder_norm = nn.LayerNorm(embed_dim)
            # 1-channel |ΔL| recon head (guard 5: do not reuse the RGB recon_head)
            self.m_recon_head = nn.Linear(embed_dim, patch_size * patch_size * m_in_ch)
            # Learned null-motion token: replaces an M-encoder(ΔL=0) full forward
            # for the P-side static routing helper (efficiency + a well-defined,
            # always-on routing signal).
            self.null_motion_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.null_motion_token, std=0.02)

    # ----------------------------------------------------------------------
    # EMA update -- no-op: the released objectives use no EMA teacher.
    # (Kept so a standard training loop can call it unconditionally.)
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        return

    # ----------------------------------------------------------------------
    # Encoder / sequence helpers
    # ----------------------------------------------------------------------

    def _encode_p_unmasked(self, p_channel: torch.Tensor) -> torch.Tensor:
        """Full-sequence P (appearance) encoder forward -> [B, 1+N, D]."""
        B = p_channel.shape[0]
        patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_p.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1) + self.pos_embed_p
        for block in self.blocks_p:
            x = block(x, freqs_cis=None)
        return self.norm_p(x)

    def _encode_m_unmasked(self, m_channel: torch.Tensor) -> torch.Tensor:
        """Full-sequence M (motion) encoder forward -> [B, 1+N, D]."""
        B = m_channel.shape[0]
        patches = self.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_m.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1) + self.pos_embed_m
        for block in self.blocks_m:
            x = block(x, freqs_cis=None)
        return self.norm_m(x)

    def _encode_m_masked(self, m_channel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Visible-only M encoder forward (masked patches dropped) -> [B, 1+N_vis, D]."""
        B = m_channel.shape[0]
        patches = self.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_m.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos_embed_m
        return self._encode_stream_visible(tokens, mask, self.blocks_m, self.norm_m)

    def _student_p_encode_visible(
        self, p_channel: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Visible-only P encoder forward (masked patches dropped) -> [B, 1+N_vis, D]."""
        B = p_channel.shape[0]
        patches = self.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        cls = self.cls_token_p.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos_embed_p
        return self._encode_stream_visible(tokens, mask, self.blocks_p, self.norm_p)

    def _build_full_seq_p(self, p_visible: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Inject P mask tokens at masked positions and add decoder APE -> [B, 1+N, D]."""
        p_full = self._inject_mask_tokens(p_visible, mask, self.mask_token_p)
        return p_full + self.dec_pos_embed_p

    # ----------------------------------------------------------------------
    # P-recon (pixel prediction) -- shared by both objectives
    # ----------------------------------------------------------------------

    def _predict_pixels(
        self, p_visible: torch.Tensor, mask: torch.Tensor,
        m_routing: torch.Tensor, target_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unified P-recon mechanism: visible -> inject mask tokens -> motion
        routing (p_motion_decoder) -> recon_head -> pixels. Loss on masked
        positions only. m_routing=null -> in-place recovery; m_routing=real ->
        motion-conditioned future prediction.
        """
        full_seq = self._build_full_seq_p(p_visible, mask)   # [B, 1+N, D], mask token + dec_pos
        p_state = full_seq
        for step in self.p_motion_decoder:
            p_state = step(p_state, m_routing)
        p_state = self.p_motion_decoder_norm(p_state)
        patch_pred = self.recon_head(p_state[:, 1:])          # [B, N, ps^2 * 3]
        patch_target = self._patchify(target_image)
        err = ((patch_pred - patch_target) ** 2).mean(dim=-1)
        denom = mask.float().sum().clamp(min=1.0)
        loss = (err * mask.float()).sum() / denom
        if self.lambda_ssim > 0:
            loss = loss + self.lambda_ssim * ssim_loss(self._unpatchify(patch_pred), target_image)
        return loss, patch_pred

    def _forward_pair_pixel(self, image_current: torch.Tensor, image_future: torch.Tensor) -> dict:
        """Plain cross-modal MAE control (2-frame): in-place recovery (M=null) x2 +
        future prediction (M=real) x1, all in pixel space. No EMA teacher / JEPA.

        Loss = lambda_recon * (L_t + L_tk) + lambda_pred * L_pred.
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
        mask_t = self._random_mask(B, device, self.mask_ratio_p)
        mask_tk = self._random_mask(B, device, self.mask_ratio_p)
        p_t_visible = self._student_p_encode_visible(p_channel_t, mask_t)
        p_tk_visible = self._student_p_encode_visible(p_channel_tk, mask_tk)

        # Real motion M(t,tk); null motion M(x,x)=ΔL=0 (self-pair), reused for L_t/L_tk.
        # RGB-routed control (routing_source="p"): routing ignores M -> skip M encoder.
        if self.routing_source == "p":
            m_real = m_null = None
        else:
            m_real = self._encode_m_unmasked(self.preprocessing.compute_m_channel(image_current, image_future))
            m_null = self._encode_m_unmasked(self.preprocessing.compute_m_channel(image_current, image_current))

        # 3 unified pixel predictions (gap=0 recovery x2 + real-gap prediction x1)
        loss_t, patch_pred_t = self._predict_pixels(p_t_visible, mask_t, m_null, image_current)
        loss_tk, patch_pred_tk = self._predict_pixels(p_tk_visible, mask_tk, m_null, image_future)
        loss_pred, patch_pred_pred = self._predict_pixels(p_t_visible, mask_t, m_real, image_future)

        loss = self.lambda_recon * (loss_t + loss_tk) + self.lambda_pred * loss_pred

        zero = torch.zeros((), device=device, dtype=loss_t.dtype)
        cls_p_repr = p_t_visible[:, 0]
        m_out = m_real if m_real is not None else torch.zeros_like(p_t_visible)
        cls_m_repr = m_out[:, 0]
        return {
            "loss": loss,
            "loss_t": loss_t, "loss_tn": loss_t, "loss_tk": loss_tk,
            "loss_pred": loss_pred, "loss_pred_short": loss_pred,
            "loss_pred_step": loss_pred, "loss_pred_long": loss_pred,
            "loss_m_jepa": zero, "loss_compose": zero, "loss_dino": zero,
            "loss_var": zero,
            "pred_t": self._unpatchify(patch_pred_t), "pred_tk": self._unpatchify(patch_pred_pred),
            "mask_p": mask_t, "mask_m": mask_t,
            "m_features": m_out[:, 1:], "p_features_t": p_t_visible[:, 1:],
            "p_features_tk": p_tk_visible[:, 1:], "cls_m": cls_m_repr, "cls_p": cls_p_repr,
            "predicted_tk_repr": p_tk_visible, "target_tk_repr": p_tk_visible,
            "m_local_short": m_out, "m_local_step": m_out, "m_local_long": m_out,
            "m_compose_target": m_out, "m_predicted": m_out,
            "student_dino_cls": cls_p_repr,
            "teacher_dino_cls": torch.zeros(B, self.embed_dim, device=device, dtype=loss.dtype),
            "teacher_proto_logits": torch.zeros(B, 1, device=device, dtype=loss.dtype),
        }

    # ----------------------------------------------------------------------
    # CoMP-MAE: M-recon mirror branch
    # ----------------------------------------------------------------------

    def _null_routing(self, B: int, device: torch.device) -> torch.Tensor:
        """Learned null-motion routing helper (replaces an M-encoder(ΔL=0) forward).

        Position-structured (pos_embed_m), not a uniform blur; the routing block
        re-normalizes it with norm_m, so a raw token + APE suffices. Used for the
        P-side in-place recovery (L_t / L_tk).
        """
        tok = self.null_motion_token.expand(B, self.num_patches + 1, -1).to(device)
        return tok + self.pos_embed_m

    def _recon_dL(
        self, m_channel: torch.Tensor, p_helper_full: torch.Tensor, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CoMP-MAE M-recon: masked |ΔL| -> P-grouping routing -> 1ch |ΔL| recovery.
        Loss on masked positions only.

        guard 6: M uses a *separate* masked pass (no full-pass reuse -> no target leak).
        guard 7: per-patch weight w = floor + scale * |ΔL_patch| (static=floor calibration).
        guard 8: mask_ratio_m_recon < P so motion patches remain visible.

        Args:
            m_channel:     [B, 1, H, W] -- |ΔL| target (Case A=ΔL(t,t)≈0 / Case B=ΔL(t,tk))
            p_helper_full: [B, 1+N, D]  -- P-encoder full-pass of frame_t (grad on, guard 4)
        Returns (loss, patch_pred, mask_m).
        """
        B = m_channel.shape[0]
        mask_m = self._random_mask(B, device, self.mask_ratio_m_recon)
        m_visible = self._encode_m_masked(m_channel, mask_m)              # separate masked pass (guard 6)
        m_full = self._inject_mask_tokens(m_visible, mask_m, self.mask_token_m)
        m_state = m_full + self.dec_pos_embed_m                           # mask token inject + APE
        for step in self.m_recon_decoder:
            m_state = step(m_state, p_helper_full)   # owner=M(V), helper=P_full(Q/K) => P->P, gather M
        m_state = self.m_recon_decoder_norm(m_state)
        patch_pred = self.m_recon_head(m_state[:, 1:])                    # [B, N, ps^2 * 1]
        patch_target = self._patchify(m_channel)                         # [B, N, ps^2 * 1]
        err = ((patch_pred - patch_target) ** 2).mean(dim=-1)            # [B, N]
        # per-patch |ΔL| weighting (guard 7); floor>0 keeps a static-patch calibration signal.
        weight = self.m_recon_weight_floor + self.m_recon_weight_scale * patch_target.abs().mean(dim=-1)
        wm = weight * mask_m.float()
        loss = (err * wm).sum() / wm.sum().clamp(min=1e-6)
        return loss, patch_pred, mask_m

    def _forward_pair_comp(self, image_current: torch.Tensor, image_future: torch.Tensor) -> dict:
        """CoMP-MAE forward (2-frame): P-recon (same as the plain control) + M-recon mirror.

        P-side (deliverable encoder): L_t/L_tk = in-place recovery (M=null routing)
                                      + L_pred = future (M=real).
        M-side (grounding):           L_M = Case A (ΔL(t,t)=0) + Case B (ΔL(t,tk)),
                                      P-helper = frame_t only (Sec. 3.1).
        Loss = lambda_recon*(L_t+L_tk) + lambda_pred*L_pred + lambda_M*(caseA_w*L_M_A + L_M_B).
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

        # ── P channels (RGB) + M channels (|ΔL|) ──────────────────────────
        p_channel_t = self.preprocessing.compute_p_channel(image_current)
        p_channel_tk = self.preprocessing.compute_p_channel(image_future)
        m_chan_real = self.preprocessing.compute_m_channel(image_current, image_future)   # ΔL(t,tk)
        m_chan_null = self.preprocessing.compute_m_channel(image_current, image_current)  # ΔL(t,t)=0

        # ── P-recon (same as plain control): in-place x2 (null) + future x1 (real) ─
        mask_t = self._random_mask(B, device, self.mask_ratio_p)
        mask_tk = self._random_mask(B, device, self.mask_ratio_p)
        p_t_visible = self._student_p_encode_visible(p_channel_t, mask_t)
        p_tk_visible = self._student_p_encode_visible(p_channel_tk, mask_tk)
        m_real = self._encode_m_unmasked(m_chan_real)   # full M (P-side routing helper)
        m_null = self._null_routing(B, device)          # learned null-motion token
        loss_t, patch_pred_t = self._predict_pixels(p_t_visible, mask_t, m_null, image_current)
        loss_tk, patch_pred_tk = self._predict_pixels(p_tk_visible, mask_tk, m_null, image_future)
        loss_pred, patch_pred_pred = self._predict_pixels(p_t_visible, mask_t, m_real, image_future)

        # ── M-recon mirror: P-helper = frame_t full-pass only (Sec. 3.1, guard 4) ─
        #   Case B (real ΔL) is the core branch, always run (ensures grad on m_recon).
        #   Case A (static calibration) is probabilistically skipped for efficiency.
        p_helper_t = self._encode_p_unmasked(p_channel_t)
        loss_m_B, patch_pred_dL, mask_m = self._recon_dL(m_chan_real, p_helper_t, device)  # Case B: real ΔL
        run_caseA = self.caseA_prob >= 1.0 or torch.rand(1).item() < self.caseA_prob
        if run_caseA:
            loss_m_A, _, _ = self._recon_dL(m_chan_null, p_helper_t, device)    # Case A: static calibration
        else:
            loss_m_A = torch.zeros((), device=device, dtype=loss_m_B.dtype)
        loss_m_recon = self.caseA_weight * loss_m_A + loss_m_B

        # ── Total ─────────────────────────────────────────────────────────
        loss = (
            self.lambda_recon * (loss_t + loss_tk)
            + self.lambda_pred * loss_pred
            + self.lambda_m_recon * loss_m_recon
        )

        zero = torch.zeros((), device=device, dtype=loss_t.dtype)
        cls_p_repr = p_t_visible[:, 0]
        cls_m_repr = m_real[:, 0]
        return {
            "loss": loss,
            "loss_t": loss_t, "loss_tn": loss_t, "loss_tk": loss_tk,
            "loss_pred": loss_pred, "loss_pred_short": loss_pred,
            "loss_pred_step": loss_pred, "loss_pred_long": loss_pred,
            # M-recon diagnostics (reuse the loss_m_jepa slot so existing loggers surface it)
            "loss_m_jepa": loss_m_recon, "loss_m_recon": loss_m_recon,
            "loss_m_caseA": loss_m_A, "loss_m_caseB": loss_m_B,
            "loss_compose": zero, "loss_dino": zero, "loss_var": zero,
            "pred_t": self._unpatchify(patch_pred_t), "pred_tk": self._unpatchify(patch_pred_pred),
            "mask_p": mask_t, "mask_m": mask_m,
            "m_features": m_real[:, 1:], "p_features_t": p_t_visible[:, 1:],
            "p_features_tk": p_tk_visible[:, 1:], "cls_m": cls_m_repr, "cls_p": cls_p_repr,
            "predicted_tk_repr": p_tk_visible, "target_tk_repr": p_tk_visible,
            "m_local_short": m_real, "m_local_step": m_real, "m_local_long": m_real,
            "m_compose_target": m_real, "m_predicted": m_real,
            "student_dino_cls": cls_p_repr,
            "teacher_dino_cls": torch.zeros(B, self.embed_dim, device=device, dtype=loss.dtype),
            "teacher_proto_logits": torch.zeros(B, 1, device=device, dtype=loss.dtype),
        }

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def forward(
        self,
        image_current: torch.Tensor,          # frame_t   [B, 3, 224, 224], range [0, 1]
        image_short:   torch.Tensor,          # frame_t+k [B, 3, 224, 224], range [0, 1]
        image_future:  torch.Tensor = None,   # unused in pair mode (kept for API compat)
    ) -> dict:
        """Dispatch to the CoMP-MAE (comp_mae) or plain control (pixel_pred) pair forward.

        The returned dict always exposes the same keys (extra keys are zero
        placeholders) so downstream probing / logging code is objective-agnostic.
        The representation used downstream is the P encoder output (`p_features_t`
        / `cls_p`); see the probing harness in eval/.
        """
        if self.comp_mae:
            return self._forward_pair_comp(image_current, image_short)
        return self._forward_pair_pixel(image_current, image_short)

    def compute_loss(
        self,
        image_current: torch.Tensor,
        image_short: torch.Tensor,
        image_future: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training-loop convenience wrapper -> (loss, pred_tk)."""
        out = self.forward(image_current, image_short, image_future)
        return out["loss"], out["pred_tk"]
