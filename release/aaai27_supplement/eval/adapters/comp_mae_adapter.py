"""CoMP-MAE encoder adapter -- P_t (+) P_tk concat for the BC-T policy.

Wraps CoMPMAE(pair_mode=True, use_sobel=False, masked_anchor=True):
  - P channel = raw RGB (3ch), identical to the pretraining input ([0,1] raw)
    -> train/inference preprocessing parity holds.
  - Input: a (prev, curr) 2-frame pair -> P encoder x2 -> patch mean -> concat.
  - Output: (B, T, 2*D) = mean-pooled P_t patches (+) mean-pooled P_tk patches
    (or 3*D if use_m=True, appending the M stream).
  - Only the P encoder (and optionally the M encoder) is used; the motion-routing
    decoder is not exercised at inference.

The architecture (embed_dim / m_depth / comp_mae) is inferred from the checkpoint,
so this adapter handles both ViT-S (CoMP-MAE-S, D=384) and ViT-B (D=768).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import EncoderAdapter


def _infer_routing_mode(state_dict) -> str:
    """Infer the motion-routing mode from checkpoint parameter names.

    v_from_m (standard cross-attn, plain control) has routing.q_p / routing.kv_m;
    v_from_p (value-ownership routing, CoMP-MAE) has routing.qk_m / routing.v_p.
    """
    if any(".routing.q_p." in k or ".routing.kv_m." in k for k in state_dict):
        return "v_from_m"
    return "v_from_p"


class CoMPMAEAdapter(EncoderAdapter):
    """P_t (+) P_tk readout. pooling='mean' or 'attentive' (per-stream learnable
    query; only the query is trainable while the encoder stays frozen)."""

    img_size = 224

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
        device: str = "cpu",
        pooling: str = "mean",
        use_m: bool = False,
        embed_dim: Optional[int] = None,
        m_depth: Optional[int] = None,
        comp_mae: Optional[bool] = None,
        **kwargs,  # ignore extra kwargs passed by build_adapter
    ):
        super().__init__(freeze=freeze)
        from model.comp_mae import CoMPMAE

        if checkpoint_path is not None:
            # finetune path: infer arch from the pretrain checkpoint + load encoder weights.
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            sd = ckpt.get("model_state_dict", ckpt)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            # arch inference (head_dim=64 standard).
            _ed = next(v.shape[-1] for k, v in sd.items() if k == "pos_embed_p")
            _md = len({k.split(".")[1] for k in sd if k.startswith("blocks_m.")})
            _comp = any("m_recon" in k for k in sd)
            _routing = _infer_routing_mode(sd)
        else:
            # self-contained path (e.g. rollout): arch given explicitly, weights
            # overwritten externally (from the policy state dict).
            assert None not in (embed_dim, m_depth, comp_mae), (
                "checkpoint_path=None requires embed_dim/m_depth/comp_mae "
                "(inferred from the policy state dict and passed in)"
            )
            sd = None
            _ed, _md, _comp = embed_dim, m_depth, comp_mae
            _routing = "v_from_p" if _comp else "v_from_m"
        self.base_dim = _ed
        self.use_m = use_m
        # P_t (+) P_tk [(+) M] -- use_m adds the M (|ΔL| curr-prev) stream.
        self.n_streams = 3 if use_m else 2
        self.embed_dim = _ed * self.n_streams
        self.pooling = pooling

        # Build the model exactly as the checkpoint was trained so it loads
        # cleanly. pixel_pred=(not comp_mae) selects the plain-control objective;
        # only the (objective-independent) P/M encoders are used at inference, so
        # this choice does not affect the extracted representation.
        self.model = CoMPMAE(
            embed_dim=_ed, num_heads=_ed // 64, m_depth=_md,
            comp_mae=_comp, pixel_pred=not _comp, routing_mode=_routing,
            pair_mode=True, use_sobel=False, masked_anchor=True,
        ).to(device)
        if sd is not None:
            missing, _ = self.model.load_state_dict(sd, strict=False)
            enc_missing = [k for k in missing
                           if k.startswith(("blocks_p", "patch_embed_p", "cls_token_p", "pos_embed_p"))]
            assert not enc_missing, f"P encoder weights not loaded: {enc_missing[:5]}"
        self.model.eval()

        # attentive pooling: one learnable query per stream (single-head, minimal capacity).
        if pooling == "attentive":
            self.pool_q = nn.Parameter(torch.randn(self.n_streams, _ed) * 0.02)
            self.pool_scale = _ed ** -0.5
        elif pooling != "mean":
            raise ValueError(f"pooling must be 'mean'|'attentive', got {pooling}")

        if freeze:
            # Freeze only the encoder backbone; the attentive pool_q stays trainable.
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

        self.prev_obs: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        # Frozen encoder always stays in eval mode (train<->inference parity).
        super().train(mode)
        self.model.eval()
        return self

    def reset(self) -> None:
        self.prev_obs = None

    def _attn_pool(self, tokens: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # tokens (BT, N, D), q (D,) -> softmax-weighted patch sum (BT, D)
        attn = (tokens @ q) * self.pool_scale          # (BT, N)
        attn = attn.softmax(dim=-1)
        return (attn.unsqueeze(-1) * tokens).sum(dim=1)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = obs_seq.shape

        # P_t = previous frame, P_tk = current frame (matches the training pair order).
        if T > 1:
            prev = torch.cat([obs_seq[:, :1], obs_seq[:, :-1]], dim=1)
        else:
            if self.prev_obs is None:
                prev = obs_seq.clone()
            else:
                prev = self.prev_obs
            self.prev_obs = obs_seq.detach()

        img_prev = prev.reshape(B * T, C, H, W)
        img_curr = obs_seq.reshape(B * T, C, H, W)

        # Frozen encoder forward needs no grad; the trainable pool_q acts on the
        # encoder *output* outside no_grad, so gradients flow only into pool_q.
        with torch.set_grad_enabled(not self._freeze):
            # no-Sobel: compute_p_channel returns [0,1] RGB 3ch unchanged (== training input)
            p_prev = self.model.preprocessing.compute_p_channel(img_prev)
            p_curr = self.model.preprocessing.compute_p_channel(img_curr)

            tok_t = self.model._encode_p_unmasked(p_prev)[:, 1:]    # (B*T, N, D)
            tok_tk = self.model._encode_p_unmasked(p_curr)[:, 1:]

            feats = [tok_t, tok_tk]
            if self.use_m:
                # M = |ΔL| (curr, prev) motion. NOTE: rollout gap=1 (consecutive frames)
                # vs. training gap ~15 -- a distribution shift to be aware of.
                m_chan = self.model.preprocessing.compute_m_channel(img_prev, img_curr)
                feats.append(self.model._encode_m_unmasked(m_chan)[:, 1:])

        if self.pooling == "attentive":
            pooled = [self._attn_pool(t, self.pool_q[i]) for i, t in enumerate(feats)]
        else:
            pooled = [t.mean(dim=1) for t in feats]

        token = torch.cat(pooled, dim=-1)
        return token.reshape(B, T, -1)
