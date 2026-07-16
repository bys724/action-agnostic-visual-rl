"""Encoder-swap variant of the LIBERO official BC-Transformer policy.

Subclasses the LIBERO `BCTransformerPolicy` and replaces only the image
encoder with our adapter. Everything else (TemporalTransformer + GMM head +
language token + extra modality) is used exactly as in the LIBERO official
implementation.

Design notes:
  - Encoder-stage FiLM language conditioning is removed. Language is injected
    only via the BC-T text token, so all encoders are compared fairly.
  - Each encoder's native input format is handled inside the adapter itself.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import robomimic.utils.tensor_utils as TensorUtils

# LIBERO is imported from the conda env site-packages (documented runtime dep).
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.modules.language_modules import (
    IdentityEncoder,
    MLPEncoder,
)
from libero.lifelong.models.modules.transformer_modules import (
    SinusoidalPositionEncoding,
    TransformerDecoder,
)
from libero.lifelong.models.policy_head import GMMHead
from libero.lifelong.models.bc_transformer_policy import ExtraModalityTokens

from eval.adapters import EncoderAdapter, build_adapter


class AdaptedBCTransformerPolicy(BasePolicy):
    """LIBERO BCTransformerPolicy structure + our encoder adapter.

    The original `spatial_encode` maps camera inputs through ResNet+FiLM.
    This class instead maps camera inputs through our adapter
    (`EncoderAdapter`) -> projection -> BC-T embed_size.

    All cameras share a single adapter (to save memory). Per-camera adapters
    could be added later if needed.
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        embed_size = policy_cfg.embed_size

        # -- 1. Encoder adapter (shared by all cameras) --------------------
        encoder_cfg = cfg.encoder
        self.encoder_type = encoder_cfg.type
        self.adapter: EncoderAdapter = build_adapter(
            encoder_type=encoder_cfg.type,
            checkpoint_path=encoder_cfg.get("checkpoint", None),
            **dict(encoder_cfg.get("adapter_kwargs", {})),
        )
        self.adapter_embed_dim = self.adapter.embed_dim

        # Collect camera names (rgb/depth modality)
        self.image_names = [
            n for n in shape_meta["all_shapes"].keys()
            if "rgb" in n or "depth" in n
        ]
        if len(self.image_names) == 0:
            raise ValueError("No image modality found in shape_meta")

        # -- 2. Per-camera projection (adapter output -> embed_size) -------
        # Separate projection per camera (different viewpoints have different
        # feature distributions).
        self.image_projections = nn.ModuleDict({
            name: nn.Linear(self.adapter_embed_dim, embed_size)
            for name in self.image_names
        })

        # -- 3. Language encoder (LIBERO official -- task_emb -> embed_size)
        lang_cfg = policy_cfg.language_encoder.network_kwargs
        lang_cfg = dict(lang_cfg)
        lang_cfg["output_size"] = embed_size
        self.language_encoder = MLPEncoder(**lang_cfg)

        # -- 4. Extra modality (joint/gripper/ee -> embed_size) ------------
        self.extra_encoder = ExtraModalityTokens(
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
            extra_num_layers=policy_cfg.extra_num_layers,
            extra_hidden_size=policy_cfg.extra_hidden_size,
            extra_embedding_size=embed_size,
        )

        # -- 5. Temporal Transformer (LIBERO official) ---------------------
        pe_cfg = dict(policy_cfg.temporal_position_encoding.network_kwargs)
        pe_cfg["input_size"] = embed_size
        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(**pe_cfg)

        self.temporal_transformer = TransformerDecoder(
            input_size=embed_size,
            num_layers=policy_cfg.transformer_num_layers,
            num_heads=policy_cfg.transformer_num_heads,
            head_output_size=policy_cfg.transformer_head_output_size,
            mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
            dropout=policy_cfg.transformer_dropout,
        )

        # -- 6. GMM policy head (LIBERO official) --------------------------
        head_cfg = dict(policy_cfg.policy_head.network_kwargs)
        head_cfg["input_size"] = embed_size
        head_cfg["output_size"] = shape_meta["ac_dim"]
        loss_cfg = dict(policy_cfg.policy_head.loss_kwargs)
        self.policy_head = GMMHead(**loss_cfg, **head_cfg)

        # rollout state
        self.latent_queue: List[torch.Tensor] = []
        self.max_seq_len = policy_cfg.transformer_max_seq_len

    # ====================================================================
    # Image encoding via adapter
    # ====================================================================

    def _encode_camera(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """data["obs"][name] -> BC-T embed_size token sequence.

        x: (B, T, C, H, W) -- caller must already resize to the adapter's
           native size.
        return: (B, T, 1, embed_size) -- BC-T adds a modality dim.
        """
        # adapter: (B, T_in, C, H, W) -> (B, T_out, adapter_dim)
        # T_in == T_out for single-frame / pair encoders (video encoders may
        # shrink the time dim).
        tokens = self.adapter(x)  # (B, T_out, adapter_dim)
        proj = self.image_projections[name](tokens)  # (B, T_out, embed_size)
        return proj.unsqueeze(2)  # (B, T_out, 1, embed_size)

    # ====================================================================
    # spatial / temporal encode (LIBERO official flow preserved)
    # ====================================================================

    def temporal_encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, num_modalities, E) -> (B, T, E) [last token (image)]."""
        pos_emb = self.temporal_position_encoding_fn(x)
        x = x + pos_emb.unsqueeze(1)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)
        x = TensorUtils.join_dimensions(x, 1, 2)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)
        return x[:, :, 0]

    def spatial_encode(self, data: dict) -> torch.Tensor:
        """data["obs"] -> (B, T, num_modalities, E).

        Modality order: text, extra, image (same as LIBERO official).
        T is determined by the image adapter's output length (a video encoder
        may shorten it relative to the input).
        """
        # 1) Extra (joint/gripper/ee): (B, T_ext, num_extra, E)
        extra = self.extra_encoder(data["obs"])
        T_ext = extra.shape[1]

        # 2) Image encoding via adapter
        image_tokens = []
        T_img = None
        for name in self.image_names:
            img_seq = data["obs"][name]  # (B, T_in, C, H, W)
            tok = self._encode_camera(name, img_seq)  # (B, T_out, 1, E)
            image_tokens.append(tok)
            T_img = tok.shape[1]

        # If the image encoder shortens the time dim (T_img < T_ext), align the
        # other modalities to T_img by keeping the most recent T_img timesteps
        # (preserves causality).
        if T_img is not None and T_img != T_ext:
            extra = extra[:, -T_img:]
            T_ext = T_img

        # 3) Language: (B, E) -> (B, T, 1, E)
        text = self.language_encoder(data)
        text_token = text.view(text.shape[0], 1, 1, -1).expand(-1, T_ext, -1, -1)

        # Concat: text + extra + images -> (B, T, num_modalities, E)
        encoded = [text_token, extra] + image_tokens
        return torch.cat(encoded, dim=-2)

    def forward(self, data: dict) -> dict:
        x = self.spatial_encode(data)
        x = self.temporal_encode(x)
        return self.policy_head(x)

    def get_action(self, data: dict):
        self.eval()
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            x = self.spatial_encode(data)  # (B, 1, num_mod, E) single step
            self.latent_queue.append(x)
            if len(self.latent_queue) > self.max_seq_len:
                self.latent_queue.pop(0)
            x = torch.cat(self.latent_queue, dim=1)
            x = self.temporal_encode(x)
            dist = self.policy_head(x[:, -1])
        action = dist.sample().detach().cpu()
        return action.view(action.shape[0], -1).numpy()

    def reset(self):
        self.latent_queue = []
        if hasattr(self.adapter, "reset"):
            self.adapter.reset()
