"""Single-frame encoder 어댑터 (DINOv2 / SigLIP / VC-1).

각 인코더는 native 1-frame 입력. D3 정책상 motion 정보 노출을 위해
(obs_{t-1}, obs_t) 둘을 각자 1-frame씩 인코딩 후 concat → 1 token/timestep.

출력 차원: 2 × encoder_hidden (예: 2 × 768 = 1536)
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import EncoderAdapter


# Encoder type → (HF model id, normalization mean/std, hidden_size 추정)
_ENCODER_SPECS = {
    "dinov2": {
        "hf_id": "facebook/dinov2-base",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "img_size": 224,
        "use_cls": False,  # patch mean
    },
    "siglip": {
        "hf_id": "google/siglip-base-patch16-224",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "img_size": 224,
        "use_cls": False,
    },
    "vc1": {
        "hf_id": "facebook/vc1-base",  # HF 미등록 시 ckpt 직접 로딩 fallback
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "img_size": 224,
        "use_cls": True,  # VC-1 공식: CLS 사용
    },
}


class SingleFrameAdapter(EncoderAdapter):
    """1-frame encoder × 2-frame concat 어댑터.

    DINOv2 / SigLIP은 huggingface AutoModel로 로드.
    VC-1은 HF 미등록일 가능성 — checkpoint_path 명시 시 우선 사용.
    """

    def __init__(
        self,
        encoder_type: str,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__(freeze=freeze)
        spec = _ENCODER_SPECS[encoder_type]
        self.encoder_type = encoder_type
        self.use_cls = spec["use_cls"]
        self.img_size = spec["img_size"]

        # ── Backbone 로드 ─────────────────────────────────────────────────
        if encoder_type == "vc1":
            self._load_vc1(spec, checkpoint_path)
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(spec["hf_id"])

        # ── Hidden dim ────────────────────────────────────────────────────
        if hasattr(self.model, "config"):
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                # SigLIPVisionModel은 vision_config.hidden_size에 있을 수 있음
                hidden_size = getattr(
                    getattr(self.model.config, "vision_config", None),
                    "hidden_size", 768,
                )
        else:
            hidden_size = 768
        self._hidden_size = hidden_size
        self.embed_dim = 2 * hidden_size  # concat (prev, curr)

        # ── Normalization buffers (encoder native) ────────────────────────
        self.register_buffer(
            "_mean",
            torch.tensor(spec["mean"]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(spec["std"]).view(1, 3, 1, 1),
            persistent=False,
        )

        if freeze:
            self.freeze_encoder()
        self.prev_obs: Optional[torch.Tensor] = None

    def _load_vc1(self, spec: dict, checkpoint_path: Optional[str]) -> None:
        """VC-1 로딩. HF id 시도 → 실패 시 timm/torch.hub fallback."""
        from transformers import AutoModel
        try:
            self.model = AutoModel.from_pretrained(spec["hf_id"])
        except Exception as e_hf:
            if checkpoint_path is None:
                raise RuntimeError(
                    f"VC-1 HF load 실패 ({e_hf}). "
                    "checkpoint_path를 명시하거나 facebook/eai-vc 코드 통합 필요."
                )
            # fallback: timm ViT-B 16 (224)에 ckpt 로드
            import timm
            model = timm.create_model(
                "vit_base_patch16_224", pretrained=False, num_classes=0,
            )
            sd = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"VC-1 timm fallback: missing={len(missing)}, unexpected={len(unexpected)}")
            self.model = model

    def reset(self) -> None:
        self.prev_obs = None

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 3, H, W) in [0, 1]
        return (x - self._mean) / self._std

    def _encode_one(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) → (N, hidden_size)."""
        x = self._normalize(x)
        # HF transformers: pixel_values=...
        out = self.model(pixel_values=x) if hasattr(self.model, "config") else self.model(x)
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state  # (N, 1+P, D)
            if self.use_cls:
                return hidden[:, 0]
            return hidden[:, 1:].mean(dim=1)
        # timm fallback: forward_features → (N, D) directly
        return out

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = obs_seq.shape

        if T > 1:
            prev = torch.cat([obs_seq[:, :1], obs_seq[:, :-1]], dim=1)
        else:
            prev = obs_seq.clone() if self.prev_obs is None else self.prev_obs
            self.prev_obs = obs_seq.detach()

        prev_flat = prev.reshape(B * T, C, H, W)
        curr_flat = obs_seq.reshape(B * T, C, H, W)

        # 두 프레임 각각 인코딩 → concat
        was_training = self.model.training
        self.model.eval()
        with torch.set_grad_enabled(not self._freeze):
            e_prev = self._encode_one(prev_flat)  # (B*T, D)
            e_curr = self._encode_one(curr_flat)  # (B*T, D)
        if was_training and not self._freeze:
            self.model.train()

        token = torch.cat([e_prev, e_curr], dim=-1)  # (B*T, 2D)
        return token.reshape(B, T, -1)
