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
        "pool": "patch_mean_skip_cls",  # CLS at index 0, skip it
    },
    "siglip": {
        "hf_id": "google/siglip-base-patch16-224",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "img_size": 224,
        "pool": "patch_mean_no_cls",   # SigLIP 출력엔 CLS 없음
    },
    "vc1": {
        # VC-1은 facebookresearch/eai-vc 패키지 (`vc_models`)에서 로드
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "img_size": 224,
        "pool": "vc1_direct",          # vc_models 호출 결과는 (B, 768) CLS
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
        self.pool = spec["pool"]
        self.img_size = spec["img_size"]

        # ── Backbone 로드 (probe_action.py와 동일 정책) ─────────────────────
        if encoder_type == "vc1":
            from vc_models.models.vit import model_utils
            model, _, _, _ = model_utils.load_model(model_utils.VC1_BASE_NAME)
            self.model = model
            hidden_size = 768  # ViT-B
        elif encoder_type == "siglip":
            # SiglipModel은 forward에서 input_ids 요구 → VisionModel 단독 로드
            from transformers import SiglipVisionModel
            self.model = SiglipVisionModel.from_pretrained(spec["hf_id"])
            hidden_size = self.model.config.hidden_size
        else:  # dinov2
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(spec["hf_id"])
            hidden_size = self.model.config.hidden_size
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

    def reset(self) -> None:
        self.prev_obs = None

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 3, H, W) in [0, 1]
        return (x - self._mean) / self._std

    def _encode_one(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) → (N, hidden_size)."""
        x = self._normalize(x)
        if self.pool == "vc1_direct":
            # vc_models VC-1: encoder(img) → (N, 768) CLS
            return self.model(x)
        # HF VisionModel: pixel_values 인자
        out = self.model(pixel_values=x)
        hidden = out.last_hidden_state  # (N, *, D)
        if self.pool == "patch_mean_skip_cls":
            return hidden[:, 1:].mean(dim=1)
        if self.pool == "patch_mean_no_cls":
            return hidden.mean(dim=1)
        raise ValueError(f"Unknown pool: {self.pool}")

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
