"""V-JEPA 2.1 ViT-B 384 어댑터 (16-frame 누적 sliding window).

각 timestep마다 [obs_{t-15}, ..., obs_t] 16-frame clip을 V-JEPA에 forward.

학습 시: caller가 (B, T_obs + 15, 3, 384, 384) sequence 제공 → (B, T_obs, embed_dim)
        T_obs = T_in - 15. (LIBERO 표준 T_obs=10 → 25-frame 입력 필요)
Rollout: T_in=1 → 내부 history_buf (max 15 obs) 사용, episode 시작 시 obs_0 padding.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

from .base import EncoderAdapter

_VJEPA_REPO = Path(__file__).resolve().parents[3] / "external" / "vjepa2"


def _ensure_vjepa_path():
    p = str(_VJEPA_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


class VJEPA2Adapter(EncoderAdapter):
    """V-JEPA 2.1 ViT-B 384, 16-frame 누적 sliding window."""

    img_size = 384
    num_frames_window = 16  # tubelet_size=2 → 짝수
    history_required = 15   # = num_frames_window - 1

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__(freeze=freeze)
        if checkpoint_path is None:
            checkpoint_path = (
                "/proj/external_group/mrg/checkpoints/vjepa2_official/"
                "vjepa2_1_vitb_384.pt"
            )

        _ensure_vjepa_path()
        from app.vjepa_2_1.models.vision_transformer import vit_base

        encoder = vit_base(
            patch_size=16,
            img_size=(self.img_size, self.img_size),
            num_frames=self.num_frames_window,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("target_encoder", ckpt)
        # 'module.' / 'backbone.' prefix 제거
        sd = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in sd.items()
        }
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        if len(missing) > 5 or len(unexpected) > 5:
            print(f"V-JEPA 2.1 load: missing={len(missing)} unexpected={len(unexpected)} "
                  f"(missing first 3: {missing[:3]})")

        self.model = encoder
        self.embed_dim = encoder.embed_dim  # 768

        # ImageNet normalization (V-JEPA 2 공식 preprocess)
        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1),
            persistent=False,
        )

        if freeze:
            self.freeze_encoder()

        # rollout 상태: 과거 15 obs (or fewer)
        self.history_buf: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.history_buf = None

    def _normalize_clip(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (B, C, T, H, W) → mean/std broadcast (1, 3, 1, 1, 1)
        return (clip - self._mean) / self._std

    def _encode_clip(self, clip_btchw: torch.Tensor) -> torch.Tensor:
        """clip: (B, T, 3, H, W) → (B, embed_dim).

        Mean-pool over all output spatio-temporal tokens.
        """
        # V-JEPA video input: (B, C, T, H, W)
        video = clip_btchw.permute(0, 2, 1, 3, 4).contiguous()
        video = self._normalize_clip(video)

        was_training = self.model.training
        self.model.eval()
        with torch.set_grad_enabled(not self._freeze):
            tokens = self.model(video)  # (B, num_st_tokens, D)
        if was_training and not self._freeze:
            self.model.train()
        return tokens.mean(dim=1)  # (B, D)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """obs_seq:
          · 학습: (B, T_obs + 15, 3, H, W) → 출력 (B, T_obs, embed_dim)
          · Rollout: (B, 1, 3, H, W) → 출력 (B, 1, embed_dim)
        """
        B, T_in, C, H, W = obs_seq.shape
        N = self.num_frames_window
        Hpad = self.history_required

        if T_in == 1:
            # ── Rollout 단일 step ───────────────────────────────────────
            if self.history_buf is None:
                # 첫 step: 모든 16 frame을 obs[0]으로 채움
                clip = obs_seq.expand(-1, N, -1, -1, -1)
                self.history_buf = obs_seq.detach().clone()
            else:
                buf_len = self.history_buf.shape[1]
                if buf_len < Hpad:
                    # padding: buffer 첫 frame replicate
                    n_pad = Hpad - buf_len
                    pad = self.history_buf[:, :1].expand(-1, n_pad, -1, -1, -1)
                    hist = torch.cat([pad, self.history_buf], dim=1)  # (B, 15, ...)
                else:
                    hist = self.history_buf[:, -Hpad:]
                clip = torch.cat([hist, obs_seq], dim=1)  # (B, 16, ...)
                # buffer 업데이트 (max 15)
                new_buf = torch.cat([self.history_buf, obs_seq.detach()], dim=1)
                self.history_buf = new_buf[:, -Hpad:]

            emb = self._encode_clip(clip)  # (B, D)
            return emb.unsqueeze(1)        # (B, 1, D)

        # ── 학습: 시퀀스 sliding window ─────────────────────────────────
        T_out = T_in - Hpad
        if T_out <= 0:
            raise ValueError(
                f"V-JEPA 2.1 학습 입력 부족: T_in={T_in} < {N}. "
                f"caller가 obs sequence를 (T_obs+{Hpad}) 길이로 제공해야 함."
            )

        # 각 t에 대해 obs_seq[:, t:t+N] → clip
        # vectorize via unfold (메모리 효율은 후속 최적화)
        clips = []
        for t in range(T_out):
            clips.append(obs_seq[:, t:t + N])  # (B, 16, 3, H, W)
        clips_b = torch.stack(clips, dim=1)    # (B, T_out, 16, 3, H, W)
        clips_flat = clips_b.reshape(B * T_out, N, C, H, W)

        emb = self._encode_clip(clips_flat)    # (B*T_out, D)
        return emb.reshape(B, T_out, -1)
