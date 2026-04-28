"""Two-Stream v11 어댑터 (A+D' mode = M encoder patches + P motion-routing 후).

probe_action_v11.py의 step-by-step forward를 재사용.
출력: (B, T, 1536) — A (768) ⊕ D' (768)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import sys
import torch

from .base import EncoderAdapter

# probe_action_v11이 src.models를 사용하므로 project root sys.path 보장
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TwoStreamV11Adapter(EncoderAdapter):
    """v11 (A+D') 어댑터. 2-frame pair 입력, A+D' concat 출력.

    학습 (T>1): obs_seq에서 t=0은 (obs[0], obs[0]) 복제, t≥1은 (obs[t-1], obs[t])
    Rollout (T=1): self.prev_obs 버퍼와 pair 형성. reset()으로 episode 시작 초기화.
    """

    embed_dim = 768 * 2  # A + D'
    img_size = 224

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        p_depth: int = 12,
        m_depth: int = 6,
        freeze: bool = True,
        device: str = "cpu",
    ):
        super().__init__(freeze=freeze)
        from scripts.eval.probe_action_v11 import (
            _full_forward_with_d_prime,
            load_v11_model,
        )

        if checkpoint_path is None:
            raise ValueError("Two-Stream v11 어댑터는 checkpoint 필수")

        self.model = load_v11_model(
            checkpoint_path, p_depth=p_depth, m_depth=m_depth, device=device,
        )
        self._step_forward = _full_forward_with_d_prime

        if freeze:
            self.freeze_encoder()

        # rollout 상태
        self.register_buffer("_prev_obs_buf", torch.zeros(0), persistent=False)
        self.prev_obs: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.prev_obs = None

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        # obs_seq: (B, T, 3, H, W) in [0,1]
        B, T, C, H, W = obs_seq.shape

        if T > 1:
            # 학습: 시퀀스에서 직접 pair 형성
            prev = torch.cat([obs_seq[:, :1], obs_seq[:, :-1]], dim=1)
        else:
            # rollout: prev_obs 버퍼 사용
            if self.prev_obs is None:
                prev = obs_seq.clone()  # episode 첫 step → duplicate
            else:
                prev = self.prev_obs
            self.prev_obs = obs_seq.detach()

        img_prev = prev.reshape(B * T, C, H, W)
        img_curr = obs_seq.reshape(B * T, C, H, W)

        # _full_forward_with_d_prime은 @torch.no_grad — frozen이라 OK
        out = self._step_forward(self.model, img_prev, img_curr)

        # A = M encoder patch mean (CLS 제외)
        A = out["m_encoded"][:, 1:].mean(dim=1)
        # D' = P state after motion-routing (interpreter_2 전)
        D_prime = out["p_state_routing"][:, 1:].mean(dim=1)

        token = torch.cat([A, D_prime], dim=-1)  # (B*T, 1536)
        return token.reshape(B, T, -1)
