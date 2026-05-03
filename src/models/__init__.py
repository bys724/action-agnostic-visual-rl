"""Models for action-agnostic visual representation learning.

| Model | Status | 비고 |
|-------|--------|------|
| TwoStreamV11Model | **active main** | Motion-guided attention routing + dual-target. ep44 final champion (R²=0.288) |
| VideoMAEModel | **active baseline** | Masked autoencoding controlled comparison. ep50 R²=0.326 |
| TwoStreamModel | **legacy reference** | M/P channel separation with CLS exchange (v4~v10). 폐기 lineup이지만 v6 챔피언 (R²=0.259) 재현 / probing / 시각화에 keep |
| TwoStreamV12Model | **post-CoRL follow-up** | v11 + CLS-level semantic residual + EMA teacher. 코드만 있음, 학습 X |

Phase 마감 후 legacy / follow-up 정리 검토 — `docs/refactor_plan_2026-05-03.md` 참조.
"""

from .two_stream import TwoStreamModel, TwoStreamEncoder
from .two_stream_v11 import TwoStreamV11Model, TwoStreamV11Encoder
from .two_stream_v12 import TwoStreamV12Model
from .videomae import VideoMAEModel, VideoMAEEncoderForVLA as VideoMAEEncoder
from .common import TwoStreamPreprocessing

__all__ = [
    "TwoStreamModel",
    "TwoStreamV11Model",
    "TwoStreamV12Model",
    "VideoMAEModel",
    "TwoStreamEncoder",
    "TwoStreamV11Encoder",
    "VideoMAEEncoder",
    "TwoStreamPreprocessing",
]
