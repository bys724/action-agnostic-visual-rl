"""Models for action-agnostic visual representation learning.

Active (paper main + active baseline):
| Model | Status | 비고 |
|-------|--------|------|
| TwoStreamV15Model | **🏆 paper main** | Layered specialization: predictor-only V-JEPA P + V-JEPA-M (Option B) + L_compose + 3-frame triple. ep32 P_t+P_tk = +0.390 ★ |
| VideoMAEModel | **active baseline** | Masked autoencoding controlled comparison. ep50 R²=0.326 |

Reference (paper §method history 인용용, 학습 ckpt 보존):
| Model | Status | 비고 |
|-------|--------|------|
| TwoStreamV11Model | reference | Motion-guided attention routing + dual-target. ep44 A+B+D' = +0.288 (이전 champion) |
| TwoStreamModel | reference | M/P channel separation with CLS exchange (v4~v10). v6 ep8 = +0.259 |

(v12/13/14 = deprecated, 2026-06-23 코드 삭제. 결과/이력은 git history·docs 참조.)
"""

from .two_stream import TwoStreamModel, TwoStreamEncoder
from .two_stream_v11 import TwoStreamV11Model, TwoStreamV11Encoder
from .two_stream_v15 import TwoStreamV15Model
from .videomae import VideoMAEModel, VideoMAEEncoderForVLA as VideoMAEEncoder
from .siammae import SiamMAEModel, SiamMAEEncoderForVLA as SiamMAEEncoder
from .common import TwoStreamPreprocessing

__all__ = [
    "TwoStreamModel",
    "TwoStreamV11Model",
    "TwoStreamV15Model",
    "VideoMAEModel",
    "SiamMAEModel",
    "TwoStreamEncoder",
    "TwoStreamV11Encoder",
    "VideoMAEEncoder",
    "SiamMAEEncoder",
    "TwoStreamPreprocessing",
]
