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

Deprecated (paper 미사용, 폐기 lineup, 코드만 보존):
| Model | Status | 비고 |
|-------|--------|------|
| TwoStreamV12Model | deprecated | v11 + CLS semantic residual + EMA teacher. sanity 단계 cls_p collapse |
| TwoStreamV13Model | deprecated | Dual-frame recon + motion-routed latent + DINO global CLS. ep10+ uniform collapse |
| TwoStreamV14Model | deprecated | Stream-wise paradigm specialization (P=MAE+V-JEPA, M=DINO). ep20 cancel + EgoDex probing R²=-0.065 |
"""

from .two_stream import TwoStreamModel, TwoStreamEncoder
from .two_stream_v11 import TwoStreamV11Model, TwoStreamV11Encoder
from .two_stream_v12 import TwoStreamV12Model
from .two_stream_v13 import TwoStreamV13Model
from .two_stream_v14 import TwoStreamV14Model
from .two_stream_v15 import TwoStreamV15Model
from .videomae import VideoMAEModel, VideoMAEEncoderForVLA as VideoMAEEncoder
from .common import TwoStreamPreprocessing

__all__ = [
    "TwoStreamModel",
    "TwoStreamV11Model",
    "TwoStreamV12Model",
    "TwoStreamV13Model",
    "TwoStreamV14Model",
    "TwoStreamV15Model",
    "VideoMAEModel",
    "TwoStreamEncoder",
    "TwoStreamV11Encoder",
    "VideoMAEEncoder",
    "TwoStreamPreprocessing",
]
