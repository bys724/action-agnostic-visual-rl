"""Models for action-agnostic visual representation learning.

| Model | Status | 비고 |
|-------|--------|------|
| TwoStreamV14Model | **post-CoRL follow-up** | Stream-wise paradigm specialization. P=MAE+V-JEPA, M=DINO. v13 paradigm conflict 진단 후 redesign |
| TwoStreamV13Model | **active V3 main** | Dual-frame recon + motion-routed latent + DINO global CLS. v12 cls_p collapse 본질 해결 시도 (학습 33833830 진행 중) |
| TwoStreamV11Model | **active baseline** | Motion-guided attention routing + dual-target. ep44 final champion (R²=0.288) |
| VideoMAEModel | **active baseline** | Masked autoencoding controlled comparison. ep50 R²=0.326 |
| TwoStreamModel | **legacy reference** | M/P channel separation with CLS exchange (v4~v10). 폐기 lineup이지만 v6 챔피언 (R²=0.259) 재현 / probing / 시각화에 keep |
| TwoStreamV12Model | **post-CoRL follow-up** | v11 + CLS-level semantic residual + EMA teacher. 코드만 있음, sanity 단계에서 cls_p collapse → v13으로 redesign |

Phase 마감 후 legacy / follow-up 정리 검토 — `docs/refactor_plan_2026-05-03.md` 참조.
"""

from .two_stream import TwoStreamModel, TwoStreamEncoder
from .two_stream_v11 import TwoStreamV11Model, TwoStreamV11Encoder
from .two_stream_v12 import TwoStreamV12Model
from .two_stream_v13 import TwoStreamV13Model
from .two_stream_v14 import TwoStreamV14Model
from .videomae import VideoMAEModel, VideoMAEEncoderForVLA as VideoMAEEncoder
from .common import TwoStreamPreprocessing

__all__ = [
    "TwoStreamModel",
    "TwoStreamV11Model",
    "TwoStreamV12Model",
    "TwoStreamV13Model",
    "TwoStreamV14Model",
    "VideoMAEModel",
    "TwoStreamEncoder",
    "TwoStreamV11Encoder",
    "VideoMAEEncoder",
    "TwoStreamPreprocessing",
]
