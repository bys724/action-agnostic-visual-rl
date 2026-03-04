# Legacy Scripts

이 디렉토리는 더 이상 사용되지 않지만 참고용으로 보관하는 스크립트들입니다.

## run_full_training.sh (중복)

**Status**: Deprecated (중복)
**Replaced by**: `scripts/train_aws.sh`

3개 모델을 순차 학습하는 스크립트였으나, `train_aws.sh`가 더 완전한 기능을 제공합니다:
- S3 sync (데이터 다운로드)
- Git pull (코드 업데이트)
- Sanity test 모드
- 개별 모델 선택 가능

**Migration**:
```bash
# Before
bash scripts/run_full_training.sh

# After
bash scripts/train_aws.sh
```

## run_pretrain.sh (특정 환경)

**Status**: Environment-specific
**Environment**: Docker (libero-env) + ETRI workstation

BridgeV2 pretraining + EgoDex action probing 파이프라인이지만, 특정 환경에 의존:
- Docker 이미지: `libero-env:latest`
- 고정 경로: `/home/etri/action-agnostic-visual-rl`
- BridgeV2 데이터셋 (현재 프로젝트는 EgoDex 중심)

**현재 접근 방식**:
- Phase 1: EgoDex pretraining (AWS) → `scripts/train_aws.sh`
- Phase 2: Action probing → `scripts/eval/probe_action.py`
- Phase 3: LIBERO fine-tuning → `scripts/eval/finetune_libero.py`

환경별로 독립적으로 실행하는 것이 더 유연합니다.

## test_encoders.py (개발용)

**Status**: Development test script
**Purpose**: 인코더 통합 테스트

모델 리팩토링 전에 사용했던 테스트 스크립트입니다.
현재는 새로운 모델 구조 (TwoStreamModel, SingleStreamModel, VideoMAEModel)를 사용하므로 업데이트 없이는 작동하지 않습니다.

**현재 대안**:
```python
# Quick encoder test
from src.models import TwoStreamModel, SingleStreamModel, VideoMAEModel
import torch

model = TwoStreamModel()
img_current = torch.rand(2, 3, 224, 224)
img_future = torch.rand(2, 3, 224, 224)
pred, emb = model(img_current, img_future)
print(f"Output: {pred.shape}, Embedding: {emb.shape}")
```

## 정리 방침

**보관 기준**:
- 중복 스크립트: 마이그레이션 경로 문서화 후 보관
- 환경 특정: 참고용으로 보관 (다른 환경에서 재사용 가능)
- 개발 테스트: 간단한 대안 제시 후 보관

**삭제 고려**:
- 6개월 이상 참조되지 않은 경우
- Git history에서 복원 가능
