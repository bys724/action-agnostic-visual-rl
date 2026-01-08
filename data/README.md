# 데이터셋 디렉토리

이 디렉토리는 모든 데이터셋을 관리합니다.

## 구조

```
data/
├── trajectories/      # SimplerEnv에서 수집한 trajectory
│   └── *.pkl         # collect_trajectories.py 출력
├── results/          # 평가 결과
│   ├── *.json        # 개별 모델 결과
│   └── *.csv         # 다중 모델 비교 결과
├── checkpoints/      # 모델 체크포인트
│   ├── openvla/      # OpenVLA 체크포인트
│   ├── lapa/         # LAPA 체크포인트
│   └── custom/       # 개발 모델 체크포인트
└── datasets/         # 외부 데이터셋 (필요시 다운로드)
    ├── droid/        # DROID 데이터셋
    └── bridge_v2/    # Bridge V2 데이터셋
```

## 사용법

### Trajectory 수집
```bash
python src/collect_trajectories.py \
    --type openvla \
    --checkpoint "openvla/openvla-7b" \
    --output ./data/trajectories
```

### 평가 결과 저장
```bash
python src/eval_simpler.py \
    --models "base:openvla:openvla/openvla-7b" \
    --output ./data/results
```

### 체크포인트 사용
```bash
# Fine-tuned 모델 평가
python src/eval_simpler.py \
    --model "./data/checkpoints/openvla/finetuned_model"
```

## 주의사항
- 이 디렉토리는 git에서 제외됩니다 (.gitignore)
- 대용량 데이터셋은 필요시 다운로드하세요