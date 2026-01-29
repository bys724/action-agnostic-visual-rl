# 데이터 디렉토리

```
data/
├── checkpoints/        # 모델 체크포인트
│   ├── lapa/           # LAPA 모델
│   ├── openvla/        # OpenVLA 모델
│   ├── openvla-libero/ # OpenVLA LIBERO fine-tuned
│   ├── pi0/            # Pi0 모델
│   └── two_stream/     # Two-Stream 모델 (자체 학습)
│       └── YYYYMMDD_HHMMSS/  # 학습 실행별 폴더
│           ├── config.json
│           ├── history.json
│           ├── best_model.pt
│           ├── latest.pt
│           └── checkpoint_epoch*.pt
├── datasets/
│   ├── bridge_v2/      # Bridge V2 로봇 데이터셋
│   └── libero_rlds/    # LIBERO RLDS 포맷
├── egodex/             # EgoDex 사람 비디오 데이터셋
│   └── test/           # 테스트 세트 (17GB)
├── libero/
│   ├── results/        # LIBERO 평가 결과 JSON
│   └── videos/         # 에피소드 비디오
└── trajectories/       # 수집한 trajectory
```

## LIBERO 결과 파일 형식

```json
{
  "task_suite": "libero_spatial",
  "overall_success_rate": 0.4,
  "total_successes": 40,
  "total_episodes": 100,
  "task_results": [...],
  "metadata": { "model": "openvla", "seed": 7, ... }
}
```

결과 비교: `python3 src/compare_results.py --results-dir data/libero/results`

이 디렉토리는 `.gitignore`에 포함됩니다.
