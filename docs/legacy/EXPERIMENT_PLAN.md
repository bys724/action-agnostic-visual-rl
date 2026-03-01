# 실험 계획

**목표**: Two-Stream Interleaved ViT가 움직임 정보를 효과적으로 인코딩하여 로봇 조작 성능을 향상시킴을 입증

---

## 1. 핵심 주장 검증

움직임 정보를 실제로 인코딩하는지 직접 검증

| 실험 | 방법 | 성공 기준 |
|-----|------|----------|
| Optical Flow Probe | 인코더 freeze → linear probe로 flow 예측 | DINOv2/CLIP 대비 EPE 20%+ 낮음 |
| Temporal Order | 프레임 쌍 시간 순서 분류 | 90%+ 정확도 |
| Action Clustering | CLS 임베딩 t-SNE | 같은 행동끼리 클러스터링 |

---

## 2. 컴포넌트 분석 (Ablation)

각 모듈의 기여도 분석

| 변형 | 설명 |
|-----|------|
| M-only | P채널 제거 |
| P-only | M채널 제거 |
| No Cross-Attn | 교환 없이 최종 concat만 |
| Late Fusion | 끝에서만 fusion |
| Early Fusion | 6ch 통합 단일 ViT |
| 1/2/3 Exchanges | 교환 횟수 변화 |

---

## 3. 로봇 태스크 평가

| 벤치마크 | 비교 대상 | 핵심 메트릭 |
|---------|----------|------------|
| LIBERO | OpenVLA, Pi0 | Success Rate |
| Few-shot | 10/20/50 demos | Sample Efficiency |

---

## 4. 시각화

- Attention Map: M-branch vs P-branch가 주목하는 영역 차이
- CLS Evolution: 각 stage별 CLS 변화 추적

---

## 일정

| 단계 | 기간 | 내용 |
|-----|------|------|
| Phase 1 | 2주 | 모델 구현 완료, 학습 파이프라인 |
| Phase 2 | 3주 | Probe 실험, Ablation |
| Phase 3 | 3주 | LIBERO 평가, Few-shot |
| Phase 4 | 2주 | 분석, 시각화, 논문 작성 |

---

## 파일 구조

```
src/models/
├── two_stream_vit.py      # 메인 인코더
├── two_stream_preprocessing.py  # M/P 채널 전처리
└── action_decoder.py      # Action Decoder

src/training/
├── train.py               # 학습 스크립트
└── trainer.py             # Trainer 클래스

src/eval/
├── flow_probe.py          # Optical flow probe 평가
├── temporal_order.py      # Temporal order 평가
└── eval_libero.py         # LIBERO 벤치마크
```
