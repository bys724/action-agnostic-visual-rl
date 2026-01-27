# Vision Encoder Research Plan

## 연구 목표

**핵심 가설**: 로봇 영상 데이터로 사전학습한 비전 인코더가 기존 VLA 모델의 성능을 향상시킨다.

**검증 방법**:
1. 새로운 비전 인코더 개발
2. 로봇 관련 데이터(Bridge V2 등)로 사전학습
3. 기존 VLA 모델의 비전 인코더를 교체
4. SIMPLER 벤치마크에서 성능/효율 비교

**장점**: 실제 로봇 없이 시뮬레이션으로 효용성 입증 가능

---

## 타겟 모델

| 모델 | 비전 인코더 | 프레임워크 | SIMPLER 지원 | 우선순위 |
|------|------------|-----------|--------------|---------|
| **OpenVLA** | DINOv2 + SigLIP | PyTorch | ✅ 공식 | 1순위 |
| **Pi0** | SigLIP (PaliGemma) | JAX | ⚠️ 통합 필요 | 2순위 |

### OpenVLA
- GitHub: https://github.com/openvla/openvla
- 구조: Prismatic VLM 기반, config 수정으로 인코더 교체 용이
- 체크포인트: `openvla/openvla-7b` (HuggingFace)

### Pi0
- GitHub: https://github.com/Physical-Intelligence/openpi
- 라이선스: Apache 2.0 / MIT
- 체크포인트: HuggingFace 공개
- 특징: 2025년 1월 공개, 최신 SOTA

---

## 데이터셋

### 사전학습용
| 데이터셋 | 규모 | 용도 | 우선순위 |
|---------|------|------|---------|
| **Bridge V2** | 60K trajectories | 메인 사전학습 | 필수 |
| EgoDex | 829h, 194 tasks | 추가 사전학습 | 선택 |
| Something-Something V2 | 220K videos | 추가 사전학습 | 선택 |

### 평가용
- SIMPLER (WidowX 환경, BridgeData V2 tasks)

---

## 평가 메트릭

| 메트릭 | 측정 방법 | 의미 |
|--------|----------|------|
| **Success Rate** | SIMPLER task 성공률 | 최종 성능 |
| **Sample Efficiency** | 같은 성능까지 필요한 finetuning 데이터 | 학습 효율 |
| **Convergence Speed** | 수렴까지 step 수 | 학습 속도 |

---

## 실험 설계

### Baseline 구성
```
실험군:   새 인코더 (Bridge 사전학습) → VLA finetuning → SIMPLER 평가
대조군 A: 원본 인코더 (DINOv2+SigLIP)  → 동일 finetuning → SIMPLER 평가
대조군 B: 랜덤 초기화 인코더           → 동일 finetuning → SIMPLER 평가
```

### 인코더 동결 전략
- **메인 실험**: 인코더 동결 (사전학습 효과 명확히 분리)
- **보조 실험**: 전체 학습 (최고 성능 확인)

---

## Todo List

### Phase 1: 환경 구축
- [ ] Pi0 레포지토리 클론 및 환경 설정
- [ ] Pi0 SIMPLER 통합 (OpenVLA 통합 코드 참고)
- [ ] Bridge V2 데이터셋 다운로드 및 전처리
- [ ] OpenVLA 비전 인코더 교체 지점 코드 분석
- [ ] Pi0 비전 인코더 교체 지점 코드 분석

### Phase 2: 비전 인코더 개발
- [ ] 인코더 아키텍처 설계
- [ ] 사전학습 목표 함수 설계 (contrastive, reconstruction 등)
- [ ] 학습 파이프라인 구현
- [ ] Bridge V2로 사전학습 실행

### Phase 3: 인코더 교체 및 통합
- [ ] OpenVLA 인코더 교체 구현
- [ ] Pi0 인코더 교체 구현
- [ ] 교체 후 inference 테스트

### Phase 4: Finetuning 실험
- [ ] OpenVLA + 새 인코더 finetuning
- [ ] OpenVLA + 원본 인코더 finetuning (baseline)
- [ ] Pi0 + 새 인코더 finetuning
- [ ] Pi0 + 원본 인코더 finetuning (baseline)

### Phase 5: 평가 및 분석
- [ ] SIMPLER 벤치마크 평가 (전 모델)
- [ ] Sample efficiency 측정
- [ ] Ablation study (인코더 구조별 비교)
- [ ] 결과 시각화 및 분석

---

## 리소스

### 코드 레포지토리
| 프로젝트 | URL |
|---------|-----|
| OpenVLA | https://github.com/openvla/openvla |
| Pi0 (openpi) | https://github.com/Physical-Intelligence/openpi |
| SIMPLER | https://github.com/simpler-env/SimplerEnv |
| SimplerEnv-OpenVLA | https://github.com/DelinQu/SimplerEnv-OpenVLA |

### 체크포인트
| 모델 | 경로 |
|------|------|
| OpenVLA | `openvla/openvla-7b` (HuggingFace) |
| Pi0 | Physical-Intelligence (HuggingFace) |

---

## 일정 (예상)

| Phase | 기간 | 산출물 |
|-------|------|--------|
| Phase 1 | 2주 | 환경 구축 완료, Pi0 SIMPLER 통합 |
| Phase 2 | 3주 | 비전 인코더 사전학습 완료 |
| Phase 3 | 1주 | 인코더 교체 구현 완료 |
| Phase 4 | 2주 | Finetuning 실험 완료 |
| Phase 5 | 2주 | 평가 및 분석 완료 |

**총 예상 기간**: 10주

---

*최종 수정: 2025-01-27*
