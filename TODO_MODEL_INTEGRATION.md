# TODO: SimplerEnv 모델 통합 작업

## LAPA 참고 자료
**LAPA (Latent Action Pretraining)** - SimplerEnv를 사용한 최신 연구

### 개요
- 액션 라벨 없이 비디오만으로 VLA 모델 사전학습
- OpenVLA 대비 +6.22% 성능 향상, 30배 빠른 학습
- SimplerEnv에서 평가 완료

### 참고 자료
- **공식 저장소**: https://github.com/LatentActionPretraining/LAPA
- **프로젝트 페이지**: https://latentactionpretraining.github.io/
- **모델 체크포인트**: https://huggingface.co/latent-action-pretraining/LAPA-7B-openx
- **논문**: https://arxiv.org/abs/2410.11758
- **SimplerEnv 평가 스크립트**: `scripts/lapa_bridge.sh`

### SimplerEnv 사용법
```bash
# LAPA 환경 설정
conda create -n lapa python=3.10 -y
conda activate lapa
git clone https://github.com/LatentActionPretraining/LAPA.git
pip install -r requirements.txt

# SimplerEnv 평가 실행
./scripts/lapa_bridge.sh
```

### 주요 특징
- 인간 비디오로 학습 가능 (Something-Something V2)
- SimplerEnv Bridge에서 71.2% 성공률
- 4x 80GB A100 GPU 필요
- CoRL 2024 Best Paper Award

---

## SimplerEnv 학습/평가 가이드

### 공식 문서
- **메인 저장소**: https://github.com/simpler-env/SimplerEnv
- **프로젝트 페이지**: https://simpler-env.github.io/
- **ManiSkill3 통합**: GPU 가속으로 10-15배 빠른 실행

### 새로운 정책 추가 프로세스
1. `simpler_env/policies/{your_policy}/` 디렉토리 생성
2. RT-1, Octo 코드 참고하여 구현
3. `simpler_env/main_inference.py` 수정
4. `scripts/` 폴더에 평가 스크립트 작성
5. `tools/calc_metrics.py`로 메트릭 계산

### 평가 방식
- **Visual Matching**: 실제 이미지 오버레이
- **Variant Aggregation**: 다양한 환경 변형 평균

### 디버깅 도구
- `demo_manual_control_custom_envs.py`: 수동 제어
- `simpler_env/utils/debug/`: 정책 디버깅 유틸

---

## 1. OpenVLA 통합
**우선순위**: 높음 (공식 지원 존재)

### 작업 내용
- OpenVLA 모델을 SimplerEnv에 통합
- 기존 RT-1, Octo와 함께 평가 파이프라인 구성
- Visual matching과 variant aggregation 평가 수행

### 참고 자료
- **공식 Fork**: https://github.com/DelinQu/SimplerEnv-OpenVLA
- **OpenVLA 저장소**: https://github.com/openvla/openvla
- **설치 가이드**: SimplerEnv-OpenVLA README 참조
- **모델 체크포인트**: Hugging Face에서 제공

### 예상 작업
```bash
# 패키지 설치
pip install torch==2.3.1 torchvision==0.18.1
pip install timm==0.9.10 tokenizers==0.15.2 accelerate==0.32.1
pip install flash-attn==2.6.1 --no-build-isolation
```

---

## 2. Pi0 (π₀) 통합
**우선순위**: 중간 (최신 모델, 성능 검증 필요)

### 작업 내용
- Physical Intelligence의 Pi0 모델 통합
- Zero-shot 및 파인튜닝 성능 평가
- DROID 데이터셋 기반 변형 테스트

### 참고 자료
- **공식 저장소**: https://github.com/Physical-Intelligence/openpi
- **커뮤니티 구현**: https://github.com/allenzren/open-pi-zero
- **논문**: https://arxiv.org/html/2410.24164v1
- **블로그**: https://www.physicalintelligence.company/blog/pi0
- **모델 변형**: Pi0-base, Pi0-FAST, Pi0-DROID

### 주요 특징
- 3B 파라미터 VLM 기반
- 7개 로봇 플랫폼, 68개 작업 사전학습
- 1-20시간 데이터로 파인튜닝 가능
- Zero-shot 성능: 20-50% (SimplerEnv)

### 예상 작업
```python
# openpi 저장소 클론 및 설치
git clone https://github.com/Physical-Intelligence/openpi
cd openpi
pip install -e .

# 체크포인트 다운로드 및 추론 코드 작성
# simpler_env/policies/pi0/ 디렉토리 생성
```

---

## 3. 추가 작업 사항

### 벤치마크 평가 자동화
- 모든 모델(RT-1, Octo, OpenVLA, Pi0)에 대한 통합 평가 스크립트
- 성능 비교 테이블 자동 생성
- Visual matching vs Variant aggregation 비교

### 문서화
- 각 모델별 설치 및 실행 가이드
- 성능 벤치마크 결과 정리
- 트러블슈팅 가이드

---

## 참고: 기존 지원 모델

### RT-1 / RT-1-X
- 이미 SimplerEnv에 통합됨
- gsutil로 체크포인트 다운로드 필요

### Octo
- 이미 SimplerEnv에 통합됨
- Hugging Face 자동 다운로드

---

## 일정 계획
1. **Phase 1**: OpenVLA 통합 (공식 Fork 활용)
2. **Phase 2**: Pi0 통합 (openpi 저장소 기반)
3. **Phase 3**: 통합 평가 및 벤치마크
4. **Phase 4**: 문서화 및 최적화