# Dataset Download TODO

후속 연구용 데이터셋 확보 현황 (B 옵션: 움직임 풍부 ego/video 중심).

**기준 일자**: 2026-04-14
**저장소 quota**: 50 TB (여유 ~40 TB, 증량 완료)

## 현황

| 데이터셋 | 상태 | 크기 | 비고 |
|---------|------|------|------|
| ✅ EgoDex | 보유 | ~4 TB | 본 논문 학습에 사용 중 |
| ✅ DROID | 보유 | 3.4 TB | Cross-domain probing용 |
| 🔄 Nymeria (subset) | **다운로드 중** | ~2.14 TB | video_main_rgb + body_motion + narration (v0.0) |
| ⏳ Epic-Kitchens-100 | Nymeria 완료 후 자동 시작 | ~400 GB | mp4 videos only, annotations 포함 |
| ⏳ Ego4D | **승인 대기 중** (EULA 서명 2026-04-14) | ~4 TB (video_540ss) 또는 7 TB (full_scale) | AWS credentials 이메일 수신 대기 (~48h) |
| ⬜ SSv2 | 라이선스 신청 필요 | ~220 GB | Qualcomm academic form |
| ⬜ HoloAssist | 라이선스 신청 필요 | ~300 GB | Microsoft Research form |

## 진행 순서

1. **Nymeria → Epic-Kitchens 순차 다운로드 (현재 실행 중)**
   - 동일 로그인 세션, 하나 끝나면 다음 자동 시작
2. **Ego4D 승인 이메일 수신 시**
   - AWS credentials 설정 → `download_ego4d.sh` 작성 후 순차 대기열에 추가
   - 옵션 결정 필요: `video_540ss` (4 TB, 권장) vs `full_scale` (7 TB)
3. **SSv2, HoloAssist 라이선스 신청**
   - SSv2: https://developer.qualcomm.com/software/ai-datasets/something-something
   - HoloAssist: https://holoassist.github.io/
   - 승인 후 각각 다운로드 스크립트 작성

## 사전 준비 완료

- [x] Ego4D CLI 설치 (`ego4d` v1.7.3 in `aavrl-extract` conda env)
- [x] Nymeria manifest 다운로드 + filter script로 subset 추출
- [x] Epic-Kitchens download script 작성

## 중요 주의사항

- 모든 다운로드는 **로그인 노드**에서 (compute 노드는 외부 네트워크 제한)
- **동시 2개 이상 다운로드 금지** (프로세스 누적으로 사용자 limit 초과 위험 — 2026-04-14 사건)
- Nymeria JSON URL 유효기간: **14일 (~2026-04-28 만료)**
- 모든 다운로드 스크립트는 resume 지원 → 끊어지면 재실행
