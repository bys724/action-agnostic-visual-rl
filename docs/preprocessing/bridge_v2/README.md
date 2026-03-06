# Bridge V2 전처리 결정 기록

**데이터셋**: Bridge V2 (robot manipulation trajectories)
**원본 해상도**: 480x640 (3:4)
**최종 출력**: 256x256 JPEG
**결정일**: 2026-03-05

## 전처리 파이프라인

1. 리사이즈: 480x640 → 256x256 (crop 없음)
2. 학습 시 RandomCrop(224), 평가 시 CenterCrop(224)

## 결정 근거

- 원본 비율이 3:4 → 1:1 왜곡이 미미
- 로봇 팔이 이미지 전반에 걸쳐 움직이므로 전체 장면 보존이 중요
- RT-2, Octo, OpenVLA 등 주요 모델도 동일 방식(직접 리사이즈) 채택

## 스크립트

```bash
python scripts/data/extract_bridge_frames.py \
    --bridge-root data/bridge_v2 \
    --output-dir data/bridge_frames
```
