# EgoDex 전처리 결정 기록

**데이터셋**: EgoDex (ego-centric manipulation videos)
**원본 해상도**: 1920x1080 (16:9)
**최종 출력**: 256x256 JPEG (quality 95)
**결정일**: 2026-03-06

## 전처리 파이프라인

1. 센터크롭: 1920x1080 → 1080x1080 (정사각형)
2. 리사이즈: 1080x1080 → 256x256
3. 학습 시 RandomCrop(224), 평가 시 CenterCrop(224)

## Crop 비율 검토

원본이 16:9이므로 좌우 840px을 잘라내야 함. 여러 비율을 비교함:

| 비율 | 좌측 제거 | 우측 제거 | 판정 |
|------|----------|----------|------|
| 50:50 (센터) | 420px | 420px | 채택 |
| 55:45 | 462px | 378px | - |
| 60:40 | 504px | 336px | - |
| 65:35 | 546px | 294px | 검토했으나 센터 채택 |

### 결정 근거

- EgoDex 영상의 작업 대상은 대부분 이미지 중앙에 위치
- 비대칭 크롭(65:35)은 특정 task에서 유리하나, task 간 편차가 큼
- 센터크롭이 가장 안정적이고 범용적
- 비율 왜곡 없이 원본 종횡비 유지

### 샘플 이미지

비교 이미지는 로컬에서 검토 후 삭제됨.

## 스크립트

```bash
# 프레임 추출
python scripts/data/extract_frames.py \
    --egodex-root data/egodex \
    --split part1 \
    --output-dir data/egodex_frames_part1

# 전체 파이프라인 (S3 다운로드 → 추출 → 업로드 → 정리)
bash scripts/process_egodex_part.sh part1
```

## S3 구조

- 원본: `s3://bys724-research-2026/datasets/egodex-full/partN/`
- 프레임: `s3://bys724-research-2026/egodex_frames_partN/`
