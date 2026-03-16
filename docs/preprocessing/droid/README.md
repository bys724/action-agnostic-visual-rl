# DROID 전처리 결정

## 데이터 소스

- **버전**: DROID v1.0.1 (`droid_101`, `gs://gresearch/robotics/droid/1.0.1`)
- **포맷**: TFRecord/RLDS
- **규모**: 95,658 에피소드, 1.7TB
- **v1.0.0 (`r2d2_faceblur`)은 구버전** — 에피소드 누락(92,233개), 언어 어노테이션 부족. 사용하지 않음.

## 카메라

| 필드 | 해상도 | 설명 |
|------|--------|------|
| `exterior_image_1_left` | 180x320 | 외부 카메라 1 |
| `exterior_image_2_left` | 180x320 | 외부 카메라 2 |
| `wrist_image_left` | 180x320 | 손목 카메라 |

3개 카메라 모두 추출. Action probing에는 ext1을 기본 사용.

## 전처리 방식

**리사이즈 (180x320 → 256x256, crop 없음)**

### 검토한 옵션

| 옵션 | 방식 | 결과 |
|------|------|------|
| A: Resize | 전체 리사이즈 | **채택** — 전체 장면 보존 |
| B: CenterCrop | 좌우 각 70px 제거 → 180x180 | 양 끝 물체 잘림 |
| C: Pad+Resize | 위아래 패딩 → 320x320 | 28% 면적 낭비 |

### Resize 선택 이유

1. 로봇 팔이 이미지 전반에 걸쳐 움직임 → 전체 장면 보존 중요
2. 180x320 자체가 저해상도라 crop할 여유 없음
3. 16:9 → 1:1 왜곡은 미미 (OpenVLA, Octo 등도 동일 방식)
4. Wrist 카메라도 동일 처리 — 그리퍼 위치가 에피소드마다 다르므로 crop 비실용적

### 비교 이미지

`samples/` 디렉토리에 5개 에피소드 비교 이미지 저장:
- `ep*_compare.jpg` — exterior 카메라 전처리 옵션 비교
- `ep*_wrist_compare.jpg` — wrist 카메라 crop 비율 비교

## 출력 구조

```
droid_frames/
  ext1/ep_000000/frame_000000.jpg, ...
  ext2/ep_000000/frame_000000.jpg, ...
  wrist/ep_000000/frame_000000.jpg, ...
```

## 참고: 다른 연구의 DROID 사용

- pi0/OpenPI: exterior 2개 중 랜덤 1개 + wrist (TFRecord 직접 로드)
- Octo: 동일 패턴
- 우리: PyTorch 통일을 위해 JPG 추출 후 사용
