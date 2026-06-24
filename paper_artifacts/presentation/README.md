# Presentation assets (concept / hero images)

발표 슬라이드용 컨셉·hero 이미지. **Nano Banana(생성형)** 산출물 — 정밀 도식이 아니라
도입/타이틀/섹션 divider용 시각 자료. 정확한 아키텍처 도식은
[`../fig1_architecture/`](../fig1_architecture/) (matplotlib + mermaid).

색 컨벤션은 의도적으로 본 프로젝트와 정합: **파랑 = motion(M, where)**, **녹색 = appearance(P, what)**.

| 파일 | 내용 | 용도 |
|------|------|------|
| `hero_robothand_streams_labeled.png` | 로봇 손 + 파란 motion stream / 녹색 spatial structure (라벨 포함) | 타이틀 슬라이드 (라벨 내장) |
| `hero_gripper_streams_converge.png` | 그리퍼로 파랑·녹색 stream 수렴 (무텍스트) | 타이틀 배경 |
| `hero_hand_cube_streams.png` | 큐브 든 손 + 파란 화살표 / 녹색 mesh (무텍스트) | 타이틀/섹션 |
| `concept_two_pathways_eye_robot.png` | eye → 두 경로(모션 swirl / appearance 텍스처) → robot | "two visual pathways" 개념 슬라이드 |
| `concept_two_pathways_humanoid.png` | eye → 파란 wave / 녹색 geometric → humanoid | 개념 슬라이드 (대안) |

## 재생성

`mcp__nanobanana__generate_image` 로 생성. 핵심 프롬프트 요지(텍스트 왜곡 방지 위해
"no text/words/labels" 명시): two-stream visual learning, blue=temporal motion(arrows/flow),
green=appearance/spatial structure, robot hand manipulation, minimalist scientific keynote.

> 생성형 이미지라 디테일은 비결정적. 슬라이드 맥락에 맞춰 재생성·교체 가능.
