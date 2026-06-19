# Paper 1 — Input-Prior Robot Representation (ICRA)

> **정규 출처**: Obsidian `Projects/Input-Prior Robot Representation (ICRA)/`. 모델 명명·2논문 구조는 [`CLAUDE.md`](../CLAUDE.md) "명명 · 2논문 구조".
> **상태 (2026-06-11)**: 신설. P단독 image MAE > VideoMAE = **좁게 입증** → ablation·실로봇 필요.
> **작업 분담**: 코드 편집(standalone image MAE 학습 경로 등)은 **클러스터 세션**. 이 문서는 계획·근거 정리 (문서 전용).

## Thesis

단일프레임 **image MAE on (Sobel + RGB)** 표현이 동일 스케일 **VideoMAE를 이긴다** → 로봇 표현 학습에 video/temporal 아키텍처가 필수가 아니며 **input prior로 충분**할 수 있다.

⚠️ 현재 "이긴다"는 **unmatched**(image vs video) + confound 미해소. R3M/VC-1 등 단일프레임 표현 선례가 있어 **novelty는 헤드라인이 아니라 (a) 통제된 비교 (b) edge-channel 발견 (c) 실로봇 결과**에서 나옴.

## 코드 근거 (premise 확인됨, 2026-06-11)

Obsidian "P/M이 분리 인코더인지 확정 — standalone image MAE = P단독 성립의 전제" 미해결 질문을 코드로 해소:

- **P/M 분리 인코더 ✅**: `TwoStreamV15Model`에 student P/M encoder 별도. `TeacherPv15` = "EMA copy of student **P** encoder", `TeacherMv15` = "EMA copy of student **M** encoder ONLY" ([src/models/two_stream_v15.py](../src/models/two_stream_v15.py)). → "standalone image MAE = P단독" 전제 구조적으로 성립.
- **v15(구 teacher-anchor)의 P encoder = 순수 single-frame MAE ✅**: anchor=`teacher_p(frame_t).detach()`라 P가 V-JEPA gradient를 못 받음 → "P encoder = MAE only" (two_stream_v15.py:401-403 주석). → **기존 P단독 결과 = image MAE 결과** 임이 코드로 확정. (Parvo=code v15b는 이 anchor를 student로 바꿔 P가 gradient를 받게 한 Paper 2 검증용 — Paper 1과 무관.)
- **P channel = [∂x, ∂y, R, G, B] 5ch** = Sobel_x(L), Sobel_y(L), R, G, B ([src/models/common/preprocessing.py](../src/models/common/preprocessing.py) `compute_p_channel`). → "Sobel+RGB" = 현재 P channel 그대로.

## 입증 / 미입증

- **입증 (좁음)**: v15 P-only > VideoMAE-ours. 기존 결과 = CortexBench 21잡 (`paper_artifacts/cortexbench/v15_p_only` vs `videomae_ours`).
- **미해소 confound**: ① image vs video 구조 차이 ② Sobel(edge) vs RGB 중 효자 ③ VideoMAE baseline 공정성.
- **motion · M-stream 기여 = 0** (motion routing no-op). → 이 논문은 **motion thesis 아님** (그건 Paper 2 = [v15b_retraining_status.md](v15b_retraining_status.md)).

## Ablation — Paper 1 존재 여부를 가름 (싼 실험, 며칠)

standalone **image MAE** (M-stream/routing 제거)로 입력 채널 3종:

| # | 입력 | 채널 | 목적 |
|---|------|------|------|
| 1 | `[Sobel(L) + RGB]` | 5ch | 기존 P단독 수치 **재현 확인** (de-risk: M/EMA 누수 없었는지) |
| 2 | `[RGB]` | 3ch | **Sobel 기여 격리** |
| 3 | (선택) `[Sobel(L)]` | 2ch | edge만으로 되는지 |

→ 셋 다 **VideoMAE-ours와 비교** + **VideoMAE 공정성 점검** (budget · mask ratio · tuning).

**해석 분기**:
- 2(RGB-only)가 이미 이기면 → "image > video" (Sobel 무관). 단순하나 *"VideoMAE가 약했던 것 아니냐"* 공격에 취약.
- 2는 지고 1만 이기면 → **"edge channel이 핵심"** — 훨씬 방어 가능한 contribution.

**구현 요건 (클러스터 세션)**:
- standalone image MAE 학습 경로 — 현재 single-frame image MAE 모델은 부재 (P encoder는 `two_stream` 내부). 옵션: (a) two_stream에 M-disable flag, (b) 최소 image-MAE 분리.
- 입력 채널 토글: `preprocessing.py` `compute_p_channel` 변형 (5/3/2ch).
- eval 재사용: CortexBench (`v15_loader` p_only 패턴), EgoDex/LIBERO probing, LIBERO BC-T.

## 실로봇 — ICRA의 본체 lift

표현 품질 주장만으론 ICRA 부족 → **실로봇 manipulation 결과**가 핵심. Paper 2에서 약점이던 real-robot 부재가 여기서 본체 작업. ablation은 며칠이지만 **실로봇 deploy가 본체** — 자원·일정 별도 배정.

## 재사용 eval 자산

| 자산 | 경로 |
|------|------|
| VideoMAE-ours 모델·config | [src/models/videomae.py](../src/models/videomae.py), `src/cortexbench/conf/model/videomae_ours.yaml` |
| P-only loader | [src/cortexbench/v15_loader.py](../src/cortexbench/v15_loader.py) (`load_v15_p_only`) |
| M/P 채널 전처리 | [src/models/common/preprocessing.py](../src/models/common/preprocessing.py) |
| 기존 CortexBench 결과 | `paper_artifacts/cortexbench/{v15_p_only, videomae_ours}` |

## Open

- ✅ (해소) P/M 분리 인코더 — 코드 확인됨 (위 "코드 근거").
- standalone image MAE를 새로 학습 필수 — ablation 2/3(채널 격리)는 v15 P encoder 추출로 불가, 새 학습 필요 (클러스터).
- VideoMAE-ours 공정성 — 현재 mask 0.5 (2-frame). budget·tuning 동등성 명시 점검 필요.
