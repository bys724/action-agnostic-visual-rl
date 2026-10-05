# E3 — BC 가치 실험 결과 (정본 계획 `docs/claim_spine_v2.md` §4)

공통: libero_object · C0 인코더(`two_stream_v15b_step1_comp_mae_s/20260629_101634/latest.pt`, frozen) · BC-T attentive ·
aug-on · 50ep · 데모 10/과제(seed 고정 부분집합) · 간격 g=10 스텝(0.5 s) · copycat dropout p=0.1(현재 프레임 P 외 채널) ·
롤아웃 = 클러스터 osmesa, 10 과제 × 50 trial = 500 ep, rollout seed 7. 결과 JSON = `/proj/external_group/mrg/results/libero_rollouts/<run>_t50/`.
잡 목록 `jobs_e3_min_20261004.txt` · 제출 `submit_e3_min.sh` · 세션 로그 `docs/cluster_sessions.md` "E3 0단계 / E3 1단계".

## 0단계 — 데모 수 캘리브레이션 (10-03)

팔 ② 데모 10 → SR 39.0% (목표 20–60% 안) → 데모 10 = 저데모 지점.

## 1단계 최소 칸 + 원인 분리 (10-04 ~ 10-05) [잠정 · seed 0 하나]

| 팔 | 정책 입력 (`--motion-source`) | SR | 과제별 (%) |
|---|---|---|---|
| ① | 외형 1장 P(t) (`none`) | **47.2%** | 92/12/80/12/76/48/20/62/54/16 |
| ② | 외형 2장 P(t−g), P(t) (`rgb_prev`) | 39.0% | 34/6/86/24/6/90/58/24/32/30 |
| ④ | 외형 1장 + CoMP M (`comp_m`) | **3.4%** | 12/0/2/0/10/2/2/2/0/4 |
| ④ M 끔 | 같은 ckpt, 롤아웃 때 M = 0 (`--ablate-extra`, 학습 dropout 상태) | **0.6%** | 0/0/4/0/2/0/0/0/0/0 |

- ④ ≤ ② 절반 → 사전 등록 "7월 재발" 갈래(§4.4): 원인 분리 후 1회 재판정.
- 원인: 외형 1장 아님(① 최고). M 추가가 붕괴를 만들고, M을 끄면 더 나빠짐 = 정책이 M에 강하게 의존 →
  행동 복제의 인과 혼동/copycat(M = 직전 움직임 = 직전 행동의 흔적) [추정].
- 영상(과제 0–2, 128px): 정지·표류·떨림 아님. 물체 앞까지 가서 잡지 못하고 빈손으로 바구니로 이동,
  약 240프레임에 일제히 운반 단계로 전환 [추정]. 진단 시트는 세션 scratchpad(비보존).
- ②(과거 프레임)도 ①보다 낮음 — 같은 방향이나 seed 1개로는 약함.
- 잡기 기록: 원인 분리 롤아웃부터 에피소드별 `videos/*_trace.npz`(EE·그리퍼 qpos·명령·물체 위치) 저장 — 미분석.

## 외형 이동 칸 준비 (10-04)

LIBERO 내장 `OffScreenRenderEnv(scene_properties={"floor_style": ...})` 바닥 교체 = 물리·상태 비트 동일(obs 차 0.0),
성공 판정(BDDL 접촉·위치) 동일, 픽셀 |Δ| 42.5. 벽은 agentview에 거의 안 보임. 롤아웃 클라이언트 인자 미구현.
렌더 테스트 산출물 `/proj/external_group/mrg/logs/e3/appearance_test/`.
