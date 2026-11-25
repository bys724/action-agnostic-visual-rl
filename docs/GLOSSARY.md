# 용어 사전 (GLOSSARY)

> 목적 = 보고·계획 문서의 압축 용어·코드명을 풀어 쓴 참조 사전. **네이밍 규칙**: 새 용어·노브·arm 이름은 뜻이 드러나게 짓거나, 도입 커밋에서 본 문서에 등록할 것. 보고에서 압축 용어 첫 등장 시 한 줄 풀이 병기.
> 이론·설계 용어의 정본 = Vault `Projects/Action-Agnostic Visual Representation (AAAI)/` — 본 문서는 실행·검증·프로세스 용어 담당. 신설 2026-09-23 (STATUS.md 용어 절에서 출발).

## 모델·스트림

- **CoMP** — 이 저장소의 모델 (구 CoMP-MAE, 코드 v16). P(form)·M(dynamics) 두 스트림을 서로 재구성시켜(cross-reconstruction) 분리시킴. 코드·ckpt·CSV 키는 옛 이름 유지
- **P / M 스트림** — Parvo(형태·정체성) / Magno(움직임). **deployed-P / P-only** = 배포 시 쓰는 P 출력. P+M 이어붙임은 causal confusion으로 유해(LIBERO P-only 68.7 vs P+M 2.0)
- **v15 / v16 / v17** — `src/models/two_stream_v15.py`의 config 계보. v16 = `comp_mae` 플래그, v17 = forecast-sufficient (동결 교사 + 학생). 새 파일 아님
- **S / B** — 모델 크기 (Small ~32M / Base ~86M). **part1-5** = EgoDex 데이터 분할 (part1만 vs 전체 = 6.81×)

## 판정·게이트

- **STEP 0 / 1 / 2** — 논문 게이트: 0 = 효율 headline(3b 절대 효율), 1 = factorization 인과(M-recon 스칼펠), 2 = control-level value (2-A 효율 / 2-B 폐루프 BC)
- **same-probe 판정** — 두 런을 같은 프로브(릿지 회귀 등)로 재서 비교. 프로브를 바꾸면 비교 무효
- **서랍** — 본문에 넣지 않고 supplement에만 두는 결과(attach-only). "서랍 재판정" = 그 결과를 다시 판정할지
- **조기 게이트** — 후속 연구를 하루 안에 싸게 죽일 수 있는 선행 조건. v17: M_student 릿지 프로브 R² > M_teacher (SFA 기준선 0.52~0.70)
- **qk-norm 버그 (07-20)** — probe 로더가 QK-LayerNorm을 조용히 빼먹어 15잡 오염. 재측정으로 EgoDex 기준② FAIL→PASS, OOD 확대 4/4→2/4 반전

## 실행

- **compute-matched** — 데이터가 커지면 epoch을 줄여 총 샘플 수를 맞춘 비교 (B × part1-5 = 7ep)
- **GPFS 병목** — full-data 학습 시 랜덤 액세스 I/O로 throughput 2.3× 저하. 후속 full-data 잡 전 필수 점검

## Refinement-floor (정의 정본 = `refinement_floor_plan.md` 용어 절)

- **C0 / C1** — C0 = 제출본 CoMP-S(증강 없음) · C1 = C0 + 프레임별 독립 밝기 증강(`--bright-aug`), 이후 연구의 기준 모델
- **C1-DN** — 논문 주장 모델(09-28). C1 + 10ep: M 입력 RGB 센서 노이즈 σ~U[0,0.01](P는 무노이즈) + 쌍 공유 장면 밝기 ±1 stop, M-recon 타깃 = 밝기 보존·노이즈 제거 ΔL. 평가 팔 `PilotM10`(5ep ckpt 평가 = `PilotM`). 계획서 §10
- **R2-1~R2-4** — 라운드 2 probe 시험(§9): R2-1 배포 P(P_t⊕P_tk) 분포 이동 · R2-2 P_t⊕raw 학습 투영(`--raw-pad linear`) · R2-3 probe 학습·시험 양쪽 노이즈(`--probe-noise-sigma`) · R2-4 정확히-0 ΔL 비율
- **held-out 잡음** — 학습에 쓰지 않은 잡음 종류로 시험: `shot`(밝기 의존)·`corr`(σ=1px 블러된 상관 잡음)·`jpeg`(품질 q, 정확히-0 보존이라 참고값)
- **F1 / `raw-dl`** — 학습 없는 바닥선: CoMP M 입력과 같은 raw ΔL을 16×16 패치로 자른 것을 그대로 토큰으로 (196×256). probe 인코더 이름 `raw-dl`
- **M 단독 / `m_only`** — probe에 M 스트림 토큰만 넣는 판독 (P_t⊕M과 구분)
- **ⓘ / ⓢ 시험** — probe 학습과 시험이 같은 분포(참조용) / probe 학습에 없는 조건(판정용: 교란·suite 간 무재학습 전이)
- **P_t⊕X** — 판독기에 P 스트림의 현재 프레임 토큰(P_t)과 X(학습된 M 또는 raw ΔL 등)를 토큰 축으로 붙여 넣는 판독 (계획서 효율 사다리). 코드 `parvo --parvo-mode p_t_m`, `parvo-raw`
- **(A)/(B)/(C) 판정축** — refinement_floor_plan §6: (A) LIBERO suite 간 무재학습 전이 · (B) 라벨 효율(5%·2%) · (C) 교란 강건성(그림자). 판정은 ⓢ(분포 이동)에서만
- **관측기 / Fuse (1층)** — 동결 C1-DN의 P·M 토큰을 32개 잠재 토큰 z로 묶는 Perceiver (`scripts/eval/observer_fuse.py`). z_PP = Fuse({앵커 P, 현재 P}) = 비싼 방법 · z_PM = Fuse({앵커 P, M 조각들}) = 싼 방법 · 앵커 나이 k = 앵커~현재 프레임 수(1~20 = 1초)
- **v3 / v4 관측기** — v3 = M을 5프레임 배수 조각으로만(z 5스텝마다) · v4 = 완결 5프레임 조각 + 마지막 부분 조각 1~4프레임(z 매 스텝, `observer_fuse_v4.py`). 나이 20에서 둘은 같은 구성
- **위치 너머 몫 (beyond-position)** — R²(z + 앵커 손 위치 RFF) − R²(손 위치만). 손 위치로 공짜로 맞히는 몫을 뺀 움직임 정보 (same-probe concat 규약)
- **P+P 전용 참조 (ref, `--pp-only`)** — M 입력·M 복원 없이 P+P만으로 학습한 Fuse. 칸 간 공통 분모
- **희소 M (sparse M)** — 변화 패치(mean|ΔL|>1/255)만 M 인코더에 넣고 나머지는 mask 토큰으로 채운 인코딩 (`--m-sparse-tau`)
- **인계 파일 (hand-off)** — SC 2층이 쓰는 관측기 묶음: Fuse 가중치 + 입력 표준화 통계 + z 통계 + 메타 (`paper_artifacts/observer_fuse/ckpt/*_handoff.pt`, git 제외)

## RoPE 인코더 실험 (정의 정본 = `rope_encoder_plan.md`)

- **판 R / 판 A** — R = C1-DN 가중치에서 위치 인코딩을 전부 2D RoPE 로 바꿔 이어 학습한 판(`--rope` · 인코더 self-attn + routing 교차 주의 Q/K + 디코더 self-attn · APE 4 개 삭제) · A = 같은 레시피를 APE 그대로 이어 학습한 대조 판("에폭을 더 돌린 효과" 분리용)
- **R0** — 공학 단계 첫 변형 = 사용자 11-23 선택 "전부 RoPE"(θ = 10000 · 다른 수정 없음). 이후 변형은 로그 표에 R1, R2 … 로 한 번에 하나씩
- **R-full** — §7 처음부터 학습 판 = C1 레시피 + `--rope`(R0 와 같은 범위) 50 ep → DN 10 ep(R-full-DN). 질문 = R0 의 범용 판독 하락이 RoPE 성질인가 이어 학습의 "전환 빚"인가 · 대조 = C1-DN
- **T1** — §7-2 ③ θ 점검 재설계: θ 100 / 10000 두 판을 R-full 스케줄 첫 4 ep 만 학습 → 건강 관문 seed 3 으로 θ 채택 · 이긴 판은 같은 SUFFIX 로 이어 학습
- **해상도 외삽 / 보간 (`res` / `resi`)** — 224 보다 큰 입력에서 RoPE 자리 번호를 그대로 늘림(외삽 · 못 본 거리) / 학습 범위 0~13 으로 눌러 넣음(보간 · 좌표 × 14/g). APE 판은 둘 다 bicubic 보간
- **판독기 측 위치 임베딩 (PE 판 · `PROBE_PE=1`)** — 주의 풀링 판독기에 학습되는 [스트림, 196, D] 위치 임베딩을 토큰에 더한 뒤 풀링. 판독기가 토큰 안 절대 위치에 기대지 않게 함
- **판 선택 단계 (§8)** — θ 와 RoPE 자리를 4 ep 판으로 고르는 단계(11-26 · §7-3~§7-5 대체). **모두 판** = P · M 인코더 + routing + 디코더 전부 RoPE(= R0 범위 · `T1th100` · `T1th10000`) · **M 전용 판** = M 인코더만 RoPE · P 는 APE
- **복원 자** — 판독기 없이 학습 목표의 항을 그대로 평가하는 자(§8-3): P 제자리 복원 · P 예측 복원 · M 실제 변화 복원(Case B) · M 변화 없음(Case A). **M 의 도움 몫** = 예측 복원과 같은 입력 · 마스크에서 M 만 null 로 바꾼 오차 − 예측 복원 오차
- **확대 크롭** — 두 프레임의 같은 영역을 잘라 224 로 키워 넣는 외란. 격자 14×14 · 위치 번호는 학습 그대로이고 내용의 크기만 바뀜(해상도 변경과 구분)
- **routing 교차 주의** — `MotionRoutingBlock`: owner 스트림의 V 를 helper 스트림의 Q/K 주의로 모으는 층(P 복원 = M 이 Q/K · M 복원 = P 가 Q/K). APE 판에선 국소성이 helper 토큰에 남은 인코더 APE 에서 옴
