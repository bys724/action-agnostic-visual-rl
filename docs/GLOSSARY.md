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
- **F1 / `raw-dl`** — 학습 없는 바닥선: CoMP M 입력과 같은 raw ΔL을 16×16 패치로 자른 것을 그대로 토큰으로 (196×256). probe 인코더 이름 `raw-dl`
- **M 단독 / `m_only`** — probe에 M 스트림 토큰만 넣는 판독 (P_t⊕M과 구분)
- **ⓘ / ⓢ 시험** — probe 학습과 시험이 같은 분포(참조용) / probe 학습에 없는 조건(판정용: 교란·suite 간 무재학습 전이)
