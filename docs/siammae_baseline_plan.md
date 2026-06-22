# SiamMAE 비교군 — 타당성 분석 (구현 보류)

> **상태 (2026-06-22)**: 분석 완료, 구현 미착수. Paper 2(Action-Agnostic) baseline 후보. 모델 명명·2논문 구조는 [`CLAUDE.md`](../CLAUDE.md) "명명 · 2논문 구조".
> **작업 분담**: 코드 편집(모델·dispatch·어댑터)은 이 저장소 세션. 본 문서는 계획·근거 (문서 전용).

## 결론

EgoDex에서 **SiamMAE-objective**를 matched 조건으로 학습해 Paper 2 baseline으로 추가 = **타당하며 샤프함**. 주 비용은 재구현 충실도(공식 code 미공개). 통합 마찰은 작음 (2-frame 로더·BC-T 어댑터 재사용).

## SiamMAE 정체 (NeurIPS 2023, Gupta/Wu/Deng/Fei-Fei)

2-frame·action-agnostic·masking 기반 **correspondence** 학습. 과거 f₁은 그대로, 미래 f₂를 **95% 마스킹**, **cross-attention decoder**가 f₁ 토큰을 참조해 f₂ 가려진 패치를 예측 (normalized pixel L2). "무엇이 어디로 갔나"를 학습.

| 항목 | 공식 recipe |
|------|------|
| Encoder | Siamese ViT-S/16 (공유 가중치, 21M), inference는 **프레임별 단독 인코딩** |
| 마스킹 | f₁ 0% / **f₂ 95%** (비대칭) |
| Decoder | cross-self (cross-attn → self-attn) |
| 프레임 간격 | 4–48 frames @30fps |
| Pretrain | Kinetics-400, 2000 ep, batch 2048, 4×Titan RTX |
| Eval (원논문) | DAVIS/JHMDB/VIP label propagation (correspondence) |

## Code 가용성 — ⚠️ 공식 미공개

agrimgupta92 GitHub에 SiamMAE 없음 (JAX 기반 MAE 확장으로만 언급). PyTorch는 비공식 재구현뿐 ([Robiwan245/SiamMAE](https://github.com/Robiwan245/SiamMAE) — KTH 코스, UCF-101). → **직접 재구현 필요**. 단 [facebookresearch/mae](https://github.com/facebookresearch/mae) 대비 델타 작음: siamese forward + 비대칭 마스킹 + cross-self decoder. 기존 `src/models/videomae.py`(497줄) 규모 새 파일 1개 수준.

## 왜 샤프한 비교군인가

세 inductive bias가 깔끔히 분리:
- **VideoMAE-ours**: 대칭 tube 마스킹 + joint self-attn 복원
- **SiamMAE**: 비대칭(f₁ full / f₂ 95%) + **cross-attention 예측**
- **Parvo(v15b)**: two-stream M/P + M→P scaffold

SiamMAE = Parvo "cross-stream 예측 bias"의 **단일스트림 대조판**. Paper 2 검증질문("M→P scaffold가 input-only baseline을 넘는가")에서 — **Parvo가 SiamMAE(action 없이 cross-frame 예측만 하는 강한 baseline)를 못 넘으면 깨끗한 negative**. VideoMAE보다 Parvo 설계철학에 가까워 더 엄격한 reference.

## 통합 가능성 (재사용 지점 확인됨)

- **데이터로더**: EgoDex pair 샘플링(`src/datasets/egodex.py`, `max_gap`) 이미 존재 — Parvo·VideoMAE와 공유. SiamMAE 4–48프레임 = `max_gap` 매핑. 신규 불필요.
- **dispatch**: `scripts/pretrain.py` `choices=[...]`에 `siammae` elif 1개.
- **학습 루프**: `src/training/pretrain.py`는 model-agnostic(forward→loss). 그대로.
- **BC-T 어댑터**: `VideoMAEOursAdapter` 패턴 재사용. SiamMAE 인코더는 프레임별 단독 인코딩이라 BC-T 타임스텝 특징 추출이 오히려 더 깔끔.

## Parity·프레이밍 (정직한 caveat)

- **"EgoDex에서 SiamMAE-objective 학습"이지 공식 SiamMAE 재현 아님**. DAVIS 수치 재현 목표 아님 — 다른 baseline과 matched 조건(같은 EgoDex pair 샘플링·epoch·해상도)으로 학습, 아키텍처/objective만 변수화. [`eval_protocols.md`](eval_protocols.md) §0 parity 가드 준수.
- 2000ep→matched(~20–50ep) 축소는 결함이 아니라 controlled comparison 요건.
- SiamMAE는 correspondence용 설계, control(LIBERO) 검증 전례 없음 — 하지만 "action-agnostic correspondence pretraining이 control에 전이되는가"가 이 프로젝트 질문이라 informative baseline.
- 원논문 95% 비대칭 마스킹 충실 유지 (EgoDex M-stream aggressive 금지 철학은 Parvo 전용, baseline엔 미적용).

## 확정 spec (재현 리스크 해소, 2026-06-22)

논문 §3 + 비공식 [Robiwan245/SiamMAE](https://github.com/Robiwan245/SiamMAE) `SiamMae.py` 교차검증.

**Decoder block** (핵심 — 표준 MAE에 없는 부분):
```
# x1 = f1 인코딩(full), x2 = f2 (encoder 출력 + [MASK] token + pos embed = full set)
x = x2 + cross_attention(norm1(x1), norm1(x2))   # q=x2, k/v=x1 (f2가 f1 참조)
x = x  + self_attention(norm2(x))                 # f2 self-attn (ablation상 필수)
```
- **Decoder config**: dim 512 · depth 8 · heads 16 (`dec512d8b`, MAE 기본값 — "MAE 오픈소스 위 구축" 명시와 일치)
- **비대칭 마스킹**: f₁ `mask_ratio=0` / f₂ `mask_ratio=0.95` (random)
- **Loss**: normalized-pixel L2, 마스킹된 f₂ 패치만 `(loss*mask).sum()/mask.sum()`
- **ablation 근거**: cross-self 58.1 > cross-only 52.2 J&Fm → cross+self 둘 다 구현

**Backbone (확정 2026-06-22)**: **ViT-B/16** — VideoMAE-ours와 동일 capacity로 parity 확보, objective/구조만 변수화 (원논문 ViT-S→scale-up; 2000ep→matched와 같은 controlled-comparison 논리). native ViT-S reference는 후속 선택. 인코더 block은 VideoMAE-ours와 동일 `Block`(modeling_finetune) 재사용 → per-stream capacity 통제.

## 구현 작업 (착수 시)

1. `src/models/siammae.py` — Siamese ViT-B/16 인코더(videomae/timm 재사용) + cross-self decoder + 비대칭 95% 마스킹 + normalized pixel L2.
2. `scripts/pretrain.py` dispatch + `--model siammae`.
3. `src/encoders/adapters/siammae.py` — BC-T 어댑터 (single_frame/videomae 패턴, 프레임별 단독 인코딩).
4. probing 모드 추가 (`scripts/eval/probe_action*.py`).
5. EgoDex part1 matched epoch 학습.

## Sources

- [SiamMAE arXiv 2305.14344](https://arxiv.org/abs/2305.14344) · [project page](https://siam-mae-video.github.io/) · [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7ffb9f1b57628932518505b532301603-Abstract-Conference.html)
- 관련: [Investigating Pre-Training Objectives for Generalization in Vision-Based RL (2406.06037)](https://arxiv.org/pdf/2406.06037)
