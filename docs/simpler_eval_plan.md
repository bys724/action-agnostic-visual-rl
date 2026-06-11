# SIMPLER Evaluation — 준비 계획 (TODO)

> **목적**: 우리 action-agnostic encoder/표현을 **SIMPLER** 벤치마크로 평가하기 위한 준비.
> **분담**: 클러스터(olaf) = 데이터셋 학습 / 로컬 워크스테이션 = SIMPLER 환경 + 테스트.
> **상태**: 계획 (2026-06-11). 실제 구현은 dev session에서.
> **결정 출처**: Vault 조사 (SIMPLER 세팅·공식 학습 데이터) — `Projects/Action-Agnostic Paper/2. Experiments`.

---

## ⚠️ greenfield 아님 — 기존 인프라 활용

repo에 SIMPLER 통합이 일부 존재. 새로 짜지 말고 **확장**할 것:

- `src/policies/openvla/openvla_model.py` — OpenVLA용 **SimplerEnv wrapper** ([DelinQu/SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA) 기반). 7D action `[x,y,z,rx,ry,rz,gripper]` 변환 로직 포함 (L244-245).
- `scripts/setup/setup_openvla.py` — SimplerEnv용 **ManiSkill assets 다운로드** (`widowx250s` 로봇 asset 등, L57-64).

→ 우리 policy wrapper는 위 OpenVLA wrapper **패턴을 그대로 따라** 작성하면 됨. baseline(OpenVLA) 평가도 이 코드로 이미 가능 → **비교 기준선 확보가 쉬움**.

---

## 📋 SIMPLER 사실 요약 (조사 결과)

- **정체**: 평가 전용 벤치마크 (Li et al., **CoRL 2024**, [2405.05941](https://arxiv.org/abs/2405.05941), [repo](https://github.com/simpler-env/SimplerEnv)). 학습 데이터 미제공 — *정책은 따로 학습 후 sim에서 평가*. sim 성능이 real과 강하게 상관되도록 real-to-sim gap 최소화.
- **시뮬레이터**: SAPIEN + ManiSkill2 (CPU). GPU 가속 ManiSkill3 버전도 존재 → **둘 중 결정 필요**.
- **2개 셋업 ↔ 공식 학습 데이터**:

  | 셋업 | 로봇 | 학습 데이터 (OXE 서브셋) | 규모 |
  |---|---|---|---|
  | Google Robot | RT 시리즈 로봇 | **Fractal** (= RT-1 dataset, `fractal20220817_data`) | ~13만 ep, ~100GB+ |
  | WidowX / Bridge | WidowX 250 | **BridgeData V2** | ~6만 traj |

- **Task** (셋업당 ~4, rigid-body만):
  - Google Robot: Pick Coke Can / Move Near / Open·Close Drawer / Place in Closed Drawer
  - WidowX(Bridge): Spoon on Towel / Carrot on Plate / Stack Cube / Eggplant in Basket
- **평가 모드 2종**:
  - **Visual Matching** (주력, real 상관 높음): green screening(시뮬 에셋을 실제 배경에 오버레이) + texture matching(실제 텍스처 투영).
  - **Variant Aggregation**: 배경·조명·distractor·텍스처 교란 variant 평균 → robustness.
- **기존 baseline 숫자 공개**: RT-1, RT-1-X, RT-2-X, Octo, OpenVLA → 같은 프로토콜이면 우리 방법 직접 비교.

---

## ✅ TODO — A. 클러스터 (olaf): 데이터셋 학습 준비

- [ ] **데이터 확보**: Fractal(RT-1) + BridgeData V2 다운로드 (OXE). 저장소 quota 확인 → `docs/dataset_todo.md` 연계. Fractal ~100GB+ 주의.
- [ ] **학습 프로토콜 결정**:
  - 표현 **freeze vs finetune** (action-agnostic 검증 목적이면 freeze + BC head가 정합적)
  - action space = SimplerEnv 포맷 **7D EE-delta + gripper**에 맞춤
  - baseline(OpenVLA 등)과 **동일 데이터·epoch·action space**로 맞춰 공정 비교
- [ ] **BC policy head 학습 파이프라인** (our encoder → head). 기존 LIBERO BC 코드 재사용 가능한지 검토.
- [ ] **DDP sbatch**: 기존 olaf 2node × 4 H100 환경 재사용 (`cluster_sessions.md` 패턴).
- [ ] **checkpoint 전송 경로**: 클러스터 학습 → 로컬 sim 평가용 ckpt sync 방법 정리.

## ✅ TODO — B. 로컬 워크스테이션: SIMPLER 환경 + 테스트

- [ ] **SimplerEnv + ManiSkill2 설치** (또는 ManiSkill3 GPU 결정).
- [ ] **assets 다운로드**: 기존 `scripts/setup/setup_openvla.py` 활용 (widowx250s 등 이미 준비).
- [ ] **our policy wrapper 작성** ← *dev session 구현*. `openvla_model.py` 패턴 따라 our encoder+head → 7D action 변환.
- [ ] **WidowX/Bridge 셋업부터** 시작 (widowx250s asset 이미 있음) → 이후 Google Robot 확장.
- [ ] **Visual Matching 우선** 평가 (real 상관 높음). Variant Aggregation은 robustness 보조 지표.
- [ ] **baseline 재평가**: 기존 OpenVLA wrapper로 동일 환경에서 baseline 숫자 재현 → 우리 숫자와 사과 대 사과.

---

## 🚨 주의사항 (critical guards)

- **공정 비교 최우선**: 표현만 갈아끼우고 head 붙이는 방식이면 **baseline도 같은 head/데이터/평가셋으로 재평가**. 공개 숫자 직접 인용 시 프로토콜 차이(action space, ckpt, 평가 모드) 확인.
- **action space 일치**: SimplerEnv는 7D `[x,y,z,rx,ry,rz,gripper]`. 출력 포맷·정규화·gripper 컨벤션 불일치 시 성능 0 — `openvla_model.py:245` 참고.
- **embodiment 2종 / task 4개씩** = 좁은 범위. "cross-embodiment 일반화" 과주장 금지. action-agnostic 주장 근거로는 보조 evidence.
- **학습=클러스터 / sim=로컬** 분리 → 환경(conda, ManiSkill 버전) 이원화. 로컬 sim 추론에 GPU 필요.
- **ManiSkill2(CPU) vs ManiSkill3(GPU)** 먼저 결정 (throughput 차이 큼).
- 이 문서는 **계획만** — runnable 코드(wrapper 구현, sbatch, BC head)는 dev session에서.

---

## 🔗 Cross-references

- **Vault 결정·조사**: `Projects/Action-Agnostic Paper/2. Experiments` (SIMPLER 평가 축)
- **기존 코드**: `src/policies/openvla/openvla_model.py`, `scripts/setup/setup_openvla.py`
- **데이터 확보**: `docs/dataset_todo.md`
- **상위 계획**: `docs/RESEARCH_PLAN.md`, `docs/paper_experiments_plan.md`
- **외부**: [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv), [DelinQu/SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA), [프로젝트 페이지](https://simpler-env.github.io/)
