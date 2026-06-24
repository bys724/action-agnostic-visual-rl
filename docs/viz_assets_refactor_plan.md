# Visualization Assets Refactor Plan

> **목적**: 가시화 산출물(PNG/GIF/PDF)을 "중복 누적" 대신 "업데이트 in-place"로 관리하도록 리팩토링.
> 코드 리팩토링(legacy viz 스크립트 삭제) 이후 **출력물만 남은 고아 자산**을 정리하고, 향후 생성 컨벤션을 고정한다.
> **작성**: 2026-06-24 (Vault 세션에서 진단·계획 / 실행은 dev 세션).

---

## 1. 진단 요약

| 위치 | 규모 | 상태 |
|------|------|------|
| `docs/architecture/` | 103파일 / **367 MB** (전부 git-tracked) | ⚠️ 대부분 **고아** — 생성기(`attn_*`·`rotation_*`)는 리팩토링 때 삭제됨(`ed23ab0`). 능동적 중복 생성 X, 잔재. |
| `paper_artifacts/` | 76 이미지 / 116 MB | ✅ 컨벤션 양호 (`README.md`: 수작업 편집 금지·스크립트 생성·고정 경로, `fig1~8` 매핑, `_archive/` 분리). **확장할 모델**. |

**핵심**: `docs/architecture/`에서 어디서든 참조되는 건 **v11 attention 세트뿐**:
- `attn_v11_ep44_nomask.png` — fig8 canonical combined 후보 (`fig8_mp_attention/README.md`)
- `attn_v11_ep{48,50}.png` — final champion (`docs/artifacts.md`)
- `attn_v11_ep{4,8,12,16,20,24}_nomask.png` — progression (supplementary)

나머지(`rotation_v10_ep*/` ~240 MB, `rotation_v6_*`, `v4_*`, `ablation_compare/`, `sample_detail/`, `attn_v10_*`)는 **어디서도 참조되지 않는 구버전(v4/v6/v10) 잔재**.

능동적 "epoch마다 새 파일" 패턴이 남은 **활성 스크립트는 2개뿐** (§5).

---

## 2. 목표 컨벤션 — 2-tier

| Tier | 위치 | 정책 | 용도 |
|------|------|------|------|
| **Scratch** | `scratch/viz/` (gitignored 신설) | 자유 생성·덮어쓰기, **커밋 안 됨** | 탐색/디버깅용 per-epoch 가시화 → 저장소에 중복이 쌓이지 않음 |
| **Canonical** | `paper_artifacts/fig*/` (committed) | **고정 이름 + 덮어쓰기 in-place** | 논문/발표에 실제 쓰는 그림만 |

- "update vs duplicate"의 답 = **iteration 덤프는 gitignored scratch로, 저장소엔 canonical만 stable name으로 덮어쓰기.**
- `.gitignore`에 이미 `paper_artifacts/*_main_train_samples/`, "임시 inspection PNG" 선례 있음 → 동일 철학 확장.
- `docs/architecture/`는 폐기: 역할이 scratch(개발 덤프) + `paper_artifacts/fig8`(canonical attn)로 분리됨.

---

## 3. 자산 Manifest

### KEEP → `paper_artifacts/fig8_mp_attention/`로 승격
| 원본 | 이동 후 | 비고 |
|------|---------|------|
| `docs/architecture/attn_v11_ep44_nomask.png` | `fig8_mp_attention/combined/v11_ep44_combined.png` | fig8 README가 이미 계획한 canonical |
| `docs/architecture/attn_v11_ep48.png`, `..._ep50.png` | `fig8_mp_attention/combined/` | final champion |
| `docs/architecture/attn_v11_ep{4,8,12,16,20,24}_nomask.png` | `fig8_mp_attention/combined/progression/` | supplementary progression |

### REVIEW (보존 가치 낮음, dev 세션에서 최종 판단)
- `attn_v11_ep4.png` (masked variant — progression은 `_nomask` 세트가 canonical), `attn_v11_sanity_test2.png` → 기본 삭제.

### DELETE (참조 0, git 이력 복구 가능) — ~350 MB
- `attn_v10_ep{8,12,16,20,24,36,40,44,48}.png`
- `rotation_v10_ep{8,12,16,20,24,36,40,44,48}/` (~240 MB)
- `rotation_v6_baseline/`, `rotation_v6_rotaug_ep*.png` (6)
- `ablation_compare/`, `sample_detail/`
- `v4_comp_grad_e25/`, `v4_comp_sg_e25/`, `v4_mask30_detail/`, `v4_nomask_detail/`
- → 정리 후 `docs/architecture/` 디렉토리 자체 제거.

### paper_artifacts 내 scratch 잔재 (canonical 아님 → scratch로 또는 fig4로 큐레이션)
- `paper_artifacts/parvo_runB2cont_recon_samples/epoch_030_pair.png`
- `paper_artifacts/videomae_recon_samples/videomae_recon_seen_vs_unseen_ep49.png`
- → §5 스크립트 default가 scratch로 바뀌면 자연 해소. 기존 파일은 fig4_recon_quality 후보면 승격, 아니면 삭제.

---

## 4. 실행 단계 (dev 세션)

1. **승격**: KEEP 목록을 `paper_artifacts/fig8_mp_attention/combined/`로 `git mv` (progression은 하위 폴더).
2. **삭제**: DELETE 목록 `git rm -r`. 완료 후 빈 `docs/architecture/` 제거.
3. **scratch tier 도입**: `.gitignore`에 `scratch/` 추가. `scratch/viz/.gitkeep`만 추적(선택).
4. **활성 스크립트 default 재지정** (§5 — 별도 단계, code change).
5. **참조 업데이트** (§6).
6. **검증** (§7) 후 커밋.

> ⚠️ 1~2는 자산 파일 조작(코드 아님), 3~4는 코드/설정 변경 → **dev 세션에서 실행**.

---

## 5. 스크립트 변경 Spec (구현 X — dev 세션 작업 명세)

> 원칙: **활성 viz 스크립트의 기본 출력 = `scratch/viz/<name>/`. 논문용은 명시적 `--out-dir paper_artifacts/figN_.../`로 승격.**

| 스크립트 | 현재 (file:line) | 변경 spec |
|----------|------------------|-----------|
| `scripts/eval/visualize_v15_no_mask.py` | L261-264: default `--out-dir paper_artifacts/parvo_pair_recon_samples`, name `epoch_{ep:03d}_pair.png` | default → `scratch/viz/v15_recon/`. epoch 이름은 scratch에선 유지 OK. fig4 승격 시 `--out-dir paper_artifacts/fig4_recon_quality/` + stable name. |
| `scripts/eval/visualize_videomae_recon.py` | L202-208: default `--out-dir paper_artifacts/videomae_recon_samples`, name `..._ep{ep}.png` | default → `scratch/viz/videomae_recon/`. 동일하게 fig4 승격은 명시 인자. |
| `scripts/eval/analyze_delta_l.py` | L79: default `--output docs/architecture/delta_l_histogram.png` | 삭제될 dir 가리킴 → default `scratch/viz/delta_l_histogram.png`로 repoint. |
| `scripts/eval/visualize_mcp_mae.py` | L55: default `--out paper_artifacts/mcp_mae_sanity_recon/recon.png` | "sanity" = scratch 성격 → default `scratch/viz/mcp_mae_sanity/recon.png` 검토. |

이미 고정 경로로 덮어쓰는 스크립트(`arch_figs/make_*_fig.py`, `pca_overlay.py`, `grad_cam_arrow.py`는 `--out-dir` 필수)는 **변경 불필요** — default만 위 원칙에 맞으면 됨.

**가드** (구현 시 주의):
- 기존 `epoch_{ep}_pair.png` 네이밍 로직 자체는 건드리지 말 것 — scratch에선 epoch 비교가 의도된 동작. 바뀌는 건 **default 출력 위치**뿐.
- canonical 승격은 항상 **명시적 인자**로만 (자동 paper_artifacts 쓰기 금지).

---

## 6. 참조 업데이트 (삭제·이동으로 깨지는 링크)

| 파일:line | 현재 참조 | 조치 |
|-----------|-----------|------|
| `docs/artifacts.md:109-112` | "가시화 산출물" 섹션 (`attn_v11_ep{48,50}`, `rotation_v10_ep*/`, 기타) | fig8 새 경로로 갱신, 삭제된 `rotation_v10`·"기타 PNG" 줄 제거 |
| `paper_artifacts/fig8_mp_attention/README.md:37,38,45` | `docs/architecture/attn_v11_ep44_nomask.png` 등 | `combined/` 새 경로로 갱신 (이미 "복사/심볼릭" 계획돼 있음 → 실제 이동으로 확정) |
| `paper_artifacts/TODO.md:153` | `docs/architecture/attn_v11_ep44_nomask.png` | fig8 새 경로로 갱신 |
| `docs/archive/cluster_sessions_2026-04.md:292` | `docs/architecture/attn_v11_ep{48,50}.png` | 과거 로그 — 링크 깨짐 허용(historical), 손대지 않음 |

---

## 7. 검증 체크리스트

- [ ] `git grep -nE "docs/architecture/" -- '*.md'` → 남은 참조가 archive 로그뿐인지 확인
- [ ] `paper_artifacts/fig8_mp_attention/combined/`에 v11 attn 존재, fig8 README 링크 유효
- [ ] `du -sh .git` 직접 비교는 불가(이력 유지)하나, working tree에서 `docs/architecture/` 부재 + `scratch/`가 gitignore 처리됨 확인
- [ ] 활성 4 스크립트 `--help` default가 `scratch/viz/` 가리킴 (paper_artifacts 자동 쓰기 없음)
- [ ] scratch에 한 번 생성 후 재생성 → **새 파일 안 생기고 덮어쓰기** 되는지 1회 확인

---

## 8. Cross-references

- 산출물 인덱스: [`docs/artifacts.md`](artifacts.md)
- paper figure 파이프라인 컨벤션: [`paper_artifacts/README.md`](../paper_artifacts/README.md), [`paper_artifacts/TODO.md`](../paper_artifacts/TODO.md)
- (범위 밖) post-accept project-page viz 계획: [`docs/representation_visualization_plan.md`](representation_visualization_plan.md) — pca_overlay/gif 별도 트랙
