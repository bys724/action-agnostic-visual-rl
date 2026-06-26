# Architecture figure generation

논문/발표용 아키텍처 도식의 **생성 소스·코드** 한곳 모음. 렌더 결과(아티팩트)는
[`paper_artifacts/fig1_architecture/`](../../../paper_artifacts/fig1_architecture/) 로 출력.

> 소스(코드+mermaid)는 여기, 렌더 산출물은 paper_artifacts — source/artifact 분리.
> 저장소 전체 리팩토링은 별도 작업(코드 식별자 rename 등). 여기는 figure 생성에 한정.

## 구성

| 파일 | 역할 |
|------|------|
| `_common.py` | 공유 모듈 — 색 컨벤션·EgoDex 프레임 추출·matplotlib 드로잉 프리미티브. **단일 출처** |
| `make_comp_mae_fig.py` | **CoMP-MAE (현행 ours, code v16)** — 대칭 cross-recon mirrored 2-branch (matplotlib) |
| `make_mcp_mae_fig.py` | MCP-MAE SiamMAE-스타일 figure (predecessor — P-recon만) |
| `make_ms_jepa_fig.py` | **MS-JEPA** 동일 스타일 figure (predecessor 비교용) |
| `mcp_mae_architecture.mmd` | MCP-MAE 상세 dataflow (Mermaid 소스) |
| `ms_jepa_architecture.mmd` | MS-JEPA 상세 dataflow (Mermaid 소스) |

## 색 컨벤션 (`_common.py` 단일 출처)

P/appearance/what = green · M/motion/where = blue · routing/predictor = purple ·
head/output = orange · teacher/EMA = teal (MS-JEPA 전용) · loss = red · masked patch = gray.

## 재생성 (repo 루트에서 실행)

```bash
# matplotlib figures (실제 EgoDex 프레임 추출 포함)
python3 scripts/viz/arch_figs/make_comp_mae_fig.py paper_artifacts/fig1_architecture/comp_mae_fig.png   # 현행 메인
python3 scripts/viz/arch_figs/make_comp_mae_fig.py paper_artifacts/fig1_architecture/comp_mae_fig.pdf
python3 scripts/viz/arch_figs/make_mcp_mae_fig.py paper_artifacts/fig1_architecture/mcp_mae_fig.png
python3 scripts/viz/arch_figs/make_mcp_mae_fig.py paper_artifacts/fig1_architecture/mcp_mae_fig.pdf
python3 scripts/viz/arch_figs/make_ms_jepa_fig.py  paper_artifacts/fig1_architecture/ms_jepa_fig.png
python3 scripts/viz/arch_figs/make_ms_jepa_fig.py  paper_artifacts/fig1_architecture/ms_jepa_fig.pdf

# mermaid dataflow (claude-mermaid MCP 또는 mmdc CLI)
mmdc -i scripts/viz/arch_figs/mcp_mae_architecture.mmd -o paper_artifacts/fig1_architecture/mcp_mae_architecture.png -t neutral -b white
mmdc -i scripts/viz/arch_figs/ms_jepa_architecture.mmd  -o paper_artifacts/fig1_architecture/ms_jepa_architecture.png  -t neutral -b white
```

- **프레임 출처**: `_common.load_frame_pair()` 기본값 = `paper_artifacts/parvo_runB2cont_recon_samples/epoch_030_pair.png`
  의 EgoDex GT 쌍(녹색 물체 조작, row2). 다른 예시로 바꾸려면 `row=` 인자 조정.
- **언어**: figure는 대외 산출물 → 영어 라벨. (Times New Roman; 한글 글리프 없음.)

## 새 모델 figure 추가 시

`make_<model>_fig.py` 신설 → `import _common as C` → ax-bound wrapper로 body 작성
(기존 두 스크립트가 템플릿). 새 색이 필요하면 `_common.py`에 추가해 단일 출처 유지.
