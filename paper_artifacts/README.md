# Paper Artifacts

This directory is the **single hand-off point** between this code/research
repository and the paper-writing repository. Everything here is intended to be
consumed (read-only) by paper-side scripts that produce final figures, tables,
and LaTeX content.

## Why a separate directory

- Code repo: experiments, raw probe runs, ckpt management, scratch analysis.
- Paper repo: figure styling, table formatting, LaTeX, version-controlled
  drafts of plots that go through editorial iteration.
- Mixing the two means figure tweaking pollutes the code-repo history and
  paper-repo bloats with raw artifacts. This directory is the contract.

## Directory layout

| Path | Content | Status |
|------|---------|--------|
| `probing/` | EgoDex within-domain + DROID cross-domain probing CSVs | active |
| `architecture/` | Model diagrams + attention visualizations | active |
| `libero_bct/` | Behavior cloning fine-tune metrics (val MSE, NLL) | pending — BC jobs running |
| `libero_rollout/` | LIBERO simulator success rate (closed-loop eval) | pending — local workstation |

Each subdirectory has its own `README.md` describing:
- what each file represents,
- the experimental conditions (gap, split, ckpt provenance, hyperparameters),
- known limitations / outliers / things to filter,
- candidate figures or tables that can be produced from the data,
- which raw source files / scripts produced the artifact (for reproducibility).

## Update workflow

This directory is **not hand-edited**. CSVs are regenerated from raw probe
JSON files via:

```bash
python scripts/analysis/export_probing_summary.py
```

PNG diagrams under `architecture/` are copied/linked from `docs/architecture/`
when stable. Whenever new probe runs land or new attention plots are produced,
re-run the export script and copy refreshed PNGs.

## Provenance

- Raw probe results: [`data/probing_results/`](../data/probing_results/) (gitignored / heavy)
- Source diagrams: [`docs/architecture/`](../docs/architecture/)
- Cluster session log (which JobID produced what ckpt): [`docs/cluster_sessions.md`](../docs/cluster_sessions.md)
- Master research plan + result tables: [`docs/RESEARCH_PLAN.md`](../docs/RESEARCH_PLAN.md)

## Conventions

- All CSVs are UTF-8, `,`-separated, with a header row.
- All `r2` columns are EgoDex/DROID **action joint position** R² over a held-out test split.
- "v11" always refers to the run in `checkpoints/two_stream_v11/20260426_014333/` unless noted.
- Epoch `50` corresponds to the `latest.pt` checkpoint at end of 50-epoch training.
