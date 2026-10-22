# J — plain control convergence check (zero training · 2026-10-23)

Question (AAAI-27 review UnGc W3 · `docs/paper_finalization_plan.md` §3 J): did the submitted plain control
(`two_stream_v15b_step1_plain_xmae_s`, job 36652564, 50 ep, 8×H100) converge, or did it fail to train?

Inputs: tb event files + `history.json` of plain and CoMP-S (`two_stream_v15b_step1_comp_mae_s/20260629_101634`).
Same config (global batch 1024, lr 2.8e-4, identical cosine schedule — `lr` curves match per epoch).
Per-epoch values in `curves.json`.

Comparable terms: the P-reconstruction terms are shared by both runs
(`loss_t` = in-place frame t, `loss_tk` = in-place frame t+k, `loss_pred` = future frame with real M routing).
Total loss is not comparable (plain has no M-recon term).

| | plain min (epoch) | plain ep4 | plain ep50 | CoMP-S ep4 | CoMP-S ep50 |
|---|---|---|---|---|---|
| `loss_t` | 0.0095 (ep3) | 0.0120 | 0.0197 | 0.0085 | 0.0048 |
| `loss_pred` | 0.0136 (ep3) | 0.0156 | 0.0215 | 0.0080 | 0.0023 |
| eval total (every 5 ep) | 0.0325 (ep10) | — | 0.0619 (peak 0.1345 ep40) | — | 0.0180 |

Reading:
- plain reaches its minimum at ep3 (end of warm-up, peak LR), first spike at ep7, sustained rise from ep10,
  and oscillates 0.012–0.032 until ep50. It does not recover while LR decays to ~0 (ep50 = 2.1× its own minimum).
- CoMP-S decreases monotonically to 0.0048 on the same schedule.
- `loss_pred ≈ loss_t` in plain at every epoch: real M routing gives no benefit for future-frame prediction.
- Probe trajectory agrees: plain P identity 1.000 at ep4 → 0.800 at ep50 (STEP 1 same-probe; `docs/cluster_sessions.md` 10-01 rows).
- The C2 mirror run (also no M-recon) shows the same pattern (ep6 reconstruction collapse; `factorization_crossover_plan.md` §4.1).

Verdict [confirmed · log-based]: **plain did not converge — optimization failure after warm-up**, not a healthy
same-recipe control. STEP 2-A (OOD probing, plain 0.030 / 0.127 / 0.109 / 0.059) used the ep50 `latest.pt`,
i.e. the degraded state. Branch ③ of the plan applies (retrain once, or demote claim 3) — Vault decision.

Not affected: claim 1 (mechanism). Its plain/C2 comparison is defended at ep4, where all runs are healthy
(M motion C0 0.795 / plain 0.183 / C2 0.094; P identity 1.000 for all).
