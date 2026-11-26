"""RoPE §8 복원 평가 집계 (docs/rope_encoder_plan.md §8-5 · 수치 조작화 = docs/VAULT_SYNC_TODO.md "dev 확인 보고 ⓕ" · 결과 전 고정).

입력 = scripts/eval/recon_eval.py 의 <판>.npz (표본별 값). 출력 = recon_summary.{md,json}.
  ① 판별 표: 조건 × 항(절대 값) + 낙폭(조건 − base · 도움 몫은 base − 조건 → 클수록 나쁨)
  ② 짝 비교(--pair A B ...): Δ_A − Δ_B 의 짝 부트스트랩 CI · 칸 읽기 "A 가 낫다" = CI 전부 < 0 ∧ |차| ≥ 두 판 base 평균의 5 %
     · **판정 칸 = 못 본 장면 2 조건(libero · calvin) × 항 2 = 4 칸** (§8-5 · Vault 11-26 (2) · 1 단계 결과 전 확정).
       평행이동 3 · 확대 3 은 측정 · 표시만 하는 보고 행(0 단계에서 어떤 판 쌍도 못 가름).
     · 읽기 = 4 칸 중 3 칸 이상 낫고 ∧ 반대 칸 0 ∧ base 에서 5 % 넘게 나쁘지 않음 ∧ 토큰 등변성 M(position_<판>.json · frac_moved CI)
       이 비중첩으로 반대가 아님 → "낫다" · 그 외 "못 가름". (0 단계 표 recon_summary.md 는 옛 16 칸 규칙으로 만든 기록 — --out 을 달리 줄 것)
사용: python scripts/cluster/rope/recon_summary.py --arms A0 R0 ... --pair T1th100 T1th10000 [--pair ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parents[3] / "paper_artifacts/rope_judge/recon"
EGO = ["base", "indep", "win8", "win16", "shift8", "shift16", "shift32", "zoom1.25", "zoom1.5", "zoom2"]
JUDGE = ["libero", "calvin"]                                                # 판정 칸 (× 항 2 = 4 칸)
REPORT = ["shift8", "shift16", "shift32", "zoom1.25", "zoom1.5", "zoom2"]   # 보고 행
NB, TOL = 2000, 0.05


def load(arm):
    z = np.load(DIR / f"{arm}.npz")
    conds = sorted({k.split("/")[0] for k in z.files if "/" in k})
    return {c: {k.split("/")[1]: z[k] for k in z.files if k.startswith(c + "/")} for c in conds}


def stat(r, term, idx=None):
    """표본(행) 번호 idx 로 재표집한 값. help = Lnull − Lpred 평균 · mB = 가중 합의 비."""
    g = (lambda k: r[k] if idx is None else r[k][idx])
    if term == "help":
        return float((g("Lnull") - g("Lpred")).mean())
    if term == "mB":
        return float(g("mB_num").sum() / g("mB_den").sum())
    if term == "mB_mov":
        return float(g("mB_mov_sum").sum() / max(g("mB_mov_cnt").sum(), 1.0))
    if term == "mA":
        return float(g("mA_num").sum() / g("mA_den").sum())
    return float(g(term).mean())


def boot_idx(group, rng):
    """묶음(영상 · 시연 · 세그먼트) 단위 재표집 → 행 번호."""
    ug, inv = np.unique(group, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(ug))]
    for _ in range(NB):
        yield np.concatenate([rows[j] for j in rng.integers(0, len(ug), len(ug))])


def drop_samples(r, cond, term, seed=0):
    """낙폭의 부트스트랩 표본 [NB] (클수록 나쁨). EgoDex 조건 = base 와 같은 재표집(짝) · 못 본 장면 = 두 세트 독립 재표집.
    seed 고정 → 두 판이 같은 재표집 번호를 쓴다(판 간 짝)."""
    sign = -1.0 if term == "help" else 1.0
    rb, rc = r["base"], r[cond]
    if cond in EGO:
        it = boot_idx(rb["group"], np.random.default_rng(seed))
        return np.array([sign * (stat(rc, term, i) - stat(rb, term, i)) for i in it])
    ib, ic = boot_idx(rb["group"], np.random.default_rng(seed)), boot_idx(rc["group"], np.random.default_rng(seed + 1))
    return np.array([sign * (stat(rc, term, j) - stat(rb, term, i)) for i, j in zip(ib, ic)])


def ci(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def arm_table(arm, r):
    terms = ["Lt", "Ltk", "Lpred", "Lnull", "help", "copy", "mB", "mB_mse", "mB_mov", "mA"]
    rows = [f"### {arm}", "", "| 조건 | n | " + " | ".join(terms) + " | 낙폭 help | 낙폭 mB |", "|" + "---|" * (len(terms) + 4)]
    out = {}
    for c in [c for c in EGO + ["libero", "calvin"] if c in r]:
        v = {t: stat(r[c], t) for t in terms}
        d = {t: (-1.0 if t == "help" else 1.0) * (v[t] - stat(r["base"], t)) for t in ("help", "mB")}
        out[c] = {**v, "drop_help": d["help"], "drop_mB": d["mB"]}
        rows.append(f"| {c} | {len(r[c]['Lt'])} | " + " | ".join(f"{v[t]:.5f}" for t in terms)
                    + f" | {d['help']:+.5f} | {d['mB']:+.5f} |")
    return "\n".join(rows) + "\n", out


def compare(a, b, ra, rb_):
    """판 a 대 b: 판정 칸마다 Δ_a − Δ_b (음수 = a 가 덜 나빠짐)."""
    rows, cells = [], []
    base = {}
    for t in ("help", "mB"):
        ia = list(boot_idx(ra["base"]["group"], np.random.default_rng(0)))
        da = np.array([stat(ra["base"], t, i) - stat(rb_["base"], t, i) for i in ia])
        va, vb = stat(ra["base"], t), stat(rb_["base"], t)
        ref = (abs(va) + abs(vb)) / 2
        worse = (-da if t == "help" else da)          # 양수 = a 가 base 에서 나쁨
        lo, hi = ci(worse)
        base[t] = {"a": va, "b": vb, "diff_worse_ci": [lo, hi],
                   "a_worse": bool(lo > TOL * ref), "b_worse": bool(hi < -TOL * ref)}
    void_help = all(abs(stat(r["base"], "help")) < TOL * stat(r["base"], "Lpred") for r in (ra, rb_))
    for t in ("help", "mB"):
        ref = (abs(base[t]["a"]) + abs(base[t]["b"])) / 2
        for c in JUDGE + REPORT:
            xa, xb = drop_samples(ra, c, t), drop_samples(rb_, c, t)
            pa = (-1.0 if t == "help" else 1.0) * (stat(ra[c], t) - stat(ra["base"], t))
            pb = (-1.0 if t == "help" else 1.0) * (stat(rb_[c], t) - stat(rb_["base"], t))
            lo, hi = ci(xa - xb)
            disc = c in JUDGE   # 판정 칸 여부 (보고 행은 읽기에서 제외)
            big = abs(pa - pb) >= TOL * ref
            verdict = "a" if (hi < 0 and big) else "b" if (lo > 0 and big) else "같음"
            if t == "help" and void_help:
                verdict, disc = "자 불성립", False
            cells.append({"term": t, "cond": c, "drop_a": pa, "drop_b": pb, "diff_ci": [lo, hi], "disc": bool(disc), "verdict": verdict})
            rows.append(f"| {t} | {c} | {pa:+.5f} [{ci(xa)[0]:+.5f}, {ci(xa)[1]:+.5f}] | {pb:+.5f} [{ci(xb)[0]:+.5f}, {ci(xb)[1]:+.5f}] | "
                        f"[{lo:+.5f}, {hi:+.5f}] | {'판정' if disc else '보고'} | {verdict} |")
    kept = [x for x in cells if x["disc"]]
    na, nb = sum(x["verdict"] == "a" for x in kept), sum(x["verdict"] == "b" for x in kept)
    eq = {}
    for arm in (a, b):   # 짝 자: 토큰 등변성 M (클수록 등변) — 없으면 읽기 보류
        f = DIR.parent / f"position_{arm}.json"
        eq[arm] = json.load(open(f))["equivariance"]["M"] if f.exists() else None
    def eq_opposite(win, lose):   # 진 판의 등변성이 비중첩으로 더 높으면 반대
        return eq[lose]["frac_ci"][0] > eq[win]["frac_ci"][1]
    if len(kept) < 4:
        read = f"멈춤 (판정 칸 {len(kept)} < 4 — 자 불성립)"
    elif None in eq.values():
        read = "보류 (토큰 등변성 파일 없음)"
    elif na >= 3 and nb == 0 and not any(base[t]["a_worse"] for t in base):
        read = f"{a} 가 낫다" if not eq_opposite(a, b) else "못 가름 (복원은 a · 토큰 등변성 M 이 비중첩으로 반대)"
    elif nb >= 3 and na == 0 and not any(base[t]["b_worse"] for t in base):
        read = f"{b} 가 낫다" if not eq_opposite(b, a) else "못 가름 (복원은 b · 토큰 등변성 M 이 비중첩으로 반대)"
    else:
        read = "못 가름"
    md = [f"### {a} (a) 대 {b} (b)", "",
          "base: " + " · ".join(f"{t} a {base[t]['a']:.5f} / b {base[t]['b']:.5f} (a 가 나쁜 쪽 차 CI [{base[t]['diff_worse_ci'][0]:+.5f}, {base[t]['diff_worse_ci'][1]:+.5f}])" for t in base),
          "", "| 항 | 조건 | 낙폭 a [CI] | 낙폭 b [CI] | 낙폭 차 a − b CI | 칸 종류 | 칸 읽기 |", "|---|---|---|---|---|---|---|", *rows, "",
          "토큰 등변성 M (frac_moved [CI]): " + " · ".join(
              f"{arm} " + (f"{eq[arm]['frac_moved']:.3f} [{eq[arm]['frac_ci'][0]:.3f}, {eq[arm]['frac_ci'][1]:.3f}]" if eq[arm] else "없음") for arm in (a, b)), "",
          f"판정 칸 {len(kept)} · a 가 나은 칸 {na} · b 가 나은 칸 {nb} → **읽기 = {read}**", ""]
    return "\n".join(md), {"a": a, "b": b, "base": base, "cells": cells, "kept": len(kept), "n_a": na, "n_b": nb, "equiv_M": eq, "reading": read}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--pair", nargs=2, action="append", default=[])
    ap.add_argument("--out", default="recon_summary")
    a = ap.parse_args()
    R = {arm: load(arm) for arm in a.arms}
    md, js = ["# RoPE §8 복원 평가 집계 (자동 생성 · scripts/cluster/rope/recon_summary.py)", "",
              "값 = 가려진 패치 오차(P 항 = 픽셀 MSE · mB · mA = 학습 가중 합의 비). help = Lnull − Lpred(클수록 좋음). 낙폭은 클수록 나쁨.", ""], {"arms": {}, "pairs": []}
    for arm in a.arms:
        t, o = arm_table(arm, R[arm]); md.append(t); js["arms"][arm] = o
    for x, y in a.pair:
        t, o = compare(x, y, R[x], R[y]); md.append(t); js["pairs"].append(o)
    (DIR / f"{a.out}.md").write_text("\n".join(md))
    json.dump(js, open(DIR / f"{a.out}.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
