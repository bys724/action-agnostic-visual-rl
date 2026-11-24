"""RoPE 인코더 판정 단계 보조·보고 자 둘 (docs/rope_encoder_plan.md §3-2) — 동결 인코더 · 학습 0.

① 쉬프트 등변성 (보조 판정): 프레임 쌍을 16 px(한 패치) 오른쪽·아래로 옮기면(반사 패딩 = probe_action_libero.perturb_pair
   'shift') 원본 토큰 O[i, j] 의 내용은 S[i+1, j+1] 로 간다. 경계 패치 제외 = O 의 i, j ∈ 1..12(옮겨 간 자리 2..13 ·
   반사 띠 0 행·열과 밀려 나가는 13 행·열 제외). 패치마다 moved = cos(O[i,j], S[i+1,j+1]) · stay = cos(O[i,j], S[i,j]).
   주 지표 = **moved > stay 인 패치 비율**("이웃 칸으로 옮겨 갔는가" · 크기 척도 무관) · 보고 = moved · stay 평균.
   스트림 = P(P_t 토큰) · M(옮긴 쌍의 ΔL 토큰). CI = 시연 단위 부트스트랩 1000 회(2.5–97.5 %).
② 위치 선형 판독 (보고): SC `scripts/action_predictor_stage0.py` 위치 판독 정의 그대로 — libero_goal 앞 카메라 P 토큰 하나(384)
   → 패치 번호 196 분류 · 선형 · 학습 = 시연 35~37 · 시험 = 38~39 · 에피소드당 프레임 20(rng 0) · 학습 통계 표준화 ·
   Adam 1e-2 · 1500 회 · 배치 4096 · top-1 과 이웃 1 칸 안. 차이 = SC 는 재렌더 프레임, 여기는 hdf5 프레임(128² → 224 bilinear).
   판독기 seed 3(0·1·2 · 초기화·배치 순서만 다름).

사용: python scripts/eval/rope_position.py --checkpoint <ckpt> --out <json> [--img-size 224]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.eval.probe_action_libero import (  # noqa: E402
    build_parvo_encoder, list_demos, load_demo, perturb_pair, preprocess_frames,
)

GRID = 14
GAP = 20                    # 건강 관문(STEP 1 same-probe)과 같은 프레임 간격


def suite_hdf5(suite: str, data_root: str):
    sys.path.insert(0, str(PROJECT_ROOT / "external/LIBERO"))
    from libero.libero.benchmark import get_benchmark
    bm = get_benchmark(suite)(0)
    return [str(Path(data_root) / suite / bm.get_task_demonstration(i).split("/")[-1]) for i in range(bm.n_tasks)]


@torch.no_grad()
def tokens(model, prev, curr, dev):
    p, c = prev.to(dev), curr.to(dev)
    tp = model._encode_p_unmasked(model.preprocessing.compute_p_channel(p))[:, 1:]
    tm = model._encode_m_unmasked(model.preprocessing.compute_m_channel(p, c))[:, 1:]
    return tp.float(), tm.float()


def equivariance(model, dev, data_root, demos=range(40, 50), per_demo=5, n_boot=1000):
    """libero_object · 과제 10 × 시연 40~49 × 프레임 5 (rng 0) · gap 20 쌍."""
    rng = np.random.default_rng(0)
    i = torch.arange(1, GRID - 1)
    ii, jj = torch.meshgrid(i, i, indexing="ij")
    src = (ii * GRID + jj).flatten()                 # O[i, j]
    dst = ((ii + 1) * GRID + (jj + 1)).flatten()     # S[i+1, j+1]
    rows = {s: [] for s in ("P", "M")}               # 시연별 (frac, moved, stay)
    for hp in suite_hdf5("libero_object", data_root):
        keys = set(list_demos(hp))
        for d in demos:
            assert f"demo_{d}" in keys, (hp, d)
            fr, *_ = load_demo(hp, f"demo_{d}")       # 이름 기준 번호 (list_demos 는 문자열 정렬)
            t = np.sort(rng.choice(len(fr) - GAP, per_demo, replace=False))
            prev, curr = preprocess_frames(fr[t], 224), preprocess_frames(fr[t + GAP], 224)
            sp, sc = perturb_pair(prev, curr, "shift", 16, None)
            for s, (o, sh) in zip(("P", "M"), zip(tokens(model, prev, curr, dev), tokens(model, sp, sc, dev))):
                o = F.normalize(o[:, src], dim=-1)
                moved = (o * F.normalize(sh[:, dst], dim=-1)).sum(-1)
                stay = (o * F.normalize(sh[:, src], dim=-1)).sum(-1)
                rows[s].append([float((moved > stay).float().mean()), float(moved.mean()), float(stay.mean())])
    out, brng = {}, np.random.default_rng(1)
    for s, r in rows.items():
        r = np.asarray(r)
        boot = r[brng.integers(0, len(r), (n_boot, len(r)))].mean(1)
        out[s] = {"frac_moved": float(r[:, 0].mean()), "frac_ci": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
                  "cos_moved": float(r[:, 1].mean()), "cos_stay": float(r[:, 2].mean()), "n_demos": len(r)}
    return out


def position_readout(model, dev, data_root, seeds=(0, 1, 2)):
    rng = np.random.default_rng(0)
    feats = {"tr": [], "te": []}
    for hp in suite_hdf5("libero_goal", data_root):
        keys = set(list_demos(hp))
        for d in range(35, 40):
            assert f"demo_{d}" in keys, (hp, d)
            fr, *_ = load_demo(hp, f"demo_{d}")
            t = rng.choice(len(fr), min(20, len(fr)), replace=False)
            x = preprocess_frames(fr[t], 224).to(dev)
            with torch.no_grad():
                tp = model._encode_p_unmasked(model.preprocessing.compute_p_channel(x))[:, 1:].float()
            feats["tr" if d <= 37 else "te"].append(tp.reshape(-1, tp.shape[-1]))
    Xtr, Xte = torch.cat(feats["tr"]), torch.cat(feats["te"])
    ytr = torch.arange(GRID * GRID, device=dev).repeat(len(Xtr) // (GRID * GRID))
    yte = torch.arange(GRID * GRID, device=dev).repeat(len(Xte) // (GRID * GRID))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    res = []
    for s in seeds:
        torch.manual_seed(s)
        lin = torch.nn.Linear(Xtr.shape[1], GRID * GRID).to(dev)
        opt = torch.optim.Adam(lin.parameters(), 1e-2)
        for _ in range(1500):
            j = torch.randint(0, len(Xtr), (4096,), device=dev)
            loss = F.cross_entropy(lin((Xtr[j] - mu) / sd), ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pr = lin((Xte - mu) / sd).argmax(1)
        near = ((pr // GRID - yte // GRID).abs().le(1) & (pr % GRID - yte % GRID).abs().le(1)).float().mean()
        res.append({"seed": s, "top1": float((pr == yte).float().mean()), "near1": float(near)})
    return {"per_seed": res, "top1_mean": float(np.mean([r["top1"] for r in res])),
            "n_test_tokens": int(len(Xte)), "chance": 1 / (GRID * GRID)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="/proj/external_group/mrg/datasets/libero")
    ap.add_argument("--only", choices=["equiv", "pos"], default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_parvo_encoder(a.checkpoint, dev)
    out = {"checkpoint": a.checkpoint, "rope": bool(model.rope)}
    if a.only in (None, "equiv"):
        out["equivariance"] = equivariance(model, dev, a.data_root)
        print("equivariance", json.dumps(out["equivariance"]))
    if a.only in (None, "pos"):
        out["position_readout"] = position_readout(model, dev, a.data_root)
        print("position_readout", json.dumps(out["position_readout"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
