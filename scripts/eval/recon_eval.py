"""RoPE §8 판 선택 단계의 주 자 — 학습 목표 항별 복원 오차 (docs/rope_encoder_plan.md §8-3 · §8-4). 판독기 없음 · 학습 0.

항 (표본별 저장 · 모든 판 · 모든 조건에서 같은 표본 · 같은 마스크):
  Lt · Ltk   P 제자리 복원   = _predict_pixels(P_vis, mask, null routing, 자기 프레임)      [보고]
  Lpred      P 예측 복원     = _predict_pixels(P_t vis, mask_t, M(ΔL(t,tk)), 뒤 프레임)
  Lnull      같은 입력 · 같은 마스크에서 M 만 null routing 으로 바꾼 출력을 뒤 프레임과 비교
  → **M 의 도움 몫 = Lnull − Lpred** (클수록 좋음)                                          [판정]
  copy       앞 프레임을 그대로 뒤 프레임과 비교(가려진 패치) — 예측 난이도의 척도           [보고]
  mB         M 실제 변화 복원 = _recon_dL(ΔL(t,tk), P_t full) · 학습 가중(floor + |ΔL|)      [판정]
  mA         M 변화 없음     = _recon_dL(ΔL(t,t) = 0, P_t full) → 정답 0                      [보고]
P 항 = 가려진 패치의 픽셀 MSE(학습 손실의 MSE 성분 · SSIM 0.02 항은 그림 전체 평균이라 표본·패치 단위 짝 비교에서 제외).
M 항 = 학습 그대로의 가중 합 비(num/den 을 표본별로 저장 → 합의 비) + 가중 없는 MSE + 움직인 패치(|ΔL| 패치 평균 > 1/255)만.
model.eval() → 회전 · 밝기 · 잡음 증강 꺼짐(M 타깃 = 입력 ΔL). autocast bf16 = 학습 중 evaluate() 와 같음.

조건 (전부 격자 14×14):
  egodex: base(256² 원본의 가운데 224 창 · 두 프레임 같은 창) · indep(프레임마다 독립 무작위 창 = 학습·학습 중 평가의 방식)
          · win8/16(창을 오른쪽·아래로 k px · 패딩 없음) · shift8/16/32(base 를 k px 이동 · 반사 패딩 = §3 'shift')
          · zoom1.25/1.5/2.0(base 의 가운데 224/s 영역을 224 로 bilinear 확대)
  libero: object · goal · spatial 앞 카메라 · 과제 10 × 시연 40~43 × 프레임 5 · 간격 10(0.5 s)
  calvin: validation(D) 세그먼트 300 × 쌍 2 · 간격 15(0.5 s)

--check: indep 조건에서 학습 forward(model(img_t, img_tk))에 같은 마스크를 넣어 loss_t/tk/pred(= MSE + 0.02·SSIM) · caseA · caseB 가
         이 스크립트의 항과 일치하는지 확인하고, 합계를 학습 로그의 [Eval] 값과 비교할 수 있게 출력한다.

사용: [ROPE_THETA=…] python scripts/eval/recon_eval.py --checkpoint <ckpt> --out <prefix> [--sets egodex libero calvin] [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.eval.probe_action_libero import build_parvo_encoder, list_demos, load_demo, preprocess_frames  # noqa: E402
from scripts.eval.rope_position import suite_hdf5  # noqa: E402

D = "/proj/external_group/mrg/datasets"
MOVE_THR = 1.0 / 255.0


# ── 데이터 (전부 고정 seed · 판과 무관) ─────────────────────────────────────────
def load_egodex(n: int, root=f"{D}/egodex/frames/test"):
    """test 분할에서 영상 n 개(rng 0) × 쌍 1 · 간격 = 학습 분포(triangular 1..30 · 중심 15). → raw [n,2,3,256,256] uint8."""
    vids = sorted(v for t in sorted(Path(root).glob("*")) if t.is_dir() for v in sorted(t.glob("*")) if v.is_dir())
    rng = np.random.default_rng(0)
    vids = [vids[i] for i in np.sort(rng.choice(len(vids), n, replace=False))]
    gaps = np.arange(1, 31)
    pr = np.maximum(1.0 - np.abs(gaps - 15) / 15, 0.01); pr /= pr.sum()

    def one(args):
        v, g, u = args
        nf = len(list(v.glob("frame_*.jpg")))
        if nf - g - 1 <= 0:
            g = 1
        t = int(u * (max(0, nf - g - 1) + 1))
        return np.stack([np.asarray(Image.open(v / f"frame_{i:06d}.jpg").convert("RGB")) for i in (t, t + g)]), g

    jobs = [(v, int(rng.choice(gaps, p=pr)), float(rng.random())) for v in vids]
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, jobs))
    raw = torch.from_numpy(np.stack([r[0] for r in res])).permute(0, 1, 4, 2, 3).contiguous()
    return raw, np.arange(n), np.array([r[1] for r in res])


def load_libero(gap=10, suites=("libero_object", "libero_goal", "libero_spatial"), demos=range(40, 44), per_demo=5):
    rng = np.random.default_rng(0)
    a, b, grp = [], [], []
    for s in suites:
        for hp in suite_hdf5(s, f"{D}/libero"):
            for d in demos:
                fr, *_ = load_demo(hp, f"demo_{d}")
                t = np.sort(rng.choice(len(fr) - gap, per_demo, replace=False))
                a.append(preprocess_frames(fr[t], 224)); b.append(preprocess_frames(fr[t + gap], 224))
                grp += [len(grp) // per_demo] * per_demo
    return torch.cat(a), torch.cat(b), np.array(grp)


def load_calvin(gap=15, n_seg=300, per_seg=2):
    from src.datasets.calvin import load_segments, load_segment_frames
    sd = Path(f"{D}/calvin/task_ABCD_D/validation")
    segs = load_segments(sd)
    rng = np.random.default_rng(0)
    segs = [segs[i] for i in np.sort(rng.choice(len(segs), n_seg, replace=False))]
    a, b, grp = [], [], []
    for gi, (s0, s1, _) in enumerate(segs):
        fr = load_segment_frames(sd, s0, s1)[0]
        if len(fr) - gap < per_seg:
            continue
        t = np.sort(rng.choice(len(fr) - gap, per_seg, replace=False))
        a.append(preprocess_frames(fr[t], 224)); b.append(preprocess_frames(fr[t + gap], 224))
        grp += [gi] * per_seg
    return torch.cat(a), torch.cat(b), np.array(grp)


def egodex_conditions(raw: torch.Tensor):
    """raw [n,2,3,256,256] uint8 → {조건: (prev, curr)} 를 하나씩 내놓는 생성기 (메모리: 조건 하나씩)."""
    n = len(raw)
    f = lambda x: x.float().div(255.0)
    win = lambda o: (f(raw[:, 0, :, o:o + 224, o:o + 224]), f(raw[:, 1, :, o:o + 224, o:o + 224]))
    yield "base", win(16)
    rng = np.random.default_rng(1)
    off = rng.integers(0, 33, (n, 2, 2))   # 표본 · 프레임 · (y, x)
    crop = lambda j: torch.stack([f(raw[i, j, :, off[i, j, 0]:off[i, j, 0] + 224, off[i, j, 1]:off[i, j, 1] + 224]) for i in range(n)])
    yield "indep", (crop(0), crop(1))
    for k in (8, 16):
        yield f"win{k}", win(16 + k)
    base = win(16)
    for k in (8, 16, 32):
        yield f"shift{k}", tuple(F.pad(x, (k, 0, k, 0), mode="reflect")[..., :224, :224] for x in base)
    for s in (1.25, 1.5, 2.0):
        w = int(round(224 / s)); o = (224 - w) // 2
        yield f"zoom{s:g}", tuple(F.interpolate(x[..., o:o + w, o:o + w], size=(224, 224), mode="bilinear",
                                                align_corners=False) for x in base)


# ── 측정 ───────────────────────────────────────────────────────────────────────
def make_masks(n: int, num_patches: int, seed: int):
    """표본별 고정 마스크 4 벌 (P_t · P_tk = 0.75 / M Case B · A = 0.5). model._random_mask 와 같은 구성(무작위 순열의 앞쪽이 보임)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, ratio in (("t", 0.75), ("tk", 0.75), ("mB", 0.5), ("mA", 0.5)):
        ids = torch.argsort(torch.rand(n, num_patches, generator=g), dim=1)
        m = torch.ones(n, num_patches, dtype=torch.bool)
        m.scatter_(1, ids[:, :num_patches - int(ratio * num_patches)], False)
        out[name] = m
    return out


@torch.no_grad()
def measure(model, prev, curr, masks, dev, batch=64, check=False):
    e = model
    assert (e.mask_ratio_p, e.mask_ratio_m_recon) == (0.75, 0.5)
    keys = ("Lt", "Ltk", "Lpred", "Lnull", "copy", "mB_num", "mB_den", "mB_mse", "mB_mov_sum", "mB_mov_cnt",
            "mA_num", "mA_den", "mA_mse")
    out = {k: [] for k in keys}
    chk = []
    orig_mask = e._random_mask
    for i in range(0, len(prev), batch):
        xt, xk = prev[i:i + batch].to(dev), curr[i:i + batch].to(dev)
        mk = {k: v[i:i + batch].to(dev) for k, v in masks.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pt, pk = e.preprocessing.compute_p_channel(xt), e.preprocessing.compute_p_channel(xk)
            dl_real = e.preprocessing.compute_m_channel(xt, xk)
            dl_null = e.preprocessing.compute_m_channel(xt, xt)
            vt, vk = e._student_p_encode_visible(pt, mk["t"]), e._student_p_encode_visible(pk, mk["tk"])
            m_real, m_null = e._encode_m_unmasked(dl_real), e._null_routing(len(xt), dev)

            def p_err(vis, mask, routing, tgt):   # 가려진 패치의 픽셀 MSE (표본별)
                _, pp = e._predict_pixels(vis, mask, routing, tgt)
                err = ((pp.float() - e._patchify(tgt)) ** 2).mean(-1)
                return (err * mask).sum(1) / mask.sum(1)

            out["Lt"].append(p_err(vt, mk["t"], m_null, xt))
            out["Ltk"].append(p_err(vk, mk["tk"], m_null, xk))
            out["Lpred"].append(p_err(vt, mk["t"], m_real, xk))
            out["Lnull"].append(p_err(vt, mk["t"], m_null, xk))
            cp = ((e._patchify(xt) - e._patchify(xk)) ** 2).mean(-1)
            out["copy"].append((cp * mk["t"]).sum(1) / mk["t"].sum(1))

            helper = e._encode_p_unmasked(pt)
            for tag, dl in (("mB", dl_real), ("mA", dl_null)):
                e._random_mask = lambda B, d, r, _m=mk[tag]: _m     # _recon_dL 을 학습 코드 그대로 쓰되 마스크만 고정
                _, pp, mm = e._recon_dL(dl, helper, dev)
                e._random_mask = orig_mask
                tgt = e._patchify(dl)
                err = ((pp.float() - tgt) ** 2).mean(-1)
                amp = tgt.abs().mean(-1)
                w = (e.m_recon_weight_floor + e.m_recon_weight_scale * amp) * mm
                out[f"{tag}_num"].append((err * w).sum(1)); out[f"{tag}_den"].append(w.sum(1))
                out[f"{tag}_mse"].append((err * mm).sum(1) / mm.sum(1))
                if tag == "mB":
                    mv = mm & (amp > MOVE_THR)
                    out["mB_mov_sum"].append((err * mv).sum(1)); out["mB_mov_cnt"].append(mv.sum(1).float())

            if check:   # 학습 forward 에 같은 마스크를 호출 순서대로(P_t · P_tk · Case B · Case A) 넣어 항별 일치 확인
                q = [mk["t"], mk["tk"], mk["mB"], mk["mA"]]
                e._random_mask = lambda B, d, r: q.pop(0)
                o = e(xt, xk)
                e._random_mask = orig_mask
                assert not q
                chk.append([float(o[k]) for k in ("loss", "loss_t", "loss_tk", "loss_pred", "loss_m_caseA", "loss_m_caseB")] + [len(xt)])
    res = {k: torch.cat(v).float().cpu().numpy() for k, v in out.items()}
    if check:
        c = np.asarray(chk); w = c[:, -1] / c[:, -1].sum()
        res["_check"] = {k: float((c[:, j] * w).sum()) for j, k in enumerate(("loss", "loss_t", "loss_tk", "loss_pred", "caseA", "caseB"))}
    return res


def summarize(r):
    return {"Lt": r["Lt"].mean(), "Ltk": r["Ltk"].mean(), "Lpred": r["Lpred"].mean(), "Lnull": r["Lnull"].mean(),
            "copy": r["copy"].mean(), "help": (r["Lnull"] - r["Lpred"]).mean(),
            "mB": r["mB_num"].sum() / r["mB_den"].sum(), "mB_mse": r["mB_mse"].mean(),
            "mB_mov": r["mB_mov_sum"].sum() / max(r["mB_mov_cnt"].sum(), 1.0),
            "mA": r["mA_num"].sum() / r["mA_den"].sum(), "n": len(r["Lt"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="출력 접두사 → <out>.npz(표본별) · <out>.json(요약)")
    ap.add_argument("--sets", nargs="+", default=["egodex", "libero", "calvin"])
    ap.add_argument("--n-egodex", type=int, default=2000)
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument("--m-recon-floor", type=float, default=0.02, help="학습 인자 --v15-m-recon-floor (C1 계열 전부 0.02)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_parvo_encoder(a.checkpoint, dev)
    # build_parvo_encoder 는 인코더 가중치만 확인한다 → 복원에 쓰는 디코더 · routing · 헤드까지 빠짐없이 실렸는지 다시 확인
    sd = {k.replace("module.", ""): v for k, v in torch.load(a.checkpoint, map_location="cpu")["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"state_dict 불일치 missing={missing[:5]} unexpected={unexpected[:5]}"
    model.m_recon_weight_floor = a.m_recon_floor
    assert model.m_recon_weight_scale == 1.0 and model.caseA_weight == 1.0 and model.lambda_ssim == 0.02
    model.eval()

    arrays, summary = {}, {"checkpoint": a.checkpoint, "rope": bool(model.rope), "mask_seed": a.mask_seed,
                           "rope_theta": float(model.rope_theta) if model.rope else None, "conditions": {}}

    def run(name, prev, curr, masks, grp, check=False):
        r = measure(model, prev, curr, masks, dev, check=check)
        c = r.pop("_check", None)
        s = {k: float(v) for k, v in summarize(r).items()}
        if c:
            s["train_forward"] = c
        summary["conditions"][name] = s
        for k, v in r.items():
            arrays[f"{name}/{k}"] = v
        arrays[f"{name}/group"] = grp
        print(name, json.dumps(s), flush=True)

    if "egodex" in a.sets:
        raw, grp, gaps = load_egodex(a.n_egodex)
        arrays["egodex_gap"] = gaps
        masks = make_masks(len(raw), model.num_patches, a.mask_seed)
        for name, (p, c) in egodex_conditions(raw):
            run(name, p, c, masks, grp, check=a.check and name == "indep")
        del raw
    for s, loader in (("libero", load_libero), ("calvin", load_calvin)):
        if s in a.sets:
            p, c, grp = loader()
            run(s, p, c, make_masks(len(p), model.num_patches, a.mask_seed), grp)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out + ".npz", **arrays)
    json.dump(summary, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
