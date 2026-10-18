#!/usr/bin/env python
"""α 선행 판독 4-a · 4-b (docs/alpha_plan.md §4) — Fuse 없이 frozen 토큰을 분석 1 판독기로 직접 읽는다.

한 번 실행 = 한 칸 (P 출처 × 변화 채널 × seed):
  P 출처     ckpt = C1-DN P · dino = DINOv2-base P (observer_fuse.DinoP, 분석 1 v3 DINO 칸과 동일)
  변화 채널  comp = C1-DN M 인코더로 ΔL(앵커 → t) 한 장을 한 번 인코딩 (누적 M · 5프레임 조각 없음)
             raw  = 같은 ΔL 의 16×16 패치 raw 픽셀 (분석 1 m_source=raw 토큰화 그대로, P 차원까지 0 패딩)
             none = 앵커 P 만
  나이 a = 1..20: 앵커 = 5프레임 격자 프레임(분석 1 v4 와 같은 표본), t = 앵커 + a.

4-a (분석 1 판독 프로토콜 그대로 · observer_fuse.fit_probe / v4 lin_probe / rff_expand 재사용):
  타깃 = EE 자세 변화 앵커→t (libero_action_target(앵커, a)) · 공변량 = 앵커 EE 위치 z-score → RFF 128
  beyond(위치 너머 몫) = R²(특징 + 위치) − R²(위치만) · probe 100 ep · 같은 seed 가 probe 초기화·RFF 를 정함
  분석 1 과 다른 점(Fuse 폐지에서 오는 것): 판독기가 z_PP 가 아니라 **각 칸 자기 특징**(train)으로 학습됨 ·
  입력 = [앵커 P 토큰 ‖ 변화 토큰] 196+196, stream 별 query pool (same-probe P⊕M 규약, n_streams=2) · none 은 1 stream.
  토큰은 인코더 출력 그대로(데이터 통계 표준화 없음 · same-probe·TokenProvider 규약).
4-b: 같은 특징(평탄화 · 특징별 train z-score)에서 직전 행동 a_{t−1} (hdf5 actions[t−1], 7d OSC) 릿지(dual) R².
  비교용으로 같은 릿지로 4-a 타깃(움직임) R² 도 함께. α 는 train 뒤 10% 데모(검증)로 선택 후 train 전체 재적합.

판정 ⓐ (결과 전 고정 · alpha_plan §4): DINOv2+M > DINOv2+raw, 나이 평균 위치 너머 몫의 seed 3 범위 비중첩
  → `--summarize` 가 out-root 의 JSON 들로 계산·출력 (ⓑ C1-DN+M vs raw 는 보고).

  python scripts/eval/alpha_readout.py --p-source ckpt --change comp --seed 42 --checkpoint <C1-DN ckpt>
  python scripts/eval/alpha_readout.py --summarize --task-suite libero_object
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from observer_fuse import (  # noqa: E402  분석 1 판독 부품 그대로
    PROJECT_ROOT, DinoP, _suite_split, build_parvo_encoder, compute_metrics, fit_probe,
    libero_action_target, load_demo, score,
)
from observer_fuse_v4 import K_MAX, UNIT, lin_probe  # noqa: E402
from probe_action_libero import rff_expand  # noqa: E402

C1DN_CKPT = ("/proj/external_group/mrg/checkpoints/two_stream_v15b_refine_comp_s_denoise_augtgt/"
             "20260927_235906/checkpoint_epoch0010.pt")


class Tokens:
    """Frozen 토큰 공급: 프레임(uint8, device) · 격자 앵커의 P 토큰 캐시 · 변화 토큰은 (앵커, t) 쌍마다 즉석 인코딩."""

    @torch.no_grad()
    def __init__(self, demos, p_enc, m_enc, change, device, batch=128, view="agentview_rgb"):
        from src.models.common.preprocessing import TwoStreamPreprocessing
        self.p_enc, self.m_enc, self.change, self.device = p_enc, m_enc, change, device
        self.prep = TwoStreamPreprocessing(use_sobel=False).to(device)  # ΔL = BT.709 휘도 차 (CoMP M 입력과 동일)
        F_, ee, ori, act, start, n = [], [], [], [], [], []
        off = 0
        for hp, d, _ in demos:
            frames, ee_pos, ee_ori, actions = load_demo(hp, d, view=view)
            F_.append(torch.from_numpy(frames).to(device))
            ee.append(ee_pos); ori.append(ee_ori); act.append(actions)
            start.append(off); n.append(len(frames)); off += len(frames)
        self.frames = torch.cat(F_)
        self.ee, self.ori, self.act = np.concatenate(ee), np.concatenate(ori), np.concatenate(act)
        self.start, self.n = np.array(start), np.array(n)
        self.demo_of = np.repeat(np.arange(len(n)), n)
        self.end = (self.start + self.n)[self.demo_of]
        self.grid = np.concatenate([np.arange(s, s + m, UNIT) for s, m in zip(self.start, self.n)])
        self.p_row = -np.ones(len(self.frames), dtype=np.int64)
        self.p_row[self.grid] = np.arange(len(self.grid))
        self.P = torch.cat([self.p_tok(torch.from_numpy(self.grid[i:i + batch]).to(device))
                            for i in range(0, len(self.grid), batch)])
        self.dim = self.P.shape[-1]

    def _x(self, idx):  # = preprocess_frames ([0,1] raw, bilinear 224), ImageNet 정규화 없음
        return F.interpolate(self.frames[idx].permute(0, 3, 1, 2).float().div_(255.0), size=(224, 224),
                             mode="bilinear", align_corners=False)

    @torch.no_grad()
    def p_tok(self, idx):
        x = self._x(idx)
        if isinstance(self.p_enc, DinoP):
            return self.p_enc.p_tokens(x).half()
        return self.p_enc._encode_p_unmasked(self.p_enc.preprocessing.compute_p_channel(x))[:, 1:].half()

    @torch.no_grad()
    def change_tok(self, i0, i1):
        """ΔL(i0 → i1) 한 장 → (n, 196, P 차원) fp16. 차원이 P 보다 작으면 0 패딩 (분석 1 raw 규칙)."""
        m = self.prep.compute_m_channel(self._x(i0), self._x(i1))
        if self.change == "raw":
            t = F.unfold(m, kernel_size=16, stride=16).transpose(1, 2)
        else:
            t = self.m_enc._encode_m_unmasked(m)[:, 1:]
        return F.pad(t, (0, self.dim - t.shape[-1])).half()

    def anchors(self, k):
        return self.grid[self.grid + k < self.end[self.grid]]

    @torch.no_grad()
    def features(self, a, k, batch=128):
        """(n, 196 or 392, D) fp16 on device: [앵커 P ‖ 변화(앵커 → 앵커+k)]."""
        out = []
        for i in range(0, len(a), batch):
            at = torch.from_numpy(a[i:i + batch]).to(self.device)
            p = self.P[torch.from_numpy(self.p_row[a[i:i + batch]]).to(self.device)]
            out.append(p if self.change == "none" else torch.cat([p, self.change_tok(at, at + k)], 1))
        return torch.cat(out)

    def motion(self, a, k):
        return torch.from_numpy(np.stack([libero_action_target(self.ee, self.ori, self.act, i, k) for i in a]))

    def prev_action(self, a, k):  # a_{t−1}, t = 앵커 + k
        return torch.from_numpy(self.act[a + k - 1].astype(np.float32))


@torch.no_grad()
def ridge_dual(x_tr, x_ev, targets, val, rel_alphas=tuple(10.0 ** np.arange(-3, 4)), chunk=16384):
    """평탄화 특징의 릿지 (dual: 표본 수 ≪ 특징 수). 특징별 train 평균 제거 + stream(앵커 P / 변화)별 스칼라 하나로
    분산 맞춤 — 특징별 z-score 는 sim ΔL 의 거의-상수 픽셀(정확히 0 이 77–85%)을 eval 에서 폭발시켜 쓰지 않음.
    타깃은 train 평균 제거. targets = {name: (y_tr, y_ev)}. 타깃마다 α = rel × mean diag(K) 를 검증(val 표본)으로
    고른 뒤 train 전체로 재적합."""
    xt, xe = x_tr.reshape(len(x_tr), -1), x_ev.reshape(len(x_ev), -1)
    dev, per_stream = xt.device, 196 * x_tr.shape[2]  # stream = 196 패치 토큰
    mu = torch.cat([xt[:, j:j + chunk].float().mean(0) for j in range(0, xt.shape[1], chunk)])
    var = torch.cat([xt[:, j:j + chunk].float().var(0) for j in range(0, xt.shape[1], chunk)])
    s_scale = var.view(-1, per_stream).mean(1).clamp_min(1e-12).sqrt()        # (n_streams,)
    inv = (1.0 / s_scale).repeat_interleave(per_stream)
    K = torch.zeros(len(xt), len(xt), device=dev, dtype=torch.float32)
    Ke = torch.zeros(len(xe), len(xt), device=dev, dtype=torch.float32)
    for j in range(0, xt.shape[1], chunk):
        a = (xt[:, j:j + chunk].float() - mu[j:j + chunk]) * inv[j:j + chunk]
        b = (xe[:, j:j + chunk].float() - mu[j:j + chunk]) * inv[j:j + chunk]
        K += a @ a.T; Ke += b @ a.T
    scale = float(K.diagonal().mean())
    vt = torch.from_numpy(val).to(dev); f = ~vt
    eig_f, eig_all = torch.linalg.eigh(K[f][:, f].double()), torch.linalg.eigh(K.double())

    def coef(eig, y, alpha):
        lam, U = eig
        return (U @ ((U.T @ y.double()) / (lam + alpha)[:, None])).float()

    out = {}
    for name, (y_tr, y_ev) in targets.items():
        y = y_tr.to(dev).float(); ym = y.mean(0); y = y - ym
        val_r2 = {}
        for r in rel_alphas:
            c = coef(eig_f, y[f], r * scale)
            val_r2[float(r)] = compute_metrics((K[vt][:, f] @ c + ym).cpu().numpy(),
                                               y_tr[torch.from_numpy(val)].numpy())["r2_aggregate"]
        best = max(val_r2, key=val_r2.get)
        m = compute_metrics((Ke @ coef(eig_all, y, best * scale) + ym).cpu().numpy(), y_ev.numpy())
        out[name] = {"r2": m["r2_aggregate"], "r2_per_dim": m["r2_per_dim"], "alpha_rel": best, "val_r2": val_r2}
    return out


def evaluate(tr, ev, args, device):
    ns = 1 if args.change == "none" else 2
    n_val = max(1, len(tr.n) // 10)  # train 뒤 10% 데모 = 릿지 α 검증 (v4 검증 규약)
    out = {}
    for k in args.ages:
        t0 = time.time()
        a_tr, a_ev = tr.anchors(k), ev.anchors(k)
        y_tr, y_ev = tr.motion(a_tr, k), ev.motion(a_ev, k)
        p_tr, p_ev = torch.from_numpy(tr.ee[a_tr]).float(), torch.from_numpy(ev.ee[a_ev]).float()
        mu, sd = p_tr.mean(0), p_tr.std(0) + 1e-6
        r_tr, r_ev = rff_expand((p_tr - mu) / sd, args.seed + 2), rff_expand((p_ev - mu) / sd, args.seed + 2)
        pos_only = lin_probe(r_tr, y_tr, r_ev, y_ev, args, device)
        x_tr, x_ev = tr.features(a_tr, k), ev.features(a_ev, k)
        pr = fit_probe(x_tr, y_tr, "reg", 7, args, device, extra=r_tr, n_streams=ns)
        concat = score(pr, x_ev, y_ev, "reg", device, extra=r_ev)
        pr0 = fit_probe(x_tr, y_tr, "reg", 7, args, device, n_streams=ns)
        plain = score(pr0, x_ev, y_ev, "reg", device)
        val = tr.demo_of[a_tr] >= len(tr.n) - n_val
        ridge = ridge_dual(x_tr, x_ev, {"prev_action": (tr.prev_action(a_tr, k), ev.prev_action(a_ev, k)),
                                        "motion": (y_tr, y_ev)}, val)
        out[k] = {"n_train": len(a_tr), "n_eval": len(a_ev), "pos_only": pos_only, "concat": concat,
                  "beyond": concat - pos_only, "plain": plain, "ridge": ridge, "seconds": round(time.time() - t0, 1)}
        print(f"  age {k:2d}  pos {pos_only:.3f}  beyond {concat - pos_only:+.3f}  plain {plain:.3f}  "
              f"ridge a_(t-1) {ridge['prev_action']['r2']:.3f}  ridge motion {ridge['motion']['r2']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        del x_tr, x_ev
    return out


def summarize(args):
    """out-root JSON (같은 suite) → 칸별 나이 평균 위치 너머 몫(seed 범위) + 판정 ⓐ · ⓑ."""
    cells = {}
    for f in sorted(Path(args.out_root).glob("*.json")):
        d = json.load(open(f))
        a = d.get("args", {})
        if a.get("task_suite") != args.task_suite or a.get("max_demos"):
            continue
        view = "wrist" if a.get("view") == "eye_in_hand_rgb" else "front"   # 10-17 이전 JSON = 앞 카메라
        cells.setdefault((view, a["p_source"], a["change"]), {})[a["seed"]] = d["age_mean"]
    rows = {}
    print(f"[{args.task_suite}] 나이 평균 (seed 별)  beyond · ridge a_(t-1) · ridge motion")
    for (view, p, c), by_seed in sorted(cells.items()):
        b = [v["beyond"] for v in by_seed.values()]
        rows[(view, p, c)] = (min(b), max(b), len(b))
        print(f"  {view:5s} {p:5s} + {c:4s}  seeds {sorted(by_seed)}  beyond {np.mean(b):+.3f} [{min(b):+.3f}, {max(b):+.3f}]  "
              f"a_(t-1) {np.mean([v['ridge_prev_action'] for v in by_seed.values()]):.3f}  "
              f"motion {np.mean([v['ridge_motion'] for v in by_seed.values()]):.3f}")

    def verdict(p, view="front"):
        m, r = rows.get((view, p, "comp")), rows.get((view, p, "raw"))
        if not m or not r or m[2] < 3 or r[2] < 3:
            return "seed 3 미완 — 판정 불가"
        return "PASS (M > raw, 범위 비중첩)" if m[0] > r[1] else "FAIL (겹침 또는 역전)"
    print(f"판정 ⓐ DINOv2+M > DINOv2+raw: {verdict('dino')}")
    print(f"보고 ⓑ C1-DN+M > C1-DN+raw: {verdict('ckpt')}")
    print(f"보고 (손목) C1-DN+M > C1-DN+raw: {verdict('ckpt', 'wrist')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-source", default="ckpt", choices=["ckpt", "dino"])
    ap.add_argument("--change", default="comp", choices=["comp", "raw", "none"])
    ap.add_argument("--checkpoint", default=C1DN_CKPT, help="C1-DN (P 출처 ckpt 및 comp M 인코더)")
    ap.add_argument("--tag", default=None, help="기본 = <suite>_<p>_<change>_s<seed>")
    ap.add_argument("--task-suite", default="libero_object")
    ap.add_argument("--view", default="agentview_rgb", choices=["agentview_rgb", "eye_in_hand_rgb"],
                    help="카메라 (alpha_plan §4-a 손목 칸 = eye_in_hand_rgb · 10-17 (2))")
    ap.add_argument("--data-root", default="/proj/external_group/mrg/datasets/libero")
    ap.add_argument("--ages", type=int, nargs="+", default=list(range(1, K_MAX + 1)))
    ap.add_argument("--probe-epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42, help="분석 1 seed = 42 · 1 · 2")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--max-demos", type=int, default=None, help="스모크: split 별 데모 수 상한")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "paper_artifacts" / "alpha_readout"))
    args = ap.parse_args()
    if args.summarize:
        return summarize(args)
    args.tag = args.tag or (f"{args.task_suite}_{args.p_source}_{args.change}_s{args.seed}"
                            + ("_wrist" if args.view == "eye_in_hand_rgb" else ""))
    device = torch.device(args.device)
    t0 = time.time()

    tr_d, ev_d = _suite_split(args.task_suite, args.data_root, args.split_seed, 0.8, 99.0)
    if args.max_demos:
        tr_d, ev_d = tr_d[:args.max_demos], ev_d[:max(2, args.max_demos // 4)]
    need_ckpt = args.p_source == "ckpt" or args.change == "comp"
    enc = build_parvo_encoder(args.checkpoint, device) if need_ckpt else None
    p_enc = DinoP(device) if args.p_source == "dino" else enc
    tr = Tokens(tr_d, p_enc, enc, args.change, device, view=args.view)
    ev = Tokens(ev_d, p_enc, enc, args.change, device, view=args.view)
    print(f"[{args.tag}] train frames {len(tr.frames)} grid {len(tr.grid)} / eval frames {len(ev.frames)} "
          f"grid {len(ev.grid)} · P dim {tr.dim}  ({time.time() - t0:.0f}s)", flush=True)

    curve = evaluate(tr, ev, args, device)
    mean = lambda f: float(np.mean([f(v) for v in curve.values()]))
    res = {"args": vars(args),
           "meta": {"plan": "docs/alpha_plan.md §4-a/4-b", "p_encoder": "facebook/dinov2-base" if args.p_source == "dino" else args.checkpoint,
                    "m_encoder": args.checkpoint if args.change == "comp" else None,
                    "change_tokenization": {"comp": "C1-DN M encoder on ΔL(anchor→t), one encoding",
                                            "raw": "ΔL(anchor→t) 16×16 patches, zero-padded to P dim",
                                            "none": "anchor P only"}[args.change],
                    "anchor_grid_frames": UNIT, "n_train_demos": len(tr_d), "n_eval_demos": len(ev_d),
                    "readout": "observer_fuse.fit_probe (attentive, n_streams=%d) + RFF(anchor ee_pos) · v4 lin_probe pos-only"
                               % (1 if args.change == "none" else 2)},
           "curve": curve,
           "age_mean": {"beyond": mean(lambda v: v["beyond"]), "plain": mean(lambda v: v["plain"]),
                        "ridge_prev_action": mean(lambda v: v["ridge"]["prev_action"]["r2"]),
                        "ridge_motion": mean(lambda v: v["ridge"]["motion"]["r2"])},
           "elapsed_s": round(time.time() - t0, 1)}
    print("age mean:", json.dumps(res["age_mean"]), flush=True)
    os.makedirs(args.out_root, exist_ok=True)
    with open(f"{args.out_root}/{args.tag}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
