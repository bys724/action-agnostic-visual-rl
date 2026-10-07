"""Region-wise M-recon patch error for CoMP (code v16) on LIBERO — arm / target object / ... / background.

Why (STATUS 10-06 Vault decision): the M-recon loss weights each patch by floor + scale·mean|ΔL|.
Combined with absolute MSE (∝ |ΔL|²) the hypothesis is that small-change patches (the manipulated
object) carry ~1/1000 of the arm's loss influence, so M learns "arm moved" but smears
"object did not move". This tool measures where M-recon error actually lands, per region.

Pipeline (per frame pair, gap g sim steps = g/20 s at LIBERO 20 Hz):
  1. Re-render agentview RGB **and** segmentation from the demo `states` in the SAME process,
     after restoring the demo's fixture placement (`apply_demo_fixtures`, copied from
     source-field-alternation). Never pairs masks with hdf5 RGB (render is non-deterministic
     across processes because reset re-places fixtures; restore makes it bit-identical).
  2. Model input = [0,1] RGB, bilinear 128→224 (same as probe_action_libero.preprocess_frames,
     no ImageNet norm). ΔL = model.preprocessing.compute_m_channel(frame_t, frame_tk) (1ch).
  3. M-recon replicated from TwoStreamV15Model._recon_dL (Case B) with a GIVEN mask instead of
     the internal random one: masked M pass → mask tokens + dec APE → m_recon_decoder with
     P-helper = P full pass of frame_t → norm → head. Model code is not modified.
  4. Masks: `--rounds` R partitions. Round r draws a random 50 % mask (ratio = training default
     mask_ratio_m_recon 0.5) and also runs its complement → every patch is predicted exactly once
     per round under a training-like mask; per-patch error = mean over rounds. Mask RNG seed =
     seed·1_000_003 + pair index → identical masks for every ckpt on the same sample.

Definitions (per patch p, 16×16 px on the 224 grid, 1ch ΔL target y, prediction ŷ):
  mse_p     = mean_pixels (ŷ − y)²                     (= the training err before weighting)
  energy_p  = mean_pixels y²                           (MSE of the all-zero predictor)
  absdl_p   = mean_pixels |y|
  group abs error   = mean_p mse_p
  group rel error   = Σ mse_p / Σ energy_p             (<1 = better than predicting zero)
  loss share (wS)   = Σ_group w_p·mse_p / Σ_all w_p·mse_p with w_p = 0.02 + S·absdl_p (S=1 train, S=0)
  state "moved"     = absdl_p ≥ DL_THR (fixed before any C0 number was seen, see DL_THR below)
  state "sim_moved" = the patch's region bodies moved > 1 mm between the two states (secondary)

Region (per pixel, union over both frames with priority arm > target > receptacle > other > bg;
patch label = majority pixel label):
  arm        = bodies robot0_* / mount0_* / gripper0_* (gripper merged into arm)
  target     = first BDDL `:obj_of_interest` (the manipulated object, e.g. alphabet_soup_1)
  receptacle = remaining obj_of_interest (e.g. basket_1)
  other      = other non-robot bodies with a joint on themselves or an ancestor (free objects,
               drawers, doors)
  background = everything else (floor, table, walls, fixed fixture bodies, non-geom pixels)
  near_arm   = patch with any arm pixel in its 3×3 patch neighbourhood (subgroup of static target)

P side (added 2026-10-07, observation only — no pre-registered criterion), same pairs / regions /
patch states (state = |ΔL| of the pair, as above), `branch` column in the outputs:
  p_recon = L_t path of _forward_pair_comp: P encoder on visible patches of frame_t → mask tokens
            + dec APE → p_motion_decoder with the learned null routing (_null_routing) → recon_head;
            target = raw [0,1] RGB patches of frame_t (no per-patch normalisation, as in training).
  p_pred  = L_pred path: same visible P_t and mask, routing = M encoder full pass of ΔL(t,tk);
            target = RGB patches of frame t+gap.
  Masks: ratio = training mask_ratio_p 0.75. Round r splits the N patches into 1/(1−ratio) = 4
  disjoint visible sets; mask k hides everything except set k (147/196 masked, same count as
  _random_mask) → each patch is masked (and scored) 3× per round; per-patch error = mean over all
  masks covering it. Separate RNG stream from the M masks (M results untouched).
  energy (denominator of rel_err) per branch:
    m_recon: mean y² (all-zero predictor, unchanged)
    p_recon: mean (y − patch mean of y)² (error of predicting each patch's mean colour)
    p_pred : mean (x_{t+g} − x_t)² (error of the copy-current-frame predictor; 0 on truly static
             patches → rel_err there is undefined/large, read abs_mse instead)
  loss_share_w0 = share of the unweighted masked MSE (P has no |ΔL| weight → w1 left blank).
  Replication check (first batch): run the model's own _forward_pair_comp (eval mode, masks
  captured) and recompute L_t / L_pred with this code's path (masked MSE + λ_ssim·SSIM).

Output: paper_artifacts/tables/mrecon_region/<tag>.{json,csv} + verify/ overlay PNGs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "eval"))
from probe_action_libero import build_parvo_encoder, preprocess_frames  # noqa: E402

DATA_ROOT = Path("/proj/external_group/mrg/datasets/libero")
OUT_DIR = PROJECT_ROOT / "paper_artifacts" / "tables" / "mrecon_region"
CAM, RES, IMG = "agentview", 128, 224
MJ_GEOM = 5  # mjOBJ_GEOM — segmentation channel 0 value for geom pixels
REGIONS = ["arm", "target", "receptacle", "other", "background"]
# Moved/static threshold on patch mean|ΔL| — FIXED 2026-10-06 before any C0 number was seen.
# 1/255 = one 8-bit intensity level: sim renders are deterministic, so a truly static patch has
# ΔL exactly 0 (sim exact-zero 77–85 %, STATUS R2-4); anything ≥ one quantisation level is a real
# pixel change (motion, occlusion, shadow). Do not change after seeing results.
DL_THR = 1.0 / 255.0
SIM_MOVE_THR = 1e-3  # m — same as source-field-alternation build_segmasks_6
W_FLOOR = 0.02       # guard-7 floor used by C0 and the noscale run (C0 log: --v15-m-recon-floor 0.02)


# ── LIBERO env helpers (copied minimal from source-field-alternation/scripts/rollout_m2d_stage2.py) ──
def make_env(suite: str, task_stem: str):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    name = task_stem[:-5] if task_stem.endswith("_demo") else task_stem
    bddl = Path(get_libero_path("bddl_files")) / suite / f"{name}.bddl"
    assert bddl.exists(), f"bddl missing: {bddl}"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES,
                             camera_widths=RES, hard_reset=False)
    return env, bddl


def apply_demo_fixtures(sim, model_xml: str) -> None:
    """Restore the demo's fixture root pos/quat (reset re-places them by 1–2 cm and they are not
    in the state vector → set_init_state alone does not restore them)."""
    m = sim.model
    for bd in ET.fromstring(model_xml).find("worldbody").findall("body"):
        try:
            bid = m.body_name2id(bd.get("name"))
        except Exception:
            continue
        if bd.get("pos"):
            m.body_pos[bid] = np.array(bd.get("pos").split(), float)
        if bd.get("quat"):
            m.body_quat[bid] = np.array(bd.get("quat").split(), float)
    sim.forward()


def objs_of_interest(bddl: Path) -> list[str]:
    txt = bddl.read_text()
    i = txt.index("(:obj_of_interest")
    return txt[i + len("(:obj_of_interest"): txt.index(")", i)].split()


def body_regions(sim, ooi: list[str]) -> np.ndarray:
    """body id → region index (REGIONS)."""
    m = sim.model
    hasj = np.zeros(m.nbody, bool)
    for j in range(m.njnt):
        hasj[int(m.jnt_bodyid[j])] = True
    reg = np.full(m.nbody, REGIONS.index("background"), np.int64)
    for bid in range(1, m.nbody):
        chain, b = [], bid
        while b > 0:
            chain.append(m.body_id2name(b) or "")
            b = int(m.body_parentid[b])
        nm = chain[0]
        if nm.startswith(("robot0", "mount0", "gripper0")):
            reg[bid] = REGIONS.index("arm")
        elif any(n == ooi[0] or n.startswith(ooi[0] + "_") for n in chain):
            reg[bid] = REGIONS.index("target")
        elif any(n == o or n.startswith(o + "_") for o in ooi[1:] for n in chain):
            reg[bid] = REGIONS.index("receptacle")
        else:
            b, mov = bid, False
            while b > 0:
                if hasj[b]:
                    mov = True
                    break
                b = int(m.body_parentid[b])
            reg[bid] = REGIONS.index("other") if mov else REGIONS.index("background")
    return reg


def render_state(env, st):
    """RGB (obs-aligned, uint8 128²) + body-id map (−1 = non-geom) + body xpos, one sim state."""
    from robosuite.utils.camera_utils import get_camera_segmentation
    obs = env.set_init_state(st)
    sim = env.sim
    rgb = np.asarray(obs[f"{CAM}_image"], np.uint8).copy()
    seg = get_camera_segmentation(sim, CAM, RES, RES)[::-1]   # [::-1] = obs alignment (SFA 09-23)
    ok = seg[..., 0] == MJ_GEOM
    gid = np.clip(seg[..., 1], 0, sim.model.ngeom - 1)
    body = np.where(ok, np.asarray(sim.model.geom_bodyid)[gid], -1)
    return rgb, body, np.array(sim.data.xpos, copy=True)


# ── per-pair patch labels ─────────────────────────────────────────────────────────────────────
def patch_labels(body_a, body_b, reg_of_body, disp):
    """Per-patch label / purity / near_arm / sim_moved on the 14×14 grid of the 224 input."""
    def reg_map(body):
        r = np.full(body.shape, REGIONS.index("background"))
        r[body >= 0] = reg_of_body[body[body >= 0]]
        return r
    ra, rb = reg_map(body_a), reg_map(body_b)
    pix = np.minimum(ra, rb)                       # priority = lower index (arm > target > ...)
    d_a = np.where(body_a >= 0, disp[np.clip(body_a, 0, None)], 0.0)
    d_b = np.where(body_b >= 0, disp[np.clip(body_b, 0, None)], 0.0)
    pdisp = np.maximum(d_a, d_b)
    up = lambda x: F.interpolate(torch.from_numpy(x[None, None].astype(np.float32)),
                                 size=(IMG, IMG), mode="nearest")[0, 0].numpy()
    pix, pdisp = up(pix).astype(np.int64), up(pdisp)
    g = IMG // 16
    pix_p = pix.reshape(g, 16, g, 16).transpose(0, 2, 1, 3).reshape(g * g, 256)
    dsp_p = pdisp.reshape(g, 16, g, 16).transpose(0, 2, 1, 3).reshape(g * g, 256)
    counts = np.stack([(pix_p == k).sum(1) for k in range(len(REGIONS))], 1)
    label = counts.argmax(1)
    purity = counts.max(1) / 256.0
    sim_moved = np.array([dsp_p[i][pix_p[i] == label[i]].max() > SIM_MOVE_THR for i in range(g * g)])
    arm_any = (counts[:, 0] > 0).reshape(g, g)
    near = np.zeros_like(arm_any)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near |= np.roll(np.roll(arm_any, dy, 0), dx, 1) & _valid_shift(g, dy, dx)
    return label, purity, near.reshape(-1), sim_moved


def _valid_shift(g, dy, dx):
    v = np.ones((g, g), bool)                      # cancel np.roll wrap-around
    if dy == 1: v[0, :] = False
    if dy == -1: v[-1, :] = False
    if dx == 1: v[:, 0] = False
    if dx == -1: v[:, -1] = False
    return v


# ── M-recon with a given mask (replica of TwoStreamV15Model._recon_dL, Case B) ─────────────────
@torch.no_grad()
def mrecon_err(model, x_t, x_tk, masks):
    """x_*: [B,3,224,224] [0,1]; masks: list of [B,N] bool (True=masked).
    Returns per-patch mse [B,N] (mean over masks that cover the patch), target [B,N,256], pred."""
    m_chan = model.preprocessing.compute_m_channel(x_t, x_tk)       # [B,1,H,W]
    p_helper = model._encode_p_unmasked(model.preprocessing.compute_p_channel(x_t))
    tgt = model._patchify(m_chan)
    se_sum = torch.zeros(tgt.shape[:2], device=tgt.device)
    cnt = torch.zeros_like(se_sum)
    pred_acc = torch.zeros_like(tgt)
    for mask in masks:
        vis = model._encode_m_masked(m_chan, mask)
        st = model._inject_mask_tokens(vis, mask, model.mask_token_m) + model.dec_pos_embed_m
        for step in model.m_recon_decoder:
            st = step(st, p_helper)
        pred = model.m_recon_head(model.m_recon_decoder_norm(st)[:, 1:])
        err = ((pred - tgt) ** 2).mean(-1)
        se_sum += err * mask
        cnt += mask
        pred_acc += pred * mask.unsqueeze(-1)
    return se_sum / cnt.clamp(min=1), tgt, pred_acc / cnt.clamp(min=1).unsqueeze(-1), cnt


def make_masks(B, N, pair_ids, seed, rounds, ratio):
    masks = []
    for r in range(rounds):
        m = torch.zeros(B, N, dtype=torch.bool)
        for b, pid in enumerate(pair_ids):
            gen = torch.Generator().manual_seed(seed * 1_000_003 + pid * 101 + r)
            idx = torch.randperm(N, generator=gen)[: int(ratio * N)]
            m[b, idx] = True
        masks += [m, ~m]
    return masks


# ── P side: L_t (null routing, target frame_t) and L_pred (M routing, target frame_tk) ──────────
@torch.no_grad()
def p_errs(model, x_t, x_tk, masks, raw=False):
    """Replica of the P part of TwoStreamV15Model._forward_pair_comp with GIVEN masks.
    Returns {branch: per-patch mse [B,N] (mean over covering masks)}, targets, and (raw=True)
    the per-mask full patch predictions for the replication check."""
    B = x_t.shape[0]
    p_chan = model.preprocessing.compute_p_channel(x_t)
    routing = {"p_recon": model._null_routing(B, x_t.device),
               "p_pred": model._encode_m_unmasked(model.preprocessing.compute_m_channel(x_t, x_tk))}
    tgt = {"p_recon": model._patchify(x_t), "p_pred": model._patchify(x_tk)}
    se = {k: torch.zeros(tgt[k].shape[:2], device=x_t.device) for k in tgt}
    cnt = torch.zeros(tgt["p_recon"].shape[:2], device=x_t.device)
    preds = {k: [] for k in tgt}
    for mask in masks:
        vis = model._student_p_encode_visible(p_chan, mask)
        for k in tgt:
            st = model._build_full_seq_p(vis, mask)
            for step in model.p_motion_decoder:
                st = step(st, routing[k])
            pred = model.recon_head(model.p_motion_decoder_norm(st)[:, 1:])
            se[k] += ((pred - tgt[k]) ** 2).mean(-1) * mask
            if raw:
                preds[k].append(pred)
        cnt += mask
    return {k: se[k] / cnt.clamp(min=1) for k in se}, tgt, preds


def make_p_masks(B, N, pair_ids, seed, rounds, ratio):
    """Round r: N patches split into K = round(1/(1−ratio)) disjoint visible sets of N−int(ratio·N)
    each (leftover patches stay masked in every mask); mask k = all but visible set k."""
    n_vis = N - int(ratio * N)
    K = int(round(1.0 / (1.0 - ratio)))
    masks = []
    for r in range(rounds):
        ms = torch.ones(K, B, N, dtype=torch.bool)
        for b, pid in enumerate(pair_ids):
            gen = torch.Generator().manual_seed(seed * 1_000_003 + pid * 101 + r + 7_777_777)
            perm = torch.randperm(N, generator=gen)
            for k in range(K):
                ms[k, b, perm[k * n_vis:(k + 1) * n_vis]] = False
        masks += list(ms)
    return masks


@torch.no_grad()
def replication_check(model, x_t, x_tk):
    """Model's own _forward_pair_comp (eval) with captured masks vs this file's P path."""
    from src.training.pretrain import ssim_loss
    captured, orig = [], model._random_mask
    def rec(B, device, ratio):
        m = orig(B, device, ratio)
        captured.append((ratio, m))
        return m
    model._random_mask = rec
    torch.manual_seed(0)
    out = model._forward_pair_comp(x_t, x_tk)
    model._random_mask = orig
    ratio_t, mask_t = captured[0]
    _, tgt, preds = p_errs(model, x_t, x_tk, [mask_t], raw=True)
    res = dict(mask_ratio_p_model=ratio_t, n_masked=int(mask_t[0].sum()), batch=int(x_t.shape[0]),
               lambda_ssim=model.lambda_ssim)
    for k, mk, img in (("p_recon", "loss_t", x_t), ("p_pred", "loss_pred", x_tk)):
        pred = preds[k][0]
        mse = (((pred - tgt[k]) ** 2).mean(-1) * mask_t).sum() / mask_t.sum()
        ssim = ssim_loss(model._unpatchify(pred), img) if model.lambda_ssim > 0 else torch.zeros(())
        mine = mse + model.lambda_ssim * ssim
        res[k] = dict(model_loss=float(out[mk]), ours_loss=float(mine), ours_mse_part=float(mse),
                      abs_diff=float(abs(out[mk] - mine)))
    return res


# ── aggregation ────────────────────────────────────────────────────────────────────────────────
def summarize(rows, branch="m_recon"):
    """rows = per-patch dicts → group stats. Groups: region × {all, moved, static} by |ΔL|,
    region × sim_moved, and target static near_arm. branch m_recon reads mse/energy/pred_energy
    (unchanged); p_recon / p_pred read mse_<branch>/energy_<branch>, no |ΔL| weight (w1 blank)."""
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    is_m = branch == "m_recon"
    MSE = R["mse"] if is_m else R[f"mse_{branch}"]
    EN = R["energy"] if is_m else R[f"energy_{branch}"]
    w1 = W_FLOOR + 1.0 * R["absdl"]
    w0 = np.full_like(w1, W_FLOOR)
    tot1, tot0 = (w1 * MSE).sum(), (w0 * MSE).sum()
    out = []

    def add(grouping, region, state, sel):
        n = int(sel.sum())
        if n == 0:
            out.append(dict(branch=branch, grouping=grouping, region=region, state=state, n_patches=0))
            return
        out.append(dict(
            branch=branch, grouping=grouping, region=region, state=state, n_patches=n,
            abs_mse=float(MSE[sel].mean()),
            energy=float(EN[sel].mean()),
            rel_err=float(MSE[sel].sum() / max(EN[sel].sum(), 1e-12)),
            pred_energy=float(R["pred_energy"][sel].mean()) if is_m else None,
            mean_absdl=float(R["absdl"][sel].mean()),
            loss_share_w1=float((w1 * MSE)[sel].sum() / tot1) if is_m else None,
            loss_share_w0=float((w0 * MSE)[sel].sum() / tot0),
            mean_purity=float(R["purity"][sel].mean()),
        ))
    moved = R["absdl"] >= DL_THR
    for k, reg in enumerate(REGIONS):
        r = R["label"] == k
        add("dl", reg, "all", r)
        add("dl", reg, "moved", r & moved)
        add("dl", reg, "static", r & ~moved)
        add("sim", reg, "sim_moved", r & R["sim_moved"])
        add("sim", reg, "sim_static", r & ~R["sim_moved"])
    t = R["label"] == REGIONS.index("target")
    add("dl_near_arm", "target", "static_near_arm", t & ~moved & R["near_arm"])
    add("dl_near_arm", "target", "static_far_arm", t & ~moved & ~R["near_arm"])
    return out


def verify_png(rgb, body, reg_of_body, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = np.array([[255, 60, 60], [60, 255, 60], [255, 200, 0], [80, 120, 255], [0, 0, 0]], np.float32)
    r = np.full(body.shape, 4)
    r[body >= 0] = reg_of_body[body[body >= 0]]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.4))
    ax[0].imshow(rgb); ax[0].set_title("rendered obs", fontsize=8)
    ax[1].imshow((0.45 * rgb + 0.55 * col[r]).clip(0, 255).astype(np.uint8))
    ax[1].set_title("arm=red target=green recept=yellow other=blue", fontsize=7)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default=None, help="output stem (default = ckpt dir + file stem)")
    ap.add_argument("--tag-suffix", default="", help="appended to the output stem (e.g. __mp)")
    ap.add_argument("--suite", default="libero_object")
    # Default sample = the FIXED comparison sample (C0 vs noscale): all 10 tasks × first 10 demos,
    # every pair (i, i+gap) with stride gap. Fixed 2026-10-06 (smoke n_target_static=26 was too few).
    ap.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)), help="index into sorted hdf5 list")
    ap.add_argument("--demos", type=int, default=10, help="first N demos (numeric order) per task")
    ap.add_argument("--gap", type=int, default=10, help="sim steps (20 Hz → 10 = 0.5 s)")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=2, help="mask partitions (each = mask + complement)")
    ap.add_argument("--mask-ratio", type=float, default=0.5, help="= pretrain mask_ratio_m_recon default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = Path(args.checkpoint)
    tag = (args.tag or f"{ck.parent.parent.name}__{ck.parent.name}__{ck.stem}") + args.tag_suffix
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "verify").mkdir(exist_ok=True)
    model = build_parvo_encoder(str(ck), dev)
    assert hasattr(model, "m_recon_decoder"), "ckpt has no M-recon branch (not CoMP)"

    files = sorted((DATA_ROOT / args.suite).glob("*.hdf5"))
    rows, diag, t_render, t_model, n_pairs = [], [], 0.0, 0.0, 0
    replication = None
    for ti in args.task_ids:
        f5 = files[ti]
        env, bddl = make_env(args.suite, f5.stem)
        env.reset()
        ooi = objs_of_interest(bddl)
        reg_of_body = body_regions(env.sim, ooi)
        with h5py.File(f5, "r") as f:
            dks = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))[: args.demos]
            demos = [(dk, np.asarray(f[f"data/{dk}/states"]), np.asarray(f[f"data/{dk}/obs/agentview_rgb"]),
                      f[f"data/{dk}"].attrs["model_file"]) for dk in dks]
        for dk, states, h5rgb, xml in demos:
            apply_demo_fixtures(env.sim, xml)
            t0 = time.time()
            cache = {}

            def at(i):
                if i not in cache:
                    cache[i] = render_state(env, states[i])
                return cache[i]
            idx = list(range(1, len(states) - args.gap, args.stride))
            pairs = [(i, i + args.gap) for i in idx]
            fr = {i: at(i) for p in pairs for i in p}
            # determinism: re-render the first state → must be bit-identical in this process
            rgb_again = render_state(env, states[idx[0]])[0]
            bit_ident = bool((rgb_again == fr[idx[0]][0]).all())
            # info only: rendered obs(state i) vs hdf5 obs(i−1) (dataset obs t ↔ states t+1)
            h5diff = float(np.abs(fr[idx[0]][0].astype(np.float32) - h5rgb[idx[0] - 1]).mean())
            t_render += time.time() - t0
            if dk == dks[0]:
                verify_png(fr[idx[0]][0], fr[idx[0]][1], reg_of_body,
                           OUT_DIR / "verify" / f"{args.suite}_t{ti}_{dk}.png", f"{f5.stem[:40]} · ooi={ooi}")
            t0 = time.time()
            for s in range(0, len(pairs), args.batch):
                chunk = pairs[s: s + args.batch]
                pids = [ti * 10_000_000 + int(dk.split("_")[1]) * 10_000 + a for a, _ in chunk]
                x_t = preprocess_frames(np.stack([fr[a][0] for a, _ in chunk]), IMG).to(dev)
                x_tk = preprocess_frames(np.stack([fr[b][0] for _, b in chunk]), IMG).to(dev)
                N = model.num_patches
                masks = [m.to(dev) for m in make_masks(len(chunk), N, pids, args.seed, args.rounds, args.mask_ratio)]
                mse, tgt, pred, _ = mrecon_err(model, x_t, x_tk, masks)
                mse, tgt, pred = mse.cpu().numpy(), tgt.cpu().numpy(), pred.cpu().numpy()
                if replication is None:
                    replication = replication_check(model, x_t, x_tk)
                    print(f"REPLICATION {json.dumps(replication)}", flush=True)
                p_masks = [m.to(dev) for m in make_p_masks(len(chunk), N, pids, args.seed, args.rounds,
                                                           model.mask_ratio_p)]
                pmse, ptgt, _ = p_errs(model, x_t, x_tk, p_masks)
                pmse = {k: v.cpu().numpy() for k, v in pmse.items()}
                pt = model._patchify(x_t)
                pc = pt.reshape(*pt.shape[:2], -1, 3)          # [B,N,ps²,3] (patchify layout = ps,ps,C)
                p_en = {"p_recon": ((pc - pc.mean(2, keepdim=True)) ** 2).mean((2, 3)).cpu().numpy(),
                        "p_pred": ((ptgt["p_pred"] - pt) ** 2).mean(-1).cpu().numpy()}
                for b, (a, c) in enumerate(chunk):
                    disp = np.linalg.norm(fr[c][2] - fr[a][2], axis=1)
                    lab, pur, near, simm = patch_labels(fr[a][1], fr[c][1], reg_of_body, disp)
                    for p in range(N):
                        rows.append(dict(label=int(lab[p]), purity=float(pur[p]), near_arm=bool(near[p]),
                                         sim_moved=bool(simm[p]), mse=float(mse[b, p]),
                                         energy=float((tgt[b, p] ** 2).mean()),
                                         absdl=float(np.abs(tgt[b, p]).mean()),
                                         pred_energy=float((pred[b, p] ** 2).mean()),
                                         mse_p_recon=float(pmse["p_recon"][b, p]),
                                         energy_p_recon=float(p_en["p_recon"][b, p]),
                                         mse_p_pred=float(pmse["p_pred"][b, p]),
                                         energy_p_pred=float(p_en["p_pred"][b, p])))
            t_model += time.time() - t0
            n_pairs += len(pairs)
            # ΔL-mass sanity for seg orientation: share of |ΔL| inside non-background patches
            b0 = fr[idx[0]][1]
            reg0 = np.where(b0 >= 0, reg_of_body[np.clip(b0, 0, None)], REGIONS.index("background"))
            diag.append(dict(task=f5.stem, demo=dk, n_pairs=len(pairs), rerender_bit_identical=bit_ident,
                             mean_abs_diff_vs_hdf5=h5diff, ooi=ooi,
                             pix_frac_first_frame={r: float((reg0 == k).mean()) for k, r in enumerate(REGIONS)}))
            print(f"[{f5.stem[:40]} {dk}] pairs={len(pairs)} bit_ident={bit_ident} |Δ| vs hdf5={h5diff:.2f}", flush=True)
        env.close()

    R = np.array([r["label"] for r in rows])
    A = np.array([r["absdl"] for r in rows])
    dl_mass_fg = float(A[R != REGIONS.index("background")].sum() / max(A.sum(), 1e-12))
    groups = summarize(rows)
    g = {(x["grouping"], x["region"], x["state"]): x for x in groups}
    groups += summarize(rows, "p_recon") + summarize(rows, "p_pred")
    headline = dict(
        target_static_abs_mse=g[("dl", "target", "static")].get("abs_mse"),
        arm_moved_rel_err=g[("dl", "arm", "moved")].get("rel_err"),
        n_target_static=g[("dl", "target", "static")]["n_patches"],
        n_arm_moved=g[("dl", "arm", "moved")]["n_patches"],
    )
    res = dict(checkpoint=str(ck), tag=tag, suite=args.suite, task_ids=args.task_ids, demos=args.demos,
               gap=args.gap, stride=args.stride, rounds=args.rounds, mask_ratio=args.mask_ratio,
               seed=args.seed, dl_thr=DL_THR, sim_move_thr_m=SIM_MOVE_THR, n_pairs=n_pairs,
               n_patches=len(rows), dl_mass_in_fg=dl_mass_fg,
               sec_per_pair_render=t_render / max(n_pairs, 1), sec_per_pair_model=t_model / max(n_pairs, 1),
               headline=headline, replication_p=replication, mask_ratio_p=model.mask_ratio_p,
               demos_diag=diag, groups=groups)
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    keys = ["branch", "grouping", "region", "state", "n_patches", "abs_mse", "energy", "rel_err", "pred_energy",
            "mean_absdl", "loss_share_w1", "loss_share_w0", "mean_purity"]
    with open(OUT_DIR / f"{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for x in groups:
            w.writerow({k: ("" if x.get(k) is None else x.get(k, "")) for k in keys})
    print(f"HEADLINE ① target static abs MSE = {headline['target_static_abs_mse']}  (n={headline['n_target_static']})")
    print(f"HEADLINE ② arm moved rel err     = {headline['arm_moved_rel_err']}  (n={headline['n_arm_moved']})")
    print(f"|ΔL| mass in non-background patches = {dl_mass_fg:.3f} (orientation sanity: should be high)")
    print(f"sec/pair render {res['sec_per_pair_render']:.3f} · model {res['sec_per_pair_model']:.3f}")
    print(f"saved → {OUT_DIR / tag}.json/.csv")


if __name__ == "__main__":
    main()
