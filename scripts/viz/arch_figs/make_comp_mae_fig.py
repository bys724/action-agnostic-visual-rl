"""CoMP-MAE (Co-reconstructive Magno-Parvo MAE, code v16) main figure.

Symmetric extension of MCP-MAE: each stream reconstructs *its own* masked
signal while the *other* stream is given full as a helper. Mirrored 2-branch
layout makes the "Co" (co-reconstruction) message the headline.

  TOP    P-recon (implemented):  P masked -> V_P ;  M full -> Q,K_M  -> predict RGB
  MID    cross-attn principle:   V = reconstruction target  ·  Q,K = helper
  BOTTOM M-recon (new):          M masked -> V_M ;  P full -> Q,K_P  -> predict dL

Cross-attn principle (asymmetric meaning, mirrored structure):
  P-recon  V_P / Q,K_M   motion correspondence ("where to gather from")
  M-recon  V_M / Q,K_P   form grouping          ("which regions move together")
Colors stay attached to streams (P=green, M=blue) regardless of role, so the
V<->QK role swap between branches is visible at a glance.

Design source: docs/comp_mae_plan.md ; Vault "1. Core Idea.md" §대칭 Cross-Reconstruction.
Code (to be implemented): src/models/two_stream_v15.py, common/blocks.py MotionRoutingBlock.

Run from repo root:
    python3 scripts/viz/arch_figs/make_comp_mae_fig.py paper_artifacts/fig1_architecture/comp_mae_fig.png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import _common as C

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/comp_mae_fig.png"

ft, ftk, dL = C.load_frame_pair()
maskP = C.random_mask(seed=7, ratio=0.75)
maskM = C.random_mask(seed=11, ratio=0.75)

fig, ax = C.new_canvas(figsize=(16.5, 11.0))

def text(*a, **k): C.text(ax, *a, **k)
def box(*a, **k): C.box(ax, *a, **k)
def arrow(*a, **k): C.arrow(ax, *a, **k)
def imgbox(*a, **k): C.imgbox(ax, *a, **k)
def patch_grid(*a, **k): C.patch_grid(ax, *a, **k)
def token_col(*a, **k): return C.token_col(ax, *a, **k)

C_P, C_P_FILL = C.C_P, C.C_P_FILL
C_M, C_M_FILL = C.C_M, C.C_M_FILL
C_RT, C_RT_FILL = C.C_RT, C.C_RT_FILL
C_HD, C_HD_FILL = C.C_HD, C.C_HD_FILL
C_LOSS = C.C_LOSS
GP, GM = "#1B5E20", "#0D47A1"   # dark green / blue for encoder text

S = 11           # input image side
XI, XP, XE, XT = 2, 16.5, 30, 44.5     # x of: input / patchified / encoder / tokens

# ---------------------------------------------------------------- one lane
def lane(y, img, color, fill, enc_name, role, *, cmap=None, vmin=None, vmax=None,
         mask=None, token_masked=(), full=False, sep=False):
    """Draw input -> patchify -> encoder -> token column at vertical center y.

    role: label shown above the patchified image (e.g. 'masked 75%' / 'full (helper)').
    sep:  draw the t,t+k pair instead of a single frame (M-stream input depiction).
    """
    edrk = GP if color == C_P else GM
    # input image (single frame, or t/t+k pair for the M stream)
    if sep:
        imgbox(ft, XI, y - S / 2 + 1.3, S * 0.62, S * 0.62, z=4)
        imgbox(ftk, XI + S * 0.6, y - S / 2 - 1.3, S * 0.62, S * 0.62, z=5)
    else:
        imgbox(img, XI, y - S / 2, S, S, cmap=cmap, vmin=vmin, vmax=vmax)
    arrow(XI + S + 0.4, y, XP - 0.4, y, color=color)
    # patchified (with mask, or plain grid if full/helper)
    imgbox(img, XP, y - S / 2, S, S, cmap=cmap, vmin=vmin, vmax=vmax, z=4)
    patch_grid(XP, y - S / 2, S, S, masked=mask, edge="#5c7cae" if color == C_M else "#777")
    text(XP + S / 2, y + S / 2 + 1.6, role, size=9.5, color=color, style="italic")
    arrow(XP + S + 0.4, y, XE - 0.6, y, color=color)
    # encoder
    box(XE, y - 5, 12, 10, fill, color, lw=2.0)
    text(XE + 6, y + 1.3, enc_name, size=12.5, weight="bold", color=edrk)
    text(XE + 6, y - 2.2, "full" if full else "visible-only", size=8.5, style="italic", color=edrk)
    arrow(XE + 12 + 0.4, y, XT - 0.4, y, color=color)
    token_col(XT, y + 5.5, 4, fill, color, masked_idx=token_masked)

# ---------------------------------------------------------------- predictor block
def predictor(yv, yq, *, vcol, qcol, vlabel, qlabel, out_img, out_cmap, out_vmin,
              out_vmax, out_mask, head_txt, pred_lbl, tgt_lbl, gap_note):
    yc = (yv + yq) / 2
    PX, PW = 52, 14
    box(PX, yq - 5.5, PW, (yv - yq) + 11, C_RT_FILL, C_RT, lw=2.3, r=3)
    text(PX + PW / 2, yc + 3.2, "Predictor", size=13.5, weight="bold", color="#311B92")
    text(PX + PW / 2, yc + 0.2, "cross-attn x N", size=10, color="#311B92")
    text(PX + PW / 2, yc - 2.8, "complete-first", size=8.5, style="italic", color="#311B92")
    # V (top) and Q,K (bottom) feeders, colored by their source stream.
    # Short tags sit in the gap between the token column and the predictor;
    # the what/where + correspondence/grouping semantics live in the center banner.
    arrow(XT + 1.5, yv, PX - 0.4, yv - 1.0, color=vcol, lw=2.2)
    arrow(XT + 1.5, yq, PX - 0.4, yq + 1.0, color=qcol, lw=2.2)
    text(49.4, yv + 2.4, vlabel, size=10, color=vcol, weight="bold", ha="center")
    text(49.4, yq - 2.4, qlabel, size=10, color=qcol, weight="bold", ha="center")
    # recon head
    arrow(PX + PW, yc, PX + PW + 1.6, yc, color=C_HD)
    box(PX + PW + 2, yc - 3, 8, 6, C_HD_FILL, C_HD, lw=1.6, r=2)
    text(PX + PW + 6, yc, head_txt, size=9.5, color="#bf360c", weight="bold")
    # predicted (top) & target (bottom) + MSE
    XO = 80
    yp, yt = yc + 6.5, yc - 6.5
    imgbox(out_img, XO, yp - 5, 10, 10, cmap=out_cmap, vmin=out_vmin, vmax=out_vmax, z=4)
    patch_grid(XO, yp - 5, 10, 10, masked=out_mask, edge="#999")
    text(XO + 5, yp + 6.4, pred_lbl, size=10.5, color=C_HD, weight="bold")
    arrow(PX + PW + 10, yc + 1, XO - 0.3, yp - 5 + 1.5, color=C_HD, lw=1.7)
    imgbox(out_img, XO, yt - 5, 10, 10, cmap=out_cmap, vmin=out_vmin, vmax=out_vmax, z=4)
    text(XO + 5, yt - 6.8, tgt_lbl, size=10.5, color=vcol, weight="bold")
    arrow(XO + 5, yp - 5, XO + 5, yt + 5, color=C_LOSS, lw=1.9, ls=(0, (4, 2)), style="<->", mut=14)
    text(XO + 6.6, yc, "MSE\n(masked)", size=9, color=C_LOSS, weight="bold", ha="left")
    text(XO + 5, yt - 9.0, gap_note, size=8.5, style="italic", color="#666")

# ================= TOP branch — P-recon (implemented) ====================
yPV, yPQ = 82, 68
text(1.5, 90.5, "P-recon  (implemented)", size=12.5, color=C_P, weight="bold", ha="left")
text(1.5, 88.2, "reconstruct masked RGB,  M full as helper", size=9.5, color=C_P, style="italic", ha="left")
lane(yPV, ft, C_P, C_P_FILL, "P Encoder", "masked 75%", mask=maskP, token_masked=(1, 3))
lane(yPQ, dL, C_M, C_M_FILL, "M Encoder", "full (helper)", cmap="bwr", vmin=-0.22, vmax=0.22, full=True)
predictor(yPV, yPQ, vcol=C_P, qcol=C_M,
          vlabel="V <- P", qlabel="Q,K <- M",
          out_img=ft, out_cmap=None, out_vmin=None, out_vmax=None, out_mask=maskP,
          head_txt="head\n-> RGB", pred_lbl="predicted RGB", tgt_lbl="target = real RGB",
          gap_note="targets: static (gap 0) + future (gap k)")

# ================= MIDDLE — cross-attn principle banner ===================
box(1.5, 47.5, 97, 5.4, "#ede7f6", C_RT, lw=1.8, r=2)
text(50, 51.4, "Cross-attention principle:   V = reconstruction target  (owns content -> the encoder must learn it)"
               "    ·    Q, K = helper  (low-bandwidth routing pattern only)",
     size=11, weight="bold", color="#311B92")
text(50, 48.7, "structure is mirrored, meaning is asymmetric:  P-recon = motion correspondence   |   "
               "M-recon = form grouping   (a natural consequence of P/M information asymmetry)",
     size=9.3, style="italic", color="#5E35B1")

# ================= BOTTOM branch — M-recon (new) =========================
yMV, yMQ = 38, 24
text(1.5, 44.0, "M-recon  (new — direct grounding for M)", size=12.5, color=C_M, weight="bold", ha="left")
text(1.5, 41.7, "reconstruct masked dL,  P full as helper", size=9.5, color=C_M, style="italic", ha="left")
lane(yMV, dL, C_M, C_M_FILL, "M Encoder", "dL masked 75%", cmap="bwr", vmin=-0.22, vmax=0.22,
     mask=maskM, token_masked=(0, 2), sep=False)
lane(yMQ, ft, C_P, C_P_FILL, "P Encoder", "full (helper)", full=True)
predictor(yMV, yMQ, vcol=C_M, qcol=C_P,
          vlabel="V <- M", qlabel="Q,K <- P",
          out_img=dL, out_cmap="bwr", out_vmin=-0.22, out_vmax=0.22, out_mask=maskM,
          head_txt="head\n-> dL (1ch)", pred_lbl="predicted dL", tgt_lbl="target = real dL",
          gap_note="M masked pass is separate (full-M would leak)")

# ================= bottom explanation boxes ==============================
b1x, b1y, b1w, b1h = 1.5, 1.5, 47, 9.5
box(b1x, b1y, b1w, b1h, "#fff8e1", "#f9a825", lw=1.6, r=1.5)
text(b1x + 2, b1y + 7.4, "Symmetric Co-reconstruction", size=11.5, weight="bold", color="#7a5c00", ha="left")
text(b1x + 2.6, b1y + 5.0, "each stream: own signal masked / partner full  ->  cross-condition", size=9.2, color="#5d4500", ha="left")
text(b1x + 2.6, b1y + 2.6, "L = L_P(RGB)  +  lambda_M * L_M(dL)        (both: gap 0 + gap k)", size=9.2, style="italic", color="#5d4500", ha="left")

b2x, b2y, b2w, b2h = 51.5, 1.5, 47, 9.5
box(b2x, b2y, b2w, b2h, "#e8eaf6", "#3949ab", lw=1.6, r=1.5)
text(b2x + 2, b2y + 7.4, "What changes  (vs MCP-MAE / MotionMAE)", size=11.5, weight="bold", color="#1a237e", ha="left")
text(b2x + 2.6, b2y + 5.0, "MCP-MAE : P-recon only  ->  M trained as router only (no-op risk)", size=9.2, color="#283593", ha="left")
text(b2x + 2.6, b2y + 2.6, "CoMP-MAE: + M-recon  ->  M grounded by pixels  +  mutual coupling", size=9.2, weight="bold", color="#1a237e", ha="left")

# ================= title =================================================
text(50, 97.4, "CoMP-MAE  —  Co-reconstructive Magno-Parvo MAE", size=18, weight="bold")
text(50, 94.3, "Each stream reconstructs its own masked signal using the other as a full helper "
               "(V = target owns content, Q,K = helper) — code v16",
     size=10.5, style="italic", color="#444")

fig.savefig(OUT, bbox_inches="tight", facecolor="white", pad_inches=0.15)
print("saved", OUT)
