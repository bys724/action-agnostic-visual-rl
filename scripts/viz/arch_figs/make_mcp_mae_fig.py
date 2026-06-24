"""MCP-MAE (Motion-Conditioned Predictive MAE) main figure.

SiamMAE Fig.1 layout adapted to MCP-MAE's two-stream structure:
  - Top row    P stream (appearance / WHAT): frame_t RGB -> mask 75% -> P Encoder -> V
  - Bottom row M stream (motion / WHERE)    : dL(t,t+k) -> M Encoder (lightweight) -> Q,K
  - Predictor (= p_motion_decoder): cross-attn  Q,K<-M / V<-P  + mask tokens -> recon_head -> frame_t+k pixels
  - target = real frame_t+k  (no EMA / no JEPA -> constant collapse impossible)

Code: src/models/two_stream_v15.py (_forward_pair_pixel / _predict_pixels),
      src/models/common/blocks.py (MotionRoutingBlock), preprocessing.py (no-Sobel).

Run from repo root:
    python3 scripts/viz/arch_figs/make_mcp_mae_fig.py paper_artifacts/fig1_architecture/mcp_mae_fig.png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import _common as C

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mcp_mae_fig.png"

ft, ftk, dL = C.load_frame_pair()
mask = C.random_mask(seed=7, ratio=0.75)
fig, ax = C.new_canvas()

# ax-bound wrappers (keep the body readable)
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

yP, yM, S = 68, 24, 14

# ---------- stream identity headers --------------------------------------
text(1.5, yP + S / 2 + 6.2, "P stream  —  appearance (WHAT)", size=13, color=C_P, weight="bold", ha="left")
text(1.5, yM + S / 2 + 5.0, "M stream  —  motion (WHERE)", size=13, color=C_M, weight="bold", ha="left")

# ---------- TOP: P stream ------------------------------------------------
text(9, yP + S / 2 + 2.4, "frame  t", size=14, color=C_P, weight="bold")
imgbox(ft, 2, yP - S / 2, S, S)
arrow(16.4, yP, 18.0, yP, color=C_P)
imgbox(ft, 18.2, yP - S / 2, S, S, z=4)
patch_grid(18.2, yP - S / 2, S, S, masked=mask)
text(25.2, yP + S / 2 + 2.4, "patchify & mask 75%", size=12, color=C_P)
arrow(32.6, yP, 34.4, yP, color=C_P)
box(34.7, yP - 9, 13, 18, C_P_FILL, C_P, lw=2.0)
text(41.2, yP + 2.0, "P Encoder", size=15, weight="bold", color="#1B5E20")
text(41.2, yP - 1.8, "ViT-S / B  ·  depth 12", size=10.5, color="#1B5E20")
text(41.2, yP - 5.0, "visible-only (25%)", size=10, style="italic", color="#1B5E20")
text(41.2, yP + 9 + 2.5, "downstream backbone", size=10.5, style="italic", color=C_P, weight="bold")
arrow(47.8, yP, 49.4, yP, color=C_P)
token_col(49.7, yP + 7, 6, C_P_FILL, C_P, masked_idx=(1, 4))

# ---------- BOTTOM: M stream --------------------------------------------
imgbox(ft, 2, yM - S / 2 + 1.6, S * 0.6, S * 0.6, z=4)
imgbox(ftk, 2 + S * 0.62, yM - S / 2 - 1.6, S * 0.6, S * 0.6, z=5)
text(8.6, yM + S / 2 + 2.0, "frame t  &  t+k", size=11.5, color=C_M)
arrow(16.4, yM, 18.0, yM, color=C_M)
imgbox(dL, 18.2, yM - S / 2, S, S, cmap="bwr", vmin=-0.22, vmax=0.22, z=4)
patch_grid(18.2, yM - S / 2, S, S, edge="#5c7cae")
text(25.2, yM + S / 2 + 2.4, "M channel  =  dL   (1ch, no mask)", size=11.5, color=C_M)
text(25.2, yM - S / 2 - 2.2, "L(t+k) - L(t)", size=11, style="italic", color=C_M)
arrow(32.6, yM, 34.4, yM, color=C_M)
box(34.7, yM - 6.5, 13, 13, C_M_FILL, C_M, lw=2.0)
text(41.2, yM + 1.5, "M Encoder", size=14.5, weight="bold", color="#0D47A1")
text(41.2, yM - 2.0, "lightweight · depth 6", size=10, color="#0D47A1")
text(41.2, yM - 4.6, "unmasked", size=10, style="italic", color="#0D47A1")
arrow(47.8, yM, 49.4, yM, color=C_M)
token_col(49.7, yM + 6, 5, C_M_FILL, C_M)

# ---------- Predictor (cross-attention) ---------------------------------
PX, PY, PW, PH = 58, 37, 17.5, 18
box(PX, PY, PW, PH, C_RT_FILL, C_RT, lw=2.4, r=3)
text(PX + PW / 2, PY + PH - 3.2, "Predictor", size=15.5, weight="bold", color="#311B92")
text(PX + PW / 2, PY + PH - 6.8, "p_motion_decoder", size=10.5, style="italic", color="#311B92")
text(PX + PW / 2, PY + PH / 2 - 0.5, "cross-attention x N", size=11.5, color="#311B92")
text(PX + PW / 2, PY + 3.2, "+ mask tokens", size=10, style="italic", color="#311B92")
arrow(53.2, yP - 4.5, PX + 4.5, PY + PH - 0.6, color=C_P, lw=2.3)
arrow(53.2, yM + 2.0, PX + 4.5, PY + 0.6, color=C_M, lw=2.3)
text(56.5, 65.0, "V  <-  P   (what)", size=11.5, color=C_P, weight="bold", ha="left")
text(56.5, 27.3, "Q, K  <-  M   (where)", size=11.5, color=C_M, weight="bold", ha="left")

# ---------- recon_head -> predicted / target ----------------------------
arrow(PX + PW, PY + PH / 2, PX + PW + 1.8, PY + PH / 2, color=C_HD)
box(77.8, PY + PH / 2 - 3.4, 9.2, 6.8, C_HD_FILL, C_HD, lw=1.8, r=2)
text(82.4, PY + PH / 2, "recon_head\n-> pixels", size=10.5, color="#bf360c", weight="bold")

ypred = yP - 2
imgbox(ftk, 89, ypred - 6, 12, 12, z=4)
patch_grid(89, ypred - 6, 12, 12, masked=mask, edge="#999")
text(95, ypred + 7.6, "predicted  frame t+k", size=12, color=C_HD, weight="bold")
arrow(87.0, PY + PH / 2 + 1.5, 89.3, ypred - 6 + 1.5, color=C_HD, lw=1.8)

ytgt = yM + 2
imgbox(ftk, 89, ytgt - 6, 12, 12, z=4)
text(95, ytgt - 8.0, "target = real  frame t+k", size=12, color=C_M, weight="bold")

C.arrow(ax, 95, ypred - 6, 95, ytgt + 6, color=C_LOSS, lw=2.0, ls=(0, (4, 2)), style="<->", mut=15)
text(96.6, (ypred + ytgt) / 2, "MSE\n(masked\npositions)", size=10, color=C_LOSS, weight="bold", ha="left")

# ============== bottom explanation boxes =================================
b1x, b1y, b1w, b1h = 2, 1.5, 46, 11.5
box(b1x, b1y, b1w, b1h, "#fff8e1", "#f9a825", lw=1.6, r=1.5)
text(b1x + 2, b1y + 9.1, "Unified mechanism  —  one predict() does both", size=11.5, weight="bold", color="#7a5c00", ha="left")
text(b1x + 2.6, b1y + 6.7, "M = real  dL(t, t+k)   ->  predict future pixels   (shown)", size=10, color="#5d4500", ha="left")
text(b1x + 2.6, b1y + 4.3, "M = null  dL(t, t)=0   ->  identity routing  ->  reconstruct (gap=0)", size=10, color="#5d4500", ha="left")
text(b1x + 2.6, b1y + 1.5, "L = lambda*(L_t + L_tk)  +  lambda*L_pred       (motion size = gap)", size=10, style="italic", color="#5d4500", ha="left")

b2x, b2y, b2w, b2h = 52, 1.5, 46, 11.5
box(b2x, b2y, b2w, b2h, "#e8eaf6", "#3949ab", lw=1.6, r=1.5)
text(b2x + 2, b2y + 9.1, "vs SiamMAE  —  what changes", size=11.5, weight="bold", color="#1a237e", ha="left")
text(b2x + 2.6, b2y + 7.1, "SiamMAE :  two RGB frames, siamese enc  ·  Q<-future / K,V<-past", size=9.6, color="#283593", ha="left")
text(b2x + 2.6, b2y + 5.1, "MCP-MAE :  factorize appearance P(RGB) & motion M(dL)", size=9.6, color="#283593", ha="left")
text(b2x + 2.6, b2y + 3.1, "   ->  Q,K <- M (where) · V <- P (what)   = what/where factorization", size=9.6, weight="bold", color="#1a237e", ha="left")
text(b2x + 2.6, b2y + 1.1, "   ->  target = real pixels (no EMA / no JEPA  ->  collapse impossible)", size=9.6, color="#283593", ha="left")

# ============== title ====================================================
text(50, 97.0, "MCP-MAE  —  Motion-Conditioned Predictive MAE", size=18, weight="bold")
text(50, 92.6, "Factorize appearance (P) and motion (M); M routes 'where' (Q,K) onto P's 'what' (V) to predict future pixels",
     size=11, style="italic", color="#444")

fig.savefig(OUT, bbox_inches="tight", facecolor="white", pad_inches=0.15)
print("saved", OUT)
