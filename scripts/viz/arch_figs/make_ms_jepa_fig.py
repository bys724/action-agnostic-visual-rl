"""MS-JEPA (Motion-Scaffolded JEPA) main figure  —  the predecessor of MCP-MAE.

Same SiamMAE-style two-stream layout as make_mcp_mae_fig.py, but MS-JEPA keeps the
V-JEPA latent-prediction objective (the part MCP-MAE later removed):

  TOP row     P stream (student): frame_t -> mask 75% -> P Encoder -> P tokens
                                  -> interpreter_1 -> recon_head -> pixels  (L_t / L_tk, MAE grounding)
  MIDDLE      scaffold: P tokens (V) + M tokens (Q,K) -> Predictor (p_motion_decoder, routing)
                        -> predicted LATENT  vs  EMA Teacher-P latent of frame_t+k  =>  L_pred
  BOTTOM row  M stream: dL(t,t+k) -> M Encoder -> M tokens (Q,K)

Key difference vs MCP-MAE: target is the EMA-teacher LATENT (self-referential) -> constant-collapse
risk -> needs target-LayerNorm (+ variance reg). MCP-MAE swaps this for a real-pixel target.

Code: src/models/two_stream_v15.py (_forward_pair / _vjepa_p_masked / _mae_one_frame, Run B-2),
      common/blocks.py (MotionRoutingBlock). Run B-2 here: lambda_m_jepa=0, lambda_var=0.

Run from repo root:
    python3 scripts/viz/arch_figs/make_ms_jepa_fig.py paper_artifacts/fig1_architecture/ms_jepa_fig.png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import _common as C

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ms_jepa_fig.png"

ft, ftk, dL = C.load_frame_pair()
mask = C.random_mask(seed=7, ratio=0.75)
fig, ax = C.new_canvas(figsize=(16.5, 9.4))

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
C_TE, C_TE_FILL = C.C_TE, C.C_TE_FILL
C_LOSS = C.C_LOSS

# ============== TOP ROW: P stream + MAE recon grounding ==================
yP, S = 83, 10
text(1.5, yP + S / 2 + 4.0, "P stream (student)  —  appearance,  + pixel-MAE grounding", size=12.5, color=C_P, weight="bold", ha="left")
text(7.2, yP + S / 2 + 1.4, "frame  t", size=12, color=C_P, weight="bold")
imgbox(ft, 1.5, yP - S / 2, S, S)
arrow(12.0, yP, 13.4, yP, color=C_P)
imgbox(ft, 13.6, yP - S / 2, S, S, z=4)
patch_grid(13.6, yP - S / 2, S, S, masked=mask)
text(18.6, yP + S / 2 + 1.4, "mask 75%", size=10.5, color=C_P)
arrow(24.0, yP, 25.4, yP, color=C_P)
box(25.6, yP - 6.5, 10.5, 13, C_P_FILL, C_P, lw=2.0)
text(30.85, yP + 1.6, "P Encoder", size=13, weight="bold", color="#1B5E20")
text(30.85, yP - 1.8, "ViT-S/B · d12", size=9.5, color="#1B5E20")
text(30.85, yP - 4.2, "visible-only", size=9, style="italic", color="#1B5E20")
arrow(36.4, yP, 37.8, yP, color=C_P)
token_col(38.1, yP + 5, 5, C_P_FILL, C_P, masked_idx=(1, 3))
# recon head branch (pixel grounding)
arrow(41.0, yP, 43.2, yP, color=C_HD)
box(43.4, yP - 3, 13, 6, C_HD_FILL, C_HD, lw=1.6, r=2)
text(49.9, yP, "interpreter_1\n-> recon_head", size=9.5, color="#bf360c", weight="bold")
arrow(56.6, yP, 58.4, yP, color=C_HD)
imgbox(ft, 58.7, yP - S / 2, S, S, z=4)
patch_grid(58.7, yP - S / 2, S, S, masked=mask, edge="#999")
text(63.7, yP + S / 2 + 1.4, "predicted pixels", size=10, color=C_HD, weight="bold")
text(76.5, yP + 1.6, "L_t , L_tk  =  MSE( . , frame t / t+k )", size=10.5, color=C_HD, weight="bold", ha="left")
text(76.5, yP - 1.6, "pixel grounding  (masked positions)", size=9.5, style="italic", color="#bf360c", ha="left")

# ============== MIDDLE: scaffold (V-JEPA latent prediction) ==============
PX, PY, PW, PH = 51, 42, 15, 15
box(PX, PY, PW, PH, C_RT_FILL, C_RT, lw=2.4, r=3)
text(PX + PW / 2, PY + PH - 3.0, "Predictor", size=14.5, weight="bold", color="#311B92")
text(PX + PW / 2, PY + PH - 6.2, "p_motion_decoder", size=9.5, style="italic", color="#311B92")
text(PX + PW / 2, PY + PH / 2 - 0.6, "cross-attn x N", size=10.5, color="#311B92")
text(PX + PW / 2, PY + 2.8, "+ mask tokens", size=9.2, style="italic", color="#311B92")
# V <- P (from P tokens above), Q,K <- M (from M tokens below)
arrow(40.0, yP - 6.5, PX + 4.5, PY + PH - 0.6, color=C_P, lw=2.3)
text(40.6, 58.0, "V <- P  (what)", size=11, color=C_P, weight="bold", ha="left")

# predicted latent (predictor output) — left
arrow(PX + PW, PY + PH / 2, 68.5, PY + PH / 2, color=C_RT)
box(68.5, 47.0, 10.0, 5.0, "#ede7f6", C_RT, lw=1.8, r=1.5)
text(73.5, 49.5, "predicted latent  z_hat", size=9.4, color="#311B92", weight="bold")

# teacher target latent — right of z_hat
box(84.0, 47.0, 10.0, 5.0, C_TE_FILL, C_TE, lw=1.8, r=1.5)
text(89.0, 49.5, "target latent  z", size=9.4, color="#006064", weight="bold")
# L_pred between z_hat and z (latent space)
C.arrow(ax, 78.7, 49.5, 83.8, 49.5, color=C_LOSS, lw=2.0, ls=(0, (4, 2)), style="<->", mut=14)
text(81.3, 53.0, "L_pred", size=10.5, color=C_LOSS, weight="bold")
text(81.3, 44.3, "smooth-L1\n(LATENT, masked)", size=8.6, style="italic", color=C_LOSS)

# teacher encoder (top-right, EMA) + frame_t+k
box(82.0, 56.0, 14.0, 9.0, C_TE_FILL, C_TE, lw=2.0)
text(89.0, 61.6, "Teacher P Encoder", size=10.5, weight="bold", color="#006064")
text(89.0, 58.0, "(EMA . stop-grad)", size=8.8, style="italic", color="#006064")
imgbox(ftk, 84.5, 67.0, 8.0, 8.0, z=4)
text(89.0, 76.4, "frame t+k", size=9.5, color=C_TE, weight="bold")
arrow(88.5, 66.8, 88.7, 65.2, color=C_TE, lw=2.0)        # frame_t+k -> teacher
arrow(89.0, 56.0, 89.0, 52.2, color=C_TE, lw=2.0)        # teacher -> target latent z
# EMA loop: student P Encoder -> Teacher P Encoder (curved through the upper band)
C.arrow(ax, 36.3, yP - 3, 82.0, 61.0, color="#00838f", lw=1.8, ls=(0, (5, 3)),
        style="-|>", mut=14, conn="arc3,rad=-0.16")
text(50.0, 71.0, "EMA update  (momentum 0.996)", size=10, color="#00838f", style="italic", weight="bold", ha="left")

# ============== BOTTOM ROW: M stream ====================================
yM = 19
text(1.5, yM + S / 2 + 3.6, "M stream  —  motion (WHERE),  routes the predictor", size=12.5, color=C_M, weight="bold", ha="left")
imgbox(ft, 1.5, yM - S / 2 + 1.2, S * 0.6, S * 0.6, z=4)
imgbox(ftk, 1.5 + S * 0.62, yM - S / 2 - 1.2, S * 0.6, S * 0.6, z=5)
text(6.4, yM + S / 2 + 1.2, "frame t & t+k", size=9.5, color=C_M)
arrow(12.0, yM, 13.4, yM, color=C_M)
imgbox(dL, 13.6, yM - S / 2, S, S, cmap="bwr", vmin=-0.22, vmax=0.22, z=4)
patch_grid(13.6, yM - S / 2, S, S, edge="#5c7cae")
text(18.6, yM + S / 2 + 1.2, "M = dL (1ch)", size=10, color=C_M)
text(18.6, yM - S / 2 - 1.8, "L(t+k) - L(t)", size=9.5, style="italic", color=C_M)
arrow(24.0, yM, 25.4, yM, color=C_M)
box(25.6, yM - 5, 10.5, 10, C_M_FILL, C_M, lw=2.0)
text(30.85, yM + 1.2, "M Encoder", size=12.5, weight="bold", color="#0D47A1")
text(30.85, yM - 2.0, "lightweight", size=9, style="italic", color="#0D47A1")
arrow(36.4, yM, 37.8, yM, color=C_M)
token_col(38.1, yM + 4, 4, C_M_FILL, C_M)
arrow(40.0, yM + 2.0, PX + 4.5, PY + 0.6, color=C_M, lw=2.3)
text(40.6, 30.5, "Q, K <- M  (where)", size=11, color=C_M, weight="bold", ha="left")

# ============== bottom explanation boxes =================================
b1x, b1y, b1w, b1h = 2, 1.0, 45.5, 11.5
box(b1x, b1y, b1w, b1h, "#fff8e1", "#f9a825", lw=1.6, r=1.5)
text(b1x + 2, b1y + 9.1, "Objective (Run B-2)", size=11.5, weight="bold", color="#7a5c00", ha="left")
text(b1x + 2.6, b1y + 6.7, "L = (L_t + L_tk)  +  lambda_pred * L_pred", size=10, color="#5d4500", ha="left")
text(b1x + 2.6, b1y + 4.3, "L_t / L_tk : pixel MAE grounding (interpreter_1)", size=9.6, color="#5d4500", ha="left")
text(b1x + 2.6, b1y + 1.7, "L_pred : M-routed LATENT prediction = scaffold (M->P grad)", size=9.6, style="italic", color="#5d4500", ha="left")

b2x, b2y, b2w, b2h = 52, 1.0, 46, 11.5
box(b2x, b2y, b2w, b2h, "#e0f2f1", "#00838f", lw=1.6, r=1.5)
text(b2x + 2, b2y + 9.1, "vs MCP-MAE  —  why it was replaced", size=11.5, weight="bold", color="#004d40", ha="left")
text(b2x + 2.6, b2y + 6.7, "target = EMA-teacher LATENT (self-referential)", size=9.6, color="#00695c", ha="left")
text(b2x + 2.6, b2y + 4.3, "   -> constant-collapse risk -> needs target-LN (+ var reg)", size=9.6, color="#00695c", ha="left")
text(b2x + 2.6, b2y + 1.7, "MCP-MAE: latent target -> REAL pixels => collapse impossible", size=9.6, weight="bold", color="#004d40", ha="left")

# ============== title ====================================================
text(50, 97.5, "MS-JEPA  —  Motion-Scaffolded JEPA   (predecessor of MCP-MAE)", size=17.5, weight="bold")
text(50, 93.6, "M routes 'where' (Q,K) onto P's 'what' (V) to predict the future in LATENT space (EMA teacher target)",
     size=10.5, style="italic", color="#444")

fig.savefig(OUT, bbox_inches="tight", facecolor="white", pad_inches=0.15)
print("saved", OUT)
