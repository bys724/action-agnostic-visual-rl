"""Shared helpers for architecture figures (MCP-MAE, MS-JEPA, ...).

Color convention, EgoDex frame extraction, and matplotlib drawing primitives
shared by the make_*_fig.py scripts in this directory.

Conventions (single source of truth — keep in sync with the mermaid companions):
  P / appearance / what  -> green
  M / motion / where     -> blue
  routing / predictor    -> purple
  head / output          -> orange
  teacher / EMA          -> teal   (MS-JEPA only)
  loss                   -> red ;  masked patch -> gray

Coordinate system: every figure draws on an axis with xlim/ylim = (0, 100).
Run scripts from the repo root so the default frame path resolves.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.font_manager import FontProperties

# ---- color convention ---------------------------------------------------
C_P, C_P_FILL = "#2E7D32", "#c8e6c9"      # P / appearance / what
C_M, C_M_FILL = "#1565C0", "#bbdefb"      # M / motion / where
C_RT, C_RT_FILL = "#5E35B1", "#d1c4e9"    # routing / predictor
C_HD, C_HD_FILL = "#E65100", "#ffe0b2"    # head / output
C_TE, C_TE_FILL = "#00838f", "#b2ebf2"    # teacher / EMA (MS-JEPA)
C_LOSS = "#c62828"
C_GRAY = "#cfd8dc"
TXT = "#1a1a1a"
SERIF = FontProperties(family="Times New Roman")

NG = 8   # display patch grid (model is 14x14; coarse grid is schematic)

# default frame source: an EgoDex pair with clear, localized object motion
DEFAULT_PAIR_PNG = "paper_artifacts/parvo_runB2cont_recon_samples/epoch_030_pair.png"


def load_frame_pair(png=DEFAULT_PAIR_PNG, col_t=(228, 546), col_tk=(562, 880),
                    row=(492, 810), size=224):
    """Extract (frame_t, frame_t+k, dL) from a recon-sample contact sheet.

    Columns 1/2 of the sheet are GT frame_t / frame_t+k; `row` selects an
    EgoDex hand-manipulation example. dL = luminance(t+k) - luminance(t)
    (the M channel, matching preprocessing.compute_m_channel no-Sobel).
    """
    src = Image.open(png).convert("RGB")
    ft = np.asarray(src.crop((col_t[0], row[0], col_t[1], row[1])).resize((size, size))).astype(float) / 255.0
    ftk = np.asarray(src.crop((col_tk[0], row[0], col_tk[1], row[1])).resize((size, size))).astype(float) / 255.0
    lum = lambda x: 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return ft, ftk, lum(ftk) - lum(ft)


def random_mask(seed=7, ratio=0.75):
    """Boolean [NG, NG] mask (True = masked) for the schematic patch grid."""
    return np.random.default_rng(seed).random((NG, NG)) < ratio


def new_canvas(figsize=(16.5, 8.4), dpi=200):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


# ---- drawing primitives (stateless; take ax first) ----------------------
def text(ax, x, y, s, size=12, color=TXT, weight="normal", style="normal", ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, fontstyle=style,
            ha=ha, va=va, fontproperties=SERIF, zorder=10)


def box(ax, x, y, w, h, fc, ec, lw=1.8, r=2.5, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, zorder=z, mutation_aspect=1))


def arrow(ax, x0, y0, x1, y1, color=TXT, lw=2.2, ls="-", z=6, mut=18, style="-|>", conn=None):
    kw = dict(arrowstyle=style, mutation_scale=mut, color=color, lw=lw, ls=ls,
              zorder=z, shrinkA=0, shrinkB=0)
    if conn is not None:
        kw["connectionstyle"] = conn   # e.g. "arc3,rad=-0.3" for a curved (EMA) loop
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), **kw))


def imgbox(ax, img, x, y, w, h, ec="#555", cmap=None, vmin=None, vmax=None, z=4):
    ax.imshow(img, extent=(x, x + w, y, y + h), aspect="auto", zorder=z, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=ec, lw=1.2, zorder=z + 1))


def patch_grid(ax, x, y, w, h, masked=None, edge="#777", z=5):
    cw, ch = w / NG, h / NG
    for i in range(NG):
        for j in range(NG):
            if masked is not None and masked[i, j]:
                ax.add_patch(Rectangle((x + j * cw, y + (NG - 1 - i) * ch), cw, ch,
                             fc=C_GRAY, ec=edge, lw=0.5, zorder=z))
            else:
                ax.add_patch(Rectangle((x + j * cw, y + (NG - 1 - i) * ch), cw, ch,
                             fill=False, ec=edge, lw=0.5, zorder=z))


def token_col(ax, x, ytop, n, fill, edge, masked_idx=(), w=2.3, gap=0.55, z=5):
    """A vertical stack of n token squares (some grayed by masked_idx) + a '...' tail."""
    th = 2.5
    for k in range(n):
        yy = ytop - k * (th + gap)
        fc = C_GRAY if k in masked_idx else fill
        ax.add_patch(Rectangle((x, yy), w, th, fc=fc, ec=edge, lw=1.0, zorder=z))
    ax.text(x + w / 2, ytop - n * (th + gap) + 0.4, "...", ha="center", va="center",
            fontsize=15, fontproperties=SERIF, zorder=z)
    return x + w / 2
