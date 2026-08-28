"""Supplementary Figure S8: weight distributions across the GONNECT-SL alpha sweep.

Two panels, the two ends of the sweep that the text argues from:

  a  encoder, alpha = 1e2   the setting the main text uses; soft links numerous
  b  encoder, alpha = 1e4   the setting where the penalty has all but closed
                            them off -- exactly one soft link stays active

Both panels are the encoder (MODULE below). At alpha = 1e4 the encoder has one
active soft link and the decoder two, so the caption's "just one soft link
becomes active" is the encoder's number.

Each panel carries four step histograms of the same encoder module, so the
soft-link weights can be read against both the GO edges they sit beside and the
fully connected model they would become if the penalty were dropped:

  Fully connected (MLP)          all 3,360,429 weights of a dense encoder
  GONNECT-SL, GO edges           the 9,561 positions GO does connect
  GONNECT-SL, soft links         the 3,350,868 positions it does not
  GONNECT, GO edges              the same 9,561 positions in original GONNECT

The axes follow Figure 5a rather than the published version of this figure,
which binned the signed weight on a linear axis. Two reasons. The claim the
figure supports is about magnitude -- that the soft links stay "at least an
order of magnitude below the strongest GO edges" -- and a linear axis crushes
eleven decades of that into a spike at zero. And the S8 caption now points the
reader at Figure 5a for the early-stopped counterpart of panel a, a comparison
only worth making if the two are drawn the same way. Set SIGNED_LINEAR_X = True
to recover the published rendering.

The GO positions are taken as the nonzero weights of the fixed-link run, the
same trick fig5.gather uses, so no mask file is needed and the positions
cannot drift from the model. They include the proxy edges held at exactly 1,
which is the spike at log10|w| = 0.

Inputs (relative to --data-dir)
------------------------------
    alpha_sweep_weights/AE_3.-1.2_none_weights.pt      fully connected
    alpha_sweep_weights/AE_3.-1.2_encoder_weights.pt   GONNECT
    alpha_sweep_weights/AE_3.0.3_encoder_weights.pt    alpha = 1e2
    alpha_sweep_weights/AE_3.2.3_encoder_weights.pt    alpha = 1e4

Each holds only the four weight matrices of the module the run constrains, cut
out of the full autoencoder checkpoint by prepare/extract_alpha_sweep_weights.py.

All four are single instances at data-split seed 1, trained for a fixed 1,000
epochs with early stopping disabled -- unlike the main-text models, which use
the same alpha as panel a but stop early and so keep the GO-weight distribution
shown in Figure 5a.

Usage
-----
    python figS8.py [--data-dir figures/data] [--out-dir figures/out]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe: no display needed
import matplotlib.pyplot as plt
import torch

import _common
from _common import PT_PANEL, PT_SMALL, PT_TITLE, figsize, pt

# ── What is drawn ────────────────────────────────────────────────────────────
MODULE = "encoder"
# nn.Sequential indices carrying a weight matrix (ReLU sits on the odd ones),
# as in fig5. Layers are pooled: the per-layer split is Supplementary Figure S9.
WEIGHTED = [0, 2, 4, 6]

FIXED_LINKS = f"AE_3.-1.2_{MODULE}"
# The MLP run constrains neither module, so it has no per-module variant;
# MODULE picks which half of it is read, as it does for every other series.
FULLY_CONNECTED = "AE_3.-1.2_none"

# (alpha label, run stem, panel letter). The title names the module too, built
# from MODULE below: figS7 titles its panels "GONNECT-SL encoder" and
# "GONNECT-SL decoder", while this figure said only the alpha and left the
# reader to infer which module the histograms belong to. Deriving both the stem
# and the title from MODULE means the title cannot name a module the data did
# not come from.
PANELS = [(r"$\alpha$ = 1$\cdot$10$^2$", f"AE_3.0.3_{MODULE}", "a"),
          (r"$\alpha$ = 1$\cdot$10$^4$", f"AE_3.2.3_{MODULE}", "b")]
PANEL_TITLE = "GONNECT-SL " + MODULE + ", {alpha}"

# (key, legend label, colour, dash). The key is internal and stable, the label
# is what the reader sees -- the same split fig5 makes, and for the same reason:
# these labels were the dict keys of gather(), so a terminology fix would
# silently rename data.
#
# Colours are Okabe-Ito, in an order verified to pass every check of the palette
# validator under the all-pairs comparison; the MLP and GONNECT entries keep the
# exact colours they carry in figS7, so the two figures read as one family.
# Figure 5a's own four fail that check badly -- its green and red sit at
# deuteranopic dE 3.9 -- which is worth fixing there too. Dashes are the
# secondary encoding the validator's contrast WARN obliges.
SERIES = [
    ("mlp",      "Fully connected (MLP)",  "#0072B2", "solid"),
    ("sl_go",    "GONNECT-SL, GO edges",   "#D55E00", (0, (5, 1.6))),
    ("sl_soft",  "GONNECT-SL, soft links", "#009E73", (0, (1.4, 1.4))),
    ("go_edges", "GONNECT, GO edges",      "#56B4E9", (0, (6, 1.6, 1.4, 1.6))),
]

SIGNED_LINEAR_X = False   # True restores the published linear signed-value axis
CLIP = 1e-12              # floors log10|w| so exact zeros do not become -inf
BINS_LOG = np.linspace(-10, 1, 111)
BINS_LINEAR = np.linspace(-2, 2, 161)

# |w| > 0.1 is the threshold the paper calls a soft link "active", so the count
# to the right of this line in the green series is the number quoted in the
# text: 1,008 active links at alpha = 1e2 against 1 at alpha = 1e4.
ACTIVE_THRESH = 0.1

# ── Layout ───────────────────────────────────────────────────────────────────
# Width is fixed by the paper's shared standard, so only the height and the
# internal division are free, and both are applied through subplots_adjust:
# save_figure keeps the full canvas width, so the row must reach both edges.
FIG_HEIGHT_IN = 5.30

MARGIN_L_IN = 1.05      # "count", its 10^n tick labels, and the panel letter
MARGIN_R_IN = 0.10
MARGIN_T_IN = 0.46
XLABEL_BAND_IN = 0.72   # tick labels and axis label under each panel
LEGEND_BAND_IN = 0.84   # the shared key, two rows of two. One row of
                        # four overflows both canvas edges: the labels
                        # carry their n, and four of those do not fit.
MARGIN_B_IN = XLABEL_BAND_IN + LEGEND_BAND_IN
PANEL_GAP_IN = 1.15     # panel b's y axis label, ticks and panel letter

LW = pt(1.0)


def load_weights(data_dir: Path, stem: str) -> list:
    """The four encoder weight matrices of one run, as numpy arrays.

    Reads the per-module extracts in `alpha_sweep_weights/` rather than the
    training checkpoints they came from: every run saves a whole autoencoder,
    and the half nothing here reads is half the deposit. See
    prepare/extract_alpha_sweep_weights.py.
    """
    path = data_dir / "alpha_sweep_weights" / f"{stem}_weights.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    return [state[f"{MODULE}.net_layers.{i}.weight"].numpy() for i in WEIGHTED]


def transform(values: np.ndarray) -> np.ndarray:
    if SIGNED_LINEAR_X:
        return values
    return np.log10(np.clip(np.abs(values), CLIP, None))


def gather(data_dir: Path, stem: str) -> dict:
    """The four series of one panel, each already binned-ready.

    GO positions come from the fixed-link checkpoint's nonzero entries: that
    model has every non-GO weight pinned at exactly 0, so the nonzero mask is
    the GO mask, proxy edges included.
    """
    fixed = load_weights(data_dir, FIXED_LINKS)
    dense = load_weights(data_dir, FULLY_CONNECTED)
    soft = load_weights(data_dir, stem)

    out = {key: [] for key, _, _, _ in SERIES}
    for w_fixed, w_dense, w_soft in zip(fixed, dense, soft):
        go = w_fixed != 0
        out["go_edges"].append(transform(w_fixed[go]))
        out["sl_go"].append(transform(w_soft[go]))
        out["sl_soft"].append(transform(w_soft[~go]))
        out["mlp"].append(transform(w_dense.ravel()))
    return {k: np.concatenate(v) for k, v in out.items()}


def plot_panel(ax, data: dict, title: str, letter: str) -> None:
    bins = BINS_LINEAR if SIGNED_LINEAR_X else BINS_LOG
    for key, label, color, dash in SERIES:
        ax.hist(data[key], bins=bins, histtype="step", lw=LW, color=color,
                ls=dash, label=f"{label} (n = {len(data[key]):,})", zorder=3)

    thresh = ACTIVE_THRESH if SIGNED_LINEAR_X else np.log10(ACTIVE_THRESH)
    ax.axvline(thresh, color="0.45", lw=LW * 0.7, ls=(0, (2, 2)), zorder=2)
    # Inside the axes, just under the top spine, rather than above it: the panel
    # title now carries the module name as well and is wide enough to reach
    # this label in the band over the axes. The top of both panels is empty at
    # the threshold -- every series has fallen far below its peak by |w| = 0.1.
    ax.text(thresh, 0.97, r"$|w|$ > 0.1", transform=ax.get_xaxis_transform(),
            fontsize=PT_SMALL, color="0.35", ha="center", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                      pad=1.5), zorder=4)

    ax.set_yscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("weight value" if SIGNED_LINEAR_X else r"$\log_{10}|w|$")
    ax.set_ylabel("count")
    ax.set_title(PANEL_TITLE.format(alpha=title), fontsize=PT_TITLE)
    ax.grid(color="0.9", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.text(-0.075, 1.015, letter, transform=ax.transAxes, fontsize=PT_PANEL,
            fontweight="bold", va="bottom", ha="left")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supplementary Figure S8: alpha-sweep weight distributions.")
    _common.add_io_args(parser)
    args = parser.parse_args()

    print(f"Reading {2 + len(PANELS)} weight files from "
          f"{args.data_dir / 'alpha_sweep_weights'} ...")

    fig, axes = plt.subplots(1, 2, figsize=figsize(FIG_HEIGHT_IN))
    axes_w = (_common.FIG_WIDTH_IN - MARGIN_L_IN - MARGIN_R_IN - PANEL_GAP_IN) / 2
    fig.subplots_adjust(
        left=MARGIN_L_IN / _common.FIG_WIDTH_IN,
        right=1.0 - MARGIN_R_IN / _common.FIG_WIDTH_IN,
        top=1.0 - MARGIN_T_IN / FIG_HEIGHT_IN,
        bottom=MARGIN_B_IN / FIG_HEIGHT_IN,
        wspace=PANEL_GAP_IN / axes_w,
    )

    for ax, (title, stem, letter) in zip(axes, PANELS):
        plot_panel(ax, gather(args.data_dir, stem), title, letter)

    # Both panels on one y range: the figure's point is that the soft-link mass
    # collapses between them, which is unreadable if each panel rescales.
    top = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0.7, top)

    # One key for both panels, in the band under them. It cannot sit inside an
    # axes: the soft-link series climbs through the upper left of both, which is
    # the only corner an in-axes legend would otherwise fit.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=PT_SMALL, frameon=False, handlelength=2.6,
               columnspacing=pt(3.0), handletextpad=pt(0.4),
               labelspacing=0.5, bbox_to_anchor=(0.5, 0.008))

    _common.report_text_overlaps(fig, "figS8")
    _common.save_figure(fig, args.out_dir, "figS8", dpi=200)


if __name__ == "__main__":
    main()
