"""Supplementary Figure S8: weight distributions across the GONNECT-SL alpha sweep.

Two panels, the two ends of the sweep that the text argues from:

  a  alpha = 1e2   the setting the main text uses -- soft links are numerous
  b  alpha = 1e4   the setting where the penalty has all but closed them off

Each panel carries four step histograms of the same encoder module, so the
soft-link weights can be read against both the GO edges they sit beside and the
fully connected model they would become if the penalty were dropped:

  Fully connected (MLP)          all 3,360,429 weights of a dense encoder
  GONNECT-SL, GO edges           the 9,561 positions GO does connect
  GONNECT-SL, soft links         the 3,350,868 positions it does not
  Fixed links, GO edges          the same 9,561 positions in original GONNECT

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
    alpha_sweep_weights/AE_3.-1.2_encoder_weights.pt   fixed links
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

FIXED_LINKS = "AE_3.-1.2_encoder"
FULLY_CONNECTED = "AE_3.-1.2_none"
PANELS = [(r"$\alpha$ = 1$\cdot$10$^2$", "AE_3.0.3_encoder", "a"),
          (r"$\alpha$ = 1$\cdot$10$^4$", "AE_3.2.3_encoder", "b")]

# (legend label, colour, dash). Okabe-Ito, in an order verified to pass every
# check of the palette validator under the all-pairs comparison; the MLP and
# fixed-link entries keep the exact colours they carry in figS7, so the two
# figures read as one family. Figure 5a's own four fail that check badly -- its
# green and red sit at deuteranopic dE 3.9 -- which is worth fixing there too.
# Dashes are the secondary encoding the validator's contrast WARN obliges.
SERIES = [
    ("Fully connected (MLP)",  "#0072B2", "solid"),
    ("GONNECT-SL, GO edges",   "#D55E00", (0, (5, 1.6))),
    ("GONNECT-SL, soft links", "#009E73", (0, (1.4, 1.4))),
    ("Fixed links, GO edges",  "#56B4E9", (0, (6, 1.6, 1.4, 1.6))),
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

    out = {label: [] for label, _, _ in SERIES}
    for w_fixed, w_dense, w_soft in zip(fixed, dense, soft):
        go = w_fixed != 0
        out["Fixed links, GO edges"].append(transform(w_fixed[go]))
        out["GONNECT-SL, GO edges"].append(transform(w_soft[go]))
        out["GONNECT-SL, soft links"].append(transform(w_soft[~go]))
        out["Fully connected (MLP)"].append(transform(w_dense.ravel()))
    return {k: np.concatenate(v) for k, v in out.items()}


def plot_panel(ax, data: dict, title: str, letter: str) -> None:
    bins = BINS_LINEAR if SIGNED_LINEAR_X else BINS_LOG
    for label, color, dash in SERIES:
        ax.hist(data[label], bins=bins, histtype="step", lw=LW, color=color,
                ls=dash, label=f"{label} (n = {len(data[label]):,})", zorder=3)

    thresh = ACTIVE_THRESH if SIGNED_LINEAR_X else np.log10(ACTIVE_THRESH)
    ax.axvline(thresh, color="0.45", lw=LW * 0.7, ls=(0, (2, 2)), zorder=2)
    ax.text(thresh, 1.006, r"$|w|$ > 0.1", transform=ax.get_xaxis_transform(),
            fontsize=PT_SMALL, color="0.35", ha="center", va="bottom")

    ax.set_yscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("weight value" if SIGNED_LINEAR_X else r"$\log_{10}|w|$")
    ax.set_ylabel("count")
    ax.set_title(title, fontsize=PT_TITLE)
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
