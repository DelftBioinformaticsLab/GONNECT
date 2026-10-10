"""Supplementary Figure S7: reconstruction MSE during the GONNECT-SL alpha sweep.

Two panels, one per biologically-informed module, each carrying six training
curves over the fixed 1,000-epoch schedule:

  a  GONNECT-SL encoder      b  GONNECT-SL decoder

The six curves are one ordered sequence, not six unrelated models. Methods puts
it plainly: a soft-link model at alpha = 0 is a fully connected MLP, and at
alpha = infinity it is the original ontology-derived GONNECT. So the series run
MLP -> 1e2 -> 1e3 -> 1e4 -> 1e5 -> GONNECT, from no constraint to full
constraint, and the panels are read in that order. "GONNECT" is the series the
manuscript calls original GONNECT; "fixed link" appears nowhere in the text, so
it is not used here either.

Inputs (relative to --data-dir)
------------------------------
    loss_traces/AE_3.-1/AE_3.-1.2_none_results.txt          MLP
    loss_traces/AE_3.-1/AE_3.-1.2_<module>_results.txt      GONNECT
    loss_traces/AE_3.0/AE_3.0.3_<module>_results.txt        alpha = 1e2
    loss_traces/AE_3.1/AE_3.1.3_<module>_results.txt        alpha = 1e3
    loss_traces/AE_3.2/AE_3.2.3_<module>_results.txt        alpha = 1e4
    loss_traces/AE_3.3/AE_3.3.3_<module>_results.txt        alpha = 1e5

Every run is a single instance at data-split seed 1, trained for a fixed 1,000
epochs with early stopping disabled (patience = 10000), unlike the models
reported in the main text. `experiment_log` lines 158-241 record the job IDs;
the trailing `.3` (`.2` for the baselines) is the full-length run, and lower
trailing numbers are earlier attempts that stopped early.

Which column is plotted
-----------------------
The last column of each file, which is not the same quantity in every file, and
that is a property of the deposited data rather than a choice made here:

  * The four alpha runs postdate the commit that added plain-MSE tracking, so
    they carry a 4th `MSE loss` column: the unregularized reconstruction MSE on
    the test split. Column 3 for these is the *regularized* objective, which
    reaches ~32 at alpha = 1e5 and is not comparable across alpha.
  * `AE_3.-1.2_none` predates it and has 3 columns, but it trained on plain
    `mse`, so its `Test loss` already is the plain test MSE.
  * `AE_3.-1.2_{encoder,decoder}` also have 3 columns and trained on
    `mse masked`, which zeroes the residuals of the 29-of-1000 genes carrying no
    GO annotation while still dividing by 1000. Their curves therefore sit below
    a plain MSE on the same weights -- measured from the saved checkpoints,
    0.23071 against 0.29414 (encoder) and 0.67214 against 0.70147 (decoder).
    Recovering a plain-MSE *trace* would need a re-run; the per-epoch history
    cannot be recomputed from a final checkpoint. The figure does not caption
    this; the manuscript caption is where it belongs, and
    prepare/alpha_sweep_numbers.py carries it per row as `mse_is_masked`.

Usage
-----
    python figS7.py [--data-dir figures/data] [--out-dir figures/out]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe: no display needed
import matplotlib.pyplot as plt

import _common
from _common import PT_PANEL, PT_SMALL, PT_TITLE, figsize, pt

# ── The sweep ────────────────────────────────────────────────────────────────
# (label, subdirectory, run stem, module-dependent?) in the order they are read,
# which is also legend order and the order of the underlying constraint.
# `module_dependent` is False for the MLP: one run serves both panels, because a
# fully connected autoencoder has no encoder/decoder variant to choose between.
SERIES = [
    ("MLP",             "AE_3.-1", "AE_3.-1.2", False),
    (r"$\alpha$ = 1e2", "AE_3.0",  "AE_3.0.3",  True),
    (r"$\alpha$ = 1e3", "AE_3.1",  "AE_3.1.3",  True),
    (r"$\alpha$ = 1e4", "AE_3.2",  "AE_3.2.3",  True),
    (r"$\alpha$ = 1e5", "AE_3.3",  "AE_3.3.3",  True),
    ("GONNECT",         "AE_3.-1", "AE_3.-1.2", True),
]

# (module, title, panel letter, direct-label the curves?). Only the decoder
# separates enough at epoch 1000 to be worth labelling -- see place_labels.
MODULES = [("encoder", "GONNECT-SL encoder", "a", False),
           ("decoder", "GONNECT-SL decoder", "b", True)]

# ── Colour ───────────────────────────────────────────────────────────────────
# Okabe-Ito, in an order verified to pass every check of the palette validator
# against a light surface, including the all-pairs CVD comparison. The paper's
# own Tableau-10 set does not: its red and green sit at deuteranopic dE 0.7,
# i.e. one colour for a red-green colourblind reader, and the preprint v3 version
# of this figure used matplotlib's tab10 defaults with the same defect. Three
# entries fall below 3:1 contrast on white, which the validator flags as
# requiring relief; the legend's visible labels and the dash patterns below
# supply it.
COLORS = ["#0072B2",  # MLP          blue
          "#E69F00",  # alpha = 1e2  orange
          "#009E73",  # alpha = 1e3  bluish green
          "#D55E00",  # alpha = 1e4  vermillion
          "#CC79A7",  # alpha = 1e5  reddish purple
          "#56B4E9"]  # GONNECT      sky blue

# Dash patterns carry the same six identities as the colours do. Under the
# all-pairs comparison the worst separation, alpha = 1e3 against alpha = 1e5,
# lands at deuteranopic dE 7.6 -- inside the 6-8 band the validator permits only
# alongside a secondary encoding, which is what this is. It also does the work
# the direct labels cannot in panel a, where five of the six curves converge
# into a band 0.04 wide and no leader could point at one unambiguously.
LINESTYLES = ["solid",
              (0, (5, 1.6)),
              (0, (1.4, 1.4)),
              (0, (6, 1.6, 1.4, 1.6)),
              (0, (3, 1.4, 1.4, 1.4, 1.4, 1.4)),
              (0, (2.6, 1.6))]

# ── Layout ───────────────────────────────────────────────────────────────────
# Width is fixed by the paper's shared standard (_common.FIG_WIDTH_IN), so only
# the height and the internal division are free. Margins are in inches and
# applied with subplots_adjust rather than tight_layout: save_figure keeps the
# full canvas width, so the row has to reach both edges itself.
FIG_HEIGHT_IN = 5.27

MARGIN_L_IN = 0.95      # "MSE", its tick labels, and the panel letter
MARGIN_R_IN = 1.60      # the right-margin direct labels of panel b
MARGIN_T_IN = 0.46      # panel titles, with the letters beside them
MARGIN_B_IN = 0.72      # "Epoch" and its tick labels
PANEL_GAP_IN = 1.15     # panel b's y axis label, ticks and panel letter. Panel
                        # a carries no right-margin labels, so the gap holds
                        # only panel b's own furniture.
AXES_H_IN = FIG_HEIGHT_IN - MARGIN_T_IN - MARGIN_B_IN

LW = pt(0.9)            # data lines, in rendered points
LABEL_X = 1.015         # direct-label anchor, in axes fractions past the right

# A z-scored input means MSE = 1 is the score for predicting every gene at its
# mean, so the curves start at 1 by construction and the axis is pinned there
# rather than left to the data. Shared by both panels: the whole point of the
# figure is that the decoder pays a price the encoder does not, which is only
# legible if the two panels are on one scale.
YLIM = (0.18, 1.03)


def read_trace(data_dir: Path, subdir: str, stem: str, module: str):
    """Return (epochs, mse) for one run.

    The plotted quantity is always the last column: `MSE loss` where the run is
    late enough to have one, `Test loss` otherwise. See the module docstring for
    why that is the right column in each case rather than a shortcut.
    """
    path = data_dir / "loss_traces" / subdir / f"{stem}_{module}_results.txt"
    frame = pd.read_csv(path, sep="\t")
    series = frame[frame.columns[-1]]
    return range(1, len(series) + 1), series.to_numpy()


def place_labels(ax, entries):
    """Lay direct labels down the right margin, joined to their curve by an elbow.

    Only called for the panel whose curves actually separate. Labelling a panel
    where they do not is worse than not labelling it: the leaders then fan out
    from one point, and a reader sees the fan as the curves themselves turning
    upward at the last epoch. Identity in that panel is carried by the dash
    patterns instead.

    `entries` is a list of (y, text, colour). The spacing is derived from the
    rendered text size and the axes height rather than hard-coded, so it follows
    a change to either.
    """
    span = YLIM[1] - YLIM[0]
    line_in = PT_SMALL / 72.0 * 1.45              # rendered height plus leading
    gap = span * line_in / AXES_H_IN

    # Sort by the value each label points at, then push apart from the bottom up
    # and settle any overshoot from the top down, so the block stays inside the
    # axes instead of walking off the top.
    ordered = sorted(entries, key=lambda e: e[0])
    ys = [e[0] for e in ordered]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    if ys[-1] > YLIM[1]:
        ys[-1] = YLIM[1]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)

    # An elbow rather than a straight leader: the run out from the axis and the
    # run into the label are both horizontal, so only the short middle segment
    # is diagonal and none of it can be mistaken for the curve continuing.
    transform = ax.get_yaxis_transform()          # x in axes fractions, y in data
    for (y_true, text, color), y in zip(ordered, ys):
        ax.plot([1.0, LABEL_X, LABEL_X + 0.014, LABEL_X + 0.022],
                [y_true, y_true, y, y], transform=transform, color=color,
                lw=LW * 0.55, alpha=0.75, clip_on=False,
                solid_capstyle="butt", zorder=2)
        ax.text(LABEL_X + 0.030, y, text, transform=transform, color=color,
                fontsize=PT_SMALL, va="center", ha="left", clip_on=False)


def plot_panel(ax, data_dir: Path, module: str, title: str, letter: str,
               label_curves: bool) -> None:
    """Draw one module's six curves."""
    labels = []
    for (label, subdir, stem, module_dependent), color, dash in zip(
            SERIES, COLORS, LINESTYLES):
        epochs, mse = read_trace(
            data_dir, subdir, stem, module if module_dependent else "none")
        ax.plot(epochs, mse, color=color, lw=LW, ls=dash, label=label, zorder=3)
        labels.append((float(mse[-1]), label, color))

    ax.set_xlim(0, len(list(epochs)))
    ax.set_ylim(*YLIM)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(title, fontsize=PT_TITLE)
    ax.grid(axis="y", color="0.9", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # The legend carries the reading order of the sweep, which the direct labels
    # cannot: they are sorted by final MSE, and in panel b that ordering happens
    # to match while in panel a it does not. Upper right is the one corner both
    # panels leave empty, the curves having dropped below 0.7 within 400 epochs.
    ax.legend(loc="upper right", fontsize=PT_SMALL, frameon=True,
              framealpha=0.9, edgecolor="0.8", borderpad=0.5,
              labelspacing=0.45, handlelength=2.6)

    if label_curves:
        place_labels(ax, labels)
    ax.text(-0.075, 1.015, letter, transform=ax.transAxes, fontsize=PT_PANEL,
            fontweight="bold", va="bottom", ha="left")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supplementary Figure S7: alpha-sweep training curves.")
    _common.add_io_args(parser)
    args = parser.parse_args()

    print(f"Reading {2 * len(SERIES)} loss traces from "
          f"{args.data_dir / 'loss_traces'} ...")

    fig, axes = plt.subplots(1, 2, figsize=figsize(FIG_HEIGHT_IN))
    axes_w = (_common.FIG_WIDTH_IN - MARGIN_L_IN - MARGIN_R_IN - PANEL_GAP_IN) / 2
    fig.subplots_adjust(
        left=MARGIN_L_IN / _common.FIG_WIDTH_IN,
        right=1.0 - MARGIN_R_IN / _common.FIG_WIDTH_IN,
        top=1.0 - MARGIN_T_IN / FIG_HEIGHT_IN,
        bottom=MARGIN_B_IN / FIG_HEIGHT_IN,
        wspace=PANEL_GAP_IN / axes_w,
    )

    for ax, (module, title, letter, label_curves) in zip(axes, MODULES):
        plot_panel(ax, args.data_dir, module, title, letter, label_curves)

    _common.report_text_overlaps(fig, "figS7")
    _common.save_figure(fig, args.out_dir, "figS7", dpi=200)


if __name__ == "__main__":
    main()
