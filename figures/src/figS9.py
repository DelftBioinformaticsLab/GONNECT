"""Supplementary Figure S9: GONNECT-SL soft-link weight stability across seeds.

A 2x3 figure whose rows are the two biologically-informed modules (encoder,
decoder). Every panel pools the AE_2.1.<seed> soft-link models, one seed per
input file:

  col 1  pairwise Spearman rho between seeds, computed on the soft-link weight
         magnitudes of all edges (edges sorted identically in every seed)
  col 2  KDE of log10(|w|) per layer, all seeds overlaid
  col 3  active soft-link consistency: the number of edges that are active
         (|w| > 0.1) in exactly k of the n seeds, stacked by layer

Inputs (relative to --data-dir)
------------------------------
    soft_link_weights/AE_2.1.<seed>_encoder_soft_links.csv.gz
    soft_link_weights/AE_2.1.<seed>_decoder_soft_links.csv.gz

Every file matching the glob is used; the seed is read from the file name, so
the panel labels and the "k / n seeds" axis follow whatever is present.

Usage
-----
    python figS9.py [--data-dir figures/data] [--out-dir figures/out]
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

from _common import (FIG_WIDTH_IN, PT_BODY, PT_PANEL, PT_SMALL, PT_SUPTITLE,
                     PT_TITLE, add_io_args, figsize, pt, report_text_overlaps,
                     save_figure)

# ── config ────────────────────────────────────────────────────────────────────
ACTIVE_THRESH = 0.1          # |w| > 0.1 => "active"
# The two biologically-informed autoencoder modules. (Not the enc/dec/both
# model naming that _common.MODULES uses.)
MODULES = ["encoder", "decoder"]

# ── layout ────────────────────────────────────────────────────────────────────
# Width is fixed by the paper's shared standard, so only the height and the
# column split are free. Both are written out in inches and handed to an
# explicit gridspec instead of tight_layout, because save_figure keeps the
# whole canvas width and the grid therefore has to reach both edges itself.
#
# The height is set by column 1: imshow keeps the rho matrix square, and its
# 5 x 5 cells have to stay wider than the "0.9873" they carry (0.68 in at
# PT_SMALL), which puts a floor of ~3.9 in under both the panel width and the
# row height.
FIG_HEIGHT_IN = 11.3

MARGIN_L_IN = 1.20      # bold row label (0.31) then the "seed n" ticks (0.65)
MARGIN_R_IN = 0.04
MARGIN_T_IN = 1.20      # suptitle, then the two-line panel titles
MARGIN_B_IN = 0.71      # the 45-degree seed labels of the bottom-left panel
ROW_GAP_IN = 1.45       # row 1's axis labels above row 2's two-line titles

# The two column gaps are far apart in what they have to hold, so they are set
# separately (a uniform wspace sized for the wider one would starve the middle
# panel of ~0.7 in). Gap 1 carries the colourbar's tick labels on its left and
# "density" plus its tick labels on its right; gap 2 only the latter pair.
# Panel titles and x-axis labels may overhang into both: they sit above and
# below the band the tick labels occupy, and the check below keeps neighbouring
# titles apart.
COL_GAP_IN = (1.68, 0.95)

# Column 1 fits the square rho matrix (its cells must stay wider than the
# "0.9873" they carry) plus the colourbar; column 3 fits its 4.4 in x-axis
# label without running off the canvas; column 2 takes what is left.
COL_WIDTHS_IN = (4.30, 3.83, 4.50)

# Title pad, shared so the three titles of a row sit on one line. Generous
# because column 3 writes the stack total just above each bar, and the tallest
# of those sits within a hair of the axes top once the 5% y-margin is applied.
TITLE_PAD_PT = pt(4.5)


# ── per-module computation ──────────────────────────────────────────────────--

def _compute_module(config: str, sl_dir: Path) -> dict:
    """Load all seeds for a module and compute everything the 3 panels need."""
    files = sorted(sl_dir.glob(f"AE_2.1.*_{config}_soft_links.csv.gz"))
    if not files:
        raise FileNotFoundError(
            f"No files matching AE_2.1.*_{config}_soft_links.csv.gz in {sl_dir}")
    seeds = [f.name.split(".")[2].split("_")[0] for f in files]
    n = len(seeds)
    print(f"\n=== {config}: seeds {seeds} ===", flush=True)

    weights:   dict[str, np.ndarray] = {}
    edge_keys: dict[str, set]        = {}   # (layer, src_idx, snk_idx)
    layer_col_ref: np.ndarray | None = None

    for f, seed in zip(files, seeds):
        print(f"  Loading seed {seed}...", flush=True)
        df = pd.read_csv(
            f,
            usecols=["layer_index", "source_index", "sink_index",
                     "weight_magnitude"],
        ).sort_values(["layer_index", "source_index", "sink_index"]).reset_index(drop=True)

        weights[seed] = df["weight_magnitude"].values
        if layer_col_ref is None:
            layer_col_ref = df["layer_index"].values

        active = df["weight_magnitude"].values > ACTIVE_THRESH
        edge_keys[seed] = set(zip(
            df.loc[active, "layer_index"].values,
            df.loc[active, "source_index"].values,
            df.loc[active, "sink_index"].values,
        ))
        print(f"    active links (|w|>{ACTIVE_THRESH}): {active.sum()}")

    layers = sorted(np.unique(layer_col_ref))

    # ── pairwise Spearman ─────────────────────────────────────────────────────
    print("  Computing pairwise Spearman...", flush=True)
    rho_mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = float(stats.spearmanr(weights[seeds[i]], weights[seeds[j]]).statistic)
            rho_mat[i, j] = rho_mat[j, i] = r
            print(f"    seed {seeds[i]} vs {seeds[j]}: rho = {r:.4f}")

    # ── active-link consistency (per seed count, broken down by layer) ────────
    all_active = set().union(*edge_keys.values())
    edge_count = {
        edge: sum(1 for s in seeds if edge in edge_keys[s])
        for edge in all_active
    }
    count_per_k_layer: dict[int, Counter] = {k: Counter() for k in range(1, n + 1)}
    for (layer, src, snk), k in edge_count.items():
        count_per_k_layer[k][layer] += 1

    return dict(
        config=config, seeds=seeds, n=n, weights=weights,
        layer_col_ref=layer_col_ref, layers=layers, rho_mat=rho_mat,
        count_per_k_layer=count_per_k_layer,
    )


# ── panel painters ────────────────────────────────────────────────────────────

def _panel_spearman(ax, d, show_ylabel):
    n, rho_mat = d["n"], d["rho_mat"]
    seed_labels = [f"seed {s}" for s in d["seeds"]]
    vmin = max(0.0, rho_mat[rho_mat < 1].min() - 0.01)
    im = ax.imshow(rho_mat, vmin=vmin, vmax=1.0, cmap="YlOrRd")
    ax.set_xticks(range(n))
    ax.set_xticklabels(seed_labels, rotation=45, ha="right", fontsize=PT_SMALL)
    ax.set_yticks(range(n))
    ax.set_yticklabels(seed_labels if show_ylabel else [""] * n, fontsize=PT_SMALL)
    ax.tick_params(length=pt(1.5), pad=pt(1.5))
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{rho_mat[i, j]:.4f}", ha="center", va="center",
                    fontsize=PT_SMALL,
                    color="white" if rho_mat[i, j] > (vmin + 1) / 2 else "black")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=PT_SMALL, length=pt(1.5), pad=pt(1.5))
    cb.outline.set_linewidth(plt.rcParams["axes.linewidth"])
    ax.set_title("Pairwise Spearman rho\n(weight magnitude, all edges)",
                 fontsize=PT_TITLE, pad=TITLE_PAD_PT)


def _panel_kde(ax, d, layer_colors):
    weights, layer_col_ref, layers = d["weights"], d["layer_col_ref"], d["layers"]
    for seed_idx, seed in enumerate(d["seeds"]):
        w_all = weights[seed]
        for layer, lcolor in zip(layers, layer_colors):
            mask = layer_col_ref == layer
            logw = np.log10(w_all[mask])
            kde = gaussian_kde(logw, bw_method=0.08)
            x_grid = np.linspace(np.percentile(logw, 0.1), np.percentile(logw, 99.9), 600)
            ax.plot(x_grid, kde(x_grid), color=lcolor, lw=pt(0.55), alpha=0.5,
                    label=f"layer {layer}" if seed_idx == 0 else None)
    ax.set_xlabel("log$_{10}$(|w|)", fontsize=PT_BODY, labelpad=pt(1.5))
    ax.set_ylabel("density", fontsize=PT_BODY, labelpad=pt(1.5))
    ax.tick_params(labelsize=PT_SMALL, length=pt(2.0), pad=pt(1.5))
    ax.set_title("Weight magnitude distribution per layer\n(all seeds overlaid)",
                 fontsize=PT_TITLE, pad=TITLE_PAD_PT)
    ax.legend(fontsize=PT_SMALL, handlelength=1.6, labelspacing=0.4)


def _panel_edge_consistency(ax, d, layer_colors):
    n = d["n"]
    layers, count_per_k_layer = d["layers"], d["count_per_k_layer"]
    k_vals = list(range(1, n + 1))
    bottoms = np.zeros(n)
    for layer, lcolor in zip(layers, layer_colors):
        heights = np.array([count_per_k_layer[k].get(layer, 0) for k in k_vals], dtype=float)
        ax.bar(k_vals, heights, bottom=bottoms, color=lcolor, label=f"layer {layer}", width=0.6)
        bottoms += heights
    for k, total in zip(k_vals, bottoms):
        if total > 0:
            ax.text(k, total + 0.3, str(int(total)), ha="center", va="bottom",
                    fontsize=PT_BODY)
    ax.set_xlabel(f"Seeds where edge is active (|w| > {ACTIVE_THRESH})",
                  fontsize=PT_BODY, labelpad=pt(1.5))
    ax.set_ylabel("Number of active edges", fontsize=PT_BODY, labelpad=pt(1.5))
    ax.set_xticks(k_vals)
    ax.set_xticklabels([f"{k}/{n}" for k in k_vals], fontsize=PT_SMALL)
    ax.tick_params(labelsize=PT_SMALL, length=pt(2.0), pad=pt(1.5))
    ax.set_title(f"Active soft-link consistency\n(|w| > {ACTIVE_THRESH}, {n} seeds)",
                 fontsize=PT_TITLE, pad=TITLE_PAD_PT)
    ax.legend(fontsize=PT_SMALL, handlelength=1.6, labelspacing=0.4)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_args(parser)
    args = parser.parse_args()

    sl_dir = args.data_dir / "soft_link_weights"
    data = [_compute_module(m, sl_dir) for m in MODULES]

    # Five grid columns, the even ones the panels and the odd ones the two
    # unequal gaps, so wspace can stay at zero and every width is an inch
    # figure taken from the block above.
    fig = plt.figure(figsize=figsize(FIG_HEIGHT_IN))
    axes_h_in = (FIG_HEIGHT_IN - MARGIN_T_IN - MARGIN_B_IN - ROW_GAP_IN) / 2
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[COL_WIDTHS_IN[0], COL_GAP_IN[0], COL_WIDTHS_IN[1],
                      COL_GAP_IN[1], COL_WIDTHS_IN[2]],
        left=MARGIN_L_IN / FIG_WIDTH_IN,
        right=1.0 - MARGIN_R_IN / FIG_WIDTH_IN,
        top=1.0 - MARGIN_T_IN / FIG_HEIGHT_IN,
        bottom=MARGIN_B_IN / FIG_HEIGHT_IN,
        wspace=0.0, hspace=ROW_GAP_IN / axes_h_in,
    )
    axes = np.array([[fig.add_subplot(gs[row, 2 * col]) for col in range(3)]
                     for row in range(2)])
    row_labels = {"encoder": "Encoder", "decoder": "Decoder"}

    for row, d in enumerate(data):
        layers = d["layers"]
        layer_colors = plt.cm.tab10(np.linspace(0, 0.35, len(layers)))

        _panel_spearman(axes[row, 0], d, show_ylabel=True)
        _panel_kde(axes[row, 1], d, layer_colors)
        _panel_edge_consistency(axes[row, 2], d, layer_colors)

        # Row label on the left edge. Anchored to the (empty) y-axis label,
        # which matplotlib places just outside the "seed n" tick labels; the
        # offset is what keeps it clear of them and is in canvas points, so it
        # scales with the type like everything else.
        axes[row, 0].annotate(
            f"{row_labels[d['config']]} ({d['n']} seeds)",
            xy=(0, 0.5), xytext=(-axes[row, 0].yaxis.labelpad - pt(6.0), 0),
            xycoords=axes[row, 0].yaxis.label, textcoords="offset points",
            ha="center", va="center", rotation=90, fontsize=PT_PANEL,
            fontweight="bold",
        )

    # Kept inside the canvas: save_figure trims to the canvas box, so a
    # suptitle placed above y=1 would be cropped off the top.
    fig.suptitle("GONNECT-SL soft-link weight stability across seeds",
                 fontsize=PT_SUPTITLE, y=1.0 - 0.10 / FIG_HEIGHT_IN, va="top")

    report_text_overlaps(fig, "figS9")
    save_figure(fig, args.out_dir, "figS9", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
