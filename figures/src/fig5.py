"""Figure 5: weight distribution, and GONNECT vs GONNECT-SL preservation.

Three panels:

  a) Encoder weight-magnitude distribution, layers pooled: the GO-edge weights
     of original GONNECT, the GONNECT-SL weights at those same GO positions,
     the remaining soft links, and the fully connected MLP. Per-layer
     breakdowns and the decoder version are supplementary figures.
  b) Encoder, per layer: per-GO-term |Pearson r| of the activations (GONNECT vs
     GONNECT-SL, same seed) and, beside each, a box of the per-seed weight
     correlation (Spearman rho of |w|, sign ignored).
  c) Same as (b) for the decoder.

The activation box pools GO terms x 5 seeds; the weight box is 5 seeds (one
|w| Spearman per seed over that layer's biological GO edges). Seed dots are
overlaid on the weight boxes so the n=5 is explicit.

Terminology follows the manuscript: "original GONNECT" or "GONNECT" for the
fixed-link model, never the abbreviation FL, and "GO term" rather than "GO
node" for what a network node is coupled to.

Inputs, all under --data-dir:
  model_checkpoints/AE_2.0/AE_2.0.<seed>_both_model.pt   fixed-link checkpoints
  model_checkpoints/AE_2.0/AE_2.0.<seed>_none_model.pt   MLP (no GO prior)
  model_checkpoints/AE_2.1/AE_2.1.<seed>_both_model.pt   soft-link checkpoints
  hard_links.csv                                      GO edge positions
  activation_preservation_per_node.csv                per-node activation stats

Note: activation_preservation_per_node.csv is a *derived* table (columns
module, layer, abs_pearson_r among others). It is produced upstream by
fig_sl_preserve/activation_preservation.py from data/go_term_activations/, and is
treated purely as an input here.

Usage:
    python fig5.py [--data-dir figures/data] [--out-dir figures/out]
                   [--preservation-csv CSV]

--preservation-csv swaps in another per-node table for panels b and c, such as
the one prepare/activation_preservation.py builds from the relabelled decoder
activations (see prepare/README.md, *The untrained control*).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe: no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from scipy.stats import spearmanr

import _common
from _common import PT_BODY, PT_PANEL, PT_SMALL, figsize, pt

# ── Layout ───────────────────────────────────────────────────────────────────
# The width is fixed by the paper's shared standard (_common.FIG_WIDTH_IN), so
# only the height and the internal division are free. Margins are written out
# in inches and applied with subplots_adjust rather than left to tight_layout:
# save_figure keeps the full canvas width, so the row has to reach both edges.
MARGIN_L_IN = 0.82      # panel a: "count" + its 10^n tick labels + panel letter
MARGIN_R_IN = 0.06
MARGIN_T_IN = 0.68      # two-line panel titles, with the letters beside them
AXES_H_IN = 3.65        # the panels themselves
PANEL_GAP_IN = 0.85     # holds the y tick labels, axis label and panel letter
                        # of b and c; at 0.62 panel b's "correlation" sat on
                        # panel a's right edge

# Panel a is the wide one: it carries a log-scaled histogram of four series
# over eleven decades. b and c only need room for four boxplot pairs.
WIDTH_RATIOS = (1.20, 1.0, 1.0)

# Panel titles wrap onto two lines. Set on one line, "Encoder: activation vs
# weight preservation" is 5.5 in at PT_BODY, still wider than the 4.35 in
# panels b and c get, so their titles would run into each other.
TITLE_A = "Encoder weight distribution"
TITLE_B = "Encoder: activation vs\nweight preservation"
TITLE_C = "Decoder: activation vs\nweight preservation"

# ── Legend band ──────────────────────────────────────────────────────────────
# One key for the whole figure, in a band under all three panels: b and c share
# their two entries exactly, so a legend each said the same thing twice. It is
# drawn by hand rather than through ax.legend(), like figures 2 and 4, because
# matplotlib indents a legend title past the handles and the paper's key sets
# its headings bold and flush with the swatches under them.
#
# The band is too wide for one column and too tall for one row, so the four
# weight-distribution series are dealt over two columns (as figure 4 deals its
# model list) and the preservation pair takes the third. Column widths are the
# widest entry each has to hold at PT_SMALL, swatch included.
XLABEL_BAND_IN = 0.62   # tick labels and axis label under each panel
LEGEND_GAP_IN = 0.30    # axis label -> legend band
LEGEND_ROW_IN = 0.32
LEGEND_PAD_IN = 0.10
LEGEND_N_ROWS = 3       # heading + two entries
LEGEND_HEIGHT_IN = 2 * LEGEND_PAD_IN + LEGEND_N_ROWS * LEGEND_ROW_IN
LEGEND_EDGE_IN = 0.06   # last block -> canvas edge
LEGEND_COL_W_IN = [3.95,   # heading + GONNECT GO edges / GONNECT-SL at those
                   4.55,   # the remaining soft links / the MLP reference
                   4.10]   # heading + the two preservation boxes
# The first two columns are one block and are set close together; the third is
# a different block and goes to the far edge, so the wide gap between them is
# what separates the two groups.
LEGEND_COL_GAP_IN = 0.40
LEGEND_SWATCH_W_IN = 0.38
LEGEND_SWATCH_H_IN = 0.16
LEGEND_SWATCH_PAD_IN = 0.13
PAD_BOT_IN = 0.08

MARGIN_B_IN = (XLABEL_BAND_IN + LEGEND_GAP_IN + LEGEND_HEIGHT_IN + PAD_BOT_IN)
FIG_HEIGHT_IN = MARGIN_T_IN + AXES_H_IN + MARGIN_B_IN

SEEDS = [2, 3, 4, 5, 6]

# nn.Sequential indices carrying a weight matrix (ReLU sits on the odd ones).
# Panel a walks these directly; panels b/c go through _common.NET_IDX.
WEIGHTED = [0, 2, 4, 6]
CLIP = 1e-12
BINS = np.linspace(-10, 1, 111)

# (key, legend label, colour). The key is internal and stable; the label is what
# the reader sees and is the only part that should change when the manuscript's
# terminology does. They were one string until Reviewer 3 asked for consistent
# naming, at which point renaming a label silently renamed a dict key in
# gather() forty lines away -- hence the split.
#
# The labels spell the model names out as the manuscript does: it calls this
# model "original GONNECT" or "GONNECT" and never uses the abbreviation FL, and
# writes "fully connected" unhyphenated.
SERIES = [
    ("fl_go",    "GONNECT (GO edges)",      "#444444"),
    ("sl_at_go", "GONNECT-SL @ GO edges",   "#1f77b4"),
    ("sl_soft",  "GONNECT-SL soft links",   "#d62728"),
    ("mlp",      "MLP (fully connected)",   "#2ca02c"),
]
ACT_COLOR = "#6a9fd8"
W_COLOR = "#e8923c"


def load_sd(data_dir: Path, variant: str, seed: int, bi: str = "both") -> dict:
    """Load one checkpoint's state dict from <data-dir>/model_checkpoints/."""
    path = (data_dir / "model_checkpoints" / variant
            / f"{variant}.{seed}_{bi}_model.pt")
    return torch.load(path, map_location="cpu", weights_only=False)


def log10_abs(a: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(np.abs(a), CLIP, None))


def bio_positions(data_dir: Path) -> dict:
    """{(module, display_layer): (rows, cols)} for non-proxy (biological) edges."""
    hl = pd.read_csv(data_dir / "hard_links.csv",
                     usecols=["component", "layer", "source_index", "sink_index",
                              "source_term_id", "sink_term_id"])
    bio = hl[~hl.source_term_id.str.startswith("Proxy:")
             & ~hl.sink_term_id.str.startswith("Proxy:")]
    positions = {}
    for (component, layer), group in bio.groupby(["component", "layer"]):
        # Keep only groups gather() can map onto a weight matrix. Without this
        # an unexpected component or layer raises a bare KeyError there. This
        # is a no-op on the released hard_links.csv, which holds exactly
        # encoder L0-L3 and decoder L5-L8.
        mapped = _common.NET_IDX.get(component)
        if mapped is None or int(layer) not in mapped:
            continue
        positions[(component, int(layer))] = (group.sink_index.to_numpy(),
                                              group.source_index.to_numpy())
    return positions


def gather(data_dir: Path):
    """Single pass over checkpoints: encoder weight dist + per-seed |w| corr."""
    pos = bio_positions(data_dir)
    dist = {key: [] for key, _, _ in SERIES}                      # encoder, pooled
    wcorr = {k: [] for k in pos}                                  # per (mod,layer): 5 rhos
    for seed in SEEDS:
        fl = load_sd(data_dir, "AE_2.0", seed)
        sl = load_sd(data_dir, "AE_2.1", seed)
        fc = load_sd(data_dir, "AE_2.0", seed, "none")
        # panel a: encoder distribution
        for li in WEIGHTED:
            key = f"encoder.net_layers.{li}.weight"
            wf, ws = fl[key].numpy(), sl[key].numpy()
            mask = wf != 0
            dist["fl_go"].append(log10_abs(wf[mask]))
            dist["sl_at_go"].append(log10_abs(ws[mask]))
            dist["sl_soft"].append(log10_abs(ws[~mask]))
            dist["mlp"].append(log10_abs(fc[key].numpy().ravel()))
        # panels b/c: per-seed |w| Spearman over biological edges
        for (mod, layer), (r, c) in pos.items():
            key = f"{mod}.net_layers.{_common.NET_IDX[mod][layer]}.weight"
            wf = np.abs(fl[key].numpy()[r, c])
            ws = np.abs(sl[key].numpy()[r, c])
            wcorr[(mod, layer)].append(spearmanr(wf, ws)[0])
    dist = {k: np.concatenate(v) for k, v in dist.items()}
    return dist, wcorr


def _finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def act_boxes(df, module, layers):
    # Undefined Pearson r comes only from FL-dead / SL-alive neurons (var_fixed==0,
    # var_soft!=0) -- genuine non-preservation -- so set |r|=0 rather than drop
    # (dropping would bias preservation upward). Verified: no both-dead nodes.
    out = []
    for L in layers:
        v = np.array(df[(df.module == module) & (df.layer == L)].abs_pearson_r,
                     dtype=float)
        v[~np.isfinite(v)] = 0.0
        out.append(v)
    return out


def add_panel_letter(ax, letter):
    ax.text(-0.075, 1.015, letter, transform=ax.transAxes,
            fontsize=PT_PANEL, fontweight="bold", va="bottom", ha="left")


ACT_LABEL = "activation |Pearson r| (per GO term)"
W_LABEL = r"weight $\rho$ of $|w|$ (per seed)"


def draw_legend(ax, dist) -> None:
    """The figure's one key, as three blocks side by side under the panels.

    ``ax`` is invisible and its data coordinates are inches from its top-left
    corner, so each block sits at its measured width and every heading is flush
    with the swatches below it. The histogram series are drawn as open boxes
    because that is how ``histtype="step"`` draws them on the panel.
    """
    ax.axis("off")
    band_w = ax.get_xlim()[1]
    col_x = [0.0,
             LEGEND_COL_W_IN[0] + LEGEND_COL_GAP_IN,
             band_w - LEGEND_COL_W_IN[2]]

    def row_y(row: int) -> float:
        return LEGEND_PAD_IN + (row + 0.5) * LEGEND_ROW_IN

    def heading(col: int, text: str) -> None:
        ax.text(col_x[col], row_y(0), text, va="center", ha="left",
                fontsize=PT_BODY, fontweight="bold")

    def swatch(col: int, row: int, text: str, *, face, edge, lw, alpha=1.0):
        y = row_y(row)
        ax.add_patch(Rectangle(
            (col_x[col], y - LEGEND_SWATCH_H_IN / 2),
            LEGEND_SWATCH_W_IN, LEGEND_SWATCH_H_IN, facecolor=face,
            edgecolor=edge, linewidth=lw, alpha=alpha, clip_on=False))
        ax.text(col_x[col] + LEGEND_SWATCH_W_IN + LEGEND_SWATCH_PAD_IN, y, text,
                va="center", ha="left", fontsize=PT_SMALL)

    heading(0, "Weight distribution (a)")
    for i, (key, label, color) in enumerate(SERIES):
        col, row = divmod(i, 2)
        n = dist[key].size
        swatch(col, row + 1, f"{label} (n={n:,})",
               face="none", edge=color, lw=pt(1.0))

    heading(2, "Preservation (b, c)")
    swatch(2, 1, ACT_LABEL, face=ACT_COLOR, edge="none", lw=0.0, alpha=0.7)
    swatch(2, 2, W_LABEL, face=W_COLOR, edge="none", lw=0.0, alpha=0.7)


def plot_panel_a(ax, dist, letter):
    for key, _, color in SERIES:
        ax.hist(dist[key], bins=BINS, histtype="step", lw=pt(1.0), color=color)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}|w|$", fontsize=PT_BODY, labelpad=pt(1.5))
    ax.set_ylabel("count", fontsize=PT_BODY, labelpad=pt(1.5))
    ax.set_title(TITLE_A, fontsize=PT_BODY, fontweight="bold", pad=pt(3.0))
    ax.tick_params(labelsize=PT_SMALL, length=pt(2.0), pad=pt(1.5))
    ax.grid(alpha=0.25)
    add_panel_letter(ax, letter)


def plot_preservation(ax, df, wcorr, module, layers, title, letter):
    xs = np.arange(len(layers))
    off = 0.2
    act = act_boxes(df, module, layers)
    wts = [_finite(wcorr[(module, L)]) for L in layers]
    ba = ax.boxplot(act, positions=xs - off, widths=0.34, patch_artist=True,
                    showfliers=False, medianprops=dict(color="k", lw=pt(0.8)))
    for p in ba["boxes"]:
        p.set_facecolor(ACT_COLOR); p.set_alpha(0.7)
    bw = ax.boxplot(wts, positions=xs + off, widths=0.34, patch_artist=True,
                    showfliers=False, medianprops=dict(color="k", lw=pt(0.8)))
    for p in bw["boxes"]:
        p.set_facecolor(W_COLOR); p.set_alpha(0.7)
    # overlay the 5 seed dots on the weight boxes (n=5 is explicit)
    for x, w in zip(xs + off, wts):
        ax.scatter(np.full_like(w, x), w, s=pt(2.4) ** 2, color="k", alpha=0.75,
                   zorder=5)
    ax.axhline(0, color="grey", lw=pt(0.5), ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_xlabel("layer", fontsize=PT_BODY, labelpad=pt(1.5))
    ax.set_ylim(-0.15, 1.05)
    ax.set_ylabel("correlation", fontsize=PT_BODY, labelpad=pt(1.5))
    ax.set_title(title, fontsize=PT_BODY, fontweight="bold", pad=pt(3.0))
    ax.tick_params(labelsize=PT_SMALL, length=pt(2.0), pad=pt(1.5))
    ax.grid(alpha=0.25, axis="y")
    add_panel_letter(ax, letter)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 5: weight distribution and FL vs SL preservation.")
    _common.add_io_args(parser)
    parser.add_argument("--preservation-csv", type=Path, default=None,
                        help="per-node activation stats for panels b and c "
                             "(default: <data-dir>/activation_preservation_per_node.csv)")
    args = parser.parse_args()

    act_csv = args.preservation_csv or args.data_dir / "activation_preservation_per_node.csv"
    print(f"Reading {act_csv}")
    df = pd.read_csv(act_csv)
    print(f"Reading {len(SEEDS) * 3} checkpoints from "
          f"{args.data_dir / 'model_checkpoints'} ...")
    dist, wcorr = gather(args.data_dir)

    fig, axes = plt.subplots(1, 3, figsize=figsize(FIG_HEIGHT_IN),
                             gridspec_kw={"width_ratios": WIDTH_RATIOS})
    fig.subplots_adjust(
        left=MARGIN_L_IN / _common.FIG_WIDTH_IN,
        right=1.0 - MARGIN_R_IN / _common.FIG_WIDTH_IN,
        top=1.0 - MARGIN_T_IN / FIG_HEIGHT_IN,
        bottom=MARGIN_B_IN / FIG_HEIGHT_IN,
        wspace=PANEL_GAP_IN / ((_common.FIG_WIDTH_IN - MARGIN_L_IN - MARGIN_R_IN
                                - 2 * PANEL_GAP_IN) / 3),
    )
    plot_panel_a(axes[0], dist, "a")
    plot_preservation(axes[1], df, wcorr, "encoder", _common.ENC_LAYERS,
                      TITLE_B, "b")
    plot_preservation(axes[2], df, wcorr, "decoder", _common.DEC_LAYERS,
                      TITLE_C, "c")

    # One key for all three panels, in the band under them.
    legend_top_in = MARGIN_T_IN + AXES_H_IN + XLABEL_BAND_IN + LEGEND_GAP_IN
    band_w = _common.FIG_WIDTH_IN - MARGIN_L_IN - LEGEND_EDGE_IN
    ax_leg = fig.add_axes([
        MARGIN_L_IN / _common.FIG_WIDTH_IN,
        1.0 - (legend_top_in + LEGEND_HEIGHT_IN) / FIG_HEIGHT_IN,
        band_w / _common.FIG_WIDTH_IN,
        LEGEND_HEIGHT_IN / FIG_HEIGHT_IN,
    ])
    ax_leg.set_xlim(0.0, band_w)              # data units are inches,
    ax_leg.set_ylim(LEGEND_HEIGHT_IN, 0.0)    # y measured downwards
    draw_legend(ax_leg, dist)

    _common.report_text_overlaps(fig, "fig5")
    _common.save_figure(fig, args.out_dir, "fig5", dpi=150)


if __name__ == "__main__":
    main()
