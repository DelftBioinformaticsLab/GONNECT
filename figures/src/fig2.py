"""Paper Figure 2 — TCGA benchmark of GONNECT against the reference models.

Panels
  a-d  Test-set MSE / SS / ARI / NMI per model: mean +- SD over the five
       matched splits, with FDR-corrected paired t-tests against MLP. SS is
       taken against the true cancer types for every model.
  e-h  t-SNE of the sample embeddings: MLP (AE_2.0 `none`) and the GONNECT
       decoder / encoder / both variants, coloured by cancer type.
  i    MSE per cancer type (heatmap), test split: the per-type breakdown of a.
  j    SS per cancer type (heatmap), test split: the per-type breakdown of b.
  k    k-NN neighbourhood purity per cancer type, k=30 (heatmap): test samples,
       neighbours drawn from the training split, plus a chance-level column.
  l    Cancer-type abundance.

Only the t-SNE panels (e-h) use every sample; they are a picture of the
embedding, not a metric.

Layout: one shared 5-column grid, sized in inches (see the Layout block below)
so the label bands -- the rotated method names under the bars and under the
heatmaps, and the colourbar strip above the heatmaps -- get the room the
paper's type scale actually needs, rather than being squeezed by ratios.

Inputs, relative to --data-dir:
  metrics/metric_data_TCGA_1000_30_new.xlsx   MLP + GONNECT MSE
  metrics/test_split/gonnect_clustering.csv   MLP + GONNECT SS / ARI / NMI
  metrics/test_split/ontovae_rand.txt         OntoVAE baseline (rescored)
  metrics/test_split/vega_rand.txt            VEGA baselines (rescored)
  metrics/mse_per_cluster_TCGA_1000_30.xlsx   per-cancer-type MSE
  metrics/test_split/per_type_ss.csv          per-cancer-type SS
  metrics/test_split/per_type_purity_k30.csv  per-cancer-type purity + chance
  latent_embeddings/AE_2.0/AE_2.0.<seed>_<module>_full_dataset.pt
  TCGA_complete_bp_top1k.csv.gz                   cancer-type labels
  cache/tsne/                                     t-SNE cache
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from _common import (
    CANCER_COLORS,
    FIG_WIDTH_IN,
    MAIN_METHODS,
    METRIC_DIRECTION,
    METRIC_ORDER,
    PRIOR_GROUP_COLORS,
    PRIOR_METHOD_GROUP,
    PT_BODY,
    PT_PANEL,
    PT_SMALL,
    PT_TINY,
    PT_TITLE,
    PURITY_K_MAIN,
    add_io_args,
    figsize,
    compute_summary,
    display,
    load_cancer_types,
    load_embedding,
    paired_t_tests,
    read_per_cluster_workbook,
    read_baseline_ontovae,
    read_baseline_vega,
    read_xlsx_metrics,
    report_text_overlaps,
    save_figure,
    stars,
    tsne,
    tsne_cached,
    test_split_paths,
)

# ── Layout ───────────────────────────────────────────────────────────────────
# Everything here is inches on the saved canvas, which is always FIG_WIDTH_IN
# wide. Working in inches rather than in gridspec ratios is what keeps the
# label bands honest: at the paper's type scale a rotated "GONNECT-SL-both" is
# 2.06 in tall under the bars and 1.76 in under the heatmaps, and those two
# bands plus the colourbar strip are what actually set the figure height.
MARGIN_LEFT_IN = 0.55    # y-tick labels of panel a
MARGIN_RIGHT_IN = 0.08
MARGIN_TOP_IN = 0.52     # panel letters a-d sit above the bar titles
MARGIN_BOTTOM_IN = 1.95  # rotated method labels under the heatmaps

# Cols 0-1: t-SNE block / bars a-b. Cols 2-3: the three heatmaps i-k, which
# share exactly the span the two heatmaps used to have. Col 4: legend (top)
# and abundance (bottom); it is sized by the legend, whose widest entry is
# 2.5 in and which therefore cannot be squeezed further.
COL_RATIOS = [1.0, 1.0, 1.0, 1.0, 0.92]
# The column gap has three jobs: keep the significance star over the last bar
# of one panel off the y-tick labels of the next, keep the heatmap titles
# apart from the bars' label band, and leave the cancer-type names of panel i
# somewhere to sit.
COL_GAP_IN = 0.55

ROW_BAR_IN = 2.75        # metric bars
ROW_MAIN_IN = 7.20       # t-SNE block / heatmaps / abundance
# Rotated bar labels ("GONNECT-SL-both" is 2.06 in tall plus its tick pad),
# then clear air, then the two-line heatmap titles and the colourbar strip.
ROW_GAP_IN = 4.05

FIG_HEIGHT_IN = (MARGIN_TOP_IN + ROW_BAR_IN + ROW_GAP_IN + ROW_MAIN_IN
                 + MARGIN_BOTTOM_IN)          # 16.47 in

# Within the t-SNE block.
EMB_GAP_X_IN = 0.20
EMB_GAP_Y_IN = 0.58      # fits the panel letter of the lower row

# The heatmap block, in inches across the span of cols 2-3:
# [names, i, gap, j, gap, k, pad]. The gap is what keeps two neighbouring
# titles apart, so it is spent on white space rather than on the heatmaps;
# what is left over sets the heatmap width, and at three heatmaps that width
# is close to the 7 x 0.19 in the rotated method labels need underneath.
HEATMAP_LABEL_PAD_IN = 0.78   # cancer-type swatches + names, left of panel i
HEATMAP_GAP_IN = 0.34
HEATMAP_RIGHT_PAD_IN = 0.05
# Swatches and names, in inches from the left edge of the heatmap they belong
# to. Inches rather than axes fractions because the heatmaps are now narrow
# enough that a fraction of their width is no longer a useful unit.
HEATMAP_SWATCH_X_IN = -0.74   # left edge of the colour block
HEATMAP_SWATCH_W_IN = 0.09
HEATMAP_SWATCH_PAD_IN = 0.04  # swatch -> name
HEATMAP_LETTER_X_IN = -0.92   # panel letter of the heatmap carrying the names
# The other two letters sit in the gap between neighbouring heatmaps, so they
# are set close to the panel they belong to and well clear of the one before.
HEATMAP_LETTER_BARE_X_IN = -0.08

# Colourbar strip above each heatmap, and the title clearing it. The bar is
# shorter than the heatmap and pushed to its right edge, which is what buys the
# panel letters of j and k somewhere to sit: at full width the bar starts
# exactly where the letter is, and the two touch.
CBAR_GAP_IN = 0.30
CBAR_HEIGHT_IN = 0.17
CBAR_WIDTH_FRAC = 0.80
HEATMAP_TITLE_PAD_IN = 0.92

# Legend block (top of col 4), laid out by hand: heading, four colour swatches,
# heading, three significance rows. Wider than the old matplotlib legend, so it
# runs past the bottom of the bar row into the label band beside it.
LEGEND_ROW_IN = 0.30
LEGEND_PAD_IN = 0.10
LEGEND_BLOCK_GAP_IN = 0.20    # extra air above the second heading
LEGEND_SWATCH_W_IN = 0.30
LEGEND_SWATCH_H_IN = 0.15
LEGEND_SWATCH_PAD_IN = 0.10   # swatch -> label, and star -> p-value
LEGEND_N_ROWS = 9
LEGEND_HEIGHT_IN = (2 * LEGEND_PAD_IN + LEGEND_N_ROWS * LEGEND_ROW_IN
                    + LEGEND_BLOCK_GAP_IN)


def _grid_space(total_in: float, gap_in: float, n: int) -> float:
    """GridSpec ``wspace``/``hspace`` giving cells separated by ``gap_in``.

    GridSpec expresses the separator as a fraction of the *mean* cell size,
    which for n cells inside ``total_in`` is ``(total - (n-1)*gap) / n``.
    """
    return gap_in * n / (total_in - (n - 1) * gap_in)


def _cell_sizes(total_in: float, gap_in: float, ratios: List[float]) -> List[float]:
    """Absolute size in inches of each cell of such a grid."""
    scale = (total_in - (len(ratios) - 1) * gap_in) / sum(ratios)
    return [r * scale for r in ratios]


# ── Method config (bars) ─────────────────────────────────────────────────────
# The ten models of MAIN_METHODS: 4 reference + 3 fixed-link + 3 soft-link.
METHOD_DISPLAY = {m: display(m) for m in MAIN_METHODS}

DISPLAY_GROUPS = [
    ["MLP"],
    ["ontovae"],
    ["vega_hallmark", "vega_reactome674"],
    ["GONNECT-enc",    "GONNECT-dec",    "GONNECT-both"],
    ["GONNECT-SL-enc", "GONNECT-SL-dec", "GONNECT-SL-both"],
]
# Gap between display groups, in bar widths, plus the padding left and right of
# the outermost bars. Both are kept small: the rotated method labels are one
# bar pitch apart, and at the paper's type scale a label is 0.23 in tall, so
# every bar pitch spent on white space is a pitch the labels do not get.
DISPLAY_GROUP_GAP = 0.35
# 0.65 puts 0.25 bar widths (0.06 in) between the axes spine and the outermost
# bar, 2.5x what 0.5 gave. It is paid for out of the bar pitch, and so out of
# the gap between two rotated labels: that drops from 0.014 to 0.008 in, which
# is 0.08 mm once the figure is placed, and is as far as this can be pushed.
BAR_XLIM_PAD = 0.65

# Reference for the dashed line, the outlined bar, and the paired t-tests.
BASELINE_METHOD = "MLP"

# Heatmaps show the 7 methods that are present in mse_per_cluster_TCGA_1000_30.xlsx;
# _common.FIG2_METHODS is the same set, and prepare/test_split_metrics.py writes
# the SS and purity tables with the same columns.
HEATMAP_METHOD_ORDER = [
    "MLP",
    "GONNECT-enc",    "GONNECT-dec",    "GONNECT-both",
    "GONNECT-SL-enc", "GONNECT-SL-dec", "GONNECT-SL-both",
]
# Panel k's extra column: chance-level purity, the cancer type's share of the
# training split the neighbours are drawn from.
PURITY_CHANCE = "Random"

# ── Embedding panels (t-SNE) ─────────────────────────────────────────────────
# Titles use the same method names as the x-tick labels in panels a-d.
EMBEDDING_PANELS = [
    ("none",    display("MLP")),
    ("decoder", display("GONNECT-dec")),
    ("encoder", display("GONNECT-enc")),
    ("both",    display("GONNECT-both")),
]
EMB_VERSION = "2.0"

# Populated dynamically in main() once cancer_types_sorted (by abundance) is known.
CANCER_COLOR_MAP: Dict[str, np.ndarray] = {}

# A run that did not converge, marked grey with an "x" so that it cannot flatten
# the colour scale for everything else. No cell trips this any more: the 12 that
# used to were Excel's locale coercion, not diverged models, and are decoded on
# read (see _common._decoerce). Kept as a guard in case a real one ever appears.
MSE_DIVERGED_THRESHOLD = 10.0


# ── Drawing ──────────────────────────────────────────────────────────────────
def _bar_x_positions(methods: List[str]) -> np.ndarray:
    x_map: Dict[str, float] = {}
    pos = 0.0
    for gi, grp in enumerate(DISPLAY_GROUPS):
        if gi > 0:
            pos += DISPLAY_GROUP_GAP
        for m in grp:
            if m in methods:
                x_map[m] = pos
                pos += 1.0
    # Append any methods not in DISPLAY_GROUPS at the end.
    for m in methods:
        if m not in x_map:
            pos += DISPLAY_GROUP_GAP
            x_map[m] = pos
            pos += 1.0
    return np.array([x_map[m] for m in methods])


def draw_metric_bar(ax, summary: pd.DataFrame, metric: str, stats_table: pd.DataFrame) -> None:
    sub = summary[summary["metric"] == metric]
    methods = [m for m in MAIN_METHODS if m in sub["method"].values]
    x = _bar_x_positions(methods)
    means, stds, colors, labels = [], [], [], []
    for m in methods:
        row = sub[sub["method"] == m]
        means.append(row["mean"].iloc[0] if not row.empty else np.nan)
        stds.append(row["std"].iloc[0] if not row.empty else np.nan)
        colors.append(PRIOR_GROUP_COLORS.get(PRIOR_METHOD_GROUP.get(m, ""), "#888888"))
        labels.append(METHOD_DISPLAY.get(m, m))

    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=3)
    for m, bar in zip(methods, bars):
        if m == BASELINE_METHOD:
            bar.set_edgecolor("#212121")
            bar.set_linewidth(1.5)

    baseline_val = sub[sub["method"] == BASELINE_METHOD]["mean"]
    if not baseline_val.empty:
        ax.axhline(float(baseline_val.iloc[0]), color="#9E9E9E",
                   linestyle="--", linewidth=0.8)

    direction = METRIC_DIRECTION.get(metric, "up")
    arrow = "↓" if direction == "down" else "↑"
    ax.set_title(f"{metric} [{arrow}]", fontsize=PT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=PT_BODY)
    ax.set_xlim(x.min() - BAR_XLIM_PAD, x.max() + BAR_XLIM_PAD)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="y", labelsize=PT_BODY)

    ms = stats_table[stats_table["metric"] == metric] if stats_table is not None else pd.DataFrame()
    if not ms.empty:
        valid = [v for v in means if not np.isnan(v)]
        if valid:
            y_max = max(valid)
            y_span = max(y_max, 1e-6)
            tops = [y_max]
            for m, bar, mean, std in zip(methods, bars, means, stds):
                if m == BASELINE_METHOD:
                    continue
                row = ms[ms["method"] == m]
                if row.empty:
                    continue
                star = stars(float(row["p_adj"].iloc[0]))
                if not star:
                    continue
                h = (mean or 0.0) + (std if not np.isnan(std) else 0.0)
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03 * y_span,
                        star, ha="center", va="bottom", fontsize=PT_BODY)
                tops.append(h)
            # A star is ~0.23 in tall, roughly a tenth of the bar panel, so the
            # headroom above the tallest bar+SD has to be that much again --
            # the old 0.18 of the mean was tuned for half the type size.
            top = max(tops) + 0.16 * y_span
            ax.set_ylim(top=top)
            # The locator rounds outwards, so it can hand back a tick above the
            # new top. That tick is never drawn, but its label artist is still
            # positioned and would show up as a phantom overlap.
            ax.set_yticks([t for t in ax.get_yticks()
                           if ax.get_ylim()[0] <= t <= top])


def draw_embedding(ax, xy: np.ndarray, labels: np.ndarray, title: str, panel_letter: str) -> None:
    for ct in CANCER_COLOR_MAP:
        mask = labels == ct
        if not mask.any():
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1], c=[CANCER_COLOR_MAP[ct]],
                   s=2.0, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.set_title(title, fontsize=PT_TITLE, pad=3)
    ax.text(-0.02, 1.04, panel_letter, transform=ax.transAxes,
            fontsize=PT_PANEL, fontweight="bold", va="bottom", ha="left")


def _axes_width_in(ax) -> float:
    """Width of ``ax`` in inches on the canvas."""
    return ax.get_position().width * ax.figure.get_size_inches()[0]


def heatmap_letter(ax, letter: str, x_in: float) -> None:
    """Panel letter placed ``x_in`` inches left of the heatmap's left edge."""
    ax.text(x_in / _axes_width_in(ax), 1.02, letter, transform=ax.transAxes,
            fontsize=PT_PANEL, fontweight="bold", va="bottom", ha="left")


def draw_heatmap(
    ax, data: np.ndarray, cancer_types: List[str], method_labels: List[str],
    cmap: str, vmin: float, vmax: float, title: str,
    yticklabel_mode: str = "right",  # "right" | "swatch" | "hidden"
    diverged_mask: Optional[np.ndarray] = None,
) -> "plt.cm.ScalarMappable":
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if diverged_mask is not None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if diverged_mask[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               linewidth=0, facecolor="#CCCCCC"))
                    ax.text(j, i, "x", ha="center", va="center", fontsize=PT_TINY,
                            color="#666666")
    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels(method_labels, rotation=90, ha="center", fontsize=PT_SMALL)
    ax.set_yticks(range(len(cancer_types)))
    if yticklabel_mode == "right":
        ax.set_yticklabels(cancer_types, fontsize=PT_SMALL)
    elif yticklabel_mode == "swatch":
        # Hide default labels; draw colored block + left-aligned name to the left of the axis.
        ax.set_yticklabels([])
        ax.tick_params(left=False)
        yt = ax.get_yaxis_transform()  # x in axes coords, y in data coords
        w_in = _axes_width_in(ax)
        swatch_x = HEATMAP_SWATCH_X_IN / w_in
        swatch_w = HEATMAP_SWATCH_W_IN / w_in
        text_x = (HEATMAP_SWATCH_X_IN + HEATMAP_SWATCH_W_IN
                  + HEATMAP_SWATCH_PAD_IN) / w_in
        for i, ct in enumerate(cancer_types):
            ax.add_patch(plt.Rectangle(
                (swatch_x, i - 0.42), swatch_w, 0.84,
                facecolor=CANCER_COLOR_MAP.get(ct, "#888888"),
                edgecolor="none", transform=yt, clip_on=False,
            ))
            ax.text(text_x, i, ct, ha="left", va="center", fontsize=PT_SMALL,
                    transform=yt, clip_on=False)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=PT_TITLE, pad=HEATMAP_TITLE_PAD_IN * 72)
    ax.tick_params(bottom=False, left=False)
    return im


def add_top_colorbar(fig, ax_parent, im) -> None:
    """Horizontal colourbar in the strip above ``ax_parent``.

    Placed in inches rather than as a percentage of the parent axes: the strip
    has to clear the parent by a fixed amount and leave room for tick labels of
    a fixed height, neither of which scales with how tall the heatmap is.

    The bar spans CBAR_WIDTH_FRAC of the heatmap and is flush with its right
    edge, so the space it gives up is on the left -- which is where the panel
    letter is.
    """
    pos = ax_parent.get_position()
    fig_h = fig.get_size_inches()[1]
    cb_w = CBAR_WIDTH_FRAC * pos.width
    cb_ax = fig.add_axes([
        pos.x1 - cb_w, pos.y1 + CBAR_GAP_IN / fig_h, cb_w, CBAR_HEIGHT_IN / fig_h,
    ])
    cb = plt.colorbar(im, cax=cb_ax, orientation="horizontal")
    # The heatmaps are ~1.9 in wide and a tick label is ~0.5 in, so the default
    # tick count would run the labels into each other.
    cb.locator = MaxNLocator(nbins=3)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=PT_SMALL, pad=1)
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")


def draw_abundance(ax, cancer_types: List[str], abundance: pd.Series) -> None:
    n = len(cancer_types)
    bar_colors = plt.cm.Blues(np.linspace(0.4, 0.85, n))
    ax.barh(range(n), abundance.loc[cancer_types].values, color=bar_colors,
            height=0.8, edgecolor="none")
    ax.set_yticks([])
    ax.set_xlabel("Abundance [n]", fontsize=PT_BODY, labelpad=2)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(left=False, bottom=True, labelsize=PT_SMALL)
    ax.spines[["top", "right"]].set_visible(False)


LEGEND_METHOD_ROWS = [
    ("Reference",   "MLP / OntoVAE / VEGA"),
    ("Enc prior",   "GONNECT enc"),
    ("Dec prior",   "GONNECT dec"),
    ("Both priors", "GONNECT both"),
]
LEGEND_SIG_ROWS = [("*", "p<=0.05"), ("**", "p<=0.01"), ("***", "p<=0.001")]


def draw_method_legend(ax) -> None:
    """The key, drawn by hand in the same style as figure 4's.

    ``ax`` is invisible and its data coordinates are inches measured from its
    top-left corner, which is what lets the two blocks share one left edge:
    swatches and significance markers at x = 0, their labels and p-values at
    one common indent, so the three star rows read as a small table.

    One column, not five like figure 4's: it sits beside the bars rather than
    under them, and "MLP / OntoVAE / VEGA" already fills the column's 2.5 in.
    """
    ax.axis("off")
    label_x = LEGEND_SWATCH_W_IN + LEGEND_SWATCH_PAD_IN
    y = LEGEND_PAD_IN + LEGEND_ROW_IN / 2

    def heading(text: str) -> None:
        ax.text(0.0, y, text, va="center", ha="left",
                fontsize=PT_BODY, fontweight="bold")

    heading("Method")
    for group, label in LEGEND_METHOD_ROWS:
        y += LEGEND_ROW_IN
        ax.add_patch(mpatches.Rectangle(
            (0.0, y - LEGEND_SWATCH_H_IN / 2),
            LEGEND_SWATCH_W_IN, LEGEND_SWATCH_H_IN,
            facecolor=PRIOR_GROUP_COLORS[group], edgecolor="none", clip_on=False,
        ))
        ax.text(label_x, y, label, va="center", ha="left", fontsize=PT_SMALL)

    y += LEGEND_ROW_IN + LEGEND_BLOCK_GAP_IN
    heading("Significance (FDR p)")
    for marker, p_text in LEGEND_SIG_ROWS:
        y += LEGEND_ROW_IN
        ax.text(0.0, y, marker, va="center", ha="left", fontsize=PT_SMALL)
        ax.text(label_x, y, p_text, va="center", ha="left", fontsize=PT_SMALL)


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Figure 2 (bars + embedding panels + per-cancer heatmaps)."
    )
    add_io_args(p)
    p.add_argument("--seed", type=int, default=2,
                   help="model seed / run used for the embedding panels (default: 2)")
    p.add_argument("--recompute-tsne", action="store_true",
                   help="ignore the cached t-SNE coordinates and recompute them")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metric_file = args.data_dir / "metrics" / "metric_data_TCGA_1000_30_new.xlsx"
    per_ct_file = args.data_dir / "metrics" / "mse_per_cluster_TCGA_1000_30.xlsx"
    tcga_file = args.data_dir / "TCGA_complete_bp_top1k.csv.gz"
    emb_dir = args.data_dir / "latent_embeddings"
    cache_dir = args.data_dir / "cache"
    tsne_cache_dir = cache_dir / "tsne"

    # 1. Metric bars data, all on the test split: MSE for the GONNECT family
    #    from the xlsx, its SS / ARI / NMI from metrics/test_split/, and every
    #    OntoVAE / VEGA metric from the rescored arm files beside it. The xlsx's
    #    own SS / ARI / NMI span all samples, so they are dropped here.
    print("Loading metric data ...")
    paths = test_split_paths(args.data_dir)
    for key in ("gonnect", "ss_per_type", "purity_per_type"):
        if not paths[key].exists():
            raise SystemExit(f"missing {paths[key]}; build it with prepare/test_split_metrics.py")
    ontovae_path, vega_path = paths["ontovae"], paths["vega"]
    workbook = read_xlsx_metrics(metric_file, strict_repeats=True)
    frames = [workbook[workbook["metric"] == "MSE"],
              pd.read_csv(paths["gonnect"], usecols=["metric", "method", "repeat", "value"])]
    if ontovae_path.exists():
        frames.append(read_baseline_ontovae(ontovae_path))
    else:
        print(f"  WARNING: missing {ontovae_path}; OntoVAE bars will be skipped.")
    if vega_path.exists():
        frames.append(read_baseline_vega(vega_path))
    else:
        print(f"  WARNING: missing {vega_path}; VEGA bars will be skipped.")
    data = pd.concat(frames, ignore_index=True)
    data = data[data["method"].isin(MAIN_METHODS)].reset_index(drop=True)
    summary = compute_summary(data)
    stats_table = paired_t_tests(data, baseline=BASELINE_METHOD)
    if not stats_table.empty:
        print(stats_table.to_string(index=False))

    # 2. Per-cancer-type heatmap data, all on held-out samples: MSE from the
    # workbook, read through _common so its locale-coerced cells are decoded on
    # the way in (see _common._decoerce); SS and purity from metrics/test_split/.
    # The workbook's own SS sheet spans all samples, so it is not used.
    print("Loading per-cancer-type data ...")
    mse_df = read_per_cluster_workbook(per_ct_file)["MSE"]
    ss_df  = pd.read_csv(paths["ss_per_type"], index_col=0)
    heatmap_methods = [m for m in HEATMAP_METHOD_ORDER if m in mse_df.columns]
    mse_df = mse_df[heatmap_methods]
    ss_df  = ss_df[heatmap_methods]

    # 3. Cancer-type labels + abundance
    print("Loading TCGA labels ...")
    cancer_labels = load_cancer_types(tcga_file)
    abundance = cancer_labels.value_counts().rename("Abundance")
    cancer_types_sorted = [c for c in abundance.index if c in mse_df.index]
    mse_df    = mse_df.loc[cancer_types_sorted]
    ss_df     = ss_df.loc[cancer_types_sorted]
    abundance = abundance.loc[cancer_types_sorted]

    # Cancer-type color assignment, by abundance order (matches heatmap rows + panels e-h).
    CANCER_COLOR_MAP.clear()
    for i, ct in enumerate(cancer_types_sorted):
        CANCER_COLOR_MAP[ct] = CANCER_COLORS[i % len(CANCER_COLORS)]

    mse_vals = mse_df.values.astype(float)
    mse_diverged = mse_vals > MSE_DIVERGED_THRESHOLD
    mse_display  = np.where(mse_diverged, np.nan, mse_vals)
    ss_vals = ss_df.values.astype(float)

    # 4. k-NN purity per cancer type: test samples, neighbours from the training
    # split. Lines up column for column with i and j, plus the chance-level
    # column after them.
    print("Loading k-NN purity ...")
    purity_methods = heatmap_methods + [PURITY_CHANCE]
    purity_df = pd.read_csv(paths["purity_per_type"], index_col=0)
    purity_vals = purity_df.loc[cancer_types_sorted, purity_methods].values.astype(float)

    # 5. t-SNE for the four embedding panels
    print("Loading / computing t-SNEs ...")
    tsne_xy: Dict[str, np.ndarray] = {}
    for module, _ in EMBEDDING_PANELS:
        tsne_xy[module] = tsne_cached(
            tsne_cache_dir,
            f"AE_{EMB_VERSION}.{args.seed}_{module}",
            lambda m=module: tsne(load_embedding(emb_dir, EMB_VERSION, args.seed, m)),
            args.recompute_tsne,
        )
    sample_labels = cancer_labels.values
    n_emb_rows = next(iter(tsne_xy.values())).shape[0]
    if n_emb_rows != len(sample_labels):
        raise SystemExit(
            f"Embedding rows ({n_emb_rows}) != TCGA rows ({len(sample_labels)}); mismatch."
        )

    # 6. Build figure — single shared 5-column grid; tight spacing.
    # Cols 1-2: embedding panels (each bar above maps onto one embedding column).
    # Cols 3-4: the three per-cancer-type heatmaps (panels i-k), which share the
    #           span of those two columns.
    # Col 5:   abundance bar (panel l) / method legend.
    print("Drawing figure ...")
    fig = plt.figure(figsize=figsize(FIG_HEIGHT_IN))
    grid_w = FIG_WIDTH_IN - MARGIN_LEFT_IN - MARGIN_RIGHT_IN
    grid_h = ROW_BAR_IN + ROW_GAP_IN + ROW_MAIN_IN
    col_w = _cell_sizes(grid_w, COL_GAP_IN, COL_RATIOS)
    outer = fig.add_gridspec(
        2, 5,
        left=MARGIN_LEFT_IN / FIG_WIDTH_IN,
        right=1.0 - MARGIN_RIGHT_IN / FIG_WIDTH_IN,
        bottom=MARGIN_BOTTOM_IN / FIG_HEIGHT_IN,
        top=1.0 - MARGIN_TOP_IN / FIG_HEIGHT_IN,
        height_ratios=[ROW_BAR_IN, ROW_MAIN_IN],
        width_ratios=COL_RATIOS,
        hspace=_grid_space(grid_h, ROW_GAP_IN, 2),
        wspace=_grid_space(grid_w, COL_GAP_IN, len(COL_RATIOS)),
    )

    # ── Top row: 4 metric bars in cols 0-3, method legend in col 4 ─────────
    for i, metric in enumerate(METRIC_ORDER):
        ax = fig.add_subplot(outer[0, i])
        draw_metric_bar(ax, summary, metric, stats_table)
        ax.text(-0.10, 1.05, "abcd"[i], transform=ax.transAxes,
                fontsize=PT_PANEL, fontweight="bold", va="bottom", ha="left")
    # The legend is placed in inches rather than in outer[0, 4]: it is taller
    # than the bar row and simply runs on into the label band beside panel d.
    legend_x0 = MARGIN_LEFT_IN + sum(col_w[:4]) + 4 * COL_GAP_IN
    ax_method_legend = fig.add_axes([
        legend_x0 / FIG_WIDTH_IN,
        1.0 - (MARGIN_TOP_IN + LEGEND_HEIGHT_IN) / FIG_HEIGHT_IN,
        col_w[4] / FIG_WIDTH_IN,
        LEGEND_HEIGHT_IN / FIG_HEIGHT_IN,
    ])
    ax_method_legend.set_xlim(0.0, col_w[4])       # data units are inches,
    ax_method_legend.set_ylim(LEGEND_HEIGHT_IN, 0.0)   # y measured downwards
    draw_method_legend(ax_method_legend)

    # ── Bottom row ─────────────────────────────────────────────────────────
    # Embedding block: 2x2 across cols 0-1 of bottom row.
    emb_block_w = col_w[0] + COL_GAP_IN + col_w[1]
    emb_gs = outer[1, 0:2].subgridspec(
        2, 2,
        wspace=_grid_space(emb_block_w, EMB_GAP_X_IN, 2),
        hspace=_grid_space(ROW_MAIN_IN, EMB_GAP_Y_IN, 2),
    )
    emb_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    emb_letters = ["e", "f", "g", "h"]
    for (module, title), (row, col), letter in zip(EMBEDDING_PANELS, emb_positions, emb_letters):
        ax = fig.add_subplot(emb_gs[row, col])
        draw_embedding(ax, tsne_xy[module], sample_labels, title, letter)

    # Heatmaps i-k, in one sub-grid over the span of cols 2-3:
    # [names, i, gap, j, gap, k, pad], all in inches, laid out with wspace=0 so
    # the ratios are read as widths. The cancer-type names are drawn outside
    # panel i with clip_on=False and live in the first cell. Each panel is as
    # wide as its column count, so k's chance column costs no cell width.
    hm_span = col_w[2] + COL_GAP_IN + col_w[3]
    n_cols = [len(heatmap_methods), len(heatmap_methods), len(purity_methods)]
    cell_w = (hm_span - HEATMAP_LABEL_PAD_IN - HEATMAP_RIGHT_PAD_IN
              - 2 * HEATMAP_GAP_IN) / sum(n_cols)
    hm_gs = outer[1, 2:4].subgridspec(
        1, 7, wspace=0.0,
        width_ratios=[HEATMAP_LABEL_PAD_IN, n_cols[0] * cell_w, HEATMAP_GAP_IN,
                      n_cols[1] * cell_w, HEATMAP_GAP_IN, n_cols[2] * cell_w,
                      HEATMAP_RIGHT_PAD_IN],
    )

    # MSE heatmap (panel i): the one carrying the cancer-type swatches + names.
    ax_mse_ct = fig.add_subplot(hm_gs[0, 1])
    im_mse = draw_heatmap(
        ax_mse_ct, mse_display, cancer_types_sorted, heatmap_methods,
        cmap="YlOrRd",
        vmin=float(np.nanmin(mse_display)),
        vmax=float(np.nanpercentile(mse_display, 95)),
        title="MSE\nper c.t. [↓]",
        yticklabel_mode="swatch", diverged_mask=mse_diverged,
    )
    add_top_colorbar(fig, ax_mse_ct, im_mse)
    heatmap_letter(ax_mse_ct, "i", HEATMAP_LETTER_X_IN)

    # SS heatmap (panel j): same rows, same columns, own scale.
    ax_ss_ct = fig.add_subplot(hm_gs[0, 3])
    im_ss = draw_heatmap(
        ax_ss_ct, ss_vals, cancer_types_sorted, heatmap_methods,
        cmap="RdYlGn",
        vmin=float(np.nanmin(ss_vals)), vmax=float(np.nanmax(ss_vals)),
        title="SS\nper c.t. [↑]",
        yticklabel_mode="hidden",
    )
    add_top_colorbar(fig, ax_ss_ct, im_ss)
    heatmap_letter(ax_ss_ct, "j", HEATMAP_LETTER_BARE_X_IN)

    # k-NN purity heatmap (panel k). A fraction of neighbours, so the scale is
    # the full 0-1 rather than the data range, and higher is better -- same
    # colormap as SS, so green reads the same way in both. The last column is
    # chance level, split off from the models by a white rule.
    ax_pur_ct = fig.add_subplot(hm_gs[0, 5])
    im_pur = draw_heatmap(
        ax_pur_ct, purity_vals, cancer_types_sorted, purity_methods,
        cmap="RdYlGn", vmin=0.0, vmax=1.0,
        title=f"Purity (k={PURITY_K_MAIN})\nper c.t. [↑]",
        yticklabel_mode="hidden",
    )
    ax_pur_ct.axvline(len(heatmap_methods) - 0.5, color="white", linewidth=2.5)
    add_top_colorbar(fig, ax_pur_ct, im_pur)
    heatmap_letter(ax_pur_ct, "k", HEATMAP_LETTER_BARE_X_IN)

    # Abundance bar (col 5, aligned with method legend above) — panel l
    ax_ab = fig.add_subplot(outer[1, 4])
    draw_abundance(ax_ab, cancer_types_sorted, abundance)
    ax_ab.text(-0.10, 1.02, "l", transform=ax_ab.transAxes,
               fontsize=PT_PANEL, fontweight="bold", va="bottom", ha="left")

    report_text_overlaps(fig, "fig2")
    save_figure(fig, args.out_dir, "fig2")


if __name__ == "__main__":
    main()
