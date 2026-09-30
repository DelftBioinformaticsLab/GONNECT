"""Figure 3: effect of GO graph randomization on model performance.

Two bar panels sharing one legend column:
  a  MSE [down]   b  SS [up]

Bars are grouped so each model variant sits next to its degree-preserving (DP)
and fully-random counterparts. Colours encode the randomization level only:
  blue   = true graph
  green  = degree-preserving random
  orange = fully random
The true-graph bar of every group is outlined; the DP / random bars carry FDR
significance stars from a paired t-test against that within-group baseline. A
star is drawn a line higher than its left-hand neighbour where the two would
otherwise run into each other: at the paper's shared type scale "***" is wider
than the gap between two bars.

Every value is on held-out samples, as in Figure 2: MSE is each run's test loss,
and SS / ARI / NMI score its test split, with the silhouette taken against the
true cancer types for every model. The true-graph bars are Figure 2's.

Inputs, relative to --data-dir (default figures/data):
  metrics/metric_data_TCGA_1000_30_new.xlsx
                                             MSE of the true-graph and
                                             degree-preserving GONNECT arms
  metrics/test_split/gonnect_clustering.csv  their SS / ARI / NMI
  metrics/test_split/randomized_metrics.csv  all four metrics of the fully
                                             random and randomized soft-link
                                             arms (-RR-, -R-SL-, -RR-SL-)
  metrics/test_split/ontovae_rand.txt        OntoVAE, every graph arm (rescored)
  metrics/test_split/vega_rand.txt           VEGA Hallmark + Reactome, every arm

Supplementary Figure S4 (ARI + NMI) is the same figure over the other two
metrics; figS4.py renders it by calling main() from here.

Run: python fig3.py   ->   out/fig3.png, out/fig3.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import stats

from _common import (
    FIG_WIDTH_IN,
    MAIN_METHODS,
    METRIC_DIRECTION,
    METRIC_ORDER,
    PT_BODY,
    PT_PANEL,
    PT_SMALL,
    PT_TITLE,
    RANDOMIZATION_GROUP_COLORS,
    RANDOMIZATION_LEGEND,
    RANDOMIZED_GRAPH_ARMS,
    add_io_args,
    adjust_fdr,
    compute_summary,
    display,
    figsize,
    pt,
    read_baseline_ontovae,
    read_baseline_vega,
    read_xlsx_metrics,
    report_text_overlaps,
    save_figure,
    stars,
    test_split_paths,
)

METRIC_FILE = "metric_data_TCGA_1000_30_new.xlsx"

# Used as the command-line description only. The figure carries no title of its
# own: what it shows belongs in the caption, not on the canvas.
TITLE_MAIN = "Effect of GO graph randomization on model performance"
TITLE_SUPP = ("Effect of GO graph randomization on clustering quality "
              "(supplementary)")

# ── Canvas geometry ──────────────────────────────────────────────────────────
# The width is fixed at FIG_WIDTH_IN, so only the height is ours to choose, and
# nothing may hang off the sides: save_figure keeps the full canvas width and
# trims only vertically, so anything past the edge is lost rather than framed.
#
# The vertical budget, top to bottom: panel letters + panel titles, the bar
# axes, and the rotated group labels underneath. The group labels are the
# expensive part -- "VEGA (Reactome)" set at PT_BODY and turned on its side is
# 2.05 in tall -- so they get a fixed 2.2 in at the bottom and the axes take the
# ~2.75 in that is left. That leaves the bars roughly as slender as they were
# before the type grew. The band the suptitle used to occupy is simply trimmed
# away on save.
FIG_HEIGHT_IN = 5.8
AXES_TOP = 0.848      # axes top, in figure fractions
AXES_BOTTOM = 0.379   # axes bottom; the 2.2 in below is all group labels
AXES_LEFT = 0.030     # room for the y tick labels and the panel letters
AXES_RIGHT = 0.995    # fill the canvas
PANEL_WSPACE = 0.16

# Width the legend column has to have for its longest entry
# ("Significance vs group baseline (FDR p)" set bold at PT_BODY) to stay on the
# canvas. Set in inches rather than as a ratio so it survives a change in the
# number of metric panels; a legend that overruns the right edge is now
# silently truncated instead of widening the saved figure.
LEGEND_WIDTH_IN = 4.9

# ── Legend block ─────────────────────────────────────────────────────────────
# Drawn by hand rather than through ax.legend(), for the same reason figures 2
# and 4 are: matplotlib indents a legend title past the handles, and the paper's
# key sets its headings bold and flush with the swatches underneath them. The
# block is taller than the bar row and simply runs on into the empty label band
# beside it, so its height is set here rather than taken from the gridspec cell.
LEGEND_ROW_IN = 0.30
LEGEND_PAD_IN = 0.10
LEGEND_BLOCK_GAP_IN = 0.24    # extra air above the second heading
LEGEND_SWATCH_W_IN = 0.30
LEGEND_SWATCH_H_IN = 0.15
LEGEND_SWATCH_PAD_IN = 0.10   # swatch -> label, and star -> p-value
LEGEND_N_ROWS = 9             # 2 headings + 4 variant rows + 3 significance rows
LEGEND_HEIGHT_IN = (2 * LEGEND_PAD_IN + LEGEND_N_ROWS * LEGEND_ROW_IN
                    + LEGEND_BLOCK_GAP_IN)


# ── Method config (display names from _common) ───────────────────────────────
# Randomization level → colour group.
METHOD_GROUP: Dict[str, str] = {}
# The true-graph arm is exactly the ten models of Figure 2.
_TRUE_METHODS = set(MAIN_METHODS)
_DP_METHODS = {
    "ontovae_degree_preserving",
    "vega_hallmark_degree_preserving",
    "vega_reactome_degree_preserving",
    "GONNECT-R-enc", "GONNECT-R-dec", "GONNECT-R-both",
    "GONNECT-R-SL-enc", "GONNECT-R-SL-dec", "GONNECT-R-SL-both",
}
_RAND_METHODS = {
    "ontovae_random",
    "vega_hallmark_random",
    "vega_reactome_random",
    "GONNECT-RR-enc", "GONNECT-RR-dec", "GONNECT-RR-both",
    "GONNECT-RR-SL-enc", "GONNECT-RR-SL-dec", "GONNECT-RR-SL-both",
}
for m in _TRUE_METHODS:
    METHOD_GROUP[m] = "True"
for m in _DP_METHODS:
    METHOD_GROUP[m] = "DP"
for m in _RAND_METHODS:
    METHOD_GROUP[m] = "Random"

# Each group is one model variant followed by its DP and fully-random counterparts.
DISPLAY_GROUPS: List[List[str]] = [
    ["MLP"],
    ["ontovae",          "ontovae_degree_preserving",          "ontovae_random"],
    ["vega_hallmark",    "vega_hallmark_degree_preserving",    "vega_hallmark_random"],
    ["vega_reactome674", "vega_reactome_degree_preserving",    "vega_reactome_random"],
    ["GONNECT-enc",     "GONNECT-R-enc",     "GONNECT-RR-enc"],
    ["GONNECT-dec",     "GONNECT-R-dec",     "GONNECT-RR-dec"],
    ["GONNECT-both",    "GONNECT-R-both",    "GONNECT-RR-both"],
    ["GONNECT-SL-enc",  "GONNECT-R-SL-enc",  "GONNECT-RR-SL-enc"],
    ["GONNECT-SL-dec",  "GONNECT-R-SL-dec",  "GONNECT-RR-SL-dec"],
    ["GONNECT-SL-both", "GONNECT-R-SL-both", "GONNECT-RR-SL-both"],
]
DISPLAY_GROUP_GAP: float = 0.7

# Each group's first method = the "true graph" reference for within-group tests.
GROUP_BASELINES: set = {grp[0] for grp in DISPLAY_GROUPS}


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data(data_dir: Path) -> pd.DataFrame:
    """Every source that contributes a bar, restricted to the plotted methods.

    All on held-out samples. The workbook gives MSE only, and only for the arms
    whose runs have deposited embeddings (true graph, degree-preserving); the
    other three GONNECT arms take all four metrics from randomized_metrics.csv,
    which reads MSE from the training logs. Every baseline arm, the true graph
    included, comes from the rescored files, as Figure 2's true-graph bars do.
    The repeat filter is deliberately non-strict, matching the original script:
    ids that do not look like ``run-N`` are kept.
    """
    paths = test_split_paths(data_dir)
    for key, step in (("gonnect", "test_split_metrics.py"),
                      ("randomized", "cluster_test_metrics.py fig3"),
                      ("ontovae", "rescore_baselines.py"), ("vega", "rescore_baselines.py")):
        if not paths[key].exists():
            raise SystemExit(f"missing {paths[key]}; build it with prepare/{step}")
    columns = ["metric", "method", "repeat", "value"]
    randomized = pd.read_csv(paths["randomized"])
    workbook = read_xlsx_metrics(data_dir / "metrics" / METRIC_FILE, strict_repeats=False)
    arms = ("true",) + tuple(RANDOMIZED_GRAPH_ARMS)
    frames: List[pd.DataFrame] = [
        workbook[(workbook["metric"] == "MSE")
                 & ~workbook["method"].isin(randomized["method"].unique())],
        pd.read_csv(paths["gonnect"], usecols=columns),
        randomized[columns],
        read_baseline_ontovae(paths["ontovae"], graph_types=arms, strict_repeats=False),
        read_baseline_vega(paths["vega"], graph_types=arms, strict_repeats=False),
    ]

    data = pd.concat(frames, ignore_index=True)
    all_methods = [m for grp in DISPLAY_GROUPS for m in grp]
    return data[data["method"].isin(all_methods)].reset_index(drop=True)


# ── Stats ────────────────────────────────────────────────────────────────────
def within_group_t_tests(data: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """For each DISPLAY_GROUP triplet, test DP and Random variants against the
    group's 'true' variant (first method in the group). FDR-adjust within each metric."""
    rows = []
    for metric in sorted(data["metric"].unique()):
        md = data[data["metric"] == metric]
        for grp in DISPLAY_GROUPS:
            if len(grp) < 2:
                continue
            baseline = grp[0]
            base = md[md["method"] == baseline]
            if base.empty:
                continue
            for method in grp[1:]:
                cmp = md[md["method"] == method]
                merged = base.merge(cmp, on="repeat", suffixes=("_base", "_cmp"))
                if len(merged) < 2:
                    continue
                stat, p = stats.ttest_rel(merged["value_cmp"], merged["value_base"])
                rows.append({
                    "metric": metric, "method": method,
                    "group_baseline": baseline,
                    "n": int(len(merged)),
                    "t_stat": float(stat), "p_value": float(p),
                })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_adj"] = np.nan
    for _, sub in result.groupby("metric"):
        result.loc[sub.index, "p_adj"] = adjust_fdr(sub["p_value"].tolist())
    result["significant"] = result["p_adj"] < alpha
    return result


# ── Plot ─────────────────────────────────────────────────────────────────────
def _build_x_positions(methods: List[str]):
    """Bar x positions + group-center x positions + the label for each group."""
    x_map: Dict[str, float] = {}
    pos = 0.0
    group_centers: List[float] = []
    group_labels: List[str] = []
    for gi, grp in enumerate(DISPLAY_GROUPS):
        present = [m for m in grp if m in methods]
        if not present:
            continue
        if group_centers:
            pos += DISPLAY_GROUP_GAP
        start = pos
        for m in present:
            x_map[m] = pos
            pos += 1.0
        last_bar = pos - 1.0
        group_centers.append((start + last_bar) / 2.0)
        # Label = display of the "true" (first) method in the group. Its name is
        # already free of a randomization suffix, so take it verbatim — splitting
        # on " (" would collapse "VEGA (Hallmark)" and "VEGA (Reactome)" to "VEGA".
        group_labels.append(display(grp[0]))
    return (
        np.array([x_map[m] for m in methods]),
        np.array(group_centers),
        group_labels,
    )


def _text_size_data(ax, text: str, fontsize: float):
    """Size of ``text`` as (width, height) in this axes' data coordinates."""
    probe = ax.text(0, 0, text, fontsize=fontsize, ha="center", va="bottom")
    box = probe.get_window_extent(ax.figure.canvas.get_renderer())
    probe.remove()
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(0, 0), (box.width, box.height)])
    return abs(x1 - x0), abs(y1 - y0)


def _stack_marks(ax, marks):
    """Nudge each mark up until it clears the ones to its left.

    Returns ``[(x, text, y), ...]`` plus the highest top reached. Bars sit one
    x-unit apart, and at the paper's type scale "***" is more than twice that
    wide, so the DP and FR markers of a group collide head-on whenever both are
    significant. Lifting the right-hand one onto its own line is the only fix
    that keeps every marker centred over the bar it belongs to without setting
    the stars smaller than the rest of the paper's PT_SMALL text.
    """
    sizes = {t: _text_size_data(ax, t, PT_SMALL) for _, t, _ in marks}
    placed, out, highest = [], [], -np.inf
    for x, text, y_base in sorted(marks, key=lambda m: m[0]):
        w, h = sizes[text]
        gap = 0.12 * h
        y = y_base
        for _ in range(len(placed) + 1):
            lifted = False
            for px, pw, py, ph in placed:
                if (abs(x - px) < (w + pw) / 2.0
                        and y < py + ph + gap and y + h > py - gap):
                    y = py + ph + gap
                    lifted = True
            if not lifted:
                break
        placed.append((x, w, y, h))
        out.append((x, text, y))
        highest = max(highest, y + h)
    return out, highest


def _draw_significance_marks(ax, marks, y_max: float, y_span: float) -> None:
    """Place the FDR stars and open up whatever headroom they end up needing.

    Stacking is measured in display space, so the y limit has to be settled
    before the marks are laid out -- and raising it to fit them changes the
    scale again. Two or three passes converge.
    """
    top = y_max + 0.18 * y_span
    ax.set_ylim(top=top)
    if not marks:
        return
    ax.figure.canvas.draw()
    for _ in range(6):
        placed, highest = _stack_marks(ax, marks)
        needed = highest + 0.05 * y_span
        if needed <= top:
            break
        top = needed
        ax.set_ylim(top=top)
    for x, text, y in placed:
        ax.text(x, y, text, ha="center", va="bottom", fontsize=PT_SMALL)


def _trim_yticks(ax) -> None:
    """Drop y ticks that fall outside the view.

    Matplotlib never draws them, but it keeps their label artists alive and
    positioned where the tick would have been -- for the SS panel that parks an
    invisible "0.4" above the top of the axes. Removing them changes nothing on
    the page and keeps report_text_overlaps honest.
    """
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in ax.get_yticks() if lo - 1e-9 <= t <= hi + 1e-9])


def draw_metric_bar(ax, summary: pd.DataFrame, metric: str,
                    stats_table: pd.DataFrame, methods: List[str]) -> None:
    sub = summary[summary["metric"] == metric]
    x, group_centers, group_labels = _build_x_positions(methods)
    means, stds, colors = [], [], []
    for m in methods:
        row = sub[sub["method"] == m]
        means.append(row["mean"].iloc[0] if not row.empty else np.nan)
        stds.append(row["std"].iloc[0] if not row.empty else np.nan)
        colors.append(RANDOMIZATION_GROUP_COLORS.get(METHOD_GROUP.get(m, ""), "#888888"))

    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=2.5, width=0.85)
    # Outline the within-group baseline (true-graph variant) of every group.
    for m, bar in zip(methods, bars):
        if m in GROUP_BASELINES:
            bar.set_edgecolor("#212121")
            bar.set_linewidth(1.4)

    arrow = "↓" if METRIC_DIRECTION.get(metric, "up") == "down" else "↑"
    ax.set_title(f"{metric} [{arrow}]", fontsize=PT_TITLE, pad=pt(3))
    # Group-level x-ticks (one label per model triplet, centered).
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, rotation=90, ha="center", va="top",
                       fontsize=PT_BODY)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=PT_BODY, pad=pt(1.5))

    ms = stats_table[stats_table["metric"] == metric] if stats_table is not None else pd.DataFrame()
    if not ms.empty:
        valid = [v for v in means if not np.isnan(v)]
        if valid:
            y_max = max(valid)
            y_span = max(y_max, 1e-6)
            marks = []
            for m, bar, mean, std in zip(methods, bars, means, stds):
                if m in GROUP_BASELINES:
                    continue
                row = ms[ms["method"] == m]
                if row.empty:
                    continue
                star = stars(float(row["p_adj"].iloc[0]))
                if not star:
                    continue
                h = (mean or 0.0) + (std if not np.isnan(std) else 0.0)
                marks.append((bar.get_x() + bar.get_width() / 2, star,
                              h + 0.03 * y_span))
            _draw_significance_marks(ax, marks, y_max, y_span)
    _trim_yticks(ax)


LEGEND_VARIANT_ROWS = [
    ("True",   RANDOMIZATION_LEGEND["true"]),
    ("DP",     RANDOMIZATION_LEGEND["degree_preserving"]),
    ("Random", RANDOMIZATION_LEGEND["random"]),
]
LEGEND_SIG_ROWS = [("*", "p ≤ 0.05"), ("**", "p ≤ 0.01"), ("***", "p ≤ 0.001")]


def draw_legend(ax) -> None:
    """The key, drawn by hand in the same style as figures 2 and 4.

    ``ax`` is invisible and its data coordinates are inches measured from its
    top-left corner, which is what lets the two blocks share one left edge:
    swatches and significance markers at x = 0, their labels and p-values at one
    common indent, and the headings flush with the swatches rather than indented
    past them the way ax.legend() sets a title.
    """
    ax.axis("off")
    label_x = LEGEND_SWATCH_W_IN + LEGEND_SWATCH_PAD_IN
    y = LEGEND_PAD_IN + LEGEND_ROW_IN / 2

    def heading(text: str) -> None:
        ax.text(0.0, y, text, va="center", ha="left",
                fontsize=PT_BODY, fontweight="bold")

    def swatch(face: str, edge: str, lw: float) -> None:
        ax.add_patch(Rectangle(
            (0.0, y - LEGEND_SWATCH_H_IN / 2),
            LEGEND_SWATCH_W_IN, LEGEND_SWATCH_H_IN,
            facecolor=face, edgecolor=edge, linewidth=lw, clip_on=False,
        ))

    heading("Graph variant")
    for group, label in LEGEND_VARIANT_ROWS:
        y += LEGEND_ROW_IN
        swatch(RANDOMIZATION_GROUP_COLORS[group], "none", 0.0)
        ax.text(label_x, y, label, va="center", ha="left", fontsize=PT_BODY)
    # The outlined bar of every group, drawn the same way it is on the bars.
    y += LEGEND_ROW_IN
    swatch("white", "#212121", 1.4)
    ax.text(label_x, y, "Within-group baseline (true)", va="center", ha="left",
            fontsize=PT_BODY)

    y += LEGEND_ROW_IN + LEGEND_BLOCK_GAP_IN
    heading("Significance vs group baseline (FDR p)")
    for marker, p_text in LEGEND_SIG_ROWS:
        y += LEGEND_ROW_IN
        ax.text(0.0, y, marker, va="center", ha="left", fontsize=PT_BODY)
        ax.text(label_x, y, p_text, va="center", ha="left", fontsize=PT_BODY)


def _legend_width_ratio(ncols: int, wspace: float) -> float:
    """Gridspec width ratio giving the legend column LEGEND_WIDTH_IN inches.

    The canvas width is fixed, so the legend cannot simply be given a fixed
    share of it: with more metric panels the same share is fewer inches and the
    legend labels run off the page. Solving for the ratio keeps the column the
    width the text actually needs, whatever ncols is.
    """
    usable = (AXES_RIGHT - AXES_LEFT) * FIG_WIDTH_IN
    n = ncols + 1
    # Matplotlib's wspace is a multiple of the *mean* column width, so the
    # gaps scale with the ratios and the relation stays linear in r_leg.
    gap_factor = 1.0 + wspace * (n - 1) / n
    denom = usable - LEGEND_WIDTH_IN * gap_factor
    if denom <= 0:                     # pathologically many panels
        return 0.5
    return LEGEND_WIDTH_IN * ncols * gap_factor / denom


def render(metrics: Sequence[str], summary: pd.DataFrame, stats_table: pd.DataFrame,
           methods: List[str], out_dir: Path, out_name: str) -> None:
    """One bar panel per metric, plus the shared legend column on the right."""
    ncols = len(metrics)
    fig = plt.figure(figsize=figsize(FIG_HEIGHT_IN))
    gs = fig.add_gridspec(
        1, ncols + 1,
        width_ratios=[1.0] * ncols + [_legend_width_ratio(ncols, PANEL_WSPACE)],
        wspace=PANEL_WSPACE,
        left=AXES_LEFT, right=AXES_RIGHT, top=AXES_TOP, bottom=AXES_BOTTOM,
    )
    for i, (letter, metric) in enumerate(zip("abcdefg", metrics)):
        ax = fig.add_subplot(gs[0, i])
        draw_metric_bar(ax, summary, metric, stats_table, methods)
        ax.text(-0.055, 1.015, letter, transform=ax.transAxes,
                fontsize=PT_PANEL, fontweight="bold", va="bottom", ha="left")

    # The legend keeps the gridspec's column, but takes the height its rows
    # actually need and hangs from the top of the row: it is taller than the bar
    # axes and runs on into the label band beside them, which is empty here.
    ax_leg = fig.add_subplot(gs[0, ncols])
    pos = ax_leg.get_position()
    height = LEGEND_HEIGHT_IN / FIG_HEIGHT_IN
    ax_leg.set_position([pos.x0, pos.y1 - height, pos.width, height])
    ax_leg.set_xlim(0.0, pos.width * FIG_WIDTH_IN)   # data units are inches,
    ax_leg.set_ylim(LEGEND_HEIGHT_IN, 0.0)           # y measured downwards
    draw_legend(ax_leg)

    # No suptitle: what the figure shows is the caption's job, not the canvas'.
    report_text_overlaps(fig, out_name)
    save_figure(fig, out_dir, out_name)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    add_io_args(parser)
    return parser.parse_args()


def main(metrics: Sequence[str] = ("MSE", "SS"), out_name: str = "fig3",
         title: str = TITLE_MAIN) -> None:
    """Render one randomization figure. figS4.py reuses this for ARI + NMI.

    ``title`` is the command-line description; neither figure draws a title.
    """
    args = parse_args(title)

    print("Loading metric data ...")
    data = load_data(args.data_dir)
    if data.empty:
        raise SystemExit("No data loaded — check the input file paths.")

    summary = compute_summary(data)
    stats_table = within_group_t_tests(data)

    # Methods present in data, in DISPLAY_GROUPS order.
    methods_present = set(data["method"].unique())
    methods = [m for grp in DISPLAY_GROUPS for m in grp if m in methods_present]
    print(f"Plotting {len(methods)} methods across {len(METRIC_ORDER)} metrics.")
    missing = [m for grp in DISPLAY_GROUPS for m in grp if m not in methods_present]
    if missing:
        print(f"NOTE: {len(missing)} methods not present in data and will be skipped: {missing}")

    if not stats_table.empty:
        print(stats_table.to_string(index=False))

    render(metrics, summary, stats_table, methods, args.out_dir, out_name)


if __name__ == "__main__":
    main()
