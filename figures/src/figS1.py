"""Supplementary Figure S1: supplemental bar chart, four rows (one per metric).

Rows are MSE / SS / ARI / NMI, all sharing one x axis drawn under the bottom
row. Each row holds three sections, separated by dashed vertical lines:
  1. Left   — the ten models of Figure 2 (MLP / OntoVAE / VEGA-Hallmark /
              VEGA-Reactome / GONNECT enc,dec,both / GONNECT-SL enc,dec,both),
              coloured by which module carries the GO prior. No randomized
              variants, and no significance stars (those are in Figure 2).
  2. Middle — child-term threshold (ct) 5 / 10 / 30 per GONNECT model.
  3. Right  — Gene count: 1k vs 2k per GONNECT model.
Sections 2 and 3 outline their reference bar (ct=30 resp. 1k, the setting used
in the main figure) and star the other bars by FDR-adjusted paired t-test
against it. Pure baselines (MLP, OntoVAE, VEGA*) appear only in section 1, and
the -both models only in section 1 and 2 (no ct=5/10 or 2k data).

Every value is on held-out samples, as in Figure 2: MSE is each run's test loss,
and SS / ARI / NMI score its test split, with the silhouette taken against the
true cancer types. Section 1 is exactly Figure 2a-d. The ct=30 and 1k reference
bars are Figure 2's runs, so they come from the same files. The sweep runs' MSE
comes from their training logs, which corrects ct=10 SL-enc / SL-dec: the
workbook had their loss including the soft-link penalty.

The canvas is the paper's shared figure width, so the bars get all of it: the
legend is a four-column strip across the top rather than a column beside the
rows, and the height grows with the number of metrics.

Inputs, relative to --data-dir (default figures/data):
  metrics/metric_data_TCGA_1000_30_new.xlsx  section 1 GONNECT MSE
  metrics/metric_data_TCGA_1000_{5,10,30}.xlsx
                                                 section 2 (ct sweep): which models
                                                 each setting has, and ct=30's MSE;
                                                 the ct=30 file is also section 3's
                                                 1k reference
  metrics/metric_data_TCGA_2000_30.xlsx      section 3 (2k genes): which models
  metrics/test_split/gonnect_clustering.csv  SS / ARI / NMI of section 1 and
                                                 of the ct=30 / 1k references
  metrics/test_split/{ontovae,vega}_rand.txt section 1 baselines (rescored)
  metrics/test_split/sweep_metrics.csv       all four metrics of ct=5, ct=10, 2k

Run: python figS1.py   ->   out/figS1.png, out/figS1.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    PRIOR_GROUP_COLORS,
    PRIOR_METHOD_GROUP,
    PT_BODY,
    PT_SMALL,
    PT_TITLE,
    add_io_args,
    adjust_fdr,
    compute_summary,
    display,
    figsize,
    paired_t_tests,
    pt,
    read_baseline_ontovae,
    read_baseline_vega,
    read_xlsx_metrics,
    report_text_overlaps,
    save_figure,
    stars,
    test_split_paths,
)

OUT_NAME = "figS1"

# ── Method config ────────────────────────────────────────────────────────────
# Section 1 shows the ten models of Figure 2, in the same order:
# MLP → OntoVAE → VEGA-Hallmark → VEGA-Reactome → GONNECT-* → GONNECT-SL-*
METHOD_DISPLAY: Dict[str, str] = {m: display(m) for m in MAIN_METHODS}

# ── Typography ───────────────────────────────────────────────────────────────
# All sizes come from the shared scale in _common, so this figure's text
# renders at the same size as every other figure's once both are placed at the
# paper's column width. The numbers are ~2.3x the rendered point size.
FS_TICK = PT_SMALL          # x/y tick labels
FS_TITLE = PT_TITLE         # per-row metric name on the y axis
FS_GROUP = PT_BODY          # bold model name under each section-2/3 group
FS_STAR = PT_BODY           # significance stars
FS_LEGEND = PT_SMALL
FS_LEGEND_TITLE = PT_BODY

# Outline marking the within-section reference bar (ct=30 / 1k) that the
# stars in sections 2 & 3 are tested against.
REF_OUTLINE_LW = 2.6

# Vertical drop of the bold group labels, in points below the axis. It has to
# clear the rotated tick labels, whose longest section-2/3 entry is "ct=30"
# (5 chars); rotated 90° a character occupies ~0.62*fontsize points. The pad
# below them goes through pt() so it tracks the type scale.
GROUP_LABEL_DROP = -(5 * FS_TICK * 0.62 + pt(8))


def _wrap_group_label(name: str) -> str:
    """Break "GONNECT-SL-enc" after the prefix so it fits a 2-bar group."""
    return name.replace("GONNECT-", "GONNECT-\n", 1)

# Models that get bars in sections 2 & 3 (skip pure baselines).
EXPERIMENT_METHODS = [
    "GONNECT-enc", "GONNECT-dec", "GONNECT-both",
    "GONNECT-SL-enc", "GONNECT-SL-dec", "GONNECT-SL-both",
]

# Methods to KEEP (i.e., randomized models are excluded throughout).
KEEP_METHODS = set(MAIN_METHODS)

BASELINE_METHOD = "GONNECT-SL-dec"

# Section 1: wide GONNECT family workbook.
MAIN_METRIC_FILE = "metric_data_TCGA_1000_30_new.xlsx"
# Section 2: child-term threshold (ct) files.
CT_CONFIG_FILES: List[Tuple[str, str]] = [
    ("metric_data_TCGA_1000_5.xlsx",  "ct=5"),
    ("metric_data_TCGA_1000_10.xlsx", "ct=10"),
    ("metric_data_TCGA_1000_30.xlsx", "ct=30"),
]
# Section 3: gene-count files (1k = reference from ct=30).
GENE_CONFIG_FILE: Tuple[str, str] = ("metric_data_TCGA_2000_30.xlsx", "2k")
# The `setting` each sweep configuration carries in sweep_metrics.csv. The
# ct=30 / 1k references are absent: they are Figure 2's runs.
SWEEP_SETTINGS: Dict[str, str] = {"ct=5": "ct=5", "ct=10": "ct=10", "2k": "2k genes"}

# Wide enough that the dashed section separators read as a bigger break than
# the gaps between the model groups inside sections 2 and 3.
SECTION_GAP = 1.8
# Bar pitch inside a section-2/3 group. Wider than one bar, because at this
# figure width a "***" marker is wider than the bar it sits on: at pitch 1 the
# markers over two neighbouring bars merge into one run of six asterisks.
BAR_STEP = 1.5
# The gaps between the model groups are set by their bold labels, not by the
# bars: "GONNECT-" is wider than a whole group, so the gaps are what keeps
# neighbouring labels apart. Section 3 needs the larger one because its groups
# are only two bars wide; the two values give both sections the same group
# pitch, and with it the same label spacing.
MODEL_GAP = 1.0
GENE_MODEL_GAP = MODEL_GAP + BAR_STEP
# Bars use matplotlib's default width, so each spans ±0.4 about its centre. The
# gap between a section's outermost bar and its dashed separator works out to
# SECTION_GAP/2 + 0.1; the axis edges reuse it so every bar-to-boundary gap in
# the row is the same width.
BAR_HALF_WIDTH = 0.4
EDGE_GAP = SECTION_GAP / 2 + 0.1

# ── Canvas ───────────────────────────────────────────────────────────────────
# The width is fixed at _common.FIG_WIDTH_IN for every figure in the paper, so
# only the height is ours to choose: one band per metric plus two fixed bands,
# for the legend on top and for the shared x axis under the bottom row. The
# per-metric band is generous because the type is set for the placed size, not
# for the canvas -- 30 bars and their rotated labels need the room.
ROW_HEIGHT_IN = 3.0        # plot band per metric
XLABEL_BAND_IN = 2.05      # rotated tick labels + bold group labels
# Side margins. Left holds the y tick labels and the metric name, sized for the
# widest of them ("0.30" on the SS row); right is just enough that the last
# error bar is not clipped. Everything between the two is axes, so the bars use
# the full standard width.
LEFT_MARGIN_IN = 0.90
RIGHT_MARGIN_IN = 0.12

CT_EXTRA_COLORS: List[str] = ["#AEC7E8", "#FFBB78", "#98DF8A"]  # ct=5, ct=10, ct=30
# Section 3 gets its own hue ramp so it is not confused with section 1
# (group colours) or section 2 (pastel blue/orange/green).
GENE_COLORS: List[str] = ["#BCBDDC", "#756BB1"]  # 1k, 2k

# ── Legend band ──────────────────────────────────────────────────────────────
# Drawn by hand rather than through fig.legend(), as in figures 2, 4 and 5: the
# four blocks each carry their own bold heading, and matplotlib only offers one
# title for the whole legend and indents it past the handles. The blocks sit
# side by side across the top of the figure; a column beside the axes would
# cost ~3 in of the 16.5 the bars need. Widths are the widest entry each block
# holds, at the size it is set in (report_text_overlaps catches an overrun).
LEGEND_TOP_IN = 0.06
LEGEND_ROW_IN = 0.30
LEGEND_PAD_IN = 0.10
LEGEND_N_ROWS = 6          # heading + up to five entries (the config block)
LEGEND_HEIGHT_IN = 2 * LEGEND_PAD_IN + LEGEND_N_ROWS * LEGEND_ROW_IN
LEGEND_GAP_IN = 0.25       # legend band -> top row of bars
LEGEND_BAND_IN = LEGEND_TOP_IN + LEGEND_HEIGHT_IN + LEGEND_GAP_IN
LEGEND_COL_W_IN = [1.70,   # Group: Reference / Enc / Dec / Both priors
                   3.10,   # Annotation: the dashed rule and the outlined bar
                   1.80,   # Configs: ct=5 / ct=10 / ct=30 / 1k / 2k
                   3.40]   # the p-value key, under a heading wider than it
LEGEND_SWATCH_W_IN = 0.30
LEGEND_SWATCH_H_IN = 0.15
LEGEND_SWATCH_PAD_IN = 0.10
LEGEND_EDGE_IN = RIGHT_MARGIN_IN


# ── Data loading ─────────────────────────────────────────────────────────────
def read_config_frame(path: Path,
                      cache: Optional[Dict[Path, pd.DataFrame]] = None) -> pd.DataFrame:
    """Read one metric workbook (rows with a blank repeat cell kept as row-<idx>).

    Pass ``cache`` to reuse an already parsed workbook: metric_data_TCGA_1000_30
    is both section 2's ct=30 configuration and section 3's 1k reference, and
    parsing the workbook is by far the slowest part of this script. The frames
    are only ever read, never mutated, so sharing one is safe.
    """
    if cache is not None and path in cache:
        return cache[path]
    df = read_xlsx_metrics(path, strict_repeats=False, row_fallback=True)
    if cache is not None:
        cache[path] = df
    return df


def keep_main_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Drop everything but the ten section-1 models."""
    return df[df["method"].isin(KEEP_METHODS)].reset_index(drop=True)


def on_test_split(workbook: pd.DataFrame, held_out: pd.DataFrame) -> pd.DataFrame:
    """A workbook's models, with their metrics taken from ``held_out``.

    The workbooks' MSE is each run's test loss, but their other three metrics
    span all samples. ``held_out`` always replaces SS / ARI / NMI. It replaces
    MSE too where it carries one, as sweep_metrics.csv does; see
    prepare/cluster_test_metrics.py for why. Only models the workbook has an MSE
    for are taken, so the set of bars does not change.
    """
    mse = workbook[workbook["metric"] == "MSE"]
    rows = held_out[held_out["method"].isin(mse["method"].unique())]
    if (rows["metric"] == "MSE").any():
        mse = mse.iloc[0:0]
    return pd.concat([mse, rows[["metric", "method", "repeat", "value"]]], ignore_index=True)


def collect_orig_data(data_dir: Path, gonnect_clustering: pd.DataFrame) -> pd.DataFrame:
    """Section 1 data, read exactly as Figure 2 reads it: the GONNECT family's
    MSE from the wide xlsx and its SS / ARI / NMI from the test split, plus the
    rescored OntoVAE / VEGA 'true' arms."""
    paths = test_split_paths(data_dir)
    frames = [on_test_split(read_config_frame(data_dir / "metrics" / MAIN_METRIC_FILE),
                            gonnect_clustering)]
    for path, reader in ((paths["ontovae"], read_baseline_ontovae),
                         (paths["vega"], read_baseline_vega)):
        if path.exists():
            frames.append(reader(path))
        else:
            print(f"WARNING: missing {path}; its baseline bars will be skipped.")
    return pd.concat(frames, ignore_index=True)


# ── Stats ────────────────────────────────────────────────────────────────────
def per_model_section_t_tests(
    configs_raw: List[Tuple[str, pd.DataFrame]],
    baseline_label: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """For each (metric, method, non-baseline config) within a section, paired
    t-test against the section's baseline config FOR THE SAME METHOD.

    e.g. for CT section: ct=5 vs ct=30 for GONNECT-enc, ct=10 vs ct=30 for
    GONNECT-enc, ct=5 vs ct=30 for GONNECT-dec, ...
    FDR-adjusted within each metric (across all method × config pairs).
    """
    base_df = next((df for label, df in configs_raw if label == baseline_label), None)
    if base_df is None or base_df.empty:
        return pd.DataFrame()
    rows = []
    for metric in sorted(base_df["metric"].unique()):
        for method in sorted(base_df["method"].unique()):
            base = base_df[(base_df["metric"] == metric) & (base_df["method"] == method)]
            if base.empty:
                continue
            for label, df in configs_raw:
                if label == baseline_label:
                    continue
                cmp = df[(df["metric"] == metric) & (df["method"] == method)]
                merged = base.merge(cmp, on="repeat", suffixes=("_base", "_cmp"))
                if len(merged) < 2:
                    continue
                stat, p = stats.ttest_rel(merged["value_cmp"], merged["value_base"])
                rows.append({
                    "metric": metric, "method": method, "config": label,
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


# ── Plotting ─────────────────────────────────────────────────────────────────
CT_BASELINE_LABEL = "ct=30"
GENE_BASELINE_LABEL = "1k"

# Order the section-1 colours appear in (Reference / Enc / Dec / Both).
LEGEND_GROUP_ORDER = ["Reference", "Enc prior", "Dec prior", "Both priors"]
LEGEND_SIG_ROWS = [("*", "p ≤ 0.05"), ("**", "p ≤ 0.01"), ("***", "p ≤ 0.001")]


def draw_legend(ax, groups, ct_labels: List[str], gene_labels: List[str]) -> None:
    """The key, as four blocks side by side above the rows.

    ``ax`` is invisible and its data coordinates are inches measured from its
    top-left corner, so each block sits at its measured width and every heading
    is flush with the swatches under it rather than indented past them the way
    fig.legend() sets a title.
    """
    ax.axis("off")
    band_w = ax.get_xlim()[1]
    gap = (band_w - sum(LEGEND_COL_W_IN)) / (len(LEGEND_COL_W_IN) - 1)
    col_x, x = [], 0.0
    for width in LEGEND_COL_W_IN:
        col_x.append(x)
        x += width + gap
    label_dx = LEGEND_SWATCH_W_IN + LEGEND_SWATCH_PAD_IN

    def row_y(row: int) -> float:
        return LEGEND_PAD_IN + (row + 0.5) * LEGEND_ROW_IN

    def heading(col: int, text: str) -> None:
        ax.text(col_x[col], row_y(0), text, va="center", ha="left",
                fontsize=FS_LEGEND_TITLE, fontweight="bold")

    def label(col: int, row: int, text: str) -> None:
        ax.text(col_x[col] + label_dx, row_y(row), text, va="center", ha="left",
                fontsize=FS_LEGEND)

    def swatch(col: int, row: int, text: str, face: str,
               edge: str = "none", lw: float = 0.0) -> None:
        y = row_y(row)
        ax.add_patch(Rectangle((col_x[col], y - LEGEND_SWATCH_H_IN / 2),
                               LEGEND_SWATCH_W_IN, LEGEND_SWATCH_H_IN,
                               facecolor=face, edgecolor=edge, linewidth=lw,
                               clip_on=False))
        label(col, row, text)

    heading(0, "Group")
    present = [g for g in LEGEND_GROUP_ORDER if g in groups]
    for row, group in enumerate(present, start=1):
        swatch(0, row, group, PRIOR_GROUP_COLORS[group])

    heading(1, "Annotation")
    y = row_y(1)
    ax.plot([col_x[1], col_x[1] + LEGEND_SWATCH_W_IN], [y, y],
            color="#9E9E9E", linestyle="--", linewidth=1, clip_on=False)
    label(1, 1, "MLP reference")
    swatch(1, 2, "Setting used in main figure", "white", edge="#212121", lw=2.0)

    heading(2, "Configs")
    row = 1
    for i, text in enumerate(ct_labels):
        swatch(2, row, text, CT_EXTRA_COLORS[i % len(CT_EXTRA_COLORS)])
        row += 1
    for i, text in enumerate(gene_labels):
        swatch(2, row, text, GENE_COLORS[i % len(GENE_COLORS)])
        row += 1

    heading(3, "Significance (FDR p)")
    for row, (marker, p_text) in enumerate(LEGEND_SIG_ROWS, start=1):
        ax.text(col_x[3], row_y(row), marker, va="center", ha="left",
                fontsize=FS_LEGEND)
        label(3, row, p_text)


def plot_summary_supp(
    summary_sec1: pd.DataFrame,
    ct_configs: List[Tuple[str, pd.DataFrame]],
    gene_configs: List[Tuple[str, pd.DataFrame]],
    out_dir: Path,
    out_name: str,
    stats_ct: Optional[pd.DataFrame] = None,
    stats_gene: Optional[pd.DataFrame] = None,
) -> None:
    if summary_sec1.empty:
        raise ValueError("No section-1 data available to plot.")

    methods_orig = [m for m in MAIN_METHODS if m in summary_sec1["method"].unique()]
    # Sections 2 & 3: -both methods dropped (missing data for ct=5/10 and 2k).
    methods_ct = [m for m in EXPERIMENT_METHODS
                  if not m.endswith("-both")
                  and any(m in s["method"].unique() for _, s in ct_configs)]
    methods_gene = [m for m in EXPERIMENT_METHODS
                    if not m.endswith("-both")
                    and any(m in s["method"].unique() for _, s in gene_configs)]

    n_orig = len(methods_orig)
    n_ct   = len(ct_configs)
    n_gene = len(gene_configs)

    # x positions ------------------------------------------------------------
    x_orig = np.arange(n_orig, dtype=float)
    x_sep1 = n_orig - 0.5 + SECTION_GAP / 2

    x_cursor = n_orig + SECTION_GAP
    ct_model_x: List[np.ndarray] = []
    for _ in methods_ct:
        ct_model_x.append(x_cursor + BAR_STEP * np.arange(n_ct, dtype=float))
        # One bar width of trailing space, as in section 1, then the gap.
        x_cursor += BAR_STEP * (n_ct - 1) + 1 + MODEL_GAP
    x_cursor -= MODEL_GAP

    x_sep2 = x_cursor - 0.5 + SECTION_GAP / 2
    x_cursor += SECTION_GAP

    gene_model_x: List[np.ndarray] = []
    for _ in methods_gene:
        gene_model_x.append(x_cursor + BAR_STEP * np.arange(n_gene, dtype=float))
        x_cursor += BAR_STEP * (n_gene - 1) + 1 + GENE_MODEL_GAP
    x_cursor -= GENE_MODEL_GAP
    x_max = x_cursor + 0.5

    # Figure -----------------------------------------------------------------
    n_metrics = len(METRIC_ORDER)
    fig_height = n_metrics * ROW_HEIGHT_IN + LEGEND_BAND_IN + XLABEL_BAND_IN
    # All four rows share the same methods/configs, so the x axis is drawn once,
    # under the bottom row only.
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize(fig_height),
                             sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    legend_groups: set = set()

    for ax_i, (ax, metric) in enumerate(zip(axes, METRIC_ORDER)):
        is_bottom = ax_i == len(axes) - 1
        sub_orig = summary_sec1[summary_sec1["metric"] == metric]

        # Section 1: original bars
        means_o, stds_o, colors_o, labels_o = [], [], [], []
        for method in methods_orig:
            row = sub_orig[sub_orig["method"] == method]
            means_o.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            stds_o.append(float(row["std"].iloc[0]) if not row.empty else np.nan)
            labels_o.append(METHOD_DISPLAY.get(method, method))
            group = PRIOR_METHOD_GROUP.get(method, "Other")
            colors_o.append(PRIOR_GROUP_COLORS.get(group, "#888888"))

        bars_o = ax.bar(x_orig, means_o, yerr=stds_o, color=colors_o, capsize=3)
        # No outline here: section 1 reports no stars, so there is no reference
        # bar to mark. The outline is reserved for the sections 2 & 3 references.
        for method in methods_orig:
            legend_groups.add(PRIOR_METHOD_GROUP.get(method, "Other"))

        # Horizontal reference line at the MLP mean (spans the whole row).
        mlp_row = sub_orig[sub_orig["method"] == "MLP"]
        mlp_value = float(mlp_row["mean"].iloc[0]) if not mlp_row.empty else None
        if mlp_value is not None:
            ax.axhline(mlp_value, color="#9E9E9E", linestyle="--", linewidth=1)

        # Section 1 significance is reported in the main-text figure; suppressed here.
        has_stars = False

        # Section separators
        ax.axvline(x_sep1, color="#444444", linestyle="--", linewidth=1.5)
        ax.axvline(x_sep2, color="#444444", linestyle="--", linewidth=1.5)

        all_x = list(x_orig)
        all_labels = list(labels_o)

        # Section 2: CT sensitivity (ct=5 / ct=10 / ct=30)
        for mi, method in enumerate(methods_ct):
            x_positions = ct_model_x[mi]
            avail_ct = []
            for _, ct_summary in ct_configs:
                sub = ct_summary[ct_summary["metric"] == metric]
                row = sub[sub["method"] == method]
                if not row.empty:
                    avail_ct.append(float(row["mean"].iloc[0]))
            placeholder_h_ct = float(np.mean(avail_ct)) if avail_ct else 0.0

            for ci, (ct_label, ct_summary) in enumerate(ct_configs):
                sub = ct_summary[ct_summary["metric"] == metric]
                row = sub[sub["method"] == method]
                color = CT_EXTRA_COLORS[ci % len(CT_EXTRA_COLORS)]
                if row.empty:
                    patch = ax.bar(
                        [x_positions[ci]], [placeholder_h_ct],
                        color="none", edgecolor="#AAAAAA", linewidth=1.5,
                    )[0]
                    patch.set_linestyle("--")
                else:
                    mean_val = float(row["mean"].iloc[0])
                    std_val = float(row["std"].iloc[0])
                    bar_ct = ax.bar([x_positions[ci]], [mean_val], yerr=[[std_val]],
                                    color=color, capsize=3)[0]
                    # Outline the within-section baseline (ct=30) for this model.
                    if ct_label == CT_BASELINE_LABEL:
                        bar_ct.set_edgecolor("#212121")
                        bar_ct.set_linewidth(REF_OUTLINE_LW)
                    # Star above non-baseline configs if significant.
                    elif stats_ct is not None and not stats_ct.empty:
                        srow = stats_ct[
                            (stats_ct["metric"] == metric)
                            & (stats_ct["method"] == method)
                            & (stats_ct["config"] == ct_label)
                        ]
                        if not srow.empty:
                            star = stars(float(srow["p_adj"].iloc[0]))
                            if star:
                                ax.text(x_positions[ci], mean_val + std_val + 0.02 * max(mean_val, 1e-3),
                                        star, ha="center", va="bottom", fontsize=FS_STAR)
                                has_stars = True
                all_x.append(x_positions[ci])
                all_labels.append(ct_label)

            if is_bottom:
                x_center = float(x_positions[0] + x_positions[-1]) / 2
                ax.annotate(_wrap_group_label(METHOD_DISPLAY.get(method, method)),
                            xy=(x_center, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, GROUP_LABEL_DROP), textcoords="offset points",
                            ha="center", va="top",
                            fontsize=FS_GROUP, fontweight="bold",
                            annotation_clip=False)

        # Section 3: gene-count (1k vs 2k) — -both methods excluded (no 2k data).
        for mi, method in enumerate(methods_gene):
            x_positions = gene_model_x[mi]
            avail_gene = []
            for _, gene_summary in gene_configs:
                sub = gene_summary[gene_summary["metric"] == metric]
                row = sub[sub["method"] == method]
                if not row.empty:
                    avail_gene.append(float(row["mean"].iloc[0]))
            placeholder_h_gene = float(np.mean(avail_gene)) if avail_gene else 0.0

            for gi, (gene_label, gene_summary) in enumerate(gene_configs):
                sub = gene_summary[gene_summary["metric"] == metric]
                row = sub[sub["method"] == method]
                color = GENE_COLORS[gi % len(GENE_COLORS)]
                if row.empty:
                    patch = ax.bar(
                        [x_positions[gi]], [placeholder_h_gene],
                        color="none", edgecolor="#AAAAAA", linewidth=1.5,
                    )[0]
                    patch.set_linestyle("--")
                else:
                    mean_val = float(row["mean"].iloc[0])
                    std_val = float(row["std"].iloc[0])
                    bar_gene = ax.bar([x_positions[gi]], [mean_val], yerr=[[std_val]],
                                      color=color, capsize=3)[0]
                    if gene_label == GENE_BASELINE_LABEL:
                        bar_gene.set_edgecolor("#212121")
                        bar_gene.set_linewidth(REF_OUTLINE_LW)
                    elif stats_gene is not None and not stats_gene.empty:
                        srow = stats_gene[
                            (stats_gene["metric"] == metric)
                            & (stats_gene["method"] == method)
                            & (stats_gene["config"] == gene_label)
                        ]
                        if not srow.empty:
                            star = stars(float(srow["p_adj"].iloc[0]))
                            if star:
                                ax.text(x_positions[gi], mean_val + std_val + 0.02 * max(mean_val, 1e-3),
                                        star, ha="center", va="bottom", fontsize=FS_STAR)
                                has_stars = True
                all_x.append(x_positions[gi])
                all_labels.append(gene_label)

            if is_bottom:
                x_center = float(x_positions[0] + x_positions[-1]) / 2
                ax.annotate(_wrap_group_label(METHOD_DISPLAY.get(method, method)),
                            xy=(x_center, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, GROUP_LABEL_DROP), textcoords="offset points",
                            ha="center", va="top",
                            fontsize=FS_GROUP, fontweight="bold",
                            annotation_clip=False)

        # Axis formatting
        ax.set_xticks(all_x)
        ax.set_xticklabels(all_labels, rotation=90, ha="center")
        ax.tick_params(axis="x", labelsize=FS_TICK, labelbottom=is_bottom)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.set_xlim(-(BAR_HALF_WIDTH + EDGE_GAP), x_max)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.relim()
        ax.autoscale_view()
        _, cur_top = ax.get_ylim()
        ax.set_ylim(top=cur_top * (1.12 if has_stars else 1.05))
        # The y label is rotated 90° CCW, which rotates its glyphs too, so the
        # arrow is pre-rotated: "←" renders as ↓ (lower is better) and "→" as ↑.
        arrow = "←" if METRIC_DIRECTION.get(metric, "up") == "down" else "→"
        ax.set_ylabel(f"{metric} [{arrow}]", fontsize=FS_TITLE)

    # Legend: four blocks -- group colours | annotations | section-2/3 configs |
    # significance -- as columns of a strip across the top of the figure rather
    # than a tall stack beside the rows, which would cost ~3 in of the 16.5 the
    # 30 bars need. Drawn by hand so each block carries its own bold heading.
    band_w = FIG_WIDTH_IN - LEFT_MARGIN_IN - LEGEND_EDGE_IN
    ax_leg = fig.add_axes([
        LEFT_MARGIN_IN / FIG_WIDTH_IN,
        1.0 - (LEGEND_TOP_IN + LEGEND_HEIGHT_IN) / fig_height,
        band_w / FIG_WIDTH_IN,
        LEGEND_HEIGHT_IN / fig_height,
    ])
    ax_leg.set_xlim(0.0, band_w)             # data units are inches,
    ax_leg.set_ylim(LEGEND_HEIGHT_IN, 0.0)   # y measured downwards
    draw_legend(ax_leg, legend_groups,
                [label for label, _ in ct_configs],
                [label for label, _ in gene_configs])

    # Laid out in inches rather than by tight_layout: the group labels hang
    # outside the axes, and the bands have to be reserved for them explicitly
    # so nothing is pushed past the canvas edge (save_figure crops to it).
    fig.subplots_adjust(
        left=LEFT_MARGIN_IN / FIG_WIDTH_IN,
        right=1.0 - RIGHT_MARGIN_IN / FIG_WIDTH_IN,
        bottom=XLABEL_BAND_IN / fig_height,
        top=1.0 - LEGEND_BAND_IN / fig_height,
        hspace=0,
    )

    # The rows are flush, so a tick the locator puts just outside the view --
    # it is not drawn, but it still carries a label -- would set that label
    # down on the neighbouring row's end tick. Drop those, once the rows have
    # their final height and the locator has settled on its tick spacing.
    fig.canvas.draw()
    for ax in axes:
        low, high = ax.get_ylim()
        ax.set_yticks([t for t in ax.get_yticks() if low <= t <= high])

    report_text_overlaps(fig, out_name)
    # PNG for quick viewing, PDF (vector) alongside it for the manuscript.
    save_figure(fig, out_dir, out_name)
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supplementary Figure S1: multi-section supp bars (no randomization).")
    add_io_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    perf_dir = args.data_dir / "metrics"
    # metric_data_TCGA_1000_30.xlsx is read by both section 2 and section 3;
    # the cache keeps it to a single parse.
    xlsx_cache: Dict[Path, pd.DataFrame] = {}

    # Held-out metrics: SS / ARI / NMI of Figure 2's runs, and all four of the sweep runs'.
    paths = test_split_paths(args.data_dir)
    for key, step in (("gonnect", "test_split_metrics.py"), ("sweep", "cluster_test_metrics.py s1")):
        if not paths[key].exists():
            raise SystemExit(f"missing {paths[key]}; build it with prepare/{step}")
    gonnect_clustering = pd.read_csv(paths["gonnect"])
    sweep_metrics = pd.read_csv(paths["sweep"])

    def clustering_for(label: str) -> pd.DataFrame:
        if label in SWEEP_SETTINGS:
            return sweep_metrics[sweep_metrics["setting"] == SWEEP_SETTINGS[label]]
        return gonnect_clustering

    # ── Section 1 data ─────────────────────────────────────────────────────
    data_orig = collect_orig_data(args.data_dir, gonnect_clustering)
    if data_orig.empty:
        raise SystemExit("No original data found.")
    data_orig = keep_main_methods(data_orig)
    summary_sec1 = compute_summary(data_orig)
    stats_table = paired_t_tests(data_orig, baseline=BASELINE_METHOD)
    if not stats_table.empty:
        print("Section 1 t-tests vs", BASELINE_METHOD)
        print(stats_table.to_string(index=False))

    # ── Section 2 (CT sensitivity) ─────────────────────────────────────────
    ct_configs: List[Tuple[str, pd.DataFrame]] = []
    ct_raw: List[Tuple[str, pd.DataFrame]] = []
    for filename, label in CT_CONFIG_FILES:
        path = perf_dir / filename
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {label}")
            continue
        df = keep_main_methods(on_test_split(read_config_frame(path, xlsx_cache),
                                             clustering_for(label)))
        if df.empty:
            print(f"WARNING: {filename} produced no records, skipping {label}")
            continue
        ct_raw.append((label, df))
        ct_configs.append((label, compute_summary(df)))

    # ── Section 3 (gene count) ─────────────────────────────────────────────
    ct30_path = perf_dir / "metric_data_TCGA_1000_30.xlsx"
    gene_raw: List[Tuple[str, pd.DataFrame]] = []
    if ct30_path.exists():
        df_1k = keep_main_methods(on_test_split(read_config_frame(ct30_path, xlsx_cache),
                                                clustering_for("1k")))
        gene_configs: List[Tuple[str, pd.DataFrame]] = [("1k", compute_summary(df_1k))]
        gene_raw.append(("1k", df_1k))
    else:
        gene_configs = [("1k", summary_sec1)]
    gene_filename, gene_label = GENE_CONFIG_FILE
    gene_path = perf_dir / gene_filename
    if gene_path.exists():
        df_2k = keep_main_methods(on_test_split(read_config_frame(gene_path, xlsx_cache),
                                                clustering_for(gene_label)))
        if not df_2k.empty:
            gene_configs.append((gene_label, compute_summary(df_2k)))
            gene_raw.append((gene_label, df_2k))

    # Within-section significance tests (per model, vs each section's baseline).
    stats_ct = per_model_section_t_tests(ct_raw, baseline_label="ct=30")
    stats_gene = per_model_section_t_tests(gene_raw, baseline_label="1k")
    if not stats_ct.empty:
        print("\nSection 2 (CT) per-model t-tests vs ct=30:")
        print(stats_ct.to_string(index=False))
    if not stats_gene.empty:
        print("\nSection 3 (gene) per-model t-tests vs 1k:")
        print(stats_gene.to_string(index=False))

    plot_summary_supp(summary_sec1, ct_configs, gene_configs,
                      out_dir=args.out_dir, out_name=OUT_NAME,
                      stats_ct=stats_ct, stats_gene=stats_gene)


if __name__ == "__main__":
    main()
