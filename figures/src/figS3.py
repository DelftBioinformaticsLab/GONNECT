"""Supplementary Figure S3: per-cancer-type embedding quality, all ten models.

WHAT THIS FIGURE SHOWS
----------------------
One shared row axis -- the 32 TCGA cancer types, in the paper's standard
abundance order (``_common.CANCER_TYPE_ORDER``) -- against five views of how
well each of the ten models separates those types, plus the abundance itself:

  a  k-NN purity, k = 10   [^]   local label mixing in the latent space
  b  k-NN purity, k = 20   [^]
  c  k-NN purity, k = 30   [^]   the k the main-text benchmark reports
  d  silhouette score (SS) [^]   cluster compactness vs. separation
  e  reconstruction MSE    [v]   how well the model reproduces the input
  f  abundance [n]               samples per cancer type

The figure carries no title: what it shows belongs in the caption, not on the
canvas.

Every heatmap carries the same ten methods in the same column order: the seven
of the main-text benchmark (MLP, GONNECT enc/dec/both, GONNECT-SL enc/dec/both)
plus the three degree-preserving randomized models (GONNECT-DPR enc/dec/both,
the AE_2.2 arm), which the main text shows only in aggregate. Putting the
randomized arm next to the true-graph models per cancer type is the point of
this figure: it shows whether the GO prior helps or hurts on a *type-by-type*
basis, not just on the dataset mean. The three purity panels add an eleventh
column, ``Random``: chance level, the purity a type would get if its
neighbours were drawn at random from the training split, which is the type's
share of that split. It is the same for every k and every model, and it is
what a purity value should be read against: a rare type's 0.3 is far above its
chance of ~0.01, while BRCA's chance level alone is ~0.11, as in Figure 2k.

THE METRICS, AND WHY THEY CAN DISAGREE
--------------------------------------
The three metrics ask genuinely different questions, so a model can win one and
lose another. Read them together, not as three copies of one ranking.

Every metric scores held-out samples: the rows each run's split left out of
training (splits 2-6; the randomized arm, numbered 22-26, trained on the same
five). *k-NN purity* (panels a-c) is the fraction of a held-out sample's k
nearest neighbours (Euclidean, in the full latent space) that carry its own
cancer-type label, averaged over the held-out samples of that type and then over
the five model seeds. The neighbours are drawn from the run's training split,
not from the held-out rows themselves: a test split holds 15% of the data, too
few for many types to fill k = 30 neighbours with their own kind. It is purely local: it sees whether the immediate
neighbourhood is contaminated, and nothing else. A cluster that is enormous and
diffuse but uncontaminated scores 1.0. Showing k = 10, 20 and 30 side by side
makes the *slope* visible -- a type whose purity falls off quickly with k sits
in a small, tight island close to other types, while a flat profile means a
genuinely isolated region. Computed by ``prepare/test_split_metrics.py``, which
also writes the chance level (the type's share of the training split) that
Figure 2k shows.

*Silhouette score* (panel d) is geometric where purity is topological: for each
sample, (distance to the nearest other cluster - mean distance within its own
cluster) / max of the two, so it rewards clusters that are simultaneously tight
and far from their neighbours, and goes negative when a type's samples sit
closer to some other type's centre than to their own. Purity can be high while
SS is near zero (a clean but sprawling cluster), which is exactly the kind of
disagreement this figure is meant to expose. It is computed over each run's
held-out rows, so it is the per-type breakdown of Figure 2b, as MSE is of 2a.
Computed by ``prepare/test_split_metrics.py``.

*Reconstruction MSE* (panel e) is the only metric here that does not mention
the labels at all -- it is the autoencoder's own training objective, per cancer
type. It is included because a model can buy label structure with
reconstruction fidelity, and the reader should be able to see that trade
directly. This is the one panel where lower is better. Read from the workbook,
which took it on each run's held-out rows.
Cells above ``MSE_DIVERGED_THRESHOLD`` would be runs that did not converge, and
are drawn grey with an "x" rather than being allowed to flatten the colour
scale, following the main-text figure. As of the locale repair in
``_common._decoerce`` no cell trips that threshold, so the mark is a guard
rather than something the current data shows.

COLOUR SCALES
-------------
The three purity panels are bounded [0, 1] and share a fixed 0-1
red-yellow-green scale, so they are directly comparable by eye -- the same
colour means the same number in panels a, b and c. Green is "good" throughout,
the convention the rest of the paper uses, and it is the same map the main-text
figure gives its purity panel so the two can be read against each other without
re-learning the colours. SS and MSE need data-driven limits. SS gets a
diverging scale with symmetric limits about zero, so that the sign (samples
nearer another type's centre than their own) reads at a glance; MSE gets a
sequential scale from its minimum to its 95th percentile, as in the main-text
figure. Each heatmap has its own colourbar, immediately above it.

LAYOUT
------
One row of five heatmaps plus the abundance bar, laid out in absolute inches
(see the "layout" block). Five panels of ten or eleven columns on a 16.5 in canvas
is the width problem this figure has to solve, and two things solve it: the
colourbars are horizontal and sit in the band *above* each heatmap, so they
cost height rather than the ~1 in of width each a vertical colourbar would take
(figS5's arrangement, which does not survive five panels), and the 32
cancer-type names are written once, beside the leftmost heatmap, since all six
panels share the row axis.

A single row was chosen over two rows because the row axis is the whole point:
a reader following one cancer type across purity, SS and MSE does it in one
horizontal sweep, and a two-row version would break that sweep in half, repeat
the rotated method labels and the row labels twice, and make the figure roughly
twice as tall for the same information.

Inputs (relative to --data-dir)
-------------------------------
    metrics/test_split/per_type_purity_k{10,20,30}.csv
        panels a-c, cancer types x methods
    metrics/test_split/per_type_ss.csv
        panel d
    metrics/mse_per_cluster_TCGA_1000_30.xlsx
        sheet "MSE" (panel e), indexed on "Cluster"; its columns are already
        named to match the method keys
    TCGA_complete_bp_top1k.csv.gz
        cancer-type labels and the abundance

Outputs
-------
    <out-dir>/figS3.png, figS3.pdf     the figure
    <out-dir>/figS3.csv                every number in it, one row per
                                       (cancer type, metric, method)

Usage
-----
    python figS3.py [--data-dir figures/data] [--out-dir figures/out]
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from _common import (
    CANCER_TYPE_ORDER,
    EMBEDDING_MODELS,
    FIG_WIDTH_IN,
    PT_BODY,
    PT_PANEL,
    PT_SMALL,
    PT_TINY,
    PT_TITLE,
    PURITY_K_VALUES,
    add_io_args,
    display,
    figsize,
    load_cancer_types,
    pt,
    read_per_cluster_workbook,
    report_text_overlaps,
    save_figure,
    test_split_paths,
)

# ── config ────────────────────────────────────────────────────────────────────
FIG_NAME = "figS3"

# The ten models, in column order, straight from _common so this figure cannot
# drift from the cached metrics: 4 fixed-link (MLP + GONNECT enc/dec/both),
# 3 soft-link, 3 degree-preserving randomized.
METHODS: List[str] = [m for _, _, m in EMBEDDING_MODELS]

# The three purity panels share one map and one pair of limits, so the same
# colour means the same number across all three. RdYlGn rather than a
# perceptually nicer sequential map because purity at k=30 also appears in the
# main-text figure, and a reader comparing the two should not have to re-learn
# the colours; green = good is the convention the paper already uses for every
# "higher is better" panel.
UNIT_CMAP = "RdYlGn"
UNIT_TICKS = [0.0, 0.5, 1.0]

SS_CMAP = "RdYlGn"           # diverging, as in the main-text figure
MSE_CMAP = "YlOrRd"          # sequential, dark = high = bad

# MSE above this is a run that did not converge, drawn grey with an "x" so it
# cannot compress every real value into the first colour step. No cell trips it
# any more: the 28 that used to were Excel's locale coercion rather than
# diverged models, and are decoded on read (see _common._decoerce). Matches the
# main-text figure's threshold and is kept as the same guard.
MSE_DIVERGED_THRESHOLD = 10.0
MSE_UPPER_PERCENTILE = 95    # colour limit, as in the main-text figure

DIVERGED_FACECOLOR = "#CCCCCC"
DIVERGED_TEXTCOLOR = "#666666"


# ── layout ────────────────────────────────────────────────────────────────────
# Inches on the standard canvas, which is FIG_WIDTH_IN wide whatever else
# changes. save_figure keeps the full width, so the numbers below have to add
# up to 16.5 exactly: a leftover margin becomes white space in the paper and
# anything past the edge is lost.
#
# The width is the whole problem here. Five heatmaps of 53 columns in all, at
# the paper's shared 6 pt tick size, leaves about 0.25 in per column against a
# rotated method label 0.19 in across, so the columns have room to spare; the
# two decisions that bought that width still hold, and would be needed again if
# a panel were ever added back: horizontal colourbars above each heatmap
# instead of vertical ones beside it (which would have cost ~1 in x 5), and the
# cancer-type names written once on the left instead of six times.
MARGIN_L = 0.86       # "BRCA" is 0.53 in + tick pad, the rotated y label 0.26
MARGIN_R = 0.18       # overhang of the abundance panel's last x tick label
GAP_X = 0.24          # between heatmaps
GAP_AB = 0.48         # last heatmap -> abundance panel
ABUND_W = 1.28        # wide enough for its own two-line title

HEAT_H = 6.80         # 32 rows -> 0.2125 in each, matching figS5

# The figure carries no suptitle -- what it shows goes in the caption -- so the
# panel letters are the topmost artists and only need a hairline of pad above
# them.
PAD_TOP = 0.06
LETTER_H = 0.46       # panel letters get their own line above the titles, so a
                      # full-width title cannot run into them. The band is what
                      # sets how far the letters sit above their titles: the
                      # letters are the topmost artists and the save crops to
                      # them, so raising a letter moves everything else down.
LETTER_DX = 0.13      # letters overhang their panel's left edge, as in the
                      # main-text figures
TITLE_H = 0.64        # two lines at PT_TITLE
TITLE_GAP = 0.08
CBAR_LABEL_H = 0.26   # colourbar tick labels, which sit above the bar
CBAR_H = 0.13
CBAR_GAP = 0.14       # colourbar -> heatmap
XTICK_H = 1.88        # the rotated method names ("GONNECT-DPR-both" is 1.71 in)
MARGIN_B = 0.06

# Colourbar inset as a fraction of the heatmap width on each side. Keeping the
# bar off the panel edges keeps its outermost tick labels inside the panel too,
# so GAP_X only has to separate the panels visually.
CBAR_INSET = 0.10

N_HEATMAPS = 5

HEAD_H = LETTER_H + TITLE_H + TITLE_GAP + CBAR_LABEL_H + CBAR_H + CBAR_GAP
FIG_H = PAD_TOP + HEAD_H + HEAT_H + XTICK_H + MARGIN_B

# Width shared by the five heatmaps. Each gets a share in proportion to its
# column count, so a cell is equally wide in every panel: the purity panels
# carry one more column than the others (chance level).
HEATS_W = FIG_WIDTH_IN - MARGIN_L - MARGIN_R - ABUND_W - GAP_AB - (N_HEATMAPS - 1) * GAP_X

# Purity's extra column: chance level, the cancer type's share of the training
# split the neighbours are drawn from, as in Figure 2k.
PURITY_CHANCE = "Random"


def _rect(x: float, top: float, w: float, h: float) -> list:
    """Inches from the top-left corner -> the figure-fraction rect add_axes wants."""
    return [x / FIG_WIDTH_IN, 1.0 - (top + h) / FIG_H, w / FIG_WIDTH_IN, h / FIG_H]


def _panel_widths(n_cols: List[int]) -> List[float]:
    """Heatmap widths in inches, in proportion to their column counts."""
    cell = HEATS_W / sum(n_cols)
    return [n * cell for n in n_cols]


# ── inputs ────────────────────────────────────────────────────────────────────

def _require_complete(frame: pd.DataFrame, what: str) -> pd.DataFrame:
    """Fail loudly on a table with holes in it.

    Everything here is reindexed onto the shared cancer-type list, so a name
    that does not match between a source and ``CANCER_TYPE_ORDER`` comes back
    as a row of NaN -- which would draw as a blank stripe across a heatmap and
    look like a real result rather than a lookup failure.
    """
    bad = frame.index[frame.isna().all(axis=1)].tolist()
    if bad:
        raise SystemExit(f"{what}: no data for cancer type(s) "
                         f"{', '.join(map(str, bad))}")
    holes = int(frame.isna().to_numpy().sum())
    if holes:
        print(f"  WARNING: {what} has {holes} missing cell(s); "
              f"they are drawn blank")
    return frame


def load_per_type_table(path, cancer_types: List[str], columns: List[str]) -> pd.DataFrame:
    """One held-out per-type table (cancer types x ``columns``).

    The purity tables carry a chance-level ``Random`` column after the ten
    models; the SS table does not.
    """
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}; build it with "
                         f"prepare/test_split_metrics.py")
    frame = pd.read_csv(path, index_col=0)
    missing = [m for m in columns if m not in frame.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing column(s): {', '.join(missing)}")
    return _require_complete(frame[columns].reindex(cancer_types), path.name)


def load_workbook_metrics(path, cancer_types: List[str]) -> Dict[str, pd.DataFrame]:
    """The "MSE" sheet of the per-cluster workbook, as a ten-column frame.

    Its "SS" sheet is not read: it spans all samples, where panel d scores the
    held-out ones.

    The workbook's columns are already the method keys used everywhere else, so
    no renaming is needed -- only selection and reordering.
    """
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    book = read_per_cluster_workbook(path)
    out = {}
    for sheet in ("MSE",):
        if sheet not in book:
            raise SystemExit(f"{path.name} has no sheet {sheet!r} "
                             f"(sheets: {', '.join(book)})")
        frame = book[sheet]
        missing = [m for m in METHODS if m not in frame.columns]
        if missing:
            raise SystemExit(f"{path.name}[{sheet}] is missing method column(s): "
                             f"{', '.join(missing)}")
        out[sheet] = _require_complete(frame[METHODS].reindex(cancer_types),
                                       f"{path.name}[{sheet}]")
    return out


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_heatmap(fig, ax, cax, data: np.ndarray, cancer_types: List[str],
                 method_labels: List[str], *, cmap: str, vmin: float,
                 vmax: float, norm=None, cbar_ticks=None,
                 y_labels: bool = False, diverged: np.ndarray | None = None):
    """One cancer-type x method heatmap with its own colourbar above it.

    ``y_labels`` writes the 32 cancer-type names down the y axis; only the
    leftmost panel does, since all seven panels share the row axis and
    repeating the names would cost the width the method labels need.

    The panel title is not set here: the colourbar occupies the space directly
    above the axes, so titles are placed by ``plot_figure`` in their own band
    above it (see the layout block).
    """
    kwargs = ({"norm": norm} if norm is not None
              else {"vmin": vmin, "vmax": vmax})
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap,
                   interpolation="nearest", **kwargs)

    if diverged is not None:
        for i, j in zip(*np.nonzero(diverged)):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=0,
                                       facecolor=DIVERGED_FACECOLOR))
            ax.text(j, i, "x", ha="center", va="center", fontsize=PT_TINY,
                    color=DIVERGED_TEXTCOLOR)

    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels(method_labels, rotation=90, ha="center", fontsize=PT_SMALL)
    ax.set_yticks(range(len(cancer_types)))
    ax.set_yticklabels(cancer_types if y_labels else [], fontsize=PT_SMALL)
    ax.tick_params(length=0, pad=pt(1.5))
    if y_labels:
        ax.set_ylabel("Cancer type", fontsize=PT_BODY, labelpad=pt(1.0))

    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)
    else:
        cb.locator = MaxNLocator(nbins=3)
        cb.update_ticks()
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(labelsize=PT_SMALL, length=pt(1.5), pad=pt(1.0),
                      width=plt.rcParams["xtick.major.width"])
    cb.outline.set_linewidth(plt.rcParams["axes.linewidth"])
    return im


def draw_abundance(ax, cancer_types: List[str], abundance: pd.Series) -> None:
    """Samples per cancer type, on the same row axis as the heatmaps."""
    n = len(cancer_types)
    ax.barh(range(n), abundance.loc[cancer_types].to_numpy(),
            color=plt.cm.Blues(np.linspace(0.4, 0.85, n)), height=0.8,
            edgecolor="none")
    ax.set_yticks([])
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(left=False, bottom=True, labelsize=PT_SMALL, pad=pt(1.5))
    ax.spines[["top", "right"]].set_visible(False)


def _panel_letter(fig, x: float, letter: str) -> None:
    """Panel letter on its own line, above and just left of the panel."""
    fig.text((x - LETTER_DX) / FIG_WIDTH_IN, 1.0 - PAD_TOP / FIG_H, letter,
             fontsize=PT_PANEL, fontweight="bold", va="top", ha="left")


def _panel_title(fig, x: float, w: float, title: str) -> None:
    """Panel title, centred over the panel in the band under the letters.

    Placed in figure coordinates rather than with ``set_title`` because the
    colourbar sits between the title band and the heatmap, and a title
    belonging to either axes would land on the other's tick labels.
    """
    fig.text((x + w / 2) / FIG_WIDTH_IN, 1.0 - (PAD_TOP + LETTER_H) / FIG_H,
             title, fontsize=PT_TITLE, ha="center", va="top",
             linespacing=1.25)


def plot_figure(panels: List[dict], cancer_types: List[str],
                abundance: pd.Series):
    """The whole figure: five heatmaps then the abundance bar, in one row."""
    fig = plt.figure(figsize=figsize(FIG_H))

    heat_top = PAD_TOP + HEAD_H
    cbar_top = heat_top - CBAR_GAP - CBAR_H

    x = MARGIN_L
    for i, (panel, w) in enumerate(zip(panels, _panel_widths([len(p["methods"]) for p in panels]))):
        ax = fig.add_axes(_rect(x, heat_top, w, HEAT_H))
        cax = fig.add_axes(_rect(x + CBAR_INSET * w, cbar_top,
                                 (1.0 - 2 * CBAR_INSET) * w, CBAR_H))
        draw_heatmap(fig, ax, cax, panel["data"], cancer_types,
                     [display(m) for m in panel["methods"]],
                     cmap=panel["cmap"], vmin=panel["vmin"], vmax=panel["vmax"],
                     norm=panel.get("norm"), cbar_ticks=panel.get("cbar_ticks"),
                     y_labels=(i == 0), diverged=panel.get("diverged"))
        if PURITY_CHANCE in panel["methods"]:
            # Split chance level off from the models, as Figure 2k does.
            ax.axvline(panel["methods"].index(PURITY_CHANCE) - 0.5, color="white", linewidth=2.5)
        _panel_letter(fig, x, "abcde"[i])
        _panel_title(fig, x, w, panel["title"])
        x += w + GAP_X

    x_ab = x - GAP_X + GAP_AB
    ax_ab = fig.add_axes(_rect(x_ab, heat_top, ABUND_W, HEAT_H))
    draw_abundance(ax_ab, cancer_types, abundance)
    _panel_letter(fig, x_ab, "f")
    _panel_title(fig, x_ab, ABUND_W, "Abundance\n[n]")

    # No suptitle: what the figure shows is the caption's job, not the canvas'.
    return fig


# ── tidy side-car ─────────────────────────────────────────────────────────────

def build_csv(panels: List[dict], cancer_types: List[str],
              abundance: pd.Series) -> pd.DataFrame:
    """Every number in the figure, one row per (cancer type, metric, method)."""
    rows = []
    for panel in panels:
        diverged = panel.get("diverged")
        for i, ct in enumerate(cancer_types):
            for j, method in enumerate(panel["methods"]):
                rows.append({
                    "cancer_type": ct,
                    "abundance": int(abundance[ct]),
                    "metric": panel["metric"],
                    "direction": panel["direction"],
                    "method": method,
                    "method_label": display(method),
                    "value": float(panel["raw"][i, j]),
                    "diverged": bool(diverged[i, j]) if diverged is not None
                                else False,
                })
    return pd.DataFrame(rows)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-cancer-type embedding quality across all ten models "
                    f"-> {FIG_NAME}")
    add_io_args(parser)
    args = parser.parse_args()

    tcga_path = args.data_dir / "TCGA_complete_bp_top1k.csv.gz"
    paths = test_split_paths(args.data_dir)
    per_ct_file = args.data_dir / "metrics" / "mse_per_cluster_TCGA_1000_30.xlsx"
    for path in (tcga_path, per_ct_file):
        if not path.exists():
            raise SystemExit(f"Input file not found: {path}")

    print("Loading TCGA labels ...")
    labels = load_cancer_types(tcga_path)
    abundance = labels.value_counts()
    cancer_types = [ct for ct in CANCER_TYPE_ORDER if ct in set(labels)]
    print(f"  {len(labels)} samples, {len(cancer_types)} cancer types, "
          f"{len(METHODS)} methods")

    print("Loading per-cancer-type purity / SS (held out) and MSE ...")
    purity = {k: load_per_type_table(paths[f"purity_per_type_k{k}"], cancer_types,
                                     METHODS + [PURITY_CHANCE])
              for k in PURITY_K_VALUES}
    ss_vals = load_per_type_table(paths["ss_per_type"], cancer_types,
                                  METHODS).to_numpy(dtype=float)
    mse_vals = load_workbook_metrics(per_ct_file, cancer_types)["MSE"].to_numpy(dtype=float)

    # MSE: hide the diverged runs from the colour scale but keep them on the
    # figure as marked cells, so an empty-looking column is never ambiguous
    # between "not run" and "did not converge".
    mse_diverged = mse_vals > MSE_DIVERGED_THRESHOLD
    mse_display = np.where(mse_diverged, np.nan, mse_vals)
    print(f"  {int(mse_diverged.sum())} of {mse_diverged.size} MSE cells above "
          f"{MSE_DIVERGED_THRESHOLD:g} (diverged), marked with 'x'")

    # SS gets a diverging map pinned at zero, because zero is what the metric's
    # sign means: below it a type's samples sit closer to some other type's
    # cluster than to their own. Plain min/max limits would put the map's
    # midpoint colour at +0.21 and quietly recolour half the negatives as
    # "average"; symmetric limits would keep the sign honest but waste the
    # whole -0.76..-0.33 half of the scale on data that does not exist and
    # flatten the panel. TwoSlopeNorm does both: zero at the midpoint colour,
    # each side scaled to the range the data actually occupies.
    ss_lo, ss_hi = float(np.nanmin(ss_vals)), float(np.nanmax(ss_vals))
    ss_norm = TwoSlopeNorm(vmin=min(ss_lo, -1e-6), vcenter=0.0,
                           vmax=max(ss_hi, 1e-6))
    # A two-slope scale needs its own ticks: the automatic locator picks one
    # step for a bar whose two halves have different steps, and here it drops
    # the negative end entirely. Both extremes plus zero, rounded inwards so
    # they stay inside the range and get drawn.
    ss_ticks = [np.ceil(ss_lo * 10) / 10, 0.0, np.floor(ss_hi * 10) / 10]

    panels: List[dict] = []
    for k in PURITY_K_VALUES:
        vals = purity[k].to_numpy(dtype=float)
        panels.append({
            "metric": f"knn_purity_k{k}", "direction": "up",
            "methods": METHODS + [PURITY_CHANCE], "data": vals, "raw": vals,
            "cmap": UNIT_CMAP, "vmin": 0.0, "vmax": 1.0,
            "cbar_ticks": UNIT_TICKS,
            "title": f"k-NN purity [↑]\nk = {k}",
        })
    panels.append({
        "metric": "SS", "direction": "up",
        "methods": METHODS, "data": ss_vals, "raw": ss_vals,
        "cmap": SS_CMAP, "vmin": ss_lo, "vmax": ss_hi, "norm": ss_norm,
        "cbar_ticks": ss_ticks, "title": "Silhouette [↑]\n(SS)",
    })
    panels.append({
        "metric": "MSE", "direction": "down",
        "methods": METHODS, "data": mse_display, "raw": mse_vals, "diverged": mse_diverged,
        "cmap": MSE_CMAP,
        "vmin": float(np.nanmin(mse_display)),
        "vmax": float(np.nanpercentile(mse_display, MSE_UPPER_PERCENTILE)),
        "title": "Reconstruction\nMSE [↓]",
    })

    for panel in panels:
        finite = panel["raw"][np.isfinite(panel["raw"])]
        print(f"  {panel['metric']:16s} range "
              f"[{finite.min():.3g}, {finite.max():.3g}]  "
              f"colour [{panel['vmin']:.3g}, {panel['vmax']:.3g}]")

    fig = plot_figure(panels, cancer_types, abundance)
    n_overlaps = report_text_overlaps(fig, FIG_NAME)
    save_figure(fig, args.out_dir, FIG_NAME)

    csv_path = args.out_dir / f"{FIG_NAME}.csv"
    build_csv(panels, cancer_types, abundance).to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    plt.close(fig)
    if n_overlaps:
        print(f"WARNING: {n_overlaps} overlapping text pair(s) remain")


if __name__ == "__main__":
    main()
