"""Supplementary Figure S5: mean GO-term activations per cancer type, encoder.

Twenty GO-term nodes of the GONNECT encoder, shown for three independently
trained instances of the same model (seeds 2, 3, 4). Rows are the 32 TCGA
cancer types, columns the 20 GO terms; a cell is the mean activation of that
node over all samples of that cancer type.

  a  signed mean activation per (cancer type, GO term), one heatmap per instance
  b  the absolute value of exactly the numbers in panel a

Panel a shows that the sign of the mean activation is essentially arbitrary
across instances -- an autoencoder hidden unit has no built-in "high = up"
convention. Panel b shows that the magnitude of some term / cancer-type pairs
is consistently high across instances. All heatmaps share one symmetric
blue-white-red scale centred on zero, so panel B reads as white-to-red.

Layout: each panel is a row of one heatmap per instance, each with its own
colourbar. This is the densest label case in the paper -- 32 cancer types down
every y axis and 20 GO terms under every x axis, each written as its process
name followed by its accession, at the paper's shared 6 pt tick size -- so the
row is laid out in absolute inches (see the constants under "layout") to give
every label its own space. The cancer types are written only beside the
leftmost heatmap of each row and the GO terms only under the bottom row, since
those are shared; the x band is measured from the labels themselves
(xtick_band_in) rather than assumed. The figure is correspondingly tall.

figS6.py is the decoder counterpart: it imports main() from here and runs it
with module="decoder".

Inputs (relative to --data-dir)
------------------------------
    go_term_activations_corrected/AE_2.0.<seed>_encoder_activations.csv.gz
        one file per model instance; columns are patient_id, four more sample
        metadata columns, then one column per GO-term node named by GO ID.
        --preprint reads go_term_activations/ instead, whose encoder files
        are the same; only its decoder files (Figure S6) are labelled one
        layer off
    TCGA_complete_bp_top1k.csv.gz
        the sample table the activations are aligned to, row for row
    hard_links.csv
        the GONNECT graph, used to restrict the columns to GO-term nodes of
        this module's layers (proxy and gene nodes are not eligible)

GO-term selection
-----------------
The 20 hand-picked GO terms of the preprint v3 figure -- "processes expected to
vary in activity across cancer types" -- are in GO_TERMS below, read off the x
axis of the preprint v3 figure in its own column order. Every id was checked
against hard_links.csv: all 20 are real GO nodes present in both modules, and
none is a bottleneck node. Set GO_TERMS to None to fall back on a
deterministic stand-in instead (the 20 nodes with the highest eta-squared,
the share of a node's variance lying between cancer types rather than within
them, computed on the first seed and held fixed across instances).

Colour scale
------------
Each column is centred on its own mean across cancer types, then each term is
scaled into [-1, 1] (normalize_terms). Both steps are needed and neither is
cosmetic.

*Scaling*, because one colour scale cannot serve these 20 terms raw: their
maxima span 37x in the encoder and 1,957x in the decoder, where one term
reaches 170 while the next largest stays under 16. Raw, the decoder's scale is
set by that single node and every other column is pale. Scaling asks what the panels
are about -- which cancer types light a process up, and whether the instances
agree -- rather than which process happens to have the largest activations.

*Centring*, because a node's overall offset comes from that instance's random
initialisation rather than the biology, and it is often the largest thing in
the column. Scaling without removing it first turns a constant column into a
saturated stripe of -1 that reads as the strongest result in the figure.
Columns with no variation at all are drawn blank and named in the run output
instead of being scaled up. None of the 20 is flat in the default inputs; under
--preprint, the decoder's GO:0006631 is, constant to 4e-09 across all 32
cancer types.

The scale factor is shared across instances rather than computed per instance,
so differences between instances survive; normalising each on its own would
flatten exactly the cross-instance agreement panel b exists to show.

``--no-normalize`` plots raw means instead -- which keeps magnitude comparable
between terms, at the cost above -- and ``--vmax`` fixes a symmetric limit,
which clips (on the decoder, +/-4 clips 5.5% of cells and +/-2 clips 10.8%).

Usage
-----
    python figS5.py [--data-dir figures/data] [--out-dir figures/out]
                    [--seeds 2 3 4] [--vmax V] [--no-normalize] [--preprint]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (CANCER_TYPE_ORDER, FIG_WIDTH_IN, LAYERS_BY_MODULE,
                     PT_BODY, PT_PANEL, PT_SMALL, PT_TITLE,
                     add_io_args, figsize, load_cancer_types, pt,
                     report_text_overlaps, save_figure)

# ── config ────────────────────────────────────────────────────────────────────
# The 20 hand-picked GO terms of the preprint v3 figure: "processes expected to
# vary in activity across cancer types". Read off the x axis of the preprint v3
# figure itself, in its column order, so this is the list rather than a
# reconstruction of it. Every id was checked against hard_links.csv: all 20 are
# real GO nodes, present in both the encoder and the decoder, and none is a
# bottleneck node.
#
# Set to None to fall back to the documented stand-in instead: the highest
# eta-squared terms, computed on the first seed and reused for every instance
# (see select_terms).
GO_TERMS: list[str] | None = [
    "GO:0006631",   # fatty acid metabolic process
    "GO:0008203",   # cholesterol metabolic process
    "GO:0008206",   # bile acid metabolic process
    "GO:0008207",   # C21-steroid hormone metabolic process
    "GO:0008209",   # androgen metabolic process
    "GO:0008210",   # estrogen metabolic process
    "GO:0071870",   # cellular response to catecholamine stimulus
    "GO:0061621",   # canonical glycolysis
    "GO:0090141",   # positive regulation of mitochondrial fission
    "GO:0000077",   # DNA damage checkpoint signaling
    "GO:0043406",   # positive regulation of MAP kinase activity
    "GO:0042102",   # positive regulation of T cell proliferation
    "GO:0050671",   # positive regulation of lymphocyte proliferation
    "GO:0030198",   # extracellular matrix organization
    "GO:0030199",   # collagen fibril organization
    "GO:0010718",   # positive regulation of epithelial to mesenchymal transition
    "GO:0031643",   # positive regulation of myelination
    "GO:0070572",   # positive regulation of neuron projection regeneration
    "GO:0046951",   # ketone body biosynthetic process
    "GO:0030195",   # negative regulation of blood coagulation
]

# The activations read. The corrected directory holds the re-extracted decoder
# files and the deposited encoder files, so only Figure S6 differs between the
# two. The preprint v3 directory's decoder columns are labelled one layer off
# (see prepare/README.md, *The untrained control*).
REVISED_ACTIVATIONS = "go_term_activations_corrected"
PREPRINT_ACTIVATIONS = "go_term_activations"

# Axis labels. GO_TERM_NAMES is read from hard_links.csv at import, so the
# names always match the graph rather than being a second hand-typed copy that
# could drift from it. LABEL_WITH_NAMES picks what goes under the x axis: the
# preprint v3 figure used the bare accessions, names are the readable choice.
# Terms with no name in the graph (the eta-squared stand-in's picks) fall back
# to their id either way.
LABEL_WITH_NAMES = True


def go_term_names(hard_links_path: Path) -> dict[str, str]:
    """{GO id: process name}, taken from both endpoint columns of the graph."""
    hl = pd.read_csv(hard_links_path,
                     usecols=["source_term_id", "source_term_name",
                              "sink_term_id", "sink_term_name"])
    names: dict[str, str] = {}
    for id_col, name_col in (("source_term_id", "source_term_name"),
                             ("sink_term_id", "sink_term_name")):
        names.update(dict(zip(hl[id_col], hl[name_col])))
    return names


def term_label(term: str, names: dict[str, str]) -> str:
    """Axis label for a GO node: ``process name (GO:id)``.

    Both, because the name is what makes the panel readable and the accession
    is what makes it checkable against the graph and the manuscript. Falls back
    to the bare id when the graph has no name for the node.
    """
    label = names.get(term)
    if not LABEL_WITH_NAMES or not isinstance(label, str) or not label.strip():
        return term
    return f"{label} ({term})"


# A column counts as flat when its variation across cancer types is negligible
# *relative to its own magnitude*. It has to be relative: in the preprint v3
# (mislabelled) decoder input, GO:0006631 sits at 0.004 and varies by 4e-9,
# which is float32 rounding on that value, not signal -- but 4e-9 clears any
# absolute threshold small enough to be safe for the terms that do vary.
#
# The gap either side of this line is enormous, so its exact value does not
# matter: that dead column comes in at a ratio of 1.1e-06, and every other
# column, in either module and either input, is above 0.1. In the default
# inputs no column is flat.
FLAT_RTOL = 1e-4


def normalize_terms(panel_data: list[pd.DataFrame], terms: list[str],
                    ) -> tuple[list[pd.DataFrame], list[str]]:
    """Centre each column, then scale each term into [-1, 1].

    Returns ``(scaled_frames, flat_terms)``.

    Two steps, and both are needed:

    1. **Centre** each (term, instance) column on its own mean across cancer
       types. A node's overall offset is arbitrary -- it comes out of that
       instance's random initialisation, not out of the biology -- and what the
       panels are asking is which cancer types stand out *within* a term. Left
       in, an offset is the largest thing in the column and swamps that.
    2. **Scale** by the largest remaining deviation the term reaches over all
       instances. One factor per term, shared across instances, so differences
       between instances survive; scaling each instance on its own would
       flatten exactly the cross-instance agreement panel b exists to show.

    Columns that do not vary at all are returned as zeros and named in
    ``flat_terms`` rather than being scaled. Dividing them by their own
    (numerically meaningless) spread is what an uncentred max-scaling does, and
    it turns a dead node into a saturated column of -1: in the preprint v3
    decoder input, GO:0006631 is constant to 4e-09 across all 32 cancer types,
    and scaling its 0.004 offset printed a solid -1 stripe that looked like
    the strongest result in the figure. A flat column should read as "nothing here", which is white.
    """
    centred = []
    for frame in panel_data:
        copy = frame.copy()
        for term in terms:
            column = frame[term].to_numpy(dtype=float)
            copy[term] = column - column.mean()
        centred.append(copy)

    scales = {t: max(float(np.abs(c[t].to_numpy()).max()) for c in centred)
              for t in terms}
    magnitude = {t: max(float(np.abs(d[t].to_numpy()).max()) for d in panel_data)
                 for t in terms}
    flat = [t for t in terms
            if scales[t] <= FLAT_RTOL * max(magnitude[t], np.finfo(float).tiny)]

    out = []
    for frame in centred:
        copy = frame.copy()
        for term in terms:
            copy[term] = 0.0 if term in flat else frame[term] / scales[term]
        out.append(copy)
    return out, flat

N_TERMS = 20                 # columns per heatmap
MODEL_VERSION = "2.0"        # GONNECT fixed-link models, AE_2.0.<seed>
CMAP = "bwr"                 # diverging blue-white-red, centred on zero
DEFAULT_SEEDS = [2, 3, 4]

# The 109-node latent (encoder L3 / decoder L5). Its activations run one to two
# orders of magnitude larger than the other layers (max |mean| 79 there against
# 3.95 elsewhere in the encoder), and the printed figure spans ~±4, so the
# hand-picked terms were evidently not bottleneck nodes. The default selection
# pool therefore excludes that layer. An explicit GO_TERMS list is unaffected
# and may name any GO node.
# Value is (layer index, which endpoint column carries that layer's output).
BOTTLENECK_LAYER = {"encoder": (3, "sink_term_id"),
                    "decoder": (5, "source_term_id")}


# ── layout ────────────────────────────────────────────────────────────────────
# Everything below is inches on the standard canvas, which is FIG_WIDTH_IN wide
# whatever --seeds asks for: the width of one heatmap is therefore not a free
# parameter but whatever is left of 16.5 in after the margins, the colourbars
# and the gaps, divided by the number of instances. The height is free, and
# this figure needs it -- 32 cancer types and 20 ten-character GO ids per
# heatmap, all set at the paper's shared 6 pt tick size, is the densest label
# case in the paper. Written out in inches rather than left to tight_layout
# because the label budget per cell is what has to be controlled here, and
# because the canvas must be filled edge to edge: save_figure keeps the full
# width, so a stray margin becomes white space in the paper and anything past
# the edge is lost.
#
# At n = 3 the numbers below give a cell of 0.203 x 0.213 in against a tick
# label 0.190 in across, so every label keeps its own lane with ~7% to spare.
# On the page (the figure is placed at 180 mm, a scale of 0.43) that is a
# 2.2 mm cell carrying 6 pt type, with ~0.7 mm of white between neighbouring
# GO ids -- tight, but this is the density the data has. The margins below are
# the measured extents of the text they hold plus ~0.02 in, so trimming any of
# them clips; the width they leave over is what sets the cell size. n = 2 has
# room to spare (0.33 in cells); a fourth instance would take the cell below
# the 0.19 in a GO id occupies and the column labels would start to collide.
MARGIN_L = 0.90      # rotated y label + the four-character cancer-type labels
MARGIN_R = 0.02
CBAR_PAD = 0.07      # heatmap edge -> its colourbar
CBAR_W = 0.12
CBAR_TEXT = 0.83     # colourbar ticks, their labels, the rotated "Activation"
CBAR_FRAC = 0.45     # colourbar height as a share of the heatmap height
GAP_X = 0.16         # one instance's colourbar text -> the next heatmap
HEAT_H = 6.80        # 32 rows -> 0.213 in each
TITLE_H = 0.70       # band above each heatmap for the two-line title
# The x labels are the GO process names, not the accessions, and set on their
# side the longest ("positive regulation of keratinocyte differentiation") is
# 4.78 in. Only the bottom row carries them: both rows show the same 20 terms
# in the same order, and labelling each would add another 5.18 in band, taking
# the figure to 26.3 in -- 287 mm once placed at 180 mm, past the ~247 mm of
# text an A4 page has. Shared, it lands at 234 mm and fits.
XTICK_PAD_IN = 0.40       # tick pad + the "GO terms" axis label under the names
XTICK_H_BARE = 0.30       # row 0: just air before the next row's titles
# No figure title -- what the figure shows belongs in the caption -- so the
# panel letters are the topmost artists and need only a hairline of pad.
PAD_TOP = 0.10
MARGIN_B = 0.05


def xtick_band_in(labels: list[str]) -> float:
    """Inches the rotated x tick labels need, measured rather than assumed.

    The band is the tallest thing in this figure after the heatmaps, and it
    changes with the label text -- adding the accession behind each name grew
    it by ~1.2 in. Measuring keeps the figure exactly as tall as its labels
    need instead of tracking a constant by hand every time they change.
    """
    probe = plt.figure(figsize=(FIG_WIDTH_IN, 2.0))
    renderer = probe.canvas.get_renderer()
    longest = 0.0
    for text in labels:
        artist = probe.text(0.5, 0.5, text, fontsize=PT_SMALL, rotation=90)
        longest = max(longest, artist.get_window_extent(renderer).height / probe.dpi)
        artist.remove()
    plt.close(probe)
    return longest + XTICK_PAD_IN


def _layout(n: int, xtick_h: float) -> dict:
    """Geometry of the 2 x n grid: figure height, heatmap width, and origins."""
    block = CBAR_PAD + CBAR_W + CBAR_TEXT          # colourbar and its labels
    heat_w = (FIG_WIDTH_IN - MARGIN_L - MARGIN_R
              - n * block - (n - 1) * GAP_X) / n
    row_h = [TITLE_H + HEAT_H + XTICK_H_BARE,
             TITLE_H + HEAT_H + xtick_h]
    return {
        "fig_h": PAD_TOP + sum(row_h) + MARGIN_B,
        "heat_w": heat_w,
        # left edge of each heatmap, and top of each row's title band
        "xs": [MARGIN_L + i * (heat_w + block + GAP_X) for i in range(n)],
        "tops": [PAD_TOP + sum(row_h[:r]) for r in range(2)],
    }


def _rect(fig_h: float, x: float, top: float, w: float, h: float) -> list:
    """Inches from the top-left corner -> the figure-fraction rect add_axes wants."""
    return [x / FIG_WIDTH_IN, 1.0 - (top + h) / fig_h,
            w / FIG_WIDTH_IN, h / fig_h]


# ── inputs ────────────────────────────────────────────────────────────────────

def _activation_path(activations_dir: Path, module: str, seed: int) -> Path:
    return activations_dir / f"AE_{MODEL_VERSION}.{seed}_{module}_activations.csv.gz"


def eligible_go_terms(hard_links_path: Path, module: str,
                      exclude_bottleneck: bool = False) -> set[str]:
    """GO-term node ids of this module's layers, per _common.LAYERS_BY_MODULE.

    Only real GO terms count: proxy nodes ("Proxy:...") and gene nodes
    (UniProt accessions) are dropped. Both endpoints of every link are
    considered, so hidden layers count regardless of which side they sit on.

    With ``exclude_bottleneck``, the 109-node latent is left out (see
    BOTTLENECK_LAYER for why the default selection pool excludes it).
    """
    hl = pd.read_csv(hard_links_path,
                     usecols=["component", "layer", "source_term_id",
                              "sink_term_id"])
    sub = hl[(hl["component"] == module)
             & (hl["layer"].isin(LAYERS_BY_MODULE[module]))]
    ids = set(sub["source_term_id"]) | set(sub["sink_term_id"])
    if exclude_bottleneck:
        # A node's activation belongs to the layer it is the output of: the
        # sink side going in (encoder), the source side coming out (decoder).
        layer, side = BOTTLENECK_LAYER[module]
        ids -= set(sub[sub["layer"] == layer][side])
    return {i for i in ids if str(i).startswith("GO:")}


def mean_per_cancer_type(path: Path, go_cols_allowed: set[str],
                         labels: pd.Series, patient_ids: pd.Series,
                         cancer_types: list[str]) -> pd.DataFrame:
    """Mean activation per (cancer type, GO node) for one model instance.

    The activation columns are already named by GO ID, so no index mapping is
    needed; they are intersected with ``go_cols_allowed`` to keep only GO-term
    nodes of this module. Cancer-type labels come from the TCGA table by row
    order, and the row alignment is verified on patient_id before use.
    """
    if not path.exists():
        raise SystemExit(f"Activation file not found: {path}")
    df = pd.read_csv(path)

    if len(df) != len(patient_ids):
        raise SystemExit(
            f"{path.name}: {len(df)} rows but TCGA table has "
            f"{len(patient_ids)}; the two are not the same samples.")
    if "patient_id" in df.columns and not df["patient_id"].reset_index(
            drop=True).equals(patient_ids.reset_index(drop=True)):
        raise SystemExit(
            f"{path.name}: patient_id order differs from the TCGA table, so "
            f"cancer-type labels cannot be assigned by row order.")

    go_cols = [c for c in df.columns if c in go_cols_allowed]
    if not go_cols:
        raise SystemExit(
            f"{path.name}: none of its columns is a GO-term node of this "
            f"module; check hard_links.csv against the file's columns.")

    grouped = df[go_cols].groupby(labels.values)
    means = grouped.mean().reindex(cancer_types)
    # eta-squared per term: the share of a node's total variance that sits
    # between cancer types rather than within them. Used to rank terms; see
    # select_terms.
    total_var = df[go_cols].var(ddof=0)
    within_var = grouped.var(ddof=0).mul(grouped.size(), axis=0).sum() / len(df)
    eta_sq = (1.0 - within_var / total_var.replace(0.0, np.nan)).fillna(0.0)
    means.attrs["eta_sq"] = eta_sq
    return means


# ── term selection ────────────────────────────────────────────────────────────

def select_terms(reference: pd.DataFrame, n_terms: int = N_TERMS) -> list[str]:
    """The GO terms to plot, in column order, fixed across model instances.

    ``GO_TERMS`` wins when set. Otherwise the ``n_terms`` nodes with the
    highest eta-squared in ``reference`` (the first seed): the share of the
    node's variance that lies between cancer types rather than within them.

    Ranking on eta-squared rather than raw variance is what the caption asks
    for -- "processes expected to vary in activity across cancer types" is a
    statement about discrimination, not magnitude. Raw variance just returns
    whichever nodes have the largest activations, which are the layers nearest
    the bottleneck and the genes; on the preprint v3 inputs that pushed the
    colour scale to ~+/-79 (encoder) and ~+/-224 (decoder), far outside the
    printed +/-4. Eta-squared
    is scale-free and bounded [0, 1], so a small, sharply cancer-type-specific
    node can outrank a large diffuse one. Ties keep file column order, so the
    result is deterministic.
    """
    if GO_TERMS is not None:
        missing = [t for t in GO_TERMS if t not in reference.columns]
        if missing:
            raise SystemExit(
                f"GO_TERMS contains {len(missing)} term(s) that are not "
                f"GO-term nodes of this module: {', '.join(missing[:5])}"
                + (" ..." if len(missing) > 5 else ""))
        return list(GO_TERMS)

    score = reference.attrs["eta_sq"].sort_values(ascending=False, kind="stable")
    return score.index[:n_terms].tolist()


# ── plotting ──────────────────────────────────────────────────────────────────

def _heatmap(fig, ax, cax, matrix: np.ndarray, terms: list[str],
             cancer_types: list[str], vmax: float, title: str,
             y_labels: bool, names: dict[str, str], x_labels: bool = True,
             normalized: bool = False) -> None:
    """One "GO Term Activation Heatmap": cancer types x GO terms.

    ``y_labels`` writes the cancer types down the y axis. Only the leftmost
    heatmap of a row does; the instances share their rows, and repeating 32
    labels three times would cost the width the GO names need.

    ``x_labels`` writes the GO process names under the x axis. Only the bottom
    row does: both rows carry the same 20 terms in the same order, and the
    names are long enough set on their side that a second band would push the
    figure past a printable page (see the layout block).
    """
    im = ax.imshow(matrix, aspect="auto", cmap=CMAP, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels([term_label(t, names) for t in terms] if x_labels else [],
                       rotation=90, fontsize=PT_SMALL)
    ax.set_yticks(range(len(cancer_types)))
    ax.set_yticklabels(cancer_types if y_labels else [], fontsize=PT_SMALL)
    ax.tick_params(length=0, pad=pt(1.5))
    if x_labels:
        ax.set_xlabel("GO terms", fontsize=PT_BODY, labelpad=pt(2.0))
    if y_labels:
        # Kept short deliberately: rotated, this label already runs nearly the
        # full height of the heatmap, and spelling the scaling out here too
        # pushes it past the panel letter. The colourbar says it instead.
        ax.set_ylabel("Mean sample activation per cancer type",
                      fontsize=PT_BODY, labelpad=pt(1.0))
    ax.set_title(title, fontsize=PT_TITLE, pad=pt(4.0))

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Activation, centred and scaled per GO term"
                 if normalized else "Activation",
                 fontsize=PT_BODY, labelpad=pt(1.0))
    cb.ax.tick_params(labelsize=PT_SMALL, length=pt(1.5), pad=pt(1.5),
                      width=plt.rcParams["ytick.major.width"])
    cb.outline.set_linewidth(plt.rcParams["axes.linewidth"])


def _panel_letter(fig, fig_h: float, top: float, letter: str) -> None:
    """Panel letter in the left margin, level with the row's titles."""
    fig.text(0.03 / FIG_WIDTH_IN, 1.0 - top / fig_h, letter,
             fontsize=PT_PANEL, fontweight="bold", va="top", ha="left")


def plot_figure(panel_data: list[pd.DataFrame], seeds: list[int],
                terms: list[str], cancer_types: list[str],
                module: str, vmax: float, names: dict[str, str],
                normalized: bool = False):
    """Panel A (signed) over panel B (absolute), one column per instance."""
    n = len(seeds)
    lay = _layout(n, xtick_band_in([term_label(t, names) for t in terms]))
    fig_h, heat_w = lay["fig_h"], lay["heat_w"]
    fig = plt.figure(figsize=figsize(fig_h))

    for row in (0, 1):
        top = lay["tops"][row]
        for col, (seed, means) in enumerate(zip(seeds, panel_data)):
            x = lay["xs"][col]
            ax = fig.add_axes(_rect(fig_h, x, top + TITLE_H, heat_w, HEAT_H))
            cax = fig.add_axes(_rect(
                fig_h, x + heat_w + CBAR_PAD,
                top + TITLE_H + (1.0 - CBAR_FRAC) / 2 * HEAT_H,
                CBAR_W, CBAR_FRAC * HEAT_H))
            signed = means[terms].to_numpy(dtype=float)
            _heatmap(fig, ax, cax,
                     signed if row == 0 else np.abs(signed),
                     terms, cancer_types, vmax,
                     f"GO Term Activation Heatmap\n"
                     f"instance {col + 1} (seed {seed})",
                     y_labels=(col == 0), names=names, x_labels=(row == 1),
                     normalized=normalized)
        _panel_letter(fig, fig_h, top, "ab"[row])

    # No suptitle: what the figure shows belongs in the caption, not on the
    # canvas. The panel letters are therefore the topmost artists.
    return fig


# ── entry point ───────────────────────────────────────────────────────────────

def main(module: str = "encoder", out_name: str = "figS5") -> None:
    """Build the activation heatmap figure for one module; figS6.py calls this."""
    parser = argparse.ArgumentParser(
        description=f"Mean GO-term activations per cancer type ({module}) "
                    f"-> {out_name}")
    add_io_args(parser)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="model instances to show (default: 2 3 4)")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                        help="plot raw mean activations instead of centring "
                             "each column and scaling each GO term to [-1, 1] "
                             "(see normalize_terms)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="symmetric colour limit; default is the largest "
                             "|mean| over all instances")
    parser.add_argument("--activations-dir", type=Path, default=None,
                        help=f"GO-term activations (default: <data-dir>/{REVISED_ACTIVATIONS}; "
                             f"--preprint: <data-dir>/{PREPRINT_ACTIVATIONS})")
    parser.add_argument("--preprint", action="store_true",
                        help="the preprint v3 figure's activations, whose decoder columns are "
                             "labelled one layer off (the encoder files are the same in both)")
    args = parser.parse_args()
    activations_dir = args.activations_dir or args.data_dir / (
        PREPRINT_ACTIVATIONS if args.preprint else REVISED_ACTIVATIONS)

    hard_links = args.data_dir / "hard_links.csv"
    tcga = args.data_dir / "TCGA_complete_bp_top1k.csv.gz"
    for path in (hard_links, tcga):
        if not path.exists():
            raise SystemExit(f"Input file not found: {path}")

    # GO-term nodes of this module; everything else in the graph is a proxy or
    # a gene node and is not eligible. With no hand-picked list, the bottleneck
    # is held out of the pool so the variance ranking is not swamped by it.
    drop_bottleneck = GO_TERMS is None
    allowed = eligible_go_terms(hard_links, module, drop_bottleneck)
    print(f"{module}: {len(allowed)} GO-term nodes in layers "
          f"{LAYERS_BY_MODULE[module]}"
          f"{' (bottleneck excluded)' if drop_bottleneck else ''}")

    # Cancer-type labels come from the TCGA table by row order; patient_id is
    # read alongside so every activation file can be checked against it.
    labels = load_cancer_types(tcga)
    patient_ids = pd.read_csv(tcga, usecols=["patient_id"])["patient_id"]
    cancer_types = [ct for ct in CANCER_TYPE_ORDER if ct in set(labels)]
    print(f"{len(labels)} samples, {len(cancer_types)} cancer types")

    panel_data = []
    for seed in args.seeds:
        path = _activation_path(activations_dir, module, seed)
        print(f"  loading seed {seed}: {path.name} ...", flush=True)
        panel_data.append(mean_per_cancer_type(path, allowed, labels,
                                               patient_ids, cancer_types))

    terms = select_terms(panel_data[0])
    rule = ("hand-picked GO_TERMS" if GO_TERMS is not None
            else f"top-{N_TERMS} by eta-squared (between-cancer-type variance share), seed "
                 f"{args.seeds[0]}")
    print(f"{len(terms)} GO terms ({rule}): {', '.join(terms)}")

    if args.normalize:
        raw_span = {t: max(float(np.abs(m[t].to_numpy()).max()) for m in panel_data)
                    for t in terms}
        lo, hi = min(raw_span.values()), max(raw_span.values())
        print(f"centring each column, then scaling each term to [-1, 1]; raw "
              f"per-term maxima span {lo:.3g} to {hi:.3g} "
              f"({hi / max(lo, 1e-12):.0f}x)")
        panel_data, flat = normalize_terms(panel_data, terms)
        if flat:
            print(f"  {len(flat)} term(s) constant across all cancer types in "
                  f"every instance, drawn blank rather than scaled up: "
                  f"{', '.join(flat)}")

    vmax = args.vmax
    if vmax is None:
        vmax = max(float(np.abs(m[terms].to_numpy()).max()) for m in panel_data)
    print(f"colour limits: [{-vmax:.3g}, {vmax:.3g}]")

    names = go_term_names(hard_links)
    fig = plot_figure(panel_data, args.seeds, terms, cancer_types, module, vmax,
                      names, normalized=args.normalize)
    report_text_overlaps(fig, out_name)
    save_figure(fig, args.out_dir, out_name, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
