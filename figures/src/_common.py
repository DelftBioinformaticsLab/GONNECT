"""Shared helpers for the GONNECT paper figure scripts.

Everything in here is used by two or more of the ``fig*.py`` scripts next to it:
paths and figure saving, the model display names, the metric readers for the
performance spreadsheets, the paired-t-test machinery, and the t-SNE cache.

Figure-specific plotting logic deliberately stays in the individual scripts, so
each figure can be read and adjusted on its own.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox
import torch
from scipy import stats
from sklearn.manifold import TSNE

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
SRC_DIR = Path(__file__).resolve().parent
FINAL_DIR = SRC_DIR.parent
DATA_DIR = FINAL_DIR / "data"
OUT_DIR = FINAL_DIR / "out"
CACHE_DIR = DATA_DIR / "cache"


def add_io_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The --data-dir / --out-dir pair every figure script accepts."""
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"input data root (default: {DATA_DIR})")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help=f"where figures are written (default: {OUT_DIR})")
    return parser


# ══════════════════════════════════════════════════════════════════════════════
# Figure geometry and type scale
#
# Every figure is saved at exactly FIG_WIDTH_IN. That is the whole trick behind
# consistent type in the paper: LaTeX scales each figure by
# (placed width / saved width), so if the saved widths are equal, one point
# size means one rendered size everywhere. Saving figures at different widths
# is what made the old set inconsistent -- they ranged 11.97 to 23.61 in, so
# the same 8 pt label came out anywhere between 2.4 and 4.7 pt on the page.
#
# Consequence: the numbers passed to fontsize= are NOT what the reader sees.
# A figure saved 16.5 in wide and placed at 180 mm is scaled by
# 180mm / 16.5in = 0.430, so 16.3 pt in code renders as 7 pt on the page.
# Always go through pt() and think in rendered sizes.
# ══════════════════════════════════════════════════════════════════════════════
FIG_WIDTH_IN = 16.5             # every figure saves at exactly this width
PAGE_WIDTH_MM = 180.0           # width the figures are placed at in the paper
PAGE_WIDTH_IN = PAGE_WIDTH_MM / 25.4
PAGE_SCALE = PAGE_WIDTH_IN / FIG_WIDTH_IN   # ≈ 0.4295


def pt(page_pt: float) -> float:
    """Point size to pass to matplotlib so text renders at ``page_pt`` on the page.

    >>> round(pt(7), 1)      # body text in a figure
    16.3
    """
    return page_pt / PAGE_SCALE


# The paper's type scale, in rendered points. Scripts should use these rather
# than bare numbers so one edit here restyles every figure.
PT_TINY = pt(4.0)       # in-cell marks, e.g. the "x" for missing heatmap cells
PT_SMALL = pt(6.0)      # tick labels, legend entries
PT_BODY = pt(7.0)       # axis labels, significance stars
PT_TITLE = pt(8.0)      # panel titles
PT_PANEL = pt(10.0)     # bold panel letters (a, b, c ...)
PT_SUPTITLE = pt(10.0)  # figure-level titles


def figsize(height_in: float) -> tuple:
    """Canvas size for a figure of the given height, at the standard width."""
    return (FIG_WIDTH_IN, height_in)


def apply_style() -> None:
    """Set the rcParams defaults every figure shares.

    Called at import of this module, so a script gets the paper's type scale
    without doing anything. Explicit fontsize= arguments still win.
    """
    plt.rcParams.update({
        "font.size": PT_BODY,
        "axes.titlesize": PT_TITLE,
        "axes.labelsize": PT_BODY,
        "xtick.labelsize": PT_SMALL,
        "ytick.labelsize": PT_SMALL,
        "legend.fontsize": PT_SMALL,
        "figure.titlesize": PT_SUPTITLE,
        # Line and marker weights are in points too, so they need the same
        # treatment or they would look hairline once the figure is scaled down.
        "axes.linewidth": 0.8 / PAGE_SCALE * 0.5,
        "xtick.major.width": 0.8 / PAGE_SCALE * 0.5,
        "ytick.major.width": 0.8 / PAGE_SCALE * 0.5,
        "grid.linewidth": 0.8 / PAGE_SCALE * 0.5,
        "patch.linewidth": 0.8 / PAGE_SCALE * 0.5,
        # Embed text as TrueType (42) rather than matplotlib's default Type 3.
        # Type 3 hides glyphs inside PDF drawing operators, so the text is not
        # selectable or searchable, and Elsevier, IEEE and PLOS all reject or
        # flag it at submission. 42 changes only how glyphs are encoded -- the
        # rendered layout is identical, verified pixel-for-pixel against the
        # published PNGs.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


apply_style()


def save_figure(fig, out_dir: Path, name: str, *, dpi: int = 200,
                tight: bool = True) -> None:
    """Write ``<out_dir>/<name>.png`` and ``.pdf``, both exactly FIG_WIDTH_IN wide.

    The full canvas width is always kept -- only vertical slack is trimmed, and
    only when ``tight``. Cropping horizontally would undo the one property the
    paper depends on: that every figure is saved at the same width, so a point
    size means the same rendered size in all of them. Scripts are responsible
    for filling the canvas width; the width is checked here rather than being
    silently corrected, because quietly rescaling would change the type size.

    ``tight=False`` additionally keeps the full height, for figures positioned
    in absolute figure coordinates where cropping would shift the labels.

    ``rasterized`` artists are written as rasters in the PDF too, which is what
    keeps the scatter panels openable: Figure 2's four t-SNE panels are ~39k
    points and Figure S2's five are ~49k, and as vector paths they make a PDF
    viewer crawl on every redraw. Marking them rasterized collapses each panel
    to one embedded image. At the default dpi they land at ~465 dpi once the
    figure is placed at 180 mm, so nothing visible is given up.

    This used to be undone for the PDF pass, because saving a vector format
    with a cropping bbox once misplaced rasterized artists -- they were drawn
    through a MixedModeRenderer built from the figure's original bbox and only
    then cropped, so the raster landed at the wrong offset and scale. That is
    fixed in the matplotlib pins in pyproject.toml (>=3.10.8, verified: an embedded
    image sits at the same place relative to its axes whether the save crops
    0.0 or 0.5 in). Should the figures ever regress to one blob of points
    outside their axes, that is the bug returning and the pin is the thing to
    check.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    width_in, height_in = fig.get_size_inches()
    if abs(width_in - FIG_WIDTH_IN) > 1e-3:
        raise ValueError(
            f"{name}: canvas is {width_in:.2f} in wide, but every figure must "
            f"be FIG_WIDTH_IN ({FIG_WIDTH_IN} in) or its text will render at a "
            f"different size than the rest of the paper. Use _common.figsize().")

    kwargs = {"dpi": dpi}
    if tight:
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        pad = 0.05
        kwargs["bbox_inches"] = Bbox([
            [0.0, max(0.0, tb.y0 - pad)],
            [width_in, min(height_in, tb.y1 + pad)],
        ])

    png = out_dir / f"{name}.png"
    fig.savefig(png, **kwargs)
    print(f"  wrote {png}")

    pdf = out_dir / f"{name}.pdf"
    fig.savefig(pdf, **kwargs)
    print(f"  wrote {pdf}")


def find_text_overlaps(fig, *, min_overlap: float = 0.12) -> List[tuple]:
    """Pairs of text artists whose drawn boxes overlap. For layout checking.

    Returns ``(text_a, text_b, fraction)`` where fraction is the shared area as
    a share of the smaller box, worst first. ``min_overlap`` ignores the
    grazing contact that normal typesetting produces -- adjacent rotated tick
    labels routinely share a pixel or two of bounding box without looking
    wrong, because the boxes are rectangles and the glyphs inside them are not.

    Only a signal, not a verdict: read it alongside the rendered figure.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Tick labels for ticks outside the view limits are never drawn, but the
    # artists still exist and report a position where the tick *would* be --
    # often right on top of a title or a neighbouring row. Every figure that
    # calls set_ylim after plotting can hit this, so filter them here rather
    # than making each script work around it.
    # Note get_ticklabels() materialises tick artists that may not otherwise
    # exist, so an axes with its axis switched off -- a panel used purely to
    # hold a legend, say -- would otherwise acquire phantom "0.0, 0.2, ..."
    # labels sitting right under the legend text. Hence the axison check.
    undrawn = set()
    for axes in fig.get_axes():
        axis_off = not getattr(axes, "axison", True)
        for axis in (axes.xaxis, axes.yaxis):
            hidden = axis_off or not axis.get_visible()
            low, high = sorted(axis.get_view_interval())
            for loc, label in zip(axis.get_ticklocs(), axis.get_ticklabels()):
                if hidden or not (low <= loc <= high):
                    undrawn.add(id(label))

    items = []
    for artist in fig.findobj(plt.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        if id(artist) in undrawn:
            continue
        try:
            box = artist.get_window_extent(renderer)
        except Exception:
            continue
        if box.width <= 0 or box.height <= 0:
            continue
        items.append((artist, box))

    hits = []
    for i, (art_a, box_a) in enumerate(items):
        for art_b, box_b in items[i + 1:]:
            # Text inside a shared parent (a legend and its own title, say) is
            # laid out as a unit and is not what we are hunting for.
            if art_a.get_figure() is not art_b.get_figure():
                continue
            dx = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
            dy = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
            if dx <= 0 or dy <= 0:
                continue
            smaller = min(box_a.width * box_a.height, box_b.width * box_b.height)
            frac = (dx * dy) / smaller if smaller else 0.0
            if frac >= min_overlap:
                hits.append((art_a, art_b, frac))
    return sorted(hits, key=lambda h: -h[2])


def report_text_overlaps(fig, label: str = "", **kwargs) -> int:
    """Print what find_text_overlaps found. Returns the number of pairs."""
    hits = find_text_overlaps(fig, **kwargs)
    if not hits:
        print(f"  no text overlaps{(' in ' + label) if label else ''}")
        return 0
    print(f"  {len(hits)} overlapping text pair(s)"
          f"{(' in ' + label) if label else ''}:")
    for art_a, art_b, frac in hits[:15]:
        print(f"    {frac:5.0%}  {art_a.get_text()[:34]!r}"
              f"  <->  {art_b.get_text()[:34]!r}")
    if len(hits) > 15:
        print(f"    ... and {len(hits) - 15} more")
    return len(hits)


# ══════════════════════════════════════════════════════════════════════════════
# Model display names
#
# Internal keys are left exactly as they appear in the source data (xlsx
# columns, txt dict keys, embedding filenames); this section only maps those
# keys to the names used in the paper.
#
# Two null ontologies exist, and the historical naming flipped the meaning of
# "R" between the GONNECT models and the OntoVAE/VEGA baselines:
#
#     concept                       legacy GONNECT  legacy baseline      PAPER
#     degree-preserving randomized  GONNECT-R-*     *_degree_preserving  DPR
#     fully randomized              GONNECT-RR-*    *_random             FR
#
# Both legacy spellings are mapped, so figures agree regardless of which
# convention their input file uses.
# ══════════════════════════════════════════════════════════════════════════════
DPR = "DPR"   # degree-preserving randomized
FR = "FR"     # fully randomized

RANDOMIZATION_DISPLAY: Dict[str, str] = {
    "true": "true",
    "degree_preserving": DPR,
    "rand_degree": DPR,
    "R": DPR,
    "random": FR,
    "rand_size": FR,
    "RR": FR,
}

RANDOMIZATION_LEGEND: Dict[str, str] = {
    "true": "True graph",
    "degree_preserving": f"Degree-preserving random ({DPR})",
    "random": f"Fully random ({FR})",
}

# VEGA annotation database -> paper name. Internal keys vary across the source
# files ("reactomes", "reactome674", the raw gmt stem), so all are mapped.
VEGA_DB_DISPLAY: Dict[str, str] = {
    "hallmark": "Hallmark",
    "hallmark_v2026_1_Hs_uniprot": "Hallmark",
    "reactomes": "Reactome",
    "reactome674": "Reactome",
    "reactomes_uniprot": "Reactome",
}

MODULES = ["enc", "dec", "both"]

# Long module name (as used in embedding filenames) -> suffix in MODULES.
# Note "both"[:3] == "bot", so slicing is not a safe abbreviation.
MODULE_ABBREV: Dict[str, str] = {
    "encoder": "enc", "decoder": "dec", "both": "both", "none": "none",
}

PRIOR_DISPLAY: Dict[str, str] = {
    "enc": "Enc prior", "dec": "Dec prior", "both": "Both priors",
}

# Figure 2a-d calls the fixed-link model plain "GONNECT", so "GONNECT-FL"
# renders without the -FL suffix.
MODEL_FAMILY_DISPLAY: Dict[str, str] = {
    "GONNECT-FL": "GONNECT",
    "GONNECT": "GONNECT",
    "GONNECT-SL": "GONNECT-SL",
    "GONNECT-R": f"GONNECT-{DPR}",
    "GONNECT-Rand": f"GONNECT-{DPR}",
    "GONNECT-RR": f"GONNECT-{FR}",
}

# Embedding-file version digit -> model family: AE_x.0 fixed link,
# AE_x.1 soft link, AE_x.2 randomized.
EMB_VERSION_DISPLAY: Dict[str, str] = {
    "2.0": "GONNECT", "2.1": "GONNECT-SL", "2.2": f"GONNECT-{DPR}",
}
EMB_VERSION_SUFFIX: Dict[str, str] = {"2.0": "", "2.1": "_SL", "2.2": "_DPR"}


def _gonnect_family() -> Dict[str, str]:
    """The 12 GONNECT entries: {fixed, SL} x {true, DPR, FR} x modules."""
    out: Dict[str, str] = {}
    variants = [("", ""), ("R-", f"{DPR}-"), ("RR-", f"{FR}-")]
    for legacy_rand, paper_rand in variants:
        for legacy_link, paper_link in [("", ""), ("SL-", "SL-")]:
            for module in MODULES:
                key = f"GONNECT-{legacy_rand}{legacy_link}{module}"
                out[key] = f"GONNECT-{paper_rand}{paper_link}{module}"
    return out


MODEL_DISPLAY: Dict[str, str] = {
    "MLP": "MLP",
    "ontovae": "OntoVAE",
    "ontovae_degree_preserving": f"OntoVAE ({DPR})",
    "ontovae_random": f"OntoVAE ({FR})",
    "vega_hallmark": "VEGA (Hallmark)",
    "vega_hallmark_degree_preserving": f"VEGA (Hallmark, {DPR})",
    "vega_hallmark_random": f"VEGA (Hallmark, {FR})",
    "vega_reactome674": "VEGA (Reactome)",
    "vega_reactome_degree_preserving": f"VEGA (Reactome, {DPR})",
    "vega_reactome_random": f"VEGA (Reactome, {FR})",
    **_gonnect_family(),
}

SHORT_DISPLAY: Dict[str, str] = {
    "MLP": "MLP",
    "ontovae": "OntoVAE",
    "vega_hallmark": "VEGA-H",
    "vega_reactome674": "VEGA-R",
    **{
        f"GONNECT-{legacy_rand}{legacy_link}{m}": f"{paper}{link}{m}"
        for legacy_rand, paper in [("", ""), ("R-", f"{DPR}-"), ("RR-", f"{FR}-")]
        for legacy_link, link in [("", ""), ("SL-", "SL-")]
        for m in MODULES
    },
}


def display(key: str, default: Optional[str] = None) -> str:
    """Paper display name for a model key."""
    for table in (MODEL_DISPLAY, MODEL_FAMILY_DISPLAY):
        if key in table:
            return table[key]
    return key if default is None else default


def short(key: str, default: Optional[str] = None) -> str:
    """Compact display name, falling back to the full name."""
    if key in SHORT_DISPLAY:
        return SHORT_DISPLAY[key]
    return display(key, default)


# ── The ten models shown in Figures 2 and S1, in plotting order ──────────────
MAIN_METHODS: List[str] = [
    "MLP", "ontovae", "vega_hallmark", "vega_reactome674",
    "GONNECT-enc", "GONNECT-dec", "GONNECT-both",
    "GONNECT-SL-enc", "GONNECT-SL-dec", "GONNECT-SL-both",
]

# Which autoencoder module carries the GO prior, for bar colouring.
PRIOR_METHOD_GROUP: Dict[str, str] = {
    "MLP": "Reference",
    "ontovae": "Reference",
    "vega_hallmark": "Reference",
    "vega_reactome674": "Reference",
    "GONNECT-enc": "Enc prior",
    "GONNECT-SL-enc": "Enc prior",
    "GONNECT-dec": "Dec prior",
    "GONNECT-SL-dec": "Dec prior",
    "GONNECT-both": "Both priors",
    "GONNECT-SL-both": "Both priors",
}

PRIOR_GROUP_COLORS: Dict[str, str] = {
    "Reference":   "#4E79A7",  # blue
    "Both priors": "#F28E2B",  # orange
    "Dec prior":   "#59A14F",  # green
    "Enc prior":   "#E15759",  # red
}

# Figure 3 colours by randomization level instead, reusing the same hues.
RANDOMIZATION_GROUP_COLORS: Dict[str, str] = {
    "True":   "#4E79A7",  # blue
    "DP":     "#59A14F",  # green
    "Random": "#F28E2B",  # orange
}


# ══════════════════════════════════════════════════════════════════════════════
# Cancer types
# ══════════════════════════════════════════════════════════════════════════════
# The 32 TCGA types, ordered by sample abundance (Supplementary Table S1).
CANCER_TYPE_ORDER: List[str] = [
    "BRCA", "LUAD", "UCEC", "LGG",  "KIRC", "HNSC", "THCA", "PRAD",
    "LUSC", "SKCM", "COAD", "OV",   "STAD", "BLCA", "LIHC", "CESC",
    "KIRP", "SARC", "ESCA", "PCPG", "PAAD", "READ", "TGCT", "LAML",
    "THYM", "MESO", "UVM",  "ACC",  "KICH", "UCS",  "DLBC", "CHOL",
]

# 32 distinguishable colours: all of tab20, then every other tab20b entry.
_c20 = plt.cm.tab20(np.linspace(0, 1, 20))
_c20b = plt.cm.tab20b(np.linspace(0, 1, 20))
CANCER_COLORS = np.vstack([_c20, _c20b[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3]]])

# Non-gene columns in data/TCGA_complete_bp_top1k.csv.gz.
TCGA_META_COLS = {
    "patient_id", "sample_type", "cancer_type",
    "tumor_tissue_site", "stage_pathologic_stage",
}


# ══════════════════════════════════════════════════════════════════════════════
# Network layers
# ══════════════════════════════════════════════════════════════════════════════
# The five-layer GONNECT graph, mirrored: encoder L0-L3, 109-node latent,
# decoder L5-L8. Layer 4 does not exist as a weight matrix.
ENC_LAYERS: List[int] = [0, 1, 2, 3]
DEC_LAYERS: List[int] = [5, 6, 7, 8]
LAYERS_BY_MODULE: Dict[str, List[int]] = {
    "encoder": ENC_LAYERS, "decoder": DEC_LAYERS,
}
# Layer index -> index into nn.Sequential (ReLU occupies the odd positions).
NET_IDX: Dict[str, Dict[int, int]] = {
    "encoder": {0: 0, 1: 2, 2: 4, 3: 6},
    "decoder": {5: 0, 6: 2, 7: 4, 8: 6},
}


# ══════════════════════════════════════════════════════════════════════════════
# Metric loading
# ══════════════════════════════════════════════════════════════════════════════
METRIC_CANONICAL = {
    "nmi": "NMI", "ari": "ARI", "mse": "MSE", "ss": "SS", "silhouette": "SS",
}
METRIC_SET = set(METRIC_CANONICAL.values())
METRIC_ORDER = ["MSE", "SS", "ARI", "NMI"]
METRIC_DIRECTION = {"MSE": "down", "SS": "up", "ARI": "up", "NMI": "up"}

# The five matched data splits used throughout the paper ("run-2" to "run-6").
ALLOWED_REPEATS = {"run-2", "run-3", "run-4", "run-5", "run-6"}


def normalize_metric(name: str) -> str:
    return METRIC_CANONICAL.get(str(name).strip().lower(), str(name))


def normalize_repeat(value: object, fallback: Optional[str] = None) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    if text.lower().startswith("run-"):
        return text
    if text.isdigit():
        return f"run-{int(text)}"
    return text


def _repeat_ok(repeat_id: Optional[str], strict: bool) -> bool:
    """Whether a repeat id should be kept.

    ``strict`` keeps only the five matched splits. Non-strict keeps anything
    that is not a ``run-N`` outside that set, which lets the supplementary
    sweeps carry rows with synthetic ids.
    """
    if repeat_id is None:
        return False
    if strict:
        return repeat_id in ALLOWED_REPEATS
    return not (repeat_id.startswith("run-") and repeat_id not in ALLOWED_REPEATS)


def read_xlsx_metrics(path: Path, *, strict_repeats: bool = True,
                      row_fallback: bool = False) -> pd.DataFrame:
    """Read a wide metric workbook: one sheet per metric, one column per method.

    ``strict_repeats`` and ``row_fallback`` reproduce the three slightly
    different readers the original scripts used. Figure 2 needs
    ``strict_repeats=True``; the supplementary sweeps (Figure S1) need
    ``row_fallback=True`` so rows with a blank repeat cell are kept.
    """
    records: List[dict] = []
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        metric_name = normalize_metric(sheet)
        df = xls.parse(sheet)
        repeat_col = None
        for col in df.columns:
            if str(col).lower() in {"repeat", "run", "seed"}:
                repeat_col = col
                break
        if repeat_col is None and len(df.columns) > 0:
            first = df.columns[0]
            sample = pd.to_numeric(df[first], errors="coerce").dropna()
            if not sample.empty and sample.astype(int).isin({2, 3, 4, 5, 6}).any():
                repeat_col = first
        if repeat_col is None:
            continue
        method_cols = [
            col for col in df.columns
            if col != repeat_col and normalize_metric(col) not in METRIC_SET
        ]
        for method in method_cols:
            series = pd.to_numeric(df[method], errors="coerce")
            for idx, value in series.items():
                if pd.isna(value):
                    continue
                fallback = f"row-{idx}" if row_fallback else None
                repeat_id = normalize_repeat(df.at[idx, repeat_col], fallback)
                if not _repeat_ok(repeat_id, strict_repeats):
                    continue
                records.append({
                    "metric": metric_name, "method": str(method),
                    "repeat": repeat_id, "value": float(value),
                })
    return pd.DataFrame.from_records(records)


# ── The per-cancer-type workbook, and Excel's locale trap ────────────────────
# mse_per_cluster_TCGA_1000_30.xlsx was written with every value as text at
# three decimals, then opened and saved in a locale where "." groups thousands.
# Excel converted each string that parses as a grouped number -- "1.219" became
# the integer 1219 with number format "#,##0" -- and left the rest alone, since
# "0.354" is not valid grouping (a leading zero group is rejected). The damage
# is therefore value-conditional: exactly the cells whose true value is >= 1,
# multiplied by 1000. In the released file that is 28 cells of the MSE sheet,
# all of them 4-digit integers, i.e. true values between 1.008 and 1.678.
#
# Three things confirm the reading: the storage types (coerced cells are ints
# with a grouping format, every other cell is text), the value distribution
# (292 cells <= 0.98, 28 cells >= 1008, nothing whatsoever in between), and the
# arithmetic -- undoing the factor reproduces the independently reported
# overall MSE to within 0.4% for both affected columns.
#
# The digits survive the coercion, so dividing by 1000 restores the original to
# its full three decimals; this is a decode, not an estimate. It is done on
# read rather than by repairing the file because opening the workbook in the
# same locale re-corrupts it -- a repaired file would silently regress, this
# does not. Cells that were never coerced are untouched, so a properly
# regenerated workbook passes through unchanged.
COERCION_MIN = 1000        # a grouped number is at least this by construction


def _decoerce(cell) -> float:
    """One per-cluster cell as a float, undoing the coercion where present."""
    value = cell.value
    if value is None:
        return float("nan")
    if isinstance(value, str):
        text = value.strip()
        return float(text) if text else float("nan")
    # Only a coerced cell is an integer carrying a thousands-grouping format.
    # A genuine number -- any float, or anything written by a fixed exporter --
    # fails at least one of those and is passed through as it stands.
    if (isinstance(value, int) and not isinstance(value, bool)
            and value >= COERCION_MIN and "," in (cell.number_format or "")):
        return value / 1000.0
    return float(value)


def read_per_cluster_workbook(path: Path, *, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Every sheet of the per-cancer-type workbook, indexed on its first column.

    Returns ``{sheet_name: DataFrame}``. Locale-coerced cells are restored (see
    the block above) and the count is reported, so a silent repair never
    happens: if the number changes, the input file changed.
    """
    from openpyxl import load_workbook

    path = Path(path)
    book = load_workbook(path, data_only=True)
    sheets: Dict[str, pd.DataFrame] = {}
    repaired: List[str] = []
    for name in book.sheetnames:
        rows = [row for row in book[name].iter_rows()]
        if not rows:
            continue
        header = [c.value for c in rows[0]]
        records: Dict[object, dict] = {}
        for row in rows[1:]:
            key = row[0].value
            if key is None:
                continue
            values = {}
            for column, cell in zip(header[1:], row[1:]):
                values[column] = _decoerce(cell)
                if isinstance(cell.value, int) and values[column] != cell.value:
                    repaired.append(f"{name}[{key}, {column}]")
            records[key] = values
        frame = pd.DataFrame.from_dict(records, orient="index")
        frame.index.name = header[0]
        sheets[name] = frame
    if verbose and repaired:
        print(f"  repaired {len(repaired)} locale-coerced cell(s) in "
              f"{path.name} (x1000 by Excel; see _common._decoerce)")
    return sheets


# For the true graph the method key matches the txt/xlsx spelling; the
# randomized variants append the graph type to a base that drops VEGA's "674".
_VEGA_ANNOTATIONS: Dict[str, tuple] = {
    "hallmark_v2026_1_Hs_uniprot": ("vega_hallmark", "vega_hallmark"),
    "reactomes_uniprot": ("vega_reactome674", "vega_reactome"),
}
RANDOMIZED_GRAPH_ARMS = ("degree_preserving", "random")


def _baseline_arm_lines(path: Path) -> Iterable[tuple]:
    """Yield ``(repeat_id, payload_dict)`` for each line of a metrics.txt."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            run_id, payload = line.split(":", 1)
            try:
                data = ast.literal_eval(payload.strip())
            except (ValueError, SyntaxError):
                continue
            yield normalize_repeat(run_id.strip()), data


def read_baseline_ontovae(path: Path, split: str = "test",
                         graph_types: Sequence[str] = ("true",),
                         *, strict_repeats: bool = True) -> pd.DataFrame:
    """``run-N: {graph_type: {split: {metric: val}}}`` for the OntoVAE baseline.

    ``graph_types=("true",)`` gives the published baseline as method
    ``ontovae``; the randomized types become ``ontovae_<graph_type>``.
    """
    records: List[dict] = []
    for repeat_id, data in _baseline_arm_lines(path):
        if not _repeat_ok(repeat_id, strict_repeats):
            continue
        for graph_type in graph_types:
            splits_data = data.get(graph_type, {})
            if not isinstance(splits_data, dict) or split not in splits_data:
                continue
            method = "ontovae" if graph_type == "true" else f"ontovae_{graph_type}"
            for metric, value in splits_data[split].items():
                records.append({
                    "metric": normalize_metric(metric), "method": method,
                    "repeat": repeat_id, "value": float(value),
                })
    return pd.DataFrame.from_records(records)


def read_baseline_vega(path: Path, split: str = "test",
                      graph_types: Sequence[str] = ("true",),
                      *, strict_repeats: bool = True) -> pd.DataFrame:
    """``run-N: {annotation: {graph_type: {split: {metric: val}}}}`` for VEGA."""
    records: List[dict] = []
    for repeat_id, data in _baseline_arm_lines(path):
        if not _repeat_ok(repeat_id, strict_repeats):
            continue
        for annotation, graph_data in data.items():
            if annotation not in _VEGA_ANNOTATIONS:
                continue
            true_name, rand_base = _VEGA_ANNOTATIONS[annotation]
            for graph_type in graph_types:
                splits_data = graph_data.get(graph_type, {})
                if not isinstance(splits_data, dict) or split not in splits_data:
                    continue
                method = true_name if graph_type == "true" else f"{rand_base}_{graph_type}"
                for metric, value in splits_data[split].items():
                    records.append({
                        "metric": normalize_metric(metric), "method": method,
                        "repeat": repeat_id, "value": float(value),
                    })
    return pd.DataFrame.from_records(records)


#: Suffix of the per-graph-arm baseline files (``ontovae_rand.txt``, ``vega_rand.txt``).
ARM_FILE_SUFFIX = "_rand"


def test_split_paths(data_dir: Path) -> Dict[str, Path]:
    """The held-out metric inputs of figures 2, 3, S1, S3 and S4, by name.

    Every model is scored on held-out samples, with the silhouette taken against
    the true cancer types. The originals took GONNECT's SS / ARI / NMI and its
    per-type SS and purity over all samples, and the baselines' silhouette
    against their k-means clusters. Built by:
      ``prepare/test_split_metrics.py``   ``gonnect`` (SS / ARI / NMI of the ten
          models with embeddings), ``ss_per_type``, ``purity_per_type_k{k}``
          against the training split (figS3), and ``purity_within_test``
          (Figure 2k: k = 10, neighbours within the test split, small types blank)
      ``prepare/rescore_baselines.py``    ``ontovae``, ``vega``: every graph arm,
          in the shape of the deposited ``metrics/*_rand.txt``, MSE included
      ``prepare/cluster_test_metrics.py`` ``sweep`` (figS1's ct=5, ct=10 and 2k
          runs, with a ``setting`` column) and ``randomized`` (fig3's fully
          random and randomized soft-link arms), all four metrics each
    The ``gonnect`` file has no MSE, which stays with the workbooks.
    """
    d = data_dir / "metrics" / "test_split"
    return {
        "gonnect": d / "gonnect_clustering.csv",
        "ontovae": d / f"ontovae{ARM_FILE_SUFFIX}.txt",
        "vega": d / f"vega{ARM_FILE_SUFFIX}.txt",
        "ss_per_type": d / "per_type_ss.csv",
        "purity_within_test": d / f"per_type_purity_within_test_k{PURITY_K_WITHIN_TEST}.csv",
        **{f"purity_per_type_k{k}": d / f"per_type_purity_k{k}.csv" for k in PURITY_K_VALUES},
        "sweep": d / "sweep_metrics.csv",
        "randomized": d / "randomized_metrics.csv",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════
def compute_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD and n per (metric, method) over the repeats."""
    return (
        data.groupby(["metric", "method"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), n=("value", "count"))
        .sort_values(["metric", "method"])
    )


def adjust_fdr(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg step-up adjustment."""
    if not p_values:
        return []
    order = np.argsort(p_values)
    ranked = np.empty(len(p_values))
    n = len(p_values)
    for i, idx in enumerate(order, start=1):
        ranked[idx] = p_values[idx] * n / i
    for i in range(n - 2, -1, -1):
        ranked[order[i]] = min(ranked[order[i]], ranked[order[i + 1]])
    return ranked.tolist()


def paired_t_tests(data: pd.DataFrame, baseline: str,
                   alpha: float = 0.05) -> pd.DataFrame:
    """Paired t-test of every method against ``baseline``, over the repeats.

    Pairing is on the repeat id, so it only holds where the models share their
    data splits. FDR is applied within each metric.
    """
    rows = []
    for metric in sorted(data["metric"].unique()):
        md = data[data["metric"] == metric]
        base = md[md["method"] == baseline]
        if base.empty:
            continue
        for method in sorted(md["method"].unique()):
            if method == baseline:
                continue
            cmp = md[md["method"] == method]
            merged = base.merge(cmp, on="repeat", suffixes=("_base", "_cmp"))
            if len(merged) < 2:
                continue
            stat, p = stats.ttest_rel(merged["value_cmp"], merged["value_base"])
            rows.append({
                "metric": metric, "method": method, "n": int(len(merged)),
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


def stars(p: float) -> Optional[str]:
    """Significance marker, or None when not significant."""
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return None


def sig_marker(p: float) -> str:
    """Significance marker that spells out non-significance as 'ns'."""
    return stars(p) or "ns"


# ══════════════════════════════════════════════════════════════════════════════
# Embeddings and t-SNE
# ══════════════════════════════════════════════════════════════════════════════
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42


def load_embedding(emb_dir: Path, version: str, seed: int, module: str) -> np.ndarray:
    """Load ``AE_<version>.<seed>_<module>_full_dataset.pt`` as a float32 array."""
    path = emb_dir / f"AE_{version}" / f"AE_{version}.{seed}_{module}_full_dataset.pt"
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    return tensor.numpy().astype(np.float32)


def tsne(X: np.ndarray) -> np.ndarray:
    """2-D t-SNE with the settings used for every embedding panel in the paper."""
    return TSNE(
        n_components=2, perplexity=TSNE_PERPLEXITY, max_iter=TSNE_N_ITER,
        init="pca", random_state=TSNE_RANDOM_STATE, verbose=0,
    ).fit_transform(X)


def tsne_cached(cache_dir: Path, key: str, compute, recompute: bool = False) -> np.ndarray:
    """t-SNE coordinates for ``key``, cached as ``<cache_dir>/tsne_<key>.npz``.

    ``compute`` is a zero-argument callable returning the coordinates. t-SNE on
    the full 9,797-sample matrix takes minutes, so the cache is worth keeping.
    """
    cache_path = cache_dir / f"tsne_{key}.npz"
    if cache_path.exists() and not recompute:
        print(f"  cache hit:  {cache_path.name}")
        return np.load(cache_path)["xy"]
    print(f"  computing:  {key} (t-SNE) ...")
    xy = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, xy=xy)
    return xy


def load_cancer_types(tcga_path: Path) -> pd.Series:
    """The per-sample cancer-type labels, in file order."""
    return pd.read_csv(tcga_path, usecols=["cancer_type"])["cancer_type"]


# ══════════════════════════════════════════════════════════════════════════════
# The models with saved embeddings
#
# Read by prepare/test_split_metrics.py, which scores them on held-out samples,
# and by the figures that lay out its per-cancer-type tables.
# ══════════════════════════════════════════════════════════════════════════════

PURITY_K_VALUES = (10, 20, 30)
PURITY_K_WITHIN_TEST = 10       # Figure 2k: neighbours searched within the test split

SEEDS_BY_VERSION: Dict[str, List[int]] = {
    "2.0": [2, 3, 4, 5, 6],
    "2.1": [2, 3, 4, 5, 6],
    "2.2": [22, 23, 24, 25, 26],   # the randomized arm uses its own seed block
}

# (version, module, method key). The method keys match the mse_per_cluster
# workbook's columns, so purity and SS line up with MSE per cancer type.
EMBEDDING_MODELS: List[tuple] = [
    ("2.0", "none",    "MLP"),
    ("2.0", "encoder", "GONNECT-enc"),
    ("2.0", "decoder", "GONNECT-dec"),
    ("2.0", "both",    "GONNECT-both"),
    ("2.1", "encoder", "GONNECT-SL-enc"),
    ("2.1", "decoder", "GONNECT-SL-dec"),
    ("2.1", "both",    "GONNECT-SL-both"),
    ("2.2", "encoder", "GONNECT-R-enc"),
    ("2.2", "decoder", "GONNECT-R-dec"),
    ("2.2", "both",    "GONNECT-R-both"),
]
