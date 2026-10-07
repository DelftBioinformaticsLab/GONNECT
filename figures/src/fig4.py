"""Figure 4: activation-enrichment agreement, per layer and per baseline.

Every violin is a column-shuffle null distribution of the median per-cancer-type
Spearman r between the per-(cancer_type, node) AUC and the GSEA -log10(NOM p) of
the same node. The red dots are the five per-seed observed values, and the red
line is the statistic on a pooled AUC matrix. For panels d and e that is always
the per-seed AUC averaged over seeds, since each baseline seed has its own test
split. For panels a-c the published figure computes one AUC on the seed-averaged
activations instead. --pool auc (with nulls built by plot_perm_nulls_layers.py
--pool auc) puts a-c on the baselines' definition, so every red line means the
same thing. A node's sign is arbitrary per seed, so averaging activations can
partly cancel, while averaging AUCs does not.

--pool seeds pools nothing: in every panel the red line is the mean of the five
red dots. Its null is computed here rather than read from the .npz files. Each
of 1000 permutations shuffles every seed's node columns independently, rescores
each seed and averages the five, so that null is narrower than a pooled one.

Panels
------
  a  GONNECT       encoder L0-L3 then decoder L5-L8        (8 violins)
  b  GONNECT-SL    encoder L0-L3 then decoder L5-L8        (8 violins)
  c  GONNECT-DPR   encoder / decoder / both, bottleneck    (3 violins)
  d  OntoVAE       L0-L10                                  (11 violins)
  e  VEGA          Hallmark and Reactome x true/DPR/FR     (6 violins)

The 109-node latent is highlighted where it appears (encoder L3 / decoder L5 in
panels a and b, all of panel c). Nulls for panels a, b, d and e are read from
precomputed .npz files; panel c is computed inline from the bottleneck
embeddings, because only its encoder null was precomputed.

Layout: panels a-c fill the top row, d-e the bottom one. One violin is the same
width in all five panels; the row that needs the most violins fixes that width
and the other row spends its leftover space on the gap between its panels. The
key is a band of side-by-side blocks below the panels, because at the paper's
type size a single legend column would be taller than the whole figure.

Inputs (relative to --data-dir)
------------------------------
    TCGA_complete_bp_top1k.csv.gz
        per-sample cancer type / sample type, in row order
    hard_links.csv
        GO graph: the bottleneck node order and the term -> layer map

    gsea_gonnect_receptive_fields/gsea_results.csv  GONNECT GO terms, every layer (receptive fields)
    gsea_baselines/gsea_results_hallmark.csv   VEGA Hallmark pathways
    gsea_baselines/gsea_results_reactomes.csv  VEGA Reactome pathways
    gsea_baselines/gsea_results_ontovae.csv    OntoVAE GO terms
    untrained_reference/fig4_untrained_{gonnect,dpr,baselines}.tsv
        the untrained boxes (see --untrained)

    Under --published instead:
    gsea_gonnect_layers/gsea_results.csv       GONNECT GO terms of panels a-b (direct annotations)
    gsea_gonnect_bottleneck/gsea_results.csv   bottleneck GO terms of panel c (receptive fields)

    perm_nulls_gonnect/AE_<version>_<module>_auc/perm_layer_<L>_arrays.npz
    perm_nulls_baselines/OntoVAE_layer_<LL>_true_auc/perm_nulls_arrays.npz
    perm_nulls_baselines/VEGA_<db>_<variant>_auc/perm_nulls_arrays.npz
        Read only under --pool activations or auc (the published figure's
        nulls); --pool seeds, the default, computes its null here. They hold
        the null distribution and the pooled observed
        value of panels a, b, d and e. They are produced by a separate and
        expensive permutation pipeline (1000 column shuffles per layer and per
        model) that is NOT part of this script. A missing file drops its violin
        from the figure; every omission is listed again at the end of the run.

    go_term_activations_corrected/AE_<version>.<seed>_<module>_activations.csv.gz
        per-seed dots of panels a and b (--published: go_term_activations/)
    latent_embeddings/AE_2.2/AE_2.2.<seed>_<module>_full_dataset.pt
        panel c, both its null and its dots
    baseline_activations/ontovae/run_seed-<seed>/pathway_activities_test_<variant>.parquet
        per-seed dots of panel d
    baseline_activations/vega/run_seed-<seed>/z_test_<variant>_<db>.csv
        per-seed dots of panel e (also supplies OntoVAE's row -> sample mapping)

    cache/auc_4row[_<hash>]/<model>_seed<seed>[_test].npz
        Per-seed (cancer type x node) AUC matrices. Building one re-reads a full
        activation/embedding file and re-runs every ROC-AUC, so they are cached;
        the baseline activation files above are only read when the cache is
        cold. --no-cache recomputes and rewrites them, --no-dots skips them.

Defaults: the revised figure
----------------------------
Without options, fig4.py draws the revised Figure 4. It differs from the
published one in five ways, each of which an option below can undo, and
--published undoes all of them at once:

    --pool seeds              (published: activations)
    --eval-set test           (published: all)
    --activations-dir <data-dir>/go_term_activations_corrected
                              (published: go_term_activations, whose decoder
                              files are labelled one layer off)
    --gsea-layers-csv and --gsea-bottleneck-csv
                              <data-dir>/gsea_gonnect_receptive_fields/gsea_results.csv
                              (published: gsea_gonnect_layers/, whose sets miss
                              the genes entering at layers 2-3, for panels a-b,
                              and gsea_gonnect_bottleneck/ for panel c)
    --untrained               <data-dir>/untrained_reference/fig4_untrained_*.tsv
                              (published: none; --no-untrained drops them)

Options
-------
    --published
        Every default above as the published figure had it. Options given
        explicitly still win.
    --untrained  <data-dir>/untrained_reference/fig4_untrained_gonnect.tsv
                 <data-dir>/untrained_reference/fig4_untrained_dpr.tsv
                 <data-dir>/untrained_reference/fig4_untrained_baselines.tsv
        Untrained references, one file or several: the statistic of every
        untrained initialization per violin. That is from
        prepare/untrained_control.py for panels a and b, from
        prepare/untrained_dpr.py for panel c, and from
        prepare/untrained_baselines.py for d and e (OntoVAE's true graph and
        all of VEGA's).
        Drawn as a box left of the violin (median, quartiles, 2.5-97.5 %
        whiskers), and summarized in fig4.csv. They must have been built with
        the same --eval-set and GSEA reference as the figure; the default ones
        are what prepare/untrained_{control,dpr,baselines}.py write with their
        own defaults.
    --no-untrained
        Draw no untrained reference.
    --activations-dir, --gonnect-nulls-dir, --cache-dir
        Point panels a and b at other GONNECT activations and nulls. Cache
        entries are not checked against the files they were computed from, so
        the default cache is per activations directory: cache/auc_4row for
        go_term_activations, as before, and cache/auc_4row_<hash of the
        directory> for any other.
    --pool {activations,auc,seeds}
        The red line's definition (see the top of this docstring). seeds reads
        no perm_nulls_* file and takes a few minutes longer.
    --eval-set {all,test}
        Which samples every AUC is computed over. 'all' (the published figure)
        scores panels a-c on every primary tumour, training samples included,
        and panels d-e on their seed's test split, of every sample type. 'test'
        puts all five panels on one footing: the primary tumours of each
        seed's own test split (gonnect.train.train.split_data at that seed, via
        prepare/test_split_metrics.split_positions; AE_2.2.22-26 trained on
        splits 2-6). The baselines' test files hold that split plus 40-55 more
        rows per seed (their id-based reload brings back every row of a test
        patient); that is checked, and only the split's own rows are scored.
        Each seed then has its own rows, so it needs --pool seeds.
        Cache entries get a `_test` suffix.
    --gsea-layers-csv CSV, --gsea-bottleneck-csv CSV
        The GSEA references for panels a-b and panel c. The default puts every
        panel on one gene-set definition, the term's receptive field (every
        gene with a path to it, i.e. its own and its descendants'
        annotations), as prepare/run_gsea.py writes it.

Usage
-----
    python fig4.py [--data-dir figures/data] [--out-dir figures/out]
                   [--n-perms 1000] [--rng-seed 42] [--no-dots] [--no-cache]
                   [--untrained TSV ...] [--activations-dir DIR]
                   [--gonnect-nulls-dir DIR] [--cache-dir DIR]
                   [--pool {activations,auc,seeds}] [--eval-set {all,test}]
                   [--gsea-layers-csv CSV] [--gsea-bottleneck-csv CSV]
                   [--no-untrained] [--published]

Writes <out-dir>/fig4.png, fig4.pdf and fig4.csv (one row per violin: pooled
observed value, permutation p, null mean/SD and the per-seed values). Under
--pool seeds the observed column is obs_mean_over_seeds, followed by the stars.
"""

import argparse
import gzip
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from _common import (CANCER_TYPE_ORDER, DEC_LAYERS, EMB_VERSION_DISPLAY,
                     ENC_LAYERS, FIG_WIDTH_IN, MODULE_ABBREV, PT_BODY,
                     PT_PANEL, PT_SMALL, PT_TITLE, RANDOMIZATION_DISPLAY,
                     VEGA_DB_DISPLAY, add_io_args, display, figsize,
                     load_embedding, pt, report_text_overlaps, save_figure,
                     sig_marker)

# Keys inside the precomputed npz files: the column-shuffle null of the
# median per-cancer-type statistic, and the matching pooled observed value.
NULL_KEY = "col__per_ct"
OBS_KEY = "observed__per_ct"

GONNECT_SEEDS = [2, 3, 4, 5, 6]
RAND_SEEDS = [22, 23, 24, 25, 26]
BASELINE_SEEDS = [2, 3, 4, 5, 6]
# Run number minus the split seed it trained on (prepare/test_split_metrics.SPLIT_OFFSET):
# AE_2.2.22-26 trained on splits 2-6. A baseline's run_seed-<s> trained on split s.
SPLIT_OFFSET = {"2.0": 0, "2.1": 0, "2.2": 20}
EVAL_SAMPLE_TYPE = "Primary Tumor"


@dataclass(frozen=True)
class TestSplit:
    """--eval-set test: which positional TCGA rows every seed is scored on.

    ``rows[split_seed]`` holds the sorted positions of that seed's test split,
    exactly as gonnect.train.train.split_data draws it (TRAIN_FRACTION 0.7,
    via prepare/test_split_metrics.split_positions). ``sample_type`` is the
    sample type of every row, so the baselines get the same primary-tumour
    filter as GONNECT. ``patient_id`` (per row, when known) lets the baseline
    check recognize the extra rows the baselines' id-based reload adds.
    """
    rows: dict
    sample_type: np.ndarray
    patient_id: np.ndarray | None = None

    @classmethod
    def build(cls, meta: pd.DataFrame, seeds) -> "TestSplit":
        prepare = str(Path(__file__).resolve().parent / "prepare")
        if prepare not in sys.path:
            sys.path.append(prepare)
        from test_split_metrics import split_positions
        labels = meta["cancer_type"].astype(str).reset_index(drop=True)
        return cls(rows={s: np.sort(split_positions(labels, s)[1]) for s in sorted(set(seeds))},
                   sample_type=meta["sample_type"].to_numpy(),
                   patient_id=meta["patient_id"].to_numpy() if "patient_id" in meta else None)

    def of(self, version: str | None, run: int) -> np.ndarray:
        """Test rows of run ``run`` of GONNECT AE_<version>, or of a baseline seed when version is None."""
        return self.rows[run - (SPLIT_OFFSET[version] if version else 0)]

    def primary(self, version: str | None, run: int) -> np.ndarray:
        """Test rows that are primary tumours."""
        rows = self.of(version, run)
        return rows[self.sample_type[rows] == EVAL_SAMPLE_TYPE]


def _cache_suffix(test_split: "TestSplit | None") -> str:
    return "" if test_split is None else "_test"


COL_FL = "#1f77b4"
COL_SL = "#2ca02c"
COL_ONTO = "#9467bd"
COL_VEGA_HM = "#ff7f0e"
COL_VEGA_RE = "#d62728"
COL_RAND = "#7f7f7f"


# ══════════════════════════════════════════════════════════════════════════════
# Figure geometry
#
# All in inches on the FIG_WIDTH_IN canvas. The layout has to fill that width
# exactly: save_figure keeps the full canvas width and trims only vertically,
# so slack left at the left or right edge stays in the saved file and shows up
# as a margin the other figures do not have.
#
# The panels are placed directly rather than through a gridspec of integer
# violin columns. The property that grid existed to guarantee -- one violin is
# the same width in every panel -- is kept: the row that needs the most violins
# fixes that width, and the shorter row spreads its leftover width into the
# gap between its groups instead of leaving it at the right edge.
# ══════════════════════════════════════════════════════════════════════════════
MARGIN_L_IN = 1.20      # y-axis label + tick labels of a row's leftmost panel
MARGIN_R_IN = 0.38      # panel c ends the top row and its title overhangs it
                        # by more than half a violin, so the row stops short of
                        # the canvas edge and the title reaches it instead
GROUP_GAP_IN = 0.45     # smallest gap between two panels of the same row

PAD_TOP_IN = 0.06
TITLE_BAND_IN = 0.42    # panel title and its pad, above each row of axes
AXES_H_IN = 2.70        # one row of violins
XTICK_TOP_IN = 0.56     # tick labels of panels a-c; a and b set them over two
                        # lines, because "enc L0" on one line leaves barely more
                        # space between labels than there is inside one
XTICK_BOT_IN = 1.16     # panel e's two-line labels, turned on their side
ROW_GAP_IN = 0.30
LEGEND_GAP_IN = 0.34
PAD_BOT_IN = 0.08

# Legend band. The blocks sit side by side; widths are the widest entry each
# block has to hold, set at PT_BODY (report_text_overlaps catches it if a
# display name ever outgrows its column).
LEGEND_COL_W_IN = [2.00,   # Models heading + GONNECT / GONNECT-SL / OntoVAE
                   2.72,   # VEGA (Hallmark) / VEGA (Reactome) / randomized
                   2.63,   # bottleneck / pooled observed / per-seed observed
                   2.85,   # Significance heading + the p-value key
                   2.70]   # the italic note on the null
LEGEND_ROW_IN = 0.32    # baseline pitch inside the band
LEGEND_PAD_IN = 0.10
LEGEND_EDGE_IN = 0.06   # gap between the last legend block and the canvas edge
SWATCH_W_IN, SWATCH_H_IN, SWATCH_PAD_IN = 0.38, 0.16, 0.13

# Line and marker weights, written as the width they render at on the printed
# page (pt() converts). Same rendered weights as before the canvas grew.
LW_HAIRLINE = pt(0.30)      # violin outlines, the zero line, marker edges
LW_DIVIDER = pt(0.35)       # dashed rule between sub-blocks of a panel
LW_OBSERVED = pt(1.40)      # the pooled observed-value marker
DOT_SIZE = pt(2.40) ** 2    # per-seed dots, as a scatter area
LEGEND_DOT_PT = pt(3.20)    # the same dot in the legend, drawn via plot()
TICK_LEN, TICK_PAD = pt(2.0), pt(1.5)

# Headroom above the tallest violin for the significance markers, so they clear
# the panel title. Enough for one line of PT_BODY text plus a little air.
SIG_HEADROOM_IN = 0.34

# The optional untrained reference (--untrained): a box left of the violin's
# centre, clear of the per-seed dots, in a dark neutral so it cannot be taken
# for the grey randomized-control violins.
COL_UNTRAINED = "#3a3a3a"
UNTRAINED_X = -0.25          # offset from the violin's centre, in violin columns
UNTRAINED_HALF_W = 0.065
LW_UNTRAINED = pt(0.60)


# ══════════════════════════════════════════════════════════════════════════════
# Baseline input configuration
# (plot_activation_vs_gsea_baselines.METHOD_DBS_STATIC / ONTOVAE_BASE_CFG, with
# the hard-coded repo paths rebased onto --data-dir)
# ══════════════════════════════════════════════════════════════════════════════

def baseline_cfgs(data_dir: Path) -> tuple[dict, dict]:
    """Where each baseline's activations and GSEA results live.

    Returns ``(method_dbs, ontovae_cfg)``. OntoVAE gets one config per layer at
    call time (a ``"layer"`` key is added); its parquet has no sample ids, so
    the row -> TCGA mapping is borrowed from VEGA's matching seed file (the
    per-seed test-set sizes match exactly, so the splits are treated as equal).
    """
    baselines = data_dir / "baseline_activations"
    gsea_baselines = data_dir / "gsea_baselines"
    method_dbs = {
        ("VEGA", "hallmark"): {
            "loader": "vega_csv",
            "dir": baselines / "vega",
            "file_template": "z_test_{variant}_hallmark_v2026_1_Hs_uniprot.csv",
            "gsea_csv": gsea_baselines / "gsea_results_hallmark.csv",
        },
        ("VEGA", "reactomes"): {
            "loader": "vega_csv",
            "dir": baselines / "vega",
            "file_template": "z_test_{variant}_reactomes_uniprot.csv",
            "gsea_csv": gsea_baselines / "gsea_results_reactomes.csv",
        },
    }
    ontovae_cfg = {
        "loader": "ontovae_layer",
        "dir": baselines / "ontovae",
        "file_template": "pathway_activities_test_{variant}.parquet",
        "vega_sample_dir": baselines / "vega",
        "vega_sample_template": "z_test_{variant}_hallmark_v2026_1_Hs_uniprot.csv",
        "gsea_csv": gsea_baselines / "gsea_results_ontovae.csv",
    }
    return method_dbs, ontovae_cfg


# ══════════════════════════════════════════════════════════════════════════════
# Baseline activation loaders and GSEA pivot
# (from plot_activation_vs_gsea_baselines.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_vega_csv(path: Path, *, seed: int, cfg: dict, variant: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load a VEGA z_test_*.csv file.

    Returns:
      activations : (n_samples, n_pathways) float
      row_idx     : (n_samples,) int — positional index into TCGA metadata
      pathways    : list of pathway column names
    """
    df = pd.read_csv(path)
    # First column is an unnamed integer index = positional row index in TCGA
    idx_col = df.columns[0]
    row_idx = df[idx_col].to_numpy(dtype=int)
    pathways = [c for c in df.columns[1:]]
    activations = df[pathways].to_numpy(dtype=float)
    return activations, row_idx, pathways


def load_ontovae_layer(path: Path, *, seed: int, cfg: dict, variant: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the OntoVAE parquet for one layer:
      - filter columns to (_, layer, _) where layer == cfg["layer"]
      - average the 3 neurons per GO term → one value per (sample, term)
      - borrow row→TCGA mapping from VEGA's matching seed file (per-seed test-set
        sizes are identical, so we treat the splits as the same)
    """
    df = pd.read_parquet(path)
    layer = cfg["layer"]

    layer_cols = [c for c in df.columns if c[1] == layer]
    if not layer_cols:
        raise ValueError(f"No columns at layer {layer} in {path}")

    by_term: dict[str, list] = {}
    for c in layer_cols:
        term = c[0]
        by_term.setdefault(term, []).append(c)
    terms = sorted(by_term)

    n_samples = len(df)
    activations = np.empty((n_samples, len(terms)), dtype=np.float64)
    for i, t in enumerate(terms):
        cols = by_term[t]
        activations[:, i] = df[cols].mean(axis=1).to_numpy(dtype=np.float64)

    # Borrow sample IDs from the matching VEGA file
    vega_path = (cfg["vega_sample_dir"] / f"run_seed-{seed}"
                 / cfg["vega_sample_template"].format(variant=variant))
    vega_idx = pd.read_csv(vega_path, usecols=[0]).iloc[:, 0].to_numpy(dtype=int)
    if len(vega_idx) != n_samples:
        raise ValueError(
            f"OntoVAE seed-{seed} variant={variant} has {n_samples} samples "
            f"but VEGA has {len(vega_idx)} — splits don't match, can't borrow IDs."
        )

    return activations, vega_idx, terms


LOADERS = {"vega_csv": load_vega_csv, "ontovae_layer": load_ontovae_layer}


def load_enrichment_pivot(gsea_csv: Path, cancer_types: list[str]) -> pd.DataFrame:
    """Load GSEA results and pivot to (cancer_type × pathway) -log10(NOM p-val)."""
    gsea = pd.read_csv(gsea_csv)
    gsea["enr_score"] = -np.log10(np.clip(gsea["NOM p-val"], 1e-3, 1.0))
    pivot = gsea.pivot_table(
        index="cancer_type", columns="Term", values="enr_score", aggfunc="first",
    )
    pivot = pivot.reindex(index=[ct for ct in cancer_types if ct in pivot.index])
    return pivot


# ══════════════════════════════════════════════════════════════════════════════
# GO graph readers
# (load_bottleneck_index from plot_activation_vs_gsea.py, reduced to the one
# column this figure uses; build_layer_map from plot_perm_nulls_layers.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_bottleneck_columns(hard_links_path: Path) -> list[str]:
    """GO term ids of the bottleneck nodes, ordered by embedding dimension."""
    hl = pd.read_csv(hard_links_path)
    enc = hl[hl["component"] == "encoder"]
    last_layer = enc["layer"].max()
    bottleneck = (
        enc[enc["layer"] == last_layer][["sink_index", "sink_term_id", "sink_term_name"]]
        .drop_duplicates()
        .sort_values("sink_index")
    )
    return bottleneck["sink_term_id"].tolist()


def build_layer_map(hard_links_path: Path, module: str) -> dict[str, int]:
    """
    Return {go_term_id: layer_num} for the chosen module.

    Encoder: term → its sink layer (output of that layer in the encoder).
    Decoder: term → its source layer (input of that layer in the decoder).
             That puts the bottleneck at decoder layer 5, then 6/7/8 going
             toward gene reconstruction. Symmetric to the encoder's 0..3.
    """
    hl = pd.read_csv(hard_links_path)
    if module == "encoder":
        sub = hl[hl["component"] == "encoder"]
        pairs = sub[["sink_term_id", "layer"]].drop_duplicates()
        return dict(zip(pairs["sink_term_id"], pairs["layer"].astype(int)))
    elif module == "decoder":
        sub = hl[hl["component"] == "decoder"]
        pairs = sub[["source_term_id", "layer"]].drop_duplicates()
        return dict(zip(pairs["source_term_id"], pairs["layer"].astype(int)))
    raise ValueError(module)


# ══════════════════════════════════════════════════════════════════════════════
# GONNECT activation loading and per-cancer-type scoring
# (from plot_perm_nulls_layers.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_activations_avg(
    activations_dir: Path,
    version: str,
    seeds: list[int],
    module: str,
    sample_type: str = "Primary Tumor",
    *,
    metric: str = "mean_abs",
    rows: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Load all-seed activations for one (version, module) combo.

    metric="mean_abs": take |·|, average across seeds, return mean |activation|
    metric="auc":      average raw values across seeds (no abs), return that.
                       AUC per (cancer_type, GO_term) is computed downstream
                       in aggregate_per_cancer_type.
    rows:              positional rows (into the TCGA row order the files are
                       written in) to keep before the sample-type filter, e.g.
                       one seed's test split; None keeps every row.

    Returns:
      meta : (n_samples, metadata_cols) after sample-type filter
      mat  : (n_samples, n_go_terms) ndarray
      go_cols : list of GO term column names in matrix order
    """
    agg_sum: np.ndarray | None = None
    meta: pd.DataFrame | None = None
    go_cols: list[str] | None = None

    # (the original drove this loop with a tqdm bar; plain prints keep the
    # script free of dependencies beyond the ones _common already needs)
    print(f"  reading seeds (AE_{version}, {module}, {metric}): {seeds}", flush=True)
    for s in seeds:
        path = activations_dir / f"AE_{version}.{s}_{module}_activations.csv.gz"
        t0 = time.time()
        with gzip.open(path, "rt") as fh:
            df = pd.read_csv(fh)
        if go_cols is None:
            go_cols = [c for c in df.columns if c.startswith("GO:")]
            meta_cols = [c for c in df.columns if c not in go_cols]
            meta = df[meta_cols].copy()
        raw = df[go_cols].to_numpy(dtype=np.float32)
        vals = np.abs(raw) if metric == "mean_abs" else raw
        agg_sum = vals if agg_sum is None else agg_sum + vals
        print(f"    seed {s}: {df.shape} loaded in {time.time()-t0:.1f}s", flush=True)

    mat = agg_sum / len(seeds)

    if rows is not None:
        if len(rows) and rows.max() >= len(meta):
            raise ValueError(f"row {rows.max()} requested from {len(meta)}-row activations")
        mat = mat[rows]
        meta = meta.iloc[rows].reset_index(drop=True)

    if sample_type != "all":
        mask = (meta["sample_type"] == sample_type).to_numpy()
        mat = mat[mask]
        meta = meta.loc[mask].reset_index(drop=True)

    return meta, mat, go_cols


def aggregate_per_cancer_type(
    mat: np.ndarray,
    meta: pd.DataFrame,
    go_cols: list[str],
    cancer_types: list[str],
    *,
    metric: str = "mean_abs",
) -> pd.DataFrame:
    """
    Build a (n_cancer_types, n_terms) score matrix.

    metric="mean_abs": per-CT mean of |activation| values across samples.
    metric="auc":      per-(CT, term) one-vs-rest ROC-AUC of the per-sample raw
                       activation, symmetrized with max(AUC, 1-AUC). Mirrors
                       fig_graph/compute_node_cancer_specificity.py.
    """
    n_terms = mat.shape[1]
    out = np.zeros((len(cancer_types), n_terms), dtype=np.float32)
    if metric == "mean_abs":
        for i, ct in enumerate(cancer_types):
            mask = (meta["cancer_type"] == ct).to_numpy()
            out[i] = mat[mask].mean(axis=0) if mask.any() else np.nan
    elif metric == "auc":
        labels = meta["cancer_type"].to_numpy()
        for i, ct in enumerate(cancer_types):
            y = (labels == ct).astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                out[i] = np.nan
                continue
            for j in range(n_terms):
                a = roc_auc_score(y, mat[:, j])
                out[i, j] = max(a, 1.0 - a)
    else:
        raise ValueError(metric)
    return pd.DataFrame(out, index=cancer_types, columns=go_cols)


# ══════════════════════════════════════════════════════════════════════════════
# Bottleneck AUC, summary statistics and the permutation null
# (from plot_perm_nulls.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_mean_auc(
    emb_dir: Path,
    meta: pd.DataFrame,
    version: str,
    module: str,
    seeds: list[int],
    sample_type: str = "Primary Tumor",
) -> tuple[np.ndarray, list[str]]:
    """
    Per-(cancer_type, GO_node) ROC-AUC matrix, mirroring
    fig_graph/compute_node_cancer_specificity.py:
      1. average raw embeddings across seeds (no abs)
      2. for each (cancer_type, dim): one-vs-rest AUC on the averaged activation
      3. symmetrize sign with max(AUC, 1-AUC)
    Returns (n_cancer_types, n_dims) and the cancer-type order.
    """
    if sample_type != "all":
        mask = (meta["sample_type"] == sample_type).to_numpy()
    else:
        mask = np.ones(len(meta), dtype=bool)
    meta_filt = meta.loc[mask].reset_index(drop=True)

    seed_embs = []
    for s in seeds:
        seed_embs.append(load_embedding(emb_dir, version, s, module)[mask])
    mean_emb = np.mean(seed_embs, axis=0)        # (n_samples, n_dims), no abs

    present = set(meta_filt["cancer_type"].unique())
    cancer_types = [ct for ct in CANCER_TYPE_ORDER if ct in present]

    labels = meta_filt["cancer_type"].to_numpy()
    n_dims = mean_emb.shape[1]
    auc_mat = np.zeros((len(cancer_types), n_dims), dtype=np.float32)
    for i, ct in enumerate(cancer_types):
        y = (labels == ct).astype(int)
        for j in range(n_dims):
            a = roc_auc_score(y, mean_emb[:, j])
            auc_mat[i, j] = max(a, 1.0 - a)
    return auc_mat, cancer_types


def _summary_stats(act: np.ndarray, enr: np.ndarray) -> tuple[float, float, float]:
    """Return (global r, median per-term r, median per-ct r)."""
    a_flat = act.ravel()
    e_flat = enr.ravel()
    valid = np.isfinite(a_flat) & np.isfinite(e_flat)
    r_global, _ = spearmanr(a_flat[valid], e_flat[valid])

    n_ct, n_terms = act.shape

    r_terms = []
    for j in range(n_terms):
        v = np.isfinite(act[:, j]) & np.isfinite(enr[:, j])
        if v.sum() < 5:
            continue
        r, _ = spearmanr(act[v, j], enr[v, j])
        r_terms.append(r)

    r_cts = []
    for i in range(n_ct):
        v = np.isfinite(act[i]) & np.isfinite(enr[i])
        if v.sum() < 5:
            continue
        r, _ = spearmanr(act[i, v], enr[i, v])
        r_cts.append(r)

    return r_global, float(np.median(r_terms)), float(np.median(r_cts))


def _vectorised_corr_along_axis(
    a_rank: np.ndarray,
    b_rank: np.ndarray,
    valid: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pearson correlation of pre-ranked arrays (= Spearman on the original
    arrays) along the requested axis, restricted to cells where ``valid`` is
    True. Returns (r, n_valid_per_slice).

    Approximation: where NaN patterns differ between act and enr within a slice,
    this re-uses the slice-wide ranks instead of re-ranking on the intersection.
    Difference vs exact Spearman is below the permutation-noise floor for
    sparse-NaN inputs.
    """
    a = np.where(valid, a_rank, 0.0)
    b = np.where(valid, b_rank, 0.0)
    n = valid.sum(axis=axis).astype(float)
    n_safe = np.where(n > 0, n, 1.0)
    a_mean = a.sum(axis=axis) / n_safe
    b_mean = b.sum(axis=axis) / n_safe
    if axis == 0:
        a_c = np.where(valid, a_rank - a_mean[None, :], 0.0)
        b_c = np.where(valid, b_rank - b_mean[None, :], 0.0)
    else:
        a_c = np.where(valid, a_rank - a_mean[:, None], 0.0)
        b_c = np.where(valid, b_rank - b_mean[:, None], 0.0)
    num = (a_c * b_c).sum(axis=axis)
    den = np.sqrt((a_c ** 2).sum(axis=axis) * (b_c ** 2).sum(axis=axis))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan), n


def run_null(
    act: np.ndarray,
    enr: np.ndarray,
    mode: str,
    n_perms: int = 1000,
    seed: int = 42,
    desc: str | None = None,
) -> dict:
    """
    Vectorised permutation null. mode: 'row', 'col', or 'rowcol'.
    Pre-ranks act and enr once (per column for the per-term statistic, per row
    for the per-cancer-type statistic), then expresses each permutation as
    indexing into the rank arrays — no per-iteration ``rankdata`` per slice.
    Pass ``desc`` to label the run in the log.
    """
    rng = np.random.default_rng(seed)
    n_ct, n_terms = act.shape

    act_valid = np.isfinite(act)
    enr_valid = np.isfinite(enr)

    # Per-column ranks (one pass each).
    act_col_rank = np.column_stack([
        rankdata(np.where(act_valid[:, j], act[:, j], np.nan), nan_policy="omit")
        for j in range(n_terms)
    ])
    enr_col_rank = np.column_stack([
        rankdata(np.where(enr_valid[:, j], enr[:, j], np.nan), nan_policy="omit")
        for j in range(n_terms)
    ])
    # Per-row ranks (one pass each).
    act_row_rank = np.vstack([
        rankdata(np.where(act_valid[i, :], act[i, :], np.nan), nan_policy="omit")
        for i in range(n_ct)
    ])
    enr_row_rank = np.vstack([
        rankdata(np.where(enr_valid[i, :], enr[i, :], np.nan), nan_policy="omit")
        for i in range(n_ct)
    ])

    null_global   = np.empty(n_perms)
    null_per_term = np.empty(n_perms)
    null_per_ct   = np.empty(n_perms)

    # (the original wrapped this loop in a tqdm bar when ``desc`` was given)
    if desc is not None:
        print(f"{desc}: {n_perms} perms …", flush=True)

    for k in range(n_perms):
        if mode == "row":
            perm_r = rng.permutation(n_ct)
            perm_c = np.arange(n_terms)
        elif mode == "col":
            perm_r = np.arange(n_ct)
            perm_c = rng.permutation(n_terms)
        elif mode == "rowcol":
            perm_r = rng.permutation(n_ct)
            perm_c = rng.permutation(n_terms)
        else:
            raise ValueError(f"unknown mode: {mode}")

        # Permuted per-column / per-row rank slices and validity
        a_cr = act_col_rank[perm_r, :][:, perm_c]
        a_rr = act_row_rank[perm_r, :][:, perm_c]
        a_p  = act[perm_r, :][:, perm_c]
        valid_p = act_valid[perm_r, :][:, perm_c] & enr_valid

        # Per-term r (across cancer types)
        r_t, n_t = _vectorised_corr_along_axis(a_cr, enr_col_rank, valid_p, axis=0)
        r_t = r_t[n_t >= 5]
        null_per_term[k] = np.nanmedian(r_t) if r_t.size else np.nan

        # Per-cancer-type r (across terms)
        r_c, n_c = _vectorised_corr_along_axis(a_rr, enr_row_rank, valid_p, axis=1)
        r_c = r_c[n_c >= 5]
        null_per_ct[k] = np.nanmedian(r_c) if r_c.size else np.nan

        # Global r — Spearman over the per-perm valid set; small flat rank.
        a_flat = a_p[valid_p]
        e_flat = enr[valid_p]
        if a_flat.size < 5:
            null_global[k] = np.nan
            continue
        a_rk = rankdata(a_flat); a_rk -= a_rk.mean()
        e_rk = rankdata(e_flat); e_rk -= e_rk.mean()
        denom = np.sqrt((a_rk ** 2).sum() * (e_rk ** 2).sum())
        null_global[k] = (a_rk * e_rk).sum() / denom if denom > 0 else np.nan

    return {
        "global":   null_global[~np.isnan(null_global)],
        "per_term": null_per_term[~np.isnan(null_per_term)],
        "per_ct":   null_per_ct[~np.isnan(null_per_ct)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per-seed AUC matrices, their on-disk cache, and the per-seed statistic
# (from plot_violin_per_seed.py)
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(cache_dir: Path, model_id: str, seed: int) -> Path:
    return cache_dir / f"{model_id}_seed{seed}.npz"


def _load_cached_auc(path: Path) -> tuple[np.ndarray, list[str], list[str]] | None:
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=True)
    return (
        npz["auc"],
        [str(x) for x in npz["cancer_types"]],
        [str(x) for x in npz["columns"]],
    )


def _save_cached_auc(
    path: Path, auc: np.ndarray, cancer_types: list[str], columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        auc=auc.astype(np.float32),
        cancer_types=np.array(cancer_types, dtype=object),
        columns=np.array(columns, dtype=object),
    )


def gonnect_auc_one_seed(
    emb_dir: Path,
    meta: pd.DataFrame,
    version: str,
    module: str,
    seed: int,
    sample_type: str = "Primary Tumor",
    rows: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """One seed's (cancer_type, dim) AUC matrix for a GONNECT module.

    Mirrors load_mean_auc but without averaging over seeds first. ``rows``
    (positional) restricts it to those samples, e.g. the seed's test split.
    """
    if sample_type != "all":
        mask = (meta["sample_type"] == sample_type).to_numpy()
    else:
        mask = np.ones(len(meta), dtype=bool)
    if rows is not None:
        mask &= np.isin(np.arange(len(meta)), rows)
    meta_filt = meta.loc[mask].reset_index(drop=True)

    emb = load_embedding(emb_dir, version, seed, module)[mask]

    present = set(meta_filt["cancer_type"].unique())
    cancer_types = [ct for ct in CANCER_TYPE_ORDER if ct in present]
    labels = meta_filt["cancer_type"].to_numpy()

    n_dims = emb.shape[1]
    auc_mat = np.zeros((len(cancer_types), n_dims), dtype=np.float32)
    for i, ct in enumerate(cancer_types):
        y = (labels == ct).astype(int)
        for j in range(n_dims):
            a = roc_auc_score(y, emb[:, j])
            auc_mat[i, j] = max(a, 1.0 - a)
    return auc_mat, cancer_types


def baseline_auc_one_seed(
    cfg: dict,
    variant: str,
    cancer_type_per_row: dict[int, str],
    seed: int,
    test_split: TestSplit | None = None,
) -> pd.DataFrame:
    """One seed's (cancer_type, pathway) AUC matrix for a baseline.

    The files hold the seed's test split, of every sample type. With
    ``test_split`` (--eval-set test) their rows must be exactly that seed's
    split_data test split, and only its primary tumours are scored, as for
    GONNECT.
    """
    loader = LOADERS[cfg["loader"]]
    path = cfg["dir"] / f"run_seed-{seed}" / cfg["file_template"].format(variant=variant)
    acts, row_idx, pathways = loader(path, seed=seed, cfg=cfg, variant=variant)
    if test_split is not None:
        acts, row_idx = restrict_to_test_primary(acts, row_idx, test_split, seed, path)

    ct_arr = np.array([cancer_type_per_row.get(int(i)) for i in row_idx])
    cts_present = [ct for ct in pd.unique(ct_arr) if ct is not None]
    cancer_types = [ct for ct in CANCER_TYPE_ORDER if ct in cts_present]
    cancer_types += sorted(set(cts_present) - set(CANCER_TYPE_ORDER))

    n_pathways = acts.shape[1]
    auc_mat = np.zeros((len(cancer_types), n_pathways), dtype=np.float32)
    for i, ct in enumerate(cancer_types):
        y = (ct_arr == ct).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        for j in range(n_pathways):
            a = roc_auc_score(y, acts[:, j])
            auc_mat[i, j] = max(a, 1.0 - a)
    return pd.DataFrame(auc_mat, index=cancer_types, columns=pathways)


def restrict_to_test_primary(acts: np.ndarray, row_idx: np.ndarray, test_split: TestSplit, seed: int,
                             source) -> tuple[np.ndarray, np.ndarray]:
    """A baseline seed's activations, cut to the primary tumours of its test split.

    The baselines' files do not hold exactly split_data's test split at
    ``seed``: the runners saved the split as patient ids and reloaded it with
    ``isin``, which brings back every row of a test patient, including rows
    split_data put in train or validation (patient_id is not unique; 1523
    rows instead of 1470 at seed 2). So this checks that the file holds every
    row of the test split, and that each extra row belongs to a patient with a
    row in it, and then scores only the test split's own rows -- the very
    samples GONNECT is scored on.
    """
    rows = np.asarray(row_idx, dtype=int)
    if len(np.unique(rows)) != len(rows):
        raise AssertionError(f"{source}: repeated rows")
    expected = test_split.of(None, seed)
    missing = np.setdiff1d(expected, rows)
    if missing.size:
        raise AssertionError(f"{source}: {missing.size} rows of split_data's test split at seed {seed} are absent")
    extra = np.setdiff1d(rows, expected)
    if extra.size and test_split.patient_id is not None:
        stray = ~np.isin(test_split.patient_id[extra], test_split.patient_id[expected])
        if stray.any():
            raise AssertionError(f"{source}: {stray.sum()} rows outside seed {seed}'s test split "
                                 f"belong to no test patient")
    keep = np.isin(rows, expected) & (test_split.sample_type[rows] == EVAL_SAMPLE_TYPE)
    return acts[keep], rows[keep]


def per_ct_median_r(act_df: pd.DataFrame, enr_pivot: pd.DataFrame) -> float:
    """Median per-CT Spearman r — restricts to the act/enr intersection."""
    cts = [ct for ct in act_df.index if ct in enr_pivot.index]
    terms = sorted(set(act_df.columns) & set(enr_pivot.columns))
    if not cts or not terms:
        return np.nan
    a = act_df.loc[cts, terms].to_numpy(dtype=float)
    e = enr_pivot.loc[cts, terms].to_numpy(dtype=float)
    _, _, obs_ct = _summary_stats(a, e)
    return obs_ct


# ══════════════════════════════════════════════════════════════════════════════
# Null / observed loaders
# ══════════════════════════════════════════════════════════════════════════════

def perm_pval(obs: float, null_arr: np.ndarray) -> float:
    """One-sided permutation p (obs > null)."""
    return float((np.sum(null_arr >= obs) + 1) / (len(null_arr) + 1))


def load_npz_null(npz_path: Path, missing: list[Path]) -> tuple[np.ndarray, float] | None:
    """Null distribution + pooled observed value from a precomputed npz.

    A missing file only drops its violin, so every absent path is recorded in
    ``missing`` and re-reported at the end of the run.
    """
    if not npz_path.exists():
        print(f"  WARNING: missing {npz_path}")
        missing.append(npz_path)
        return None
    d = np.load(npz_path)
    return d[NULL_KEY], float(d[OBS_KEY])


# ══════════════════════════════════════════════════════════════════════════════
# Per-seed dot builders
# ══════════════════════════════════════════════════════════════════════════════

def gonnect_layer_full_auc(
    activations_dir: Path, version: str, module: str, seed: int, cache_dir: Path,
    use_cache: bool = True, test_split: TestSplit | None = None,
) -> pd.DataFrame:
    """Cached (cancer_type x all-GO-node) AUC matrix for one seed/module.

    With ``test_split``, over the primary tumours of the seed's test split only.
    """
    cache_p = cache_dir / f"AE_{version}_{module}_seed{seed}{_cache_suffix(test_split)}.npz"
    if use_cache:
        cached = _load_cached_auc(cache_p)
        if cached is not None:
            auc_mat, cts, cols = cached
            return pd.DataFrame(auc_mat, index=cts, columns=cols)
    meta, mat, go_cols = load_activations_avg(
        activations_dir, version, [seed], module, metric="auc",
        rows=None if test_split is None else test_split.of(version, seed),
    )
    present = set(meta["cancer_type"].unique())
    cancer_types = [ct for ct in CANCER_TYPE_ORDER if ct in present]
    df = aggregate_per_cancer_type(mat, meta, go_cols, cancer_types, metric="auc")
    _save_cached_auc(cache_p, df.to_numpy(), list(df.index), list(df.columns))
    return df


def gonnect_layer_frames(
    activations_dir: Path, version: str, module: str, layer: int,
    layer_map: dict, cache_dir: Path, use_cache: bool, test_split: TestSplit | None = None,
) -> list[pd.DataFrame]:
    """Each seed's (cancer_type x node) AUC matrix, restricted to one layer."""
    out = []
    for s in GONNECT_SEEDS:
        df = gonnect_layer_full_auc(activations_dir, version, module, s, cache_dir, use_cache, test_split)
        out.append(df[[t for t in df.columns if layer_map.get(t) == layer]])
    return out


def gonnect_layer_dots(
    activations_dir: Path, version: str, module: str, layer: int,
    layer_map: dict, enr: pd.DataFrame, cache_dir: Path, use_cache: bool,
) -> np.ndarray:
    return np.array([per_ct_median_r(df, enr) for df in gonnect_layer_frames(
        activations_dir, version, module, layer, layer_map, cache_dir, use_cache)])


def _cached_baseline_auc(cfg: dict, variant: str, cache_name: str, cancer_type_per_row: dict,
                         cache_dir: Path, use_cache: bool, seed: int,
                         test_split: TestSplit | None = None) -> pd.DataFrame:
    cache_p = _cache_path(cache_dir, cache_name + _cache_suffix(test_split), seed)
    cached = _load_cached_auc(cache_p) if use_cache else None
    if cached is not None:
        auc_mat, cts, cols = cached
        return pd.DataFrame(auc_mat, index=cts, columns=cols)
    act_df = baseline_auc_one_seed(cfg, variant, cancer_type_per_row, seed, test_split)
    _save_cached_auc(cache_p, act_df.to_numpy(), list(act_df.index), list(act_df.columns))
    return act_df


def ontovae_frames(ontovae_cfg: dict, layer: int, cancer_type_per_row: dict, cache_dir: Path,
                   use_cache: bool, test_split: TestSplit | None = None) -> list[pd.DataFrame]:
    cfg = {**ontovae_cfg, "layer": layer}
    return [_cached_baseline_auc(cfg, "true", f"OntoVAE_layer_{layer:02d}_true", cancer_type_per_row,
                                 cache_dir, use_cache, s, test_split) for s in BASELINE_SEEDS]


def ontovae_dots(ontovae_cfg: dict, layer: int, enr: pd.DataFrame,
                 cancer_type_per_row: dict, cache_dir: Path,
                 use_cache: bool) -> np.ndarray:
    return np.array([per_ct_median_r(df, enr) for df in ontovae_frames(
        ontovae_cfg, layer, cancer_type_per_row, cache_dir, use_cache)])


def vega_frames(method_dbs: dict, db: str, variant: str, cancer_type_per_row: dict, cache_dir: Path,
                use_cache: bool, test_split: TestSplit | None = None) -> list[pd.DataFrame]:
    return [_cached_baseline_auc(method_dbs[("VEGA", db)], variant, f"VEGA_{db}_{variant}", cancer_type_per_row,
                                 cache_dir, use_cache, s, test_split) for s in BASELINE_SEEDS]


def vega_dots(method_dbs: dict, db: str, variant: str, enr: pd.DataFrame,
              cancer_type_per_row: dict, cache_dir: Path,
              use_cache: bool) -> np.ndarray:
    return np.array([per_ct_median_r(df, enr) for df in vega_frames(
        method_dbs, db, variant, cancer_type_per_row, cache_dir, use_cache)])


def seed_mean_entry(frames: list[pd.DataFrame], enr: pd.DataFrame, n_perms: int, rng_seed: int) -> dict:
    """The red line as the mean of the per-seed statistics (--pool seeds), and its null.

    Each seed is scored exactly as its red dot. In every permutation, each
    seed's GO-term columns are shuffled independently (cancer-type rows fixed),
    each seed's statistic is recomputed, and the five are averaged: the null of
    the mean. Pearson on per-row ranks, as run_null computes it.
    """
    dots = np.array([per_ct_median_r(df, enr) for df in frames])
    prepared = []
    for df in frames:
        cts = [ct for ct in df.index if ct in enr.index]
        terms = sorted(set(df.columns) & set(enr.columns))
        act = df.loc[cts, terms].to_numpy(dtype=float)
        e = enr.loc[cts, terms].to_numpy(dtype=float)
        act_valid, enr_valid = np.isfinite(act), np.isfinite(e)
        rank = lambda x, v: np.vstack([rankdata(np.where(v[i], x[i], np.nan), nan_policy="omit")
                                       for i in range(len(x))])
        prepared.append((rank(act, act_valid), act_valid, rank(e, enr_valid), enr_valid))

    rng = np.random.default_rng(rng_seed)
    null = np.empty(n_perms)
    for k in range(n_perms):
        per_seed = []
        for act_rank, act_valid, enr_rank, enr_valid in prepared:
            perm = rng.permutation(act_rank.shape[1])
            r, n = _vectorised_corr_along_axis(act_rank[:, perm], enr_rank, act_valid[:, perm] & enr_valid, axis=1)
            r = r[n >= 5]
            per_seed.append(np.nanmedian(r) if r.size else np.nan)
        null[k] = np.mean(per_seed)
    if not np.all(np.isfinite(dots)):
        print(f"  WARNING: a per-seed statistic is undefined: {dots}")
    return {"null": null[np.isfinite(null)], "obs_pooled": float(np.mean(dots)), "obs_per_seed": dots}


def rand_entry(
    emb_dir: Path, meta: pd.DataFrame, module: str, bottleneck_columns: list[str],
    enr: pd.DataFrame, n_perms: int, rng_seed: int, pool: str = "activations",
    test_split: TestSplit | None = None,
) -> dict:
    """Compute null + pooled observed + per-seed dots for randomized GONNECT.

    ``pool`` picks the pooled matrix: one AUC on the seed-averaged embeddings
    ("activations", the published figure), or the per-seed AUC averaged over
    seeds ("auc"), as the baseline panels have it. ``test_split`` scores each
    seed on its own test split (AE_2.2.<run> trained on split run - 20), which
    needs pool="seeds".
    """
    if test_split is not None and pool != "seeds":
        raise ValueError("a per-seed test split leaves nothing to pool: use pool='seeds'")
    per_seed = [pd.DataFrame(am, index=c, columns=bottleneck_columns)
                for am, c in (gonnect_auc_one_seed(emb_dir, meta, "2.2", module, s,
                                                   rows=None if test_split is None else test_split.of("2.2", s))
                              for s in RAND_SEEDS)]
    if pool == "seeds":
        return seed_mean_entry(per_seed, enr, n_perms, rng_seed)
    if pool == "auc":
        pooled_df = sum(df.astype(np.float64) for df in per_seed) / len(per_seed)
    else:
        auc_mat, cts = load_mean_auc(emb_dir, meta, "2.2", module, RAND_SEEDS)
        pooled_df = pd.DataFrame(auc_mat, index=cts, columns=bottleneck_columns)
    obs_pooled = per_ct_median_r(pooled_df, enr)

    # aligned matrices for the null
    common_cts = [ct for ct in pooled_df.index if ct in enr.index]
    terms = sorted(set(pooled_df.columns) & set(enr.columns))
    act = pooled_df.loc[common_cts, terms].to_numpy(dtype=float)
    enr_a = enr.loc[common_cts, terms].to_numpy(dtype=float)
    null = run_null(act, enr_a, mode="col", n_perms=n_perms, seed=rng_seed,
                    desc=f"    rand {module} col-null")["per_ct"]

    dots = [per_ct_median_r(df, enr) for df in per_seed]
    return {"null": null, "obs_pooled": obs_pooled, "obs_per_seed": np.array(dots)}


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def draw_untrained(ax, x: float, values: np.ndarray) -> None:
    """The untrained initializations as a box: median, quartiles, 2.5-97.5 % whiskers.

    Drawn under the observed markers, so the pooled line visibly crosses it at
    the height the trained model reached.
    """
    low, q1, median, q3, high = np.percentile(values, [2.5, 25, 50, 75, 97.5])
    centre = x + UNTRAINED_X
    ax.vlines(centre, low, high, colors=COL_UNTRAINED, linewidth=LW_UNTRAINED, zorder=3)
    ax.add_patch(Rectangle((centre - UNTRAINED_HALF_W, q1), 2 * UNTRAINED_HALF_W, q3 - q1,
                           facecolor="white", edgecolor=COL_UNTRAINED, linewidth=LW_UNTRAINED, zorder=3))
    ax.hlines(median, centre - UNTRAINED_HALF_W, centre + UNTRAINED_HALF_W, colors=COL_UNTRAINED,
              linewidth=LW_OBSERVED, zorder=3.5)


def draw_group(ax, entries: list[dict], letter: str, heading: str,
               want_dots: bool, show_ylabel: bool, sig_top: float,
               label_rotation: float = 0.0) -> None:
    """Draw one model group on its own axis. Violin positions are 0..n-1 and
    the caller sets xlim=(-0.5, n-0.5), so one violin is exactly one column
    wide; sizing every axis at ``n * violin_width()`` inches then makes that
    column the same width in all five panels.

    ``label_rotation`` turns the tick labels on their side. Panel e needs it:
    its labels are "Hallmark"/"Reactome" over a graph variant, and at the
    paper's type size those words are wider than one violin column.
    """
    n = len(entries)
    tick_half = 0.34
    # bottleneck highlight bands (behind everything)
    for i, e in enumerate(entries):
        if e.get("bottleneck"):
            ax.axvspan(i - 0.5, i + 0.5, color="#ffe08a", alpha=0.45, zorder=0)
    for i, e in enumerate(entries):
        parts = ax.violinplot([e["null"]], positions=[i], widths=0.8,
                              showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(e["color"])
            body.set_edgecolor("black")
            body.set_linewidth(LW_HAIRLINE)
            body.set_alpha(0.55)
        if e.get("untrained") is not None:
            draw_untrained(ax, i, e["untrained"])
        ax.hlines(e["obs_pooled"], i - tick_half, i + tick_half,
                  colors="#d62728", linewidth=LW_OBSERVED, zorder=4)
        if want_dots and e.get("obs_per_seed") is not None:
            rng = np.random.default_rng(42 + i)
            xs = i + rng.uniform(-0.11, 0.11, size=len(e["obs_per_seed"]))
            ax.scatter(xs, e["obs_per_seed"], s=DOT_SIZE, color="#d62728",
                       edgecolor="white", linewidth=LW_HAIRLINE, zorder=5)
        # significance marker at a fixed height shared across panels
        p = perm_pval(e["obs_pooled"], e["null"])
        ax.text(i, sig_top, sig_marker(p), ha="center", va="bottom",
                fontsize=PT_BODY, color="#333333",
                fontweight="bold" if p < 0.05 else "normal")

    for i, e in enumerate(entries):
        if e.get("divider_before"):
            ax.axvline(i - 0.5, color="grey", linewidth=LW_DIVIDER, alpha=0.5,
                       linestyle="--")

    ax.axhline(0, color="black", linewidth=LW_HAIRLINE, alpha=0.5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([e["label"] for e in entries], fontsize=PT_SMALL,
                       rotation=label_rotation, ha="center", va="top")
    ax.tick_params(axis="both", length=TICK_LEN, pad=TICK_PAD)
    if show_ylabel:
        ax.set_ylabel("median per-CT\nSpearman r", fontsize=PT_BODY,
                      labelpad=pt(2.0))
    # The panel letter is set separately from the heading so it can carry
    # PT_PANEL, the size every other figure uses for its letters; drawing them
    # as one string would tie the letter to the smaller title size.
    ax.set_title(heading, fontsize=PT_TITLE, loc="left", fontweight="bold",
                 pad=pt(3.0))
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(pt(-22.0), pt(3.0)), textcoords="offset points",
                fontsize=PT_PANEL, fontweight="bold", va="bottom", ha="left",
                annotation_clip=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def row_spans(counts: list[int], violin_w: float) -> list[tuple[float, float]]:
    """``(x0, width)`` in inches for each panel of one row.

    Every panel gets exactly ``n * violin_w``; whatever the row does not need
    of the drawing area is split evenly over the gaps between its panels, so
    the row still starts at MARGIN_L_IN and ends at the same right edge as the
    other row. The bottom row carries most of that slack (it holds two violins
    fewer than the top one), which is why the gap between OntoVAE and VEGA is
    wider than the gaps in the top row.
    """
    panel_w = FIG_WIDTH_IN - MARGIN_L_IN - MARGIN_R_IN
    gaps = len(counts) - 1
    gap = (panel_w - sum(counts) * violin_w) / gaps if gaps else 0.0
    spans, x = [], MARGIN_L_IN
    for n in counts:
        spans.append((x, n * violin_w))
        x += n * violin_w + gap
    return spans


def violin_width(rows: list[list[int]]) -> float:
    """Width of one violin column, in inches: as wide as the busiest row allows."""
    panel_w = FIG_WIDTH_IN - MARGIN_L_IN - MARGIN_R_IN
    return min((panel_w - (len(counts) - 1) * GROUP_GAP_IN) / sum(counts)
               for counts in rows)


def sig_headroom(sig_top: float, ymin: float, span: float) -> float:
    """Data-unit space to leave above ``sig_top`` for the significance markers.

    The markers are drawn at sig_top with va="bottom", so they stick out of the
    axes unless the y-limit is raised by their own height -- and straight into
    the panel title, which starts a few points above the axes. Solving for the
    limit that leaves SIG_HEADROOM_IN inches is circular (the data-to-inch
    scale depends on the limit), hence the closed form.
    """
    base = (sig_top - ymin) + 0.05 * span      # everything below sig_top
    return SIG_HEADROOM_IN * base / (AXES_H_IN - SIG_HEADROOM_IN)


def draw_legend(ax, note_lines: list[str], untrained_n: int | None = None,
                red_line_label: str = "pooled observed") -> None:
    """The key, as a band of blocks side by side under the panels.

    ``ax`` is an invisible axes whose data coordinates are inches measured from
    its top-left corner, so every block can be placed at its measured width.
    The blocks were a single stacked column when the figure was set two sizes
    smaller; at the paper's type size that column is taller than the figure, so
    it is dealt out over five columns instead.
    """
    band_w = ax.get_xlim()[1]
    gap = (band_w - sum(LEGEND_COL_W_IN)) / (len(LEGEND_COL_W_IN) - 1)
    col_x, x = [], 0.0
    for width in LEGEND_COL_W_IN:
        col_x.append(x)
        x += width + gap

    def row_y(row: int) -> float:
        return LEGEND_PAD_IN + (row + 0.5) * LEGEND_ROW_IN

    def heading(col: int, text: str) -> None:
        ax.text(col_x[col], row_y(0), text, va="center", ha="left",
                fontsize=PT_BODY, fontweight="bold")

    def label(col: int, row: int, text: str, **kw) -> None:
        ax.text(col_x[col] + SWATCH_W_IN + SWATCH_PAD_IN, row_y(row), text,
                va="center", ha="left", fontsize=PT_BODY, **kw)

    def swatch(col: int, row: int, text: str, color: str,
               alpha: float = 0.55, edge: str = "black") -> None:
        y = row_y(row)
        ax.add_patch(Rectangle((col_x[col], y - SWATCH_H_IN / 2),
                               SWATCH_W_IN, SWATCH_H_IN, facecolor=color,
                               edgecolor=edge, linewidth=LW_HAIRLINE,
                               alpha=alpha, clip_on=False))
        label(col, row, text)

    def line_row(col: int, row: int, text: str) -> None:
        y = row_y(row)
        ax.plot([col_x[col], col_x[col] + SWATCH_W_IN], [y, y],
                color="#d62728", lw=LW_OBSERVED, solid_capstyle="butt",
                clip_on=False)
        label(col, row, text)

    def dot_row(col: int, row: int, text: str) -> None:
        ax.plot([col_x[col] + SWATCH_W_IN / 2], [row_y(row)], marker="o",
                color="#d62728", markeredgecolor="white",
                markersize=LEGEND_DOT_PT, linestyle="", clip_on=False)
        label(col, row, text)

    def plain(col: int, row: int, text: str, **kw) -> None:
        ax.text(col_x[col], row_y(row), text, va="center", ha="left", **kw)

    def box_row(col: int, row: int, text: str) -> None:
        y, centre = row_y(row), col_x[col] + SWATCH_W_IN / 2
        ax.plot([centre, centre], [y - 0.11, y + 0.11], color=COL_UNTRAINED, lw=LW_UNTRAINED, clip_on=False)
        ax.add_patch(Rectangle((centre - 0.07, y - 0.06), 0.14, 0.12, facecolor="white",
                               edgecolor=COL_UNTRAINED, linewidth=LW_UNTRAINED, clip_on=False))
        ax.plot([centre - 0.07, centre + 0.07], [y, y], color=COL_UNTRAINED, lw=LW_OBSERVED,
                solid_capstyle="butt", clip_on=False)
        label(col, row, text)

    heading(0, "Models")
    swatch(0, 1, EMB_VERSION_DISPLAY["2.0"], COL_FL)
    swatch(0, 2, EMB_VERSION_DISPLAY["2.1"], COL_SL)
    swatch(0, 3, "OntoVAE", COL_ONTO)

    swatch(1, 1, display("vega_hallmark"), COL_VEGA_HM)
    swatch(1, 2, display("vega_reactome674"), COL_VEGA_RE)
    swatch(1, 3, "randomized control", COL_RAND)

    swatch(2, 1, "bottleneck layer", "#ffe08a", alpha=0.45, edge="none")
    line_row(2, 2, red_line_label)
    dot_row(2, 3, "per-seed observed")
    if untrained_n:
        box_row(2, 4, f"untrained ({untrained_n} inits)")

    heading(3, "Significance (perm. p)")
    for i, text in enumerate(["***  p < 0.001", "**   p < 0.01", "*    p < 0.05"]):
        plain(3, 1 + i, text, fontsize=PT_BODY, family="monospace")

    for i, text in enumerate(note_lines):
        plain(4, 1 + i, text, fontsize=PT_SMALL, style="italic")


# What the revised figure reads by default, under --data-dir; --published restores the published inputs.
REVISED_ACTIVATIONS = "go_term_activations_corrected"
REVISED_GSEA = Path("gsea_gonnect_receptive_fields") / "gsea_results.csv"
REVISED_UNTRAINED = [Path("untrained_reference") / f"fig4_untrained_{name}.tsv"
                     for name in ("gonnect", "dpr", "baselines")]


def resolve_defaults(args: argparse.Namespace) -> None:
    """Fill every option left unset: the revised figure's value, or under --published the published one's."""
    published, d = args.published, args.data_dir
    if args.pool is None:
        args.pool = "activations" if published else "seeds"
    if args.eval_set is None:
        args.eval_set = "all" if published else "test"
    if args.activations_dir is None:
        args.activations_dir = d / ("go_term_activations" if published else REVISED_ACTIVATIONS)
    if args.gsea_layers_csv is None:
        args.gsea_layers_csv = d / "gsea_gonnect_layers" / "gsea_results.csv" if published else d / REVISED_GSEA
    if args.gsea_bottleneck_csv is None:
        args.gsea_bottleneck_csv = d / "gsea_gonnect_bottleneck" / "gsea_results.csv" if published else d / REVISED_GSEA
    if args.no_untrained:
        args.untrained = []
    elif args.untrained is None:
        args.untrained = [] if published else [d / p for p in REVISED_UNTRAINED]


def default_cache_dir(data_dir: Path, activations_dir: Path) -> Path:
    """The AUC cache of one activations directory (its entries are keyed by model and seed only)."""
    if activations_dir.resolve() == (data_dir / "go_term_activations").resolve():
        return data_dir / "cache" / "auc_4row"
    tag = hashlib.sha1(str(activations_dir.resolve()).lower().encode()).hexdigest()[:8]
    return data_dir / "cache" / f"auc_4row_{tag}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_args(parser)
    parser.add_argument("--n-perms", type=int, default=1000,
                        help="permutations for the inline panel-c null "
                             "(the other panels use their precomputed nulls)")
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--pool", choices=["activations", "auc", "seeds"], default=None,
                        help="the red line and its null. 'activations' and 'auc' pool the seeds before "
                             "scoring: one AUC on the seed-averaged embeddings (the published figure) or "
                             "the per-seed AUC averaged, as panels d and e have it; they set panel c, while "
                             "a and b take theirs from --gonnect-nulls-dir. 'seeds' makes the red line the "
                             "mean of the per-seed values in every panel, tested against a null that "
                             "shuffles each seed's GO terms independently (seed_mean_entry). "
                             "Default: seeds (--published: activations)")
    parser.add_argument("--no-dots", action="store_true", help="Skip per-seed dots (fast).")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--untrained", type=Path, nargs="+", default=None,
                        help="untrained references: prepare/untrained_control.py's fig4_untrained_*.tsv "
                             "and/or prepare/untrained_baselines.py's fig4_untrained_baselines.tsv "
                             "(default: <data-dir>/untrained_reference/fig4_untrained_*.tsv)")
    parser.add_argument("--no-untrained", action="store_true", help="draw no untrained reference")
    parser.add_argument("--activations-dir", type=Path, default=None,
                        help="GONNECT activations (default: <data-dir>/go_term_activations_corrected; "
                             "--published: <data-dir>/go_term_activations)")
    parser.add_argument("--gonnect-nulls-dir", type=Path, default=None,
                        help="GONNECT permutation nulls (default: <data-dir>/perm_nulls_gonnect)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="per-seed AUC cache (default: <data-dir>/cache/auc_4row for go_term_activations, "
                             "auc_4row_<hash> for any other activations directory)")
    parser.add_argument("--eval-set", choices=["all", "test"], default=None,
                        help="the samples every AUC is computed over. 'all' (the published figure): every "
                             "primary tumour for panels a-c, the seed's test split of every sample type for d-e. "
                             "'test': the primary tumours of each seed's own test split in every panel; "
                             "needs --pool seeds. Default: test (--published: all)")
    parser.add_argument("--gsea-layers-csv", type=Path, default=None,
                        help="GSEA reference of panels a-b (default: "
                             "<data-dir>/gsea_gonnect_receptive_fields/gsea_results.csv; "
                             "--published: <data-dir>/gsea_gonnect_layers/gsea_results.csv)")
    parser.add_argument("--gsea-bottleneck-csv", type=Path, default=None,
                        help="GSEA reference of panel c (default: as --gsea-layers-csv; "
                             "--published: <data-dir>/gsea_gonnect_bottleneck/gsea_results.csv)")
    parser.add_argument("--published", action="store_true",
                        help="the published figure's settings and inputs for every option not given explicitly")
    args = parser.parse_args()
    resolve_defaults(args)
    if args.eval_set == "test" and args.pool != "seeds":
        parser.error("--eval-set test scores every seed on its own rows, so seed-averaged activations or "
                     "AUCs are undefined: use --pool seeds")

    emb_dir = args.data_dir / "latent_embeddings"
    activations_dir = args.activations_dir
    gonnect_nulls_dir = args.gonnect_nulls_dir or args.data_dir / "perm_nulls_gonnect"
    cache_dir = args.cache_dir or default_cache_dir(args.data_dir, activations_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    want_dots = not args.no_dots
    seeds_mode = args.pool == "seeds"
    if seeds_mode and not want_dots:
        parser.error("--pool seeds builds the red line from the per-seed values, so it needs the dots")
    use_cache = not args.no_cache
    # Every violin whose precomputed null is absent, reported again at the end.
    missing_nulls: list[Path] = []

    method_dbs, ontovae_cfg = baseline_cfgs(args.data_dir)

    print("Loading TCGA metadata …")
    with gzip.open(args.data_dir / "TCGA_complete_bp_top1k.csv.gz", "rt") as fh:
        meta = pd.read_csv(fh, usecols=["patient_id", "cancer_type", "sample_type"])
    cancer_type_per_row = {i: ct for i, ct in enumerate(meta["cancer_type"].tolist())}
    test_split = None
    if args.eval_set == "test":
        test_split = TestSplit.build(meta, GONNECT_SEEDS + BASELINE_SEEDS
                                     + [s - SPLIT_OFFSET["2.2"] for s in RAND_SEEDS])
        print("Evaluating on each seed's test split, primary tumours only: "
              + ", ".join(f"split {s} {len(test_split.primary(None, s))} of {len(rows)}"
                          for s, rows in test_split.rows.items()))

    bottleneck_columns = load_bottleneck_columns(args.data_dir / "hard_links.csv")

    print("Loading GSEA enrichment matrices …")
    enr_alllayer = load_enrichment_pivot(args.gsea_layers_csv, CANCER_TYPE_ORDER)
    enr_bottleneck = load_enrichment_pivot(args.gsea_bottleneck_csv, CANCER_TYPE_ORDER)
    print(f"GSEA reference: panels a-b {args.gsea_layers_csv}, panel c {args.gsea_bottleneck_csv}")
    enr_hallmark = load_enrichment_pivot(method_dbs[("VEGA", "hallmark")]["gsea_csv"], CANCER_TYPE_ORDER)
    enr_reactomes = load_enrichment_pivot(method_dbs[("VEGA", "reactomes")]["gsea_csv"], CANCER_TYPE_ORDER)
    enr_ontovae = load_enrichment_pivot(ontovae_cfg["gsea_csv"], CANCER_TYPE_ORDER)

    layer_maps = {
        ("2.0", "encoder"): build_layer_map(args.data_dir / "hard_links.csv", "encoder"),
        ("2.0", "decoder"): build_layer_map(args.data_dir / "hard_links.csv", "decoder"),
    }
    layer_maps[("2.1", "encoder")] = layer_maps[("2.0", "encoder")]
    layer_maps[("2.1", "decoder")] = layer_maps[("2.0", "decoder")]

    # Optional untrained reference: {(row, "enc L0"): statistic per initialization}
    untrained = {}
    if args.untrained:
        reference = pd.concat([pd.read_csv(path, sep="\t") for path in args.untrained])
        untrained = {key: group["median_r"].to_numpy() for key, group in reference.groupby(["row", "label"])}
        print(f"Untrained reference: {', '.join(str(path) for path in args.untrained)}")

    # ── build rows ────────────────────────────────────────────────────────
    def gonnect_row(version: str, color: str) -> list[dict]:
        entries = []
        plan = [("encoder", ENC_LAYERS), ("decoder", DEC_LAYERS)]
        for module, layers in plan:
            for k, L in enumerate(layers):
                if seeds_mode:
                    frames = gonnect_layer_frames(activations_dir, version, module, L,
                                                  layer_maps[(version, module)], cache_dir, use_cache,
                                                  test_split)
                    res = seed_mean_entry(frames, enr_alllayer, args.n_perms, args.rng_seed)
                    null, obs, dots = res["null"], res["obs_pooled"], res["obs_per_seed"]
                else:
                    npz_path = (gonnect_nulls_dir
                                / f"AE_{version}_{module}_auc" / f"perm_layer_{L}_arrays.npz")
                    res = load_npz_null(npz_path, missing_nulls)
                    if res is None:
                        continue
                    null, obs = res
                    dots = None
                    if want_dots:
                        dots = gonnect_layer_dots(
                            activations_dir, version, module, L,
                            layer_maps[(version, module)], enr_alllayer, cache_dir, use_cache)
                # the 109-node latent is enc L3 (paper bottleneck) / dec L5 (dec-side)
                bn = None
                label = f"{MODULE_ABBREV[module]}\nL{L}"
                if module == "encoder" and L == 3:
                    bn = "paper"
                elif module == "decoder" and L == 5:
                    bn = "dec"
                entries.append({
                    "label": label, "color": color,
                    "null": null, "obs_pooled": obs, "obs_per_seed": dots,
                    "divider_before": (module == "decoder" and k == 0),
                    "bottleneck": bn,
                    "untrained": untrained.get((EMB_VERSION_DISPLAY[version], label.replace("\n", " "))),
                })
        return entries

    print(f"\n[1/5] {EMB_VERSION_DISPLAY['2.0']} …")
    row_fl = gonnect_row("2.0", COL_FL)
    print(f"[2/5] {EMB_VERSION_DISPLAY['2.1']} …")
    row_sl = gonnect_row("2.1", COL_SL)

    print("[3/5] OntoVAE …")
    row_onto = []
    for L in range(0, 11):
        if seeds_mode:
            res = seed_mean_entry(ontovae_frames(ontovae_cfg, L, cancer_type_per_row, cache_dir, use_cache,
                                                 test_split),
                                  enr_ontovae, args.n_perms, args.rng_seed)
            null, obs, dots = res["null"], res["obs_pooled"], res["obs_per_seed"]
        else:
            npz_path = (args.data_dir / "perm_nulls_baselines"
                        / f"OntoVAE_layer_{L:02d}_true_auc" / "perm_nulls_arrays.npz")
            res = load_npz_null(npz_path, missing_nulls)
            if res is None:
                continue
            null, obs = res
            dots = (ontovae_dots(ontovae_cfg, L, enr_ontovae, cancer_type_per_row, cache_dir, use_cache)
                    if want_dots else None)
        row_onto.append({"label": f"L{L}", "color": COL_ONTO,
                         "null": null, "obs_pooled": obs, "obs_per_seed": dots,
                         "untrained": untrained.get(("OntoVAE", f"L{L}"))})

    print("[4/5] VEGA baselines (true + randomized controls) …")
    row_vega = []
    for db, color, enr in [("hallmark", COL_VEGA_HM, enr_hallmark),
                           ("reactomes", COL_VEGA_RE, enr_reactomes)]:
        abbr = VEGA_DB_DISPLAY[db]
        for vi, variant in enumerate(["true", "degree_preserving", "random"]):
            if seeds_mode:
                res = seed_mean_entry(vega_frames(method_dbs, db, variant, cancer_type_per_row, cache_dir,
                                                  use_cache, test_split), enr, args.n_perms, args.rng_seed)
                null, obs, dots = res["null"], res["obs_pooled"], res["obs_per_seed"]
            else:
                npz_path = (args.data_dir / "perm_nulls_baselines"
                            / f"VEGA_{db}_{variant}_auc" / "perm_nulls_arrays.npz")
                res = load_npz_null(npz_path, missing_nulls)
                if res is None:
                    continue
                null, obs = res
                dots = (vega_dots(method_dbs, db, variant, enr, cancer_type_per_row, cache_dir, use_cache)
                        if want_dots else None)
            # randomized variants are coloured grey (control); true keeps the db colour
            vcolor = color if variant == "true" else COL_RAND
            label = f"{abbr}\n{RANDOMIZATION_DISPLAY[variant]}"
            row_vega.append({"label": label, "color": vcolor,
                             "null": null, "obs_pooled": obs, "obs_per_seed": dots,
                             "divider_before": (db == "reactomes" and vi == 0),
                             "untrained": untrained.get(("VEGA", label.replace("\n", " ")))})

    print(f"[5/5] randomized GONNECT ({EMB_VERSION_DISPLAY['2.2']}) …")
    row_rand = []
    for module in ["encoder", "decoder", "both"]:
        r = rand_entry(emb_dir, meta, module, bottleneck_columns,
                       enr_bottleneck, args.n_perms, args.rng_seed, pool=args.pool,
                       test_split=test_split)
        if not want_dots:
            r["obs_per_seed"] = None
        row_rand.append({"label": MODULE_ABBREV[module], "color": COL_RAND, **r,
                         "untrained": untrained.get((EMB_VERSION_DISPLAY["2.2"], MODULE_ABBREV[module]))})

    # ── plot ──────────────────────────────────────────────────────────────
    # Horizontal: one violin is violin_w inches wide in every panel, so each
    # axis is exactly n * violin_w wide. Both rows run from MARGIN_L_IN to the
    # same right edge: the top row because it is the one that fixes the violin
    # width, the bottom row by widening its single gap.
    top_counts = [len(row_fl), len(row_sl), len(row_rand)]
    bot_counts = [len(row_onto), len(row_vega)]
    violin_w = violin_width([top_counts, bot_counts])
    top = row_spans(top_counts, violin_w)
    bot = row_spans(bot_counts, violin_w)

    # shared y-range + a single significance-marker height for all panels
    allv = []
    for e in row_fl + row_sl + row_rand + row_onto + row_vega:
        allv += list(e["null"]) + [e["obs_pooled"]]
        if e.get("obs_per_seed") is not None:
            allv += [v for v in e["obs_per_seed"] if np.isfinite(v)]
        if e.get("untrained") is not None:
            allv += np.percentile(e["untrained"], [2.5, 97.5]).tolist()   # the whisker ends
    ymin, ymax = float(np.min(allv)), float(np.max(allv))
    span = ymax - ymin
    sig_top = ymax + 0.03 * span

    # --n-perms only drives panel c; panels a, b, d and e always show the
    # 1000-permutation nulls that came out of the precomputed npz files, so
    # the caption only claims a single perm count when the two agree.
    if args.n_perms == 1000:
        note_lines = ["1000× column-shuffle test;"]
    else:
        note_lines = ["column-shuffle test (1000×,", f"panel c {args.n_perms}×);"]
    note_lines += ["statistic = median per-", "cancer-type Spearman r"]

    # Vertical: bands stacked from the top, in inches, so the figure is exactly
    # as tall as its contents need.
    untrained_n = len(next(iter(untrained.values()))) if untrained else None
    legend_rows = max(3 + (1 if untrained_n else 0), len(note_lines))
    legend_h = (LEGEND_PAD_IN + (1 + legend_rows) * LEGEND_ROW_IN
                + LEGEND_PAD_IN)
    fig_h = (PAD_TOP_IN + TITLE_BAND_IN + AXES_H_IN + XTICK_TOP_IN + ROW_GAP_IN
             + TITLE_BAND_IN + AXES_H_IN + XTICK_BOT_IN
             + LEGEND_GAP_IN + legend_h + PAD_BOT_IN)
    row0_top = PAD_TOP_IN + TITLE_BAND_IN
    row1_top = row0_top + AXES_H_IN + XTICK_TOP_IN + ROW_GAP_IN + TITLE_BAND_IN
    legend_top = row1_top + AXES_H_IN + XTICK_BOT_IN + LEGEND_GAP_IN

    fig = plt.figure(figsize=figsize(fig_h))

    def rect(x_span: tuple[float, float], top_in: float, h_in: float) -> list[float]:
        """Figure-fraction rect from an (x0, width) span and a distance down
        from the top of the canvas, both in inches."""
        x0, w = x_span
        return [x0 / FIG_WIDTH_IN, 1.0 - (top_in + h_in) / fig_h,
                w / FIG_WIDTH_IN, h_in / fig_h]

    ax_a = fig.add_axes(rect(top[0], row0_top, AXES_H_IN))
    ax_b = fig.add_axes(rect(top[1], row0_top, AXES_H_IN), sharey=ax_a)
    ax_c = fig.add_axes(rect(top[2], row0_top, AXES_H_IN), sharey=ax_a)
    ax_d = fig.add_axes(rect(bot[0], row1_top, AXES_H_IN), sharey=ax_a)
    ax_e = fig.add_axes(rect(bot[1], row1_top, AXES_H_IN), sharey=ax_a)

    draw_group(ax_a, row_fl,   "a", EMB_VERSION_DISPLAY["2.0"], want_dots, True,  sig_top)
    draw_group(ax_b, row_sl,   "b", EMB_VERSION_DISPLAY["2.1"], want_dots, False, sig_top)
    draw_group(ax_c, row_rand, "c", EMB_VERSION_DISPLAY["2.2"], want_dots, False, sig_top)
    draw_group(ax_d, row_onto, "d", "OntoVAE",    want_dots, True,  sig_top)
    draw_group(ax_e, row_vega, "e", "VEGA",       want_dots, False, sig_top,
               label_rotation=90.0)

    ax_a.set_ylim(ymin - 0.05 * span, sig_top + sig_headroom(sig_top, ymin, span))
    for ax in (ax_b, ax_c, ax_e):
        ax.tick_params(labelleft=False)

    # structured legend, in the band under both rows of panels
    band_w = FIG_WIDTH_IN - MARGIN_L_IN - LEGEND_EDGE_IN
    ax_leg = fig.add_axes(rect((MARGIN_L_IN, band_w), legend_top, legend_h))
    ax_leg.axis("off")
    ax_leg.set_xlim(0.0, band_w)      # data coordinates are inches, y downwards
    ax_leg.set_ylim(legend_h, 0.0)
    draw_legend(ax_leg, note_lines, untrained_n,
                red_line_label="mean over seeds" if seeds_mode else "pooled observed")

    report_text_overlaps(fig, "fig4")

    # PNG for quick viewing, PDF (vector) for the manuscript.
    save_figure(fig, args.out_dir, "fig4")
    plt.close(fig)

    # CSV dump — the numbers behind every violin
    rows_csv = []
    for row_name, row in [(EMB_VERSION_DISPLAY["2.0"], row_fl),
                          (EMB_VERSION_DISPLAY["2.1"], row_sl),
                          (EMB_VERSION_DISPLAY["2.2"], row_rand),
                          ("OntoVAE", row_onto), ("VEGA", row_vega)]:
        for e in row:
            p = perm_pval(e["obs_pooled"], e["null"])
            # Named for what the red line is; the published figure's column keeps its name
            rec = {"row": row_name, "label": e["label"].replace("\n", " "),
                   ("obs_mean_over_seeds" if seeds_mode else "obs_pooled"): e["obs_pooled"],
                   "perm_p": p,
                   "null_mean": float(np.mean(e["null"])),
                   "null_std": float(np.std(e["null"]))}
            if seeds_mode:
                rec["stars"] = sig_marker(p)
            if e.get("obs_per_seed") is not None:
                for s_i, v in enumerate(e["obs_per_seed"]):
                    rec[f"seed_{s_i}"] = float(v)
            if e.get("untrained") is not None:
                rec.update(zip(["untrained_q025", "untrained_median", "untrained_q975"],
                               np.percentile(e["untrained"], [2.5, 50, 97.5]).tolist()))
                rec["untrained_n"] = len(e["untrained"])
            rows_csv.append(rec)
    csv_path = args.out_dir / "fig4.csv"
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    # A dropped null silently removes a violin, so say so loudly.
    n_violins = len(row_fl) + len(row_sl) + len(row_rand) + len(row_onto) + len(row_vega)
    print(f"\n{n_violins} violins drawn")
    if missing_nulls:
        print(f"WARNING: {len(missing_nulls)} violin(s) omitted, missing:")
        for path in missing_nulls:
            print(f"  {path}")


if __name__ == "__main__":
    main()
