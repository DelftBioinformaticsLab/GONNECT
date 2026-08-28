"""
Repeat the GONNECT activation-vs-enrichment correlation experiment for the
VEGA / OntoVAE baselines (data/baseline_activations/).

For each (method, gene-set DB, ontology variant) combination:
  - Load model pathway activations across 5 seeds, take |abs|, average
  - Map test rows → TCGA cancer type via positional index
  - Compute mean |activation| per (cancer_type, pathway)
  - Correlate with -log10(NOM p-val) from a precomputed GSEA run
  - Save per-pathway / per-cancer-type / global correlations + a permutation null

Reuses correlation/plotting helpers from plot_activation_vs_gsea.py.

OntoVAE branch is currently a stub: needs a pathway → UniProt-gene-set GMT
from the baseline runs before its GSEA precomputation can run.

Usage:
  pixi run python fig_deg/plot_activation_vs_gsea_baselines.py             # run all VEGA combos
  pixi run python fig_deg/plot_activation_vs_gsea_baselines.py \\
      --combos VEGA:hallmark:true                                           # one combo only
"""

import argparse
import gzip
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

# Reuse the GONNECT correlation/plotting helpers — they're parameterised by
# id-string + name-lookup dict, so they work for VEGA pathways too.
from plot_activation_vs_gsea import (  # noqa: E402

    compute_correlations,
    compute_per_cancer_type_correlations,
    plot_correlation_barplot,
    plot_per_cancer_type_barplot,
    plot_overall_scatter,
    plot_side_by_side_clustermaps,
    plot_permutation_test,
    run_permutation_null,
)

from _paths import ONTOVAE_ACTIVATIONS_DIR, VEGA_ACTIVATIONS_DIR

CANCER_TYPE_ORDER = [
    "BRCA", "LUAD", "UCEC", "LGG",  "KIRC", "HNSC", "THCA", "PRAD",
    "LUSC", "SKCM", "COAD", "OV",   "STAD", "BLCA", "LIHC", "CESC",
    "KIRP", "SARC", "ESCA", "PCPG", "PAAD", "READ", "TGCT", "LAML",
    "THYM", "MESO", "UVM",  "ACC",  "KICH", "UCS",  "DLBC", "CHOL",
]

SEEDS = [2, 3, 4, 5, 6]
VARIANTS = ["true", "degree_preserving", "random"]

# Static method/db combos. OntoVAE is added dynamically (one entry per layer).
METHOD_DBS_STATIC = {
    ("VEGA", "hallmark"): {
        "loader": "vega_csv",
        "dir": VEGA_ACTIVATIONS_DIR,
        "file_template": "z_test_{variant}_hallmark_v2026_1_Hs_uniprot.csv",
        "gsea_csv": Path("fig_deg/gsea_baselines/gsea_results_hallmark.csv"),
    },
    ("VEGA", "reactomes"): {
        "loader": "vega_csv",
        "dir": VEGA_ACTIVATIONS_DIR,
        "file_template": "z_test_{variant}_reactomes_uniprot.csv",
        "gsea_csv": Path("fig_deg/gsea_baselines/gsea_results_reactomes.csv"),
    },
}

# OntoVAE: one combo per layer. The parquet stores (term, layer, neuron) tuples;
# we filter to one layer and average the 3 neurons per term. Sample IDs aren't in
# the parquet, so we borrow them from VEGA's matching seed file (per-seed test-set
# sizes match exactly, so we treat the splits as identical).
ONTOVAE_BASE_CFG = {
    "loader": "ontovae_layer",
    "dir": ONTOVAE_ACTIVATIONS_DIR,
    "file_template": "pathway_activities_test_{variant}.parquet",
    "vega_sample_dir": VEGA_ACTIVATIONS_DIR,
    "vega_sample_template": "z_test_{variant}_hallmark_v2026_1_Hs_uniprot.csv",
    "gsea_csv": Path("fig_deg/gsea_baselines/gsea_results_ontovae.csv"),
}


# ── Loaders ───────────────────────────────────────────────────────────────────

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


# ── Activation aggregation ────────────────────────────────────────────────────

def load_mean_activations(
    cfg: dict,
    variant: str,
    cancer_type_per_row: dict[int, str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    For each seed: take |abs|, group test samples by cancer type, take per-CT
    mean → (n_cancer_types_seed, n_pathways). Then average those per-CT matrices
    across seeds (only over CTs present in all seeds).

    This handles the case where each seed has a different test split, which is
    how the OntoVAE/VEGA baseline runs are structured.

    Returns:
      act_df       : DataFrame indexed by cancer_type, columns = pathways
      cancer_types : ordered list of cancer types present in every seed
    """
    loader = LOADERS[cfg["loader"]]
    pathways_ref = None
    per_seed_per_ct: list[dict[str, np.ndarray]] = []  # one dict per seed: ct → mean activation vector

    for seed in SEEDS:
        path = cfg["dir"] / f"run_seed-{seed}" / cfg["file_template"].format(variant=variant)
        acts, row_idx, pathways = loader(path, seed=seed, cfg=cfg, variant=variant)
        if pathways_ref is None:
            pathways_ref = pathways
        elif pathways != pathways_ref:
            raise ValueError(
                f"Seed {seed} has different pathway columns than seed {SEEDS[0]} for {path}"
            )

        sample_cts = np.array(
            [cancer_type_per_row.get(int(i), None) for i in row_idx],
        )
        abs_acts = np.abs(acts)

        ct_means: dict[str, np.ndarray] = {}
        for ct in pd.unique(sample_cts):
            if ct is None:
                continue
            mask = sample_cts == ct
            if mask.sum() == 0:
                continue
            ct_means[ct] = abs_acts[mask].mean(axis=0)
        per_seed_per_ct.append(ct_means)

    # Cancer types present in EVERY seed (so the seed-average is well-defined)
    common_cts = set.intersection(*[set(d.keys()) for d in per_seed_per_ct])
    present_ordered = [ct for ct in CANCER_TYPE_ORDER if ct in common_cts]
    extra = sorted(common_cts - set(CANCER_TYPE_ORDER))
    cancer_types = present_ordered + extra

    act_per_ct = np.zeros((len(cancer_types), len(pathways_ref)))
    for i, ct in enumerate(cancer_types):
        seed_vecs = np.stack([d[ct] for d in per_seed_per_ct])
        act_per_ct[i] = seed_vecs.mean(axis=0)

    act_df = pd.DataFrame(
        act_per_ct, index=cancer_types, columns=pathways_ref,
    )
    return act_df, cancer_types


# ── Vectorised permutation null ───────────────────────────────────────────────

def _vectorised_corr_along_axis(
    a_rank: np.ndarray,
    b_rank: np.ndarray,
    valid: np.ndarray,
    axis: int,
) -> np.ndarray:
    """
    Pearson correlation of pre-ranked arrays (= Spearman on the original
    arrays) along the requested axis, restricted to cells where `valid` is
    True. NaN cells are treated as zero contribution after re-centring the
    ranks within each remaining valid slice.

    a_rank, b_rank, valid all have the same shape. axis=0 → per-column r,
    axis=1 → per-row r. Returns 1-D array of correlations along the *other*
    axis; entries with <2 valid cells get NaN.
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


def run_permutation_null_fast(
    act_df: pd.DataFrame,
    enr_df: pd.DataFrame,
    overlap_terms: list[str],
    cancer_types: list[str],
    n_perms: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Vectorised replacement for run_permutation_null. Computes the same three
    summary stats per permutation (global / median per-term / median per-CT)
    but pre-ranks each column/row once, then expresses each permutation as
    matrix operations on the precomputed ranks. ~100× faster on OntoVAE-sized
    inputs (~640 terms × 31 cancer types × 1000 perms).

    Approximation note: where act and enr have different NaN patterns within a
    column (or row), the Pearson-on-pre-computed-ranks is a close approximation
    of the true Spearman-on-intersection (which would re-rank within each
    intersection). For our data NaNs are sparse (full pivots in VEGA / OntoVAE
    have <1 % missing cells), so the difference is well below the permutation
    noise floor.
    """
    act = act_df.loc[cancer_types, overlap_terms].values.astype(float)
    enr = enr_df.loc[cancer_types, overlap_terms].values.astype(float)
    n_ct, n_terms = act.shape

    act_valid_full = np.isfinite(act)
    enr_valid_full = np.isfinite(enr)

    # Per-column ranks (one pass) — rankdata with nan_policy='omit' leaves NaNs
    # at NaN positions and ranks the non-NaNs as 1..k_j inside their column.
    act_col_rank = np.column_stack([
        rankdata(np.where(act_valid_full[:, j], act[:, j], np.nan), nan_policy="omit")
        for j in range(n_terms)
    ])
    enr_col_rank = np.column_stack([
        rankdata(np.where(enr_valid_full[:, j], enr[:, j], np.nan), nan_policy="omit")
        for j in range(n_terms)
    ])
    # Per-row ranks (one pass), used for the per-cancer-type statistic.
    act_row_rank = np.row_stack([
        rankdata(np.where(act_valid_full[i, :], act[i, :], np.nan), nan_policy="omit")
        for i in range(n_ct)
    ])
    enr_row_rank = np.row_stack([
        rankdata(np.where(enr_valid_full[i, :], enr[i, :], np.nan), nan_policy="omit")
        for i in range(n_ct)
    ])

    rng = np.random.default_rng(seed)
    null_global   = np.empty(n_perms)
    null_per_term = np.empty(n_perms)
    null_per_ct   = np.empty(n_perms)

    # Pre-rank the original enr flattened-over-valid for the global statistic.
    # Each permutation needs its own re-rank because the valid set changes.
    for k in range(n_perms):
        perm = rng.permutation(n_ct)

        # ── Per-term r (across cancer types, post-perm) ────────────────────
        valid_p = act_valid_full[perm] & enr_valid_full
        r_terms, n_p = _vectorised_corr_along_axis(
            act_col_rank[perm], enr_col_rank, valid_p, axis=0,
        )
        r_terms = r_terms[n_p >= 5]
        null_per_term[k] = np.nanmedian(r_terms) if r_terms.size else np.nan

        # ── Per-cancer-type r (across terms, post-perm) ────────────────────
        r_cts, n_pr = _vectorised_corr_along_axis(
            act_row_rank[perm], enr_row_rank, valid_p, axis=1,
        )
        r_cts = r_cts[n_pr >= 5]
        null_per_ct[k] = np.nanmedian(r_cts) if r_cts.size else np.nan

        # ── Global r (Spearman of all (cell, cell) pairs, post-perm) ──────
        a_perm = act[perm]
        a_flat = a_perm[valid_p]
        e_flat = enr[valid_p]
        if a_flat.size < 5:
            null_global[k] = np.nan
            continue
        a_rank = rankdata(a_flat)
        e_rank = rankdata(e_flat)
        a_rank -= a_rank.mean()
        e_rank -= e_rank.mean()
        denom = np.sqrt((a_rank ** 2).sum() * (e_rank ** 2).sum())
        null_global[k] = (a_rank * e_rank).sum() / denom if denom > 0 else np.nan

    return {
        "global":   null_global[~np.isnan(null_global)],
        "per_term": null_per_term[~np.isnan(null_per_term)],
        "per_ct":   null_per_ct[~np.isnan(null_per_ct)],
    }


# ── OntoVAE layer discovery ───────────────────────────────────────────────────

def discover_ontovae_layers(parquet_path: Path) -> list[int]:
    """Read one parquet and return the sorted layer indices it contains."""
    df = pd.read_parquet(parquet_path, columns=None)
    layers = sorted({c[1] for c in df.columns})
    return layers


def build_method_dbs() -> dict[tuple[str, str], dict]:
    """
    Combine static (VEGA) and dynamic (OntoVAE per-layer) combos into a single
    (method, db) → cfg dict. OntoVAE's "db" slot encodes the layer (e.g. layer_05).
    """
    out = dict(METHOD_DBS_STATIC)
    sample_parquet = (ONTOVAE_BASE_CFG["dir"] / "run_seed-2"
                      / ONTOVAE_BASE_CFG["file_template"].format(variant="true"))
    if sample_parquet.exists():
        for L in discover_ontovae_layers(sample_parquet):
            cfg = dict(ONTOVAE_BASE_CFG)
            cfg["layer"] = L
            out[("OntoVAE", f"layer_{L:02d}")] = cfg
    return out


# ── GSEA loading ──────────────────────────────────────────────────────────────

def load_enrichment_pivot(gsea_csv: Path, cancer_types: list[str]) -> pd.DataFrame:
    """Load GSEA results and pivot to (cancer_type × pathway) -log10(NOM p-val)."""
    gsea = pd.read_csv(gsea_csv)
    gsea["enr_score"] = -np.log10(np.clip(gsea["NOM p-val"], 1e-3, 1.0))
    pivot = gsea.pivot_table(
        index="cancer_type", columns="Term", values="enr_score", aggfunc="first",
    )
    pivot = pivot.reindex(index=[ct for ct in cancer_types if ct in pivot.index])
    return pivot


# ── Per-combination driver ────────────────────────────────────────────────────

def run_combo(
    method: str,
    db: str,
    variant: str,
    cfg: dict,
    cancer_type_per_row: dict[int, str],
    output_root: Path,
    n_perms: int = 1000,
) -> dict:
    """Returns a summary dict for the summary grid."""
    print(f"\n=== {method} / {db} / {variant} ===")
    out_dir = output_root / method / db / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading activations …")
    act_df, cancer_types = load_mean_activations(cfg, variant, cancer_type_per_row)
    print(f"  {len(cancer_types)} cancer types × {act_df.shape[1]} pathways")

    print("Loading GSEA results …")
    enr_pivot = load_enrichment_pivot(cfg["gsea_csv"], cancer_types)
    print(f"  {len(enr_pivot)} cancer types × {enr_pivot.shape[1]} GSEA pathways")

    # Overlap: pathways present in both.
    overlap_terms = sorted(set(act_df.columns) & set(enr_pivot.columns))
    print(f"  {len(overlap_terms)} pathways in both activation matrix and GSEA results")
    if not overlap_terms:
        print("  No overlap — skipping.")
        return {"method": method, "db": db, "variant": variant, "skipped": True}

    cancer_types = [ct for ct in cancer_types if ct in enr_pivot.index]
    act_df_ov = act_df.loc[cancer_types, overlap_terms]
    nes_df    = enr_pivot.loc[cancer_types, overlap_terms]

    pathway_names = {p: p for p in overlap_terms}  # use names as their own labels

    # Per-pathway correlations
    corr_df = compute_correlations(act_df_ov, nes_df, overlap_terms, cancer_types)
    corr_df["go_term_name"] = ""  # column kept for plotter compatibility
    corr_df.to_csv(out_dir / "correlation_results.csv", index=False)
    print(f"  Median per-pathway Spearman r: {corr_df['spearman_r'].median():.3f}  "
          f"(n={len(corr_df)})")

    # Per-cancer-type correlations
    corr_ct = compute_per_cancer_type_correlations(act_df_ov, nes_df, overlap_terms, cancer_types)
    corr_ct.to_csv(out_dir / "correlation_results_per_cancer_type.csv", index=False)
    print(f"  Median per-cancer-type Spearman r: {corr_ct['spearman_r'].median():.3f}  "
          f"(n={len(corr_ct)})")

    # Global Spearman
    act_flat = act_df_ov.values.astype(float).ravel()
    enr_flat = nes_df.values.astype(float).ravel()
    valid = np.isfinite(act_flat) & np.isfinite(enr_flat)
    obs_global, p_global = spearmanr(act_flat[valid], enr_flat[valid])
    print(f"  Global Spearman r: {obs_global:.3f}  p={p_global:.2g}  n={valid.sum()}")

    # Permutation null (vectorised — see run_permutation_null_fast)
    print(f"  Permutation test ({n_perms} perms) …")
    null = run_permutation_null_fast(
        act_df_ov, nes_df, overlap_terms, cancer_types, n_perms=n_perms,
    )
    obs_per_term = corr_df["spearman_r"].median()
    obs_per_ct   = corr_ct["spearman_r"].median()

    p_perm_global = (np.sum(null["global"]   >= obs_global)   + 1) / (n_perms + 1)
    p_perm_term   = (np.sum(null["per_term"] >= obs_per_term) + 1) / (n_perms + 1)
    p_perm_ct     = (np.sum(null["per_ct"]   >= obs_per_ct)   + 1) / (n_perms + 1)

    label = f"{method} / {db} / {variant}"
    # Plots — reuse GONNECT helpers
    plot_correlation_barplot(
        corr_df, pathway_names, out_dir / "correlation_barplot_per_term.png",
        title=f"{label}: per-pathway activation vs GSEA -log10(NOM p-val)\n"
              f"({len(corr_df)} pathways in both)",
    )
    plot_per_cancer_type_barplot(
        corr_ct, out_dir / "correlation_barplot_per_cancer_type.png",
        title=f"Per-cancer-type correlation\n{label}",
        term_label="pathways",
    )
    plot_overall_scatter(
        act_df_ov, nes_df, overlap_terms, cancer_types, corr_df, corr_ct,
        out_dir / "correlation_summary.png",
        title=f"{label}: activation vs GSEA -log10(NOM p-val) summary",
        term_label="pathways",
    )
    plot_permutation_test(null, obs_global, obs_per_term, obs_per_ct, n_perms,
                          out_dir / "permutation_test.png")
    plot_side_by_side_clustermaps(
        act_df_ov, nes_df, overlap_terms, cancer_types, pathway_names,
        out_dir / "activation_vs_nes_heatmaps.png",
        title=f"{label}: activation vs GSEA -log10(NOM p-val)\n"
              f"(shared row/col order from activation clustering)",
    )

    return {
        "method": method, "db": db, "variant": variant,
        "n_pathways": len(overlap_terms),
        "n_cancer_types": len(cancer_types),
        "median_per_term_r": obs_per_term,
        "median_per_ct_r": obs_per_ct,
        "global_r": obs_global,
        "global_p_perm": p_perm_global,
        "per_term_p_perm": p_perm_term,
        "per_ct_p_perm": p_perm_ct,
        "per_pathway_r": corr_df["spearman_r"].values.tolist(),
    }


# ── Summary grid ──────────────────────────────────────────────────────────────

def plot_summary_grid(summary: list[dict], output_path: Path) -> None:
    """One row per (method, db); columns = ontology variants. Each cell shows
    the per-pathway Spearman r distribution and the global r."""
    # Group by (method, db)
    rows = sorted({(s["method"], s["db"]) for s in summary if not s.get("skipped")})
    cols = VARIANTS
    n_rows, n_cols = len(rows), len(cols)
    if n_rows == 0:
        print("No results to plot in summary grid.")
        return

    # sharey="row" lets each (method, db) row autoscale to its own histogram
    # height — VEGA / hallmark has 37 pathways while OntoVAE / layer_00 has 640,
    # so a global y-axis squashes the smaller layers flat.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.6, n_rows * 2.8),
        squeeze=False, sharex=True, sharey="row",
    )

    for i, (method, db) in enumerate(rows):
        for j, variant in enumerate(cols):
            ax = axes[i][j]
            entry = next(
                (s for s in summary
                 if s["method"] == method and s["db"] == db and s["variant"] == variant
                 and not s.get("skipped")),
                None,
            )
            if entry is None:
                ax.set_facecolor("#f6f6f6")
                ax.text(0.5, 0.5, "(missing)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="#888")
                continue

            r_vals = np.array(entry["per_pathway_r"], dtype=float)
            r_vals = r_vals[np.isfinite(r_vals)]
            ax.hist(r_vals, bins=25, color="#4878d0", edgecolor="white", linewidth=0.4)
            med = np.median(r_vals)
            ax.axvline(med, color="#d62728", linewidth=1.5, linestyle="--",
                       label=f"median = {med:.2f}")
            ax.axvline(0, color="black", linewidth=0.8)
            gl = entry.get("global_r")
            if gl is not None and np.isfinite(gl):
                p_lab = (f"\np_perm = {entry['global_p_perm']:.3g}"
                         if entry.get("global_p_perm") is not None else "")
                ax.axvline(gl, color="#2ca02c", linewidth=1.5,
                           label=f"global r = {gl:.2f}{p_lab}")
            ax.set_xlim(-1, 1)
            ax.legend(fontsize=6, frameon=False, loc="upper left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if i == 0:
                ax.set_title(variant, fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("Per-pathway Spearman r", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{method} / {db}\n# pathways", fontsize=8)

    fig.suptitle(
        "Activation-vs-enrichment correlation across ontology variants\n"
        "(per-pathway Spearman r distribution; histogram = per-pathway r, dashed = median, green = global r)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved summary grid to {output_path}")


# ── OntoVAE layer trajectory ──────────────────────────────────────────────────

def plot_ontovae_layer_trajectory(summary: list[dict], output_path: Path) -> None:
    """
    Two panels: (a) median per-pathway Spearman r vs OntoVAE layer, one line per
    ontology variant; (b) global Spearman r vs layer with permutation-significance
    markers. Skipped layers (no testable terms) are dropped.
    """
    rows = [
        s for s in summary
        if s.get("method") == "OntoVAE" and not s.get("skipped")
        and s.get("n_pathways", 0) > 0
    ]
    if not rows:
        print("No OntoVAE rows in summary — skipping layer trajectory plot.")
        return

    by_variant: dict[str, list[dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    for v in by_variant:
        by_variant[v].sort(key=lambda s: int(s["db"].split("_")[1]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.3})

    variant_styles = {
        "true":              {"color": "#2ca02c", "marker": "o"},
        "degree_preserving": {"color": "#ff7f0e", "marker": "s"},
        "random":            {"color": "#7f7f7f", "marker": "^"},
    }

    # Panel (a): median per-pathway r
    ax = axes[0]
    for variant, rs in by_variant.items():
        layers = [int(r["db"].split("_")[1]) for r in rs]
        med = [r["median_per_term_r"] for r in rs]
        sty = variant_styles.get(variant, {"color": "k", "marker": "o"})
        ax.plot(layers, med, "-", linewidth=1.4, label=variant, **sty)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("OntoVAE layer", fontsize=9)
    ax.set_ylabel("Median per-pathway Spearman r", fontsize=9)
    ax.set_title("Per-pathway correlation vs OntoVAE layer", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (b): global r, with significance markers (when p_perm is available)
    ax = axes[1]
    have_global = all(r.get("global_r") is not None for rs in by_variant.values() for r in rs)
    have_pperm  = all(r.get("global_p_perm") is not None for rs in by_variant.values() for r in rs)

    for variant, rs in by_variant.items():
        layers = [int(r["db"].split("_")[1]) for r in rs]
        sty = variant_styles.get(variant, {"color": "k", "marker": "o"})

        if have_global:
            gl = [r["global_r"] for r in rs]
            ax.plot(layers, gl, "-", linewidth=1.4, label=variant, **sty)
            if have_pperm:
                sig = [r["global_p_perm"] < 0.05 for r in rs]
                x_sig = [l for l, s in zip(layers, sig) if s]
                y_sig = [g for g, s in zip(gl, sig) if s]
                x_ns  = [l for l, s in zip(layers, sig) if not s]
                y_ns  = [g for g, s in zip(gl, sig) if not s]
                ax.scatter(x_sig, y_sig, s=45, color=sty["color"], marker=sty["marker"],
                           edgecolor=sty["color"], linewidth=1.2, zorder=3)
                ax.scatter(x_ns, y_ns, s=45, color="white", marker=sty["marker"],
                           edgecolor=sty["color"], linewidth=1.2, zorder=3)
            else:
                ax.scatter(layers, gl, s=45, color=sty["color"], marker=sty["marker"],
                           edgecolor=sty["color"], linewidth=1.2, zorder=3)
        else:
            # Fall back to median per-term r when global wasn't computed
            med = [r["median_per_term_r"] for r in rs]
            ax.plot(layers, med, "-", linewidth=1.4, label=variant, **sty)
            ax.scatter(layers, med, s=45, color=sty["color"], marker=sty["marker"],
                       edgecolor=sty["color"], linewidth=1.2, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("OntoVAE layer", fontsize=9)
    ax.set_ylabel("Global Spearman r" if have_global else "Median per-pathway r", fontsize=9)
    sig_note = ("\n(filled = perm p<0.05, hollow = ns)" if have_pperm
                else "" if have_global
                else "\n(global r unavailable — showing median per-pathway r)")
    ax.set_title(f"{'Global' if have_global else 'Median per-pathway'} correlation vs OntoVAE layer{sig_note}",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "OntoVAE: activation-vs-enrichment correlation across ontology layers",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved layer trajectory to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_combo_filter(s: str | None) -> set | None:
    if not s:
        return None
    out = set()
    for tok in s.split(","):
        parts = tok.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad --combos token {tok!r}; expected method:db:variant")
        out.add(tuple(parts))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("fig_deg/activation_vs_gsea_baselines"))
    parser.add_argument("--combos",     type=str, default=None,
                        help="Comma-separated method:db:variant filters, "
                             "e.g. VEGA:hallmark:true,VEGA:hallmark:random")
    parser.add_argument("--n-perms",    type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combo_filter = parse_combo_filter(args.combos)

    # Load TCGA metadata once
    print("Loading TCGA metadata …")
    with gzip.open(args.data_dir / "TCGA_complete_bp_top1k.csv.gz", "rt") as fh:
        meta = pd.read_csv(fh, usecols=["patient_id", "cancer_type", "sample_type"])
    cancer_type_per_row = dict(enumerate(meta["cancer_type"].tolist()))
    print(f"  {len(meta)} TCGA samples")

    method_dbs = build_method_dbs()
    print(f"  {len(method_dbs)} (method, db) combos: "
          f"{sum(1 for k in method_dbs if k[0] == 'VEGA')} VEGA + "
          f"{sum(1 for k in method_dbs if k[0] == 'OntoVAE')} OntoVAE layer(s)")

    summary = []
    for (method, db), cfg in method_dbs.items():
        for variant in VARIANTS:
            if combo_filter is not None and (method, db, variant) not in combo_filter:
                continue
            try:
                result = run_combo(
                    method, db, variant, cfg, cancer_type_per_row,
                    args.output_dir, n_perms=args.n_perms,
                )
                summary.append(result)
            except FileNotFoundError as e:
                print(f"  Skipping {method}/{db}/{variant}: {e}")
                summary.append({"method": method, "db": db, "variant": variant,
                                "skipped": True, "reason": str(e)})

    # Persist a flat summary table (one row per combo) and the headline grid
    summary_table = pd.DataFrame([
        {k: v for k, v in s.items() if k != "per_pathway_r"}
        for s in summary
    ])
    summary_table.to_csv(args.output_dir / "summary.csv", index=False)
    print(f"\nSaved summary table to {args.output_dir}/summary.csv")

    plot_summary_grid(summary, args.output_dir / "summary_grid.png")
    plot_ontovae_layer_trajectory(summary, args.output_dir / "ontovae_layer_trajectory.png")


if __name__ == "__main__":
    main()
