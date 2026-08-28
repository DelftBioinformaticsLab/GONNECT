"""
Compute column-shuffle null distributions for baseline models (VEGA, OntoVAE)
so they can be plotted alongside GONNECT in the violin figure.

For each combination of (method, db, variant) × (metric in {mean_abs, auc}):
  - Build the (n_cancer_types, n_pathways) score matrix using the existing
    baseline loaders.
    - mean_abs: per-seed mean |activation| per CT, averaged across seeds.
    - auc:      per-seed one-vs-rest ROC-AUC per (CT, pathway), symmetrized
                with max(AUC, 1-AUC), then averaged across seeds.
  - Load the matching GSEA result, restrict to the pathway intersection.
  - Run the row / col / rowcol nulls and save perm_nulls_arrays.npz in the
    same format produced by fig_deg/perm_nulls/plot_perm_nulls.py.

We use seed-wise AUC then average (rather than averaging activations then
computing AUC, as the GONNECT script does) because each baseline seed has a
different test split — there is no shared sample set to average over.

Usage:
  pixi run python fig_deg/perm_nulls_violin/compute_baseline_nulls.py
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# Reuse loaders and the vectorised null from existing modules.
THIS = Path(__file__).resolve()

from plot_activation_vs_gsea_baselines import (  # noqa: E402
    CANCER_TYPE_ORDER,
    SEEDS,
    LOADERS,
    load_enrichment_pivot,
)
from plot_perm_nulls import run_null, _summary_stats  # noqa: E402

from _paths import ONTOVAE_ACTIVATIONS_DIR, PREP_OUT_DIR, TCGA_CSV, VEGA_ACTIVATIONS_DIR


# ── Baseline configs ──────────────────────────────────────────────────────────

VEGA_HALLMARK_CFG = {
    "loader": "vega_csv",
    "dir": VEGA_ACTIVATIONS_DIR,
    "file_template": "z_test_{variant}_hallmark_v2026_1_Hs_uniprot.csv",
    "gsea_csv": Path("fig_deg/gsea_baselines/gsea_results_hallmark.csv"),
}
VEGA_REACTOMES_CFG = {
    "loader": "vega_csv",
    "dir": VEGA_ACTIVATIONS_DIR,
    "file_template": "z_test_{variant}_reactomes_uniprot.csv",
    "gsea_csv": Path("fig_deg/gsea_baselines/gsea_results_reactomes.csv"),
}
# OntoVAE: one layer, parametrised below.
ONTOVAE_BASE_CFG = {
    "loader": "ontovae_layer",
    "dir": ONTOVAE_ACTIVATIONS_DIR,
    "file_template": "pathway_activities_test_{variant}.parquet",
    "vega_sample_dir": VEGA_ACTIVATIONS_DIR,
    "vega_sample_template": "z_test_{variant}_hallmark_v2026_1_Hs_uniprot.csv",
    "gsea_csv": Path("fig_deg/gsea_baselines/gsea_results_ontovae.csv"),
}

VARIANTS = ["true", "degree_preserving", "random"]


def make_combos(
    ontovae_layers: list[int],
    *,
    include_vega: bool = True,
) -> list[tuple[str, str, dict]]:
    """Each entry: (method, db, cfg)."""
    combos: list[tuple[str, str, dict]] = []
    if include_vega:
        combos += [
            ("VEGA", "hallmark",  VEGA_HALLMARK_CFG),
            ("VEGA", "reactomes", VEGA_REACTOMES_CFG),
        ]
    for L in ontovae_layers:
        combos.append(
            ("OntoVAE", f"layer_{L:02d}", {**ONTOVAE_BASE_CFG, "layer": L})
        )
    return combos


# ── TCGA index ────────────────────────────────────────────────────────────────

def load_cancer_type_per_row(tcga_path: Path) -> dict[int, str]:
    """{positional_row_index: cancer_type} — matches the indexing used by
    the baseline z_test_*.csv files (first column = positional TCGA row index)."""
    with gzip.open(tcga_path, "rt") as fh:
        meta = pd.read_csv(fh, usecols=["cancer_type"])
    return {i: ct for i, ct in enumerate(meta["cancer_type"].tolist())}


# ── Score-matrix builders ─────────────────────────────────────────────────────

def build_mean_abs_matrix(
    cfg: dict,
    variant: str,
    cancer_type_per_row: dict[int, str],
) -> tuple[pd.DataFrame, list[str]]:
    """Per-CT mean |activation|, averaged across seeds (over CTs in every seed)."""
    loader = LOADERS[cfg["loader"]]
    pathways_ref = None
    per_seed: list[dict[str, np.ndarray]] = []

    for seed in SEEDS:
        path = cfg["dir"] / f"run_seed-{seed}" / cfg["file_template"].format(variant=variant)
        acts, row_idx, pathways = loader(path, seed=seed, cfg=cfg, variant=variant)
        if pathways_ref is None:
            pathways_ref = pathways
        elif pathways != pathways_ref:
            raise ValueError(f"Pathway column mismatch at seed {seed}")
        ct_arr = np.array([cancer_type_per_row.get(int(i)) for i in row_idx])
        abs_a = np.abs(acts)
        ct_means: dict[str, np.ndarray] = {}
        for ct in pd.unique(ct_arr):
            if ct is None:
                continue
            m = ct_arr == ct
            if m.any():
                ct_means[ct] = abs_a[m].mean(axis=0)
        per_seed.append(ct_means)

    common = set.intersection(*[set(d.keys()) for d in per_seed])
    ordered = [ct for ct in CANCER_TYPE_ORDER if ct in common]
    ordered += sorted(common - set(CANCER_TYPE_ORDER))

    mat = np.zeros((len(ordered), len(pathways_ref)))
    for i, ct in enumerate(ordered):
        mat[i] = np.stack([d[ct] for d in per_seed]).mean(axis=0)
    return pd.DataFrame(mat, index=ordered, columns=pathways_ref), ordered


def build_auc_matrix(
    cfg: dict,
    variant: str,
    cancer_type_per_row: dict[int, str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Per-seed AUC per (cancer_type, pathway), max(AUC, 1-AUC), averaged across
    seeds (over CTs that have ≥ 1 positive in every seed).
    """
    loader = LOADERS[cfg["loader"]]
    pathways_ref = None
    per_seed_auc: list[dict[str, np.ndarray]] = []

    for seed in SEEDS:
        path = cfg["dir"] / f"run_seed-{seed}" / cfg["file_template"].format(variant=variant)
        acts, row_idx, pathways = loader(path, seed=seed, cfg=cfg, variant=variant)
        if pathways_ref is None:
            pathways_ref = pathways
        elif pathways != pathways_ref:
            raise ValueError(f"Pathway column mismatch at seed {seed}")

        ct_arr = np.array([cancer_type_per_row.get(int(i)) for i in row_idx])
        n_pathways = acts.shape[1]
        ct_auc: dict[str, np.ndarray] = {}
        for ct in pd.unique(ct_arr):
            if ct is None:
                continue
            y = (ct_arr == ct).astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            aucs = np.empty(n_pathways)
            for j in range(n_pathways):
                a = roc_auc_score(y, acts[:, j])
                aucs[j] = max(a, 1.0 - a)
            ct_auc[ct] = aucs
        per_seed_auc.append(ct_auc)

    common = set.intersection(*[set(d.keys()) for d in per_seed_auc])
    ordered = [ct for ct in CANCER_TYPE_ORDER if ct in common]
    ordered += sorted(common - set(CANCER_TYPE_ORDER))

    mat = np.zeros((len(ordered), len(pathways_ref)))
    for i, ct in enumerate(ordered):
        mat[i] = np.stack([d[ct] for d in per_seed_auc]).mean(axis=0)
    return pd.DataFrame(mat, index=ordered, columns=pathways_ref), ordered


# ── Single combo runner ───────────────────────────────────────────────────────

def run_one(
    method: str,
    db: str,
    variant: str,
    cfg: dict,
    metric: str,
    cancer_type_per_row: dict[int, str],
    output_root: Path,
    n_perms: int,
    rng_seed: int,
    null_modes: tuple[str, ...] = ("row", "col", "rowcol"),
) -> None:
    print(f"\n=== {method} / {db} / {variant} / metric={metric} ===")
    builder = build_mean_abs_matrix if metric == "mean_abs" else build_auc_matrix
    act_df, cancer_types = builder(cfg, variant, cancer_type_per_row)
    print(f"  {len(cancer_types)} cancer types × {act_df.shape[1]} pathways")

    enr_pivot = load_enrichment_pivot(cfg["gsea_csv"], cancer_types)
    overlap = sorted(set(act_df.columns) & set(enr_pivot.columns))
    print(f"  {len(overlap)} pathways in both activation matrix and GSEA results")
    if not overlap:
        print("  No overlap — skipping.")
        return

    cts = [ct for ct in cancer_types if ct in enr_pivot.index]
    act = act_df.loc[cts, overlap].to_numpy(dtype=np.float64)
    enr = enr_pivot.loc[cts, overlap].to_numpy(dtype=np.float64)

    obs_g, obs_t, obs_c = _summary_stats(act, enr)
    observed = {"global": obs_g, "per_term": obs_t, "per_ct": obs_c}
    print(f"  Observed:  global={obs_g:+.3f}  per-term={obs_t:+.3f}  per-ct={obs_c:+.3f}")

    nulls = {}
    for mode in null_modes:
        nulls[mode] = run_null(
            act, enr, mode=mode, n_perms=n_perms, seed=rng_seed,
            desc=f"  {mode:>6} null ({method}/{db}/{variant}/{metric})",
        )

    metric_label = "AUC (one-vs-rest)" if metric == "auc" else "mean |activation|"
    out_dir = output_root / f"{method}_{db}_{variant}_{metric}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "perm_nulls_arrays.npz",
        metric_label=np.array(metric_label),
        **{f"{mode}__{m}": nulls[mode][m]
           for mode in null_modes
           for m in ("global", "per_term", "per_ct")},
        **{f"observed__{m}": np.float64(observed[m])
           for m in ("global", "per_term", "per_ct")},
    )
    print(f"  Saved {out_dir / 'perm_nulls_arrays.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcga-csv",       type=Path,
                        default=TCGA_CSV)
    parser.add_argument("--output-root",    type=Path,
                        default=PREP_OUT_DIR / "perm_nulls_baselines")
    parser.add_argument("--ontovae-layers", type=int, nargs="+", default=[0],
                        help="OntoVAE layers to include. Use 0..12 for all.")
    parser.add_argument("--no-vega", action="store_true",
                        help="Skip VEGA combos (e.g. when only re-running OntoVAE).")
    parser.add_argument("--n-perms",        type=int, default=1000)
    parser.add_argument("--rng-seed",       type=int, default=42)
    parser.add_argument("--variants",       type=str, nargs="+", default=VARIANTS)
    parser.add_argument("--metrics",        type=str, nargs="+", default=["mean_abs", "auc"])
    parser.add_argument("--null-modes",     type=str, nargs="+", default=["row", "col", "rowcol"],
                        choices=["row", "col", "rowcol"])
    args = parser.parse_args()

    print("Loading TCGA cancer-type-per-row mapping …")
    cancer_type_per_row = load_cancer_type_per_row(args.tcga_csv)

    combos = make_combos(args.ontovae_layers, include_vega=not args.no_vega)
    for method, db, cfg in combos:
        for variant in args.variants:
            for metric in args.metrics:
                run_one(
                    method, db, variant, cfg, metric,
                    cancer_type_per_row,
                    args.output_root, args.n_perms, args.rng_seed,
                    null_modes=tuple(args.null_modes),
                )


if __name__ == "__main__":
    main()
