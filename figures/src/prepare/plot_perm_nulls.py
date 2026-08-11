"""
Permutation null comparison: GONNECT activation vs GSEA enrichment.

Runs three null distributions on the activation matrix, with the GSEA enrichment
matrix held fixed:

  1. Row shuffle      shuffle cancer-type labels of the activation matrix.
                      Tests: does the cancer-type pairing carry signal?
  2. Column shuffle   shuffle GO-term labels of the activation matrix.
                      Tests: does the GO-term identity matter? (the GONNECT claim)
  3. Row + column     shuffle both independently.
                      Tests: does any (cancer type, GO term) alignment survive
                      after row and column distributions are shuffled?

By default runs against AE_2.1 (soft-link GONNECT). Pass --version 2.2 with
--seeds 22 23 24 25 26 to run against the randomized GONNECT (AE_2.2).

Usage:
  pixi run python fig_deg/perm_nulls/plot_perm_nulls.py
  pixi run python fig_deg/perm_nulls/plot_perm_nulls.py \\
      --version 2.2 --seeds 22 23 24 25 26 \\
      --output-dir fig_deg/perm_nulls_randomized \\
      --label "AE_2.2 (randomized GONNECT)"
"""

import argparse
import gzip
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

# Reuse loaders from the main script
from plot_activation_vs_gsea import (  # noqa: E402

    CANCER_TYPE_ORDER,
    load_bottleneck_index,
    load_mean_activations,
)

from _paths import EMBEDDINGS_DIR


# ── Alternative scoring: per-(cancer_type, GO_node) ROC-AUC ───────────────────

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
        path = emb_dir / f"AE_{version}" / f"AE_{version}.{s}_{module}_full_dataset.pt"
        emb = torch.load(path, map_location="cpu", weights_only=True).numpy()
        seed_embs.append(emb[mask])
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


# ── Statistics ────────────────────────────────────────────────────────────────

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
    Pass ``desc`` to enable a tqdm bar.
    """
    from tqdm.auto import tqdm

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

    iterator = range(n_perms)
    if desc is not None:
        iterator = tqdm(iterator, desc=desc, leave=False, ncols=80)

    for k in iterator:
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


def perm_pval(obs: float, null_arr: np.ndarray) -> float:
    return float((np.sum(null_arr >= obs) + 1) / (len(null_arr) + 1))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_combined(
    nulls_by_mode: dict,
    observed: dict,
    n_perms: int,
    output_path: Path,
    label: str = "GONNECT",
    metric_label: str = "activation",
) -> None:
    modes = [
        ("row",    "Row shuffle\n(cancer-type labels)"),
        ("col",    "Column shuffle\n(GO-term labels)"),
        ("rowcol", "Row + column\n(both independently)"),
    ]
    metrics = [
        ("global",   "Global Spearman r",            "r (all term × ct pairs)"),
        ("per_term", "Median per-term r",            "median r across GO terms"),
        ("per_ct",   "Median per-cancer-type r",     "median r across cancer types"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13, 11),
                             gridspec_kw={"wspace": 0.35, "hspace": 0.55})

    # Shared x-limits per metric column for direct comparison across nulls
    xlims = {}
    for mkey, _, _ in metrics:
        all_vals = np.concatenate([nulls_by_mode[m][mkey] for m, _ in modes])
        all_vals = np.append(all_vals, observed[mkey])
        pad = 0.05 * (all_vals.max() - all_vals.min() + 1e-9)
        xlims[mkey] = (all_vals.min() - pad, all_vals.max() + pad)

    for row, (mode_key, mode_title) in enumerate(modes):
        nulls = nulls_by_mode[mode_key]
        for col, (mkey, title, xlabel) in enumerate(metrics):
            ax = axes[row, col]
            null_arr = nulls[mkey]
            obs = observed[mkey]
            p = perm_pval(obs, null_arr)

            ax.hist(null_arr, bins=40, color="#aaaaaa", edgecolor="white",
                    linewidth=0.3, label=f"Null (n={n_perms})")
            ax.axvline(obs, color="#d62728", linewidth=2.0,
                       label=f"Observed = {obs:.3f}\np = {p:.3g}")
            ax.set_xlim(xlims[mkey])
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            if row == 0:
                ax.set_title(title, fontsize=9, pad=6)
            if col == 0:
                ax.text(-0.32, 0.5, mode_title, transform=ax.transAxes,
                        fontsize=10, ha="right", va="center", fontweight="bold")
            ax.legend(fontsize=7, frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Permutation null comparison: {label} {metric_label} vs GSEA -log10(NOM p-val)\n"
        "rows = null model, columns = test statistic",
        fontsize=11, y=0.995,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved combined permutation comparison to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=Path, default=Path("data"))
    parser.add_argument("--emb-dir",    type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--gsea-csv",   type=Path,
                        default=Path("fig_deg/gsea_bottleneck/gsea_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("fig_deg/perm_nulls"))
    parser.add_argument("--version",    type=str, default="2.1",
                        help="Model version, e.g. '2.1' (soft-link) or '2.2' (randomized).")
    parser.add_argument("--seeds",      type=int, nargs="+", default=[2, 3, 4, 5, 6],
                        help="Seeds for the chosen version. AE_2.1 uses 2-6; AE_2.2 uses 22-26.")
    parser.add_argument("--module",     type=str, default="encoder",
                        choices=["encoder", "decoder", "both"],
                        help="Which BI-module variant of the model to load.")
    parser.add_argument("--label",      type=str, default="GONNECT (AE_2.1)",
                        help="Display label used in figure title.")
    parser.add_argument("--metric",     type=str, default="mean_abs",
                        choices=["mean_abs", "auc"],
                        help="Per-(cancer_type, GO_node) score: mean |activation| "
                             "or one-vs-rest ROC-AUC (max(AUC, 1-AUC), like the "
                             "node-specificity figure in the paper).")
    parser.add_argument("--n-perms",    type=int, default=1000)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading bottleneck index …")
    bottleneck = load_bottleneck_index(args.data_dir / "hard_links.csv")

    print("Loading TCGA metadata …")
    with gzip.open(args.data_dir / "TCGA_complete_bp_top1k.csv.gz", "rt") as fh:
        meta = pd.read_csv(fh, usecols=["patient_id", "cancer_type", "sample_type"])

    if args.metric == "auc":
        print(f"Loading embeddings and computing per-(cancer_type, node) AUC "
              f"(AE_{args.version}, module={args.module}, seeds {args.seeds}) …")
        act_matrix, cancer_types = load_mean_auc(
            args.emb_dir, meta, args.version, args.module, args.seeds,
            sample_type="Primary Tumor",
        )
        metric_label = "AUC (one-vs-rest)"
    else:
        print(f"Loading embeddings and computing mean |activation| "
              f"(AE_{args.version}, module={args.module}, seeds {args.seeds}) …")
        act_matrix, cancer_types = load_mean_activations(
            args.emb_dir, meta, args.version, args.module, args.seeds,
            sample_type="Primary Tumor",
        )
        metric_label = "mean |activation|"
    act_df = pd.DataFrame(
        act_matrix,
        index=cancer_types,
        columns=bottleneck["go_term_id"].tolist(),
    )

    print("Loading GSEA results …")
    gsea = pd.read_csv(args.gsea_csv)
    gsea["enr_score"] = -np.log10(np.clip(gsea["NOM p-val"], 1e-3, 1.0))
    enr_pivot = gsea.pivot_table(index="cancer_type", columns="Term",
                                 values="enr_score", aggfunc="first")
    cancer_types = [ct for ct in cancer_types if ct in enr_pivot.index]
    enr_pivot = enr_pivot.reindex(index=cancer_types)

    overlap_terms = sorted(set(bottleneck["go_term_id"]) & set(enr_pivot.columns))
    print(f"  {len(cancer_types)} cancer types × {len(overlap_terms)} overlapping GO terms")

    nes_df = enr_pivot[overlap_terms].reindex(index=cancer_types)
    act_df_ov = act_df[overlap_terms].reindex(index=cancer_types)

    act = act_df_ov.values.astype(float)
    enr = nes_df.values.astype(float)

    # ── Observed ───────────────────────────────────────────────────────────
    obs_global, obs_term, obs_ct = _summary_stats(act, enr)
    observed = {"global": obs_global, "per_term": obs_term, "per_ct": obs_ct}
    print("\nObserved Spearman r:")
    print(f"  global          = {obs_global:.3f}")
    print(f"  median per-term = {obs_term:.3f}")
    print(f"  median per-ct   = {obs_ct:.3f}")

    # ── Three nulls ────────────────────────────────────────────────────────
    nulls_by_mode = {}
    for mode in ["row", "col", "rowcol"]:
        print(f"\nRunning {mode} null ({args.n_perms} perms) …")
        nulls_by_mode[mode] = run_null(act, enr, mode=mode,
                                       n_perms=args.n_perms, seed=args.seed)
        for mkey in ["global", "per_term", "per_ct"]:
            p = perm_pval(observed[mkey], nulls_by_mode[mode][mkey])
            print(f"  {mkey:>10}: p = {p:.3g}")

    # ── Plot & summary CSV ─────────────────────────────────────────────────
    plot_combined(
        nulls_by_mode, observed, args.n_perms,
        args.output_dir / "perm_nulls_combined.png",
        label=args.label,
        metric_label=metric_label,
    )

    rows = []
    for mode in ["row", "col", "rowcol"]:
        for mkey in ["global", "per_term", "per_ct"]:
            null_arr = nulls_by_mode[mode][mkey]
            rows.append({
                "null_mode": mode,
                "metric":    mkey,
                "observed":  observed[mkey],
                "null_mean": float(null_arr.mean()),
                "null_std":  float(null_arr.std()),
                "p_value":   perm_pval(observed[mkey], null_arr),
                "n_perms":   args.n_perms,
            })
    summary_path = args.output_dir / "perm_nulls_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\n  Saved summary CSV to {summary_path}")

    # Persist the full null arrays so downstream figures (e.g. violins) can
    # show the actual null shape, not a Gaussian approximation from mean/std.
    arrays_path = args.output_dir / "perm_nulls_arrays.npz"
    np.savez(
        arrays_path,
        metric_label=np.array(metric_label),
        **{f"{mode}__{metric}": nulls_by_mode[mode][metric]
           for mode in ("row", "col", "rowcol")
           for metric in ("global", "per_term", "per_ct")},
        **{f"observed__{metric}": np.float64(observed[metric])
           for metric in ("global", "per_term", "per_ct")},
    )
    print(f"  Saved null arrays to {arrays_path}")


if __name__ == "__main__":
    main()
