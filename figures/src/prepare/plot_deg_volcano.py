"""
Differential expression analysis: one cancer type vs. rest (one-vs-rest).

For each cancer type, runs a Wilcoxon rank-sum test per gene against all
other cancer types combined.  Uses z-scored expression (data already
standardized), so fold-change is reported as mean difference in z-scores
(i.e. effect size in SDs).

Outputs:
  One volcano plot per cancer type → <output-dir>/<CANCER_TYPE>_volcano.png
  Summary CSV with all results    → <output-dir>/deg_results.csv

Usage:
  pixi run python fig_deg/plot_deg_volcano.py [--data-dir data] [--output-dir fig_deg/volcano]
"""

import argparse
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ranksums

from _paths import DATA_DIR, PREP_OUT_DIR

# ── Configuration ─────────────────────────────────────────────────────────────
FDR_ALPHA = 0.05
MIN_SAMPLES = 10   # skip cancer types with fewer samples

# Highlight top N genes by significance on each volcano plot
TOP_LABEL_N = 10

# Colours
COL_UP   = "#d62728"   # significantly up in this cancer type
COL_DOWN = "#1f77b4"   # significantly down
COL_NS   = "#aaaaaa"   # not significant
POINT_SIZE = 4
POINT_ALPHA = 0.6

CANCER_TYPE_ORDER = [
    "BRCA", "LUAD", "UCEC", "LGG",  "KIRC", "HNSC", "THCA", "PRAD",
    "LUSC", "SKCM", "COAD", "OV",   "STAD", "BLCA", "LIHC", "CESC",
    "KIRP", "SARC", "ESCA", "PCPG", "PAAD", "READ", "TGCT", "LAML",
    "THYM", "MESO", "UVM",  "ACC",  "KICH", "UCS",  "DLBC", "CHOL",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    order = np.argsort(pvals)
    rank  = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    adjusted = pvals * n / rank
    # enforce monotonicity from the right
    adjusted_sorted = adjusted[order]
    for i in range(n - 2, -1, -1):
        if adjusted_sorted[i] > adjusted_sorted[i + 1]:
            adjusted_sorted[i] = adjusted_sorted[i + 1]
    adjusted[order] = adjusted_sorted
    return np.minimum(adjusted, 1.0)


def deg_one_vs_rest(
    X_group: np.ndarray,
    X_rest: np.ndarray,
) -> pd.DataFrame:
    """
    Run Wilcoxon rank-sum test for each gene column.
    Returns DataFrame with mean_diff, pval, padj columns.
    """
    n_genes = X_group.shape[1]
    pvals     = np.empty(n_genes)
    mean_diff = np.empty(n_genes)

    for j in range(n_genes):
        g = X_group[:, j]
        r = X_rest[:, j]
        mean_diff[j] = g.mean() - r.mean()
        stat, p = ranksums(g, r)
        pvals[j] = p

    padj = _bh_adjust(pvals)
    return pd.DataFrame({"mean_diff": mean_diff, "pval": pvals, "padj": padj})


def plot_volcano(
    stats: pd.DataFrame,
    gene_ids: list,
    cancer_type: str,
    n_group: int,
    n_rest: int,
    output_path: Path,
) -> None:
    sig_up   = (stats["padj"] < FDR_ALPHA) & (stats["mean_diff"] > 0)
    sig_down = (stats["padj"] < FDR_ALPHA) & (stats["mean_diff"] < 0)
    ns       = ~(sig_up | sig_down)

    log_padj = -np.log10(np.clip(stats["padj"], 1e-300, 1.0))

    fig, ax = plt.subplots(figsize=(6, 5))

    # ── scatter points ─────────────────────────────────────────────────────
    for mask, col, label in [
        (ns,       COL_NS,   f"n.s. ({ns.sum()})"),
        (sig_down, COL_DOWN, f"down ({sig_down.sum()})"),
        (sig_up,   COL_UP,   f"up ({sig_up.sum()})"),
    ]:
        ax.scatter(
            stats.loc[mask, "mean_diff"],
            log_padj[mask],
            c=col, s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0,
            label=label, rasterized=True,
        )

    # ── significance line ──────────────────────────────────────────────────
    sig_line = -np.log10(FDR_ALPHA)
    ax.axhline(sig_line, color="black", linewidth=0.6, linestyle="--", alpha=0.5)

    # ── label top significant genes ────────────────────────────────────────
    sig_mask = sig_up | sig_down
    if sig_mask.any():
        sig_df = stats[sig_mask].copy()
        sig_df["gene"] = [gene_ids[i] for i in sig_df.index]
        sig_df["log_padj"] = log_padj[sig_mask]
        top = sig_df.nlargest(TOP_LABEL_N, "log_padj")
        for _, row in top.iterrows():
            ax.annotate(
                row["gene"],
                xy=(row["mean_diff"], row["log_padj"]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=5.5, color="black", ha="left",
            )

    # ── formatting ─────────────────────────────────────────────────────────
    ax.set_xlabel("Mean difference in z-score (this vs. rest)", fontsize=9)
    ax.set_ylabel("-log₁₀(FDR-adjusted p-value)", fontsize=9)
    ax.set_title(
        f"{cancer_type}  vs.  rest\n"
        f"n = {n_group} vs {n_rest}  |  "
        f"{sig_up.sum()} up, {sig_down.sum()} down  (FDR < {FDR_ALPHA})",
        fontsize=9,
    )
    ax.legend(fontsize=7, frameon=False, markerscale=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DEG volcano plots (one-vs-rest per cancer type).")
    parser.add_argument("--data-dir",   type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PREP_OUT_DIR / "deg")
    parser.add_argument("--sample-type", default="Primary Tumor",
                        help="Filter to this sample_type (use 'all' to keep all).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    tcga_path = args.data_dir / "TCGA_complete_bp_top1k.csv.gz"
    print(f"Loading {tcga_path} …")
    with gzip.open(tcga_path, "rt") as fh:
        df = pd.read_csv(fh)

    if args.sample_type != "all":
        df = df[df["sample_type"] == args.sample_type].reset_index(drop=True)
        print(f"  Filtered to '{args.sample_type}': {len(df)} samples")

    meta_cols = {"patient_id", "sample_type", "cancer_type", "tumor_tissue_site", "stage_pathologic_stage"}
    gene_cols = [c for c in df.columns if c not in meta_cols]
    X = df[gene_cols].values.astype(np.float32)
    cancer_labels = df["cancer_type"].values

    print(f"  {len(df)} samples × {len(gene_cols)} genes, {df['cancer_type'].nunique()} cancer types\n")

    # ── Order cancer types ─────────────────────────────────────────────────
    present = set(df["cancer_type"].unique())
    cancer_types = [ct for ct in CANCER_TYPE_ORDER if ct in present] + \
                   sorted(present - set(CANCER_TYPE_ORDER))

    all_results = []

    for ct in cancer_types:
        mask_group = cancer_labels == ct
        mask_rest  = ~mask_group
        n_group    = mask_group.sum()
        n_rest     = mask_rest.sum()

        if n_group < MIN_SAMPLES:
            print(f"  Skipping {ct} (only {n_group} samples)")
            continue

        print(f"  {ct}: {n_group} vs {n_rest} …", end=" ", flush=True)

        X_group = X[mask_group]
        X_rest  = X[mask_rest]

        stats = deg_one_vs_rest(X_group, X_rest)
        stats.index = range(len(gene_cols))

        n_sig = ((stats["padj"] < FDR_ALPHA)).sum()
        print(f"{n_sig} significant genes")

        # Store results
        stats["gene"]        = gene_cols
        stats["cancer_type"] = ct
        all_results.append(stats)

        # Plot
        out_path = args.output_dir / f"{ct}_volcano.png"
        plot_volcano(stats, gene_cols, ct, n_group, n_rest, out_path)

    # ── Save combined CSV ──────────────────────────────────────────────────
    combined = pd.concat(all_results, ignore_index=True)
    csv_path = args.output_dir / "deg_results.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved combined results to {csv_path}")
    print(f"Saved {len(cancer_types)} volcano plots to {args.output_dir}/")


if __name__ == "__main__":
    main()
