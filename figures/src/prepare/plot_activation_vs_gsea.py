"""
Compare GONNECT bottleneck activations with GSEA NES scores.

For each GO term present in both the bottleneck embedding and the GSEA results:
  - Compute mean |activation| per cancer type (averaged over 5 seeds)
  - Correlate across cancer types with the NES from GSEA

Produces:
  fig_deg/activation_vs_gsea/activation_vs_nes_scatter.png  — scatter per GO term
  fig_deg/activation_vs_gsea/correlation_barplot.png         — Spearman r per GO term
  fig_deg/activation_vs_gsea/activation_vs_nes_heatmaps.png — side-by-side clustermaps
  fig_deg/activation_vs_gsea/correlation_results.csv         — per-term Spearman r + p

Usage:
  pixi run python fig_deg/plot_activation_vs_gsea.py
"""

import argparse
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import spearmanr

from _paths import EMBEDDINGS_DIR

CANCER_TYPE_ORDER = [
    "BRCA", "LUAD", "UCEC", "LGG",  "KIRC", "HNSC", "THCA", "PRAD",
    "LUSC", "SKCM", "COAD", "OV",   "STAD", "BLCA", "LIHC", "CESC",
    "KIRP", "SARC", "ESCA", "PCPG", "PAAD", "READ", "TGCT", "LAML",
    "THYM", "MESO", "UVM",  "ACC",  "KICH", "UCS",  "DLBC", "CHOL",
]

SEEDS = [2, 3, 4, 5, 6]
MODEL_VERSION = "2.1"
MODULE = "encoder"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bottleneck_index(hard_links_path: Path) -> pd.DataFrame:
    """Return DataFrame with columns: dim_index, go_term_id, go_term_name."""
    hl = pd.read_csv(hard_links_path)
    enc = hl[hl["component"] == "encoder"]
    last_layer = enc["layer"].max()
    bottleneck = (
        enc[enc["layer"] == last_layer][["sink_index", "sink_term_id", "sink_term_name"]]
        .drop_duplicates()
        .sort_values("sink_index")
        .rename(columns={"sink_index": "dim_index", "sink_term_id": "go_term_id", "sink_term_name": "go_term_name"})
        .reset_index(drop=True)
    )
    return bottleneck


def load_mean_activations(
    emb_dir: Path,
    meta: pd.DataFrame,
    version: str,
    module: str,
    seeds: list[int],
    sample_type: str = "Primary Tumor",
) -> np.ndarray:
    """
    Load embeddings for all seeds, take abs, average across seeds,
    then return mean activation per cancer type.
    Shape: (n_cancer_types, n_dims)
    Cancer types ordered by CANCER_TYPE_ORDER (filtered to present).
    """
    if sample_type != "all":
        mask = (meta["sample_type"] == sample_type).values
    else:
        mask = np.ones(len(meta), dtype=bool)

    meta_filt = meta[mask].reset_index(drop=True)

    # Average abs activations across seeds
    seed_embs = []
    for s in seeds:
        path = emb_dir / f"AE_{version}" / f"AE_{version}.{s}_{module}_full_dataset.pt"
        emb = torch.load(path, map_location="cpu", weights_only=True).numpy()
        seed_embs.append(np.abs(emb[mask]))
    mean_emb = np.mean(seed_embs, axis=0)  # (n_samples, n_dims)

    # Mean per cancer type
    present = set(meta_filt["cancer_type"].unique())
    cancer_types = [ct for ct in CANCER_TYPE_ORDER if ct in present]

    act_per_ct = np.zeros((len(cancer_types), mean_emb.shape[1]))
    for i, ct in enumerate(cancer_types):
        ct_mask = (meta_filt["cancer_type"] == ct).values
        act_per_ct[i] = mean_emb[ct_mask].mean(axis=0)

    return act_per_ct, cancer_types


# ── Correlation ───────────────────────────────────────────────────────────────

def compute_correlations(
    act_df: pd.DataFrame,   # cancer_type × go_term_id (activations)
    nes_df: pd.DataFrame,   # cancer_type × go_term_id (NES)
    overlap_terms: list[str],
    cancer_types: list[str],
) -> pd.DataFrame:
    """Spearman r between activation and NES across cancer types, per GO term."""
    results = []
    for term in overlap_terms:
        act_vec = act_df.loc[cancer_types, term].values
        nes_vec = nes_df.loc[cancer_types, term].values
        # Only use cancer types where both are present
        valid = np.isfinite(act_vec) & np.isfinite(nes_vec)
        if valid.sum() < 5:
            continue
        r, p = spearmanr(act_vec[valid], nes_vec[valid])
        results.append({"go_term_id": term, "spearman_r": r, "pval": p, "n": valid.sum()})
    return pd.DataFrame(results).sort_values("spearman_r", ascending=False)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_correlation_barplot(
    corr_df: pd.DataFrame,
    term_names: dict,
    output_path: Path,
    title: str | None = None,
) -> None:
    df = corr_df.copy()
    df["label"] = df["go_term_id"].map(lambda g: f"{g}  {term_names.get(g,'')}")
    df = df.sort_values("spearman_r")

    colors = ["#d62728" if r > 0 else "#1f77b4" for r in df["spearman_r"]]

    fig, ax = plt.subplots(figsize=(7, max(4, len(df) * 0.22 + 1)))
    ax.barh(range(len(df)), df["spearman_r"], color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"], fontsize=6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman r  (activation vs -log10(p), across cancer types)", fontsize=8)
    if title is None:
        title = (
            f"Correlation: GONNECT bottleneck activation vs GSEA -log10(NOM p-val)\n"
            f"({len(df)} GO terms in both bottleneck and GSEA)"
        )
    ax.set_title(title, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved barplot to {output_path}")


def plot_side_by_side_clustermaps(
    act_df: pd.DataFrame,
    nes_df: pd.DataFrame,
    overlap_terms: list[str],
    cancer_types: list[str],
    term_names: dict,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Two clustermaps side by side sharing row/col order (from activation)."""
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, leaves_list

    act = act_df.loc[cancer_types, overlap_terms].T.astype(float)  # terms × cancer_types
    nes = nes_df.loc[cancer_types, overlap_terms].T.fillna(0).astype(float)

    # Normalise activation to z-score across cancer types for visual comparability
    act_z = act.sub(act.mean(axis=1), axis=0).div(act.std(axis=1).clip(lower=1e-9), axis=0)

    def safe_linkage(mat):
        d = pdist(mat.values, metric="correlation")
        d = np.where(np.isfinite(d), d, 2.0)
        d = np.clip(d, 0, None)
        return linkage(d, method="average")

    row_link = safe_linkage(act_z)
    col_link = safe_linkage(act_z.T)
    row_order = leaves_list(row_link)
    col_order = leaves_list(col_link)

    act_ordered = act_z.iloc[row_order, :].iloc[:, col_order]
    nes_ordered = nes.iloc[row_order, :].iloc[:, col_order]

    row_labels = [f"{t}  {term_names.get(t,'')}" for t in act_ordered.index]
    col_labels = list(act_ordered.columns)

    n_rows, n_cols = act_ordered.shape
    cell_h, cell_w = 0.22, 0.38
    hm_h = max(6, n_rows * cell_h + 2)
    hm_w = max(6, n_cols * cell_w + 4)

    fig, axes = plt.subplots(1, 2, figsize=(hm_w * 2 + 1, hm_h),
                             gridspec_kw={"wspace": 0.6})

    vmax_act = float(np.nanpercentile(np.abs(act_ordered.values), 95))
    vmax_enr = float(np.nanpercentile(nes_ordered.values[np.isfinite(nes_ordered.values)], 95))
    vmax_act = max(vmax_act, 0.5)
    vmax_enr = max(vmax_enr, 1.0)

    for ax, mat, cmap, vmin, vmax, title, cbar_label in [
        (axes[0], act_ordered, "RdBu_r", -vmax_act, vmax_act, "Mean |activation| (z-scored)", "z-score"),
        (axes[1], nes_ordered, "YlOrRd",  0,         vmax_enr, "-log10(NOM p-val)",            "-log10(p)"),
    ]:
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, fontsize=7, rotation=90)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels if ax is axes[0] else [], fontsize=5.5)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=9, pad=6)
        cb = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.02)
        cb.set_label(cbar_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    if title is None:
        title = (
            "GONNECT bottleneck activation vs GSEA -log10(NOM p-val)\n"
            "(shared row/col order from activation clustering)"
        )
    fig.suptitle(title, fontsize=10, y=1.01)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved side-by-side heatmaps to {output_path}")


def compute_per_cancer_type_correlations(
    act_df: pd.DataFrame,
    nes_df: pd.DataFrame,
    overlap_terms: list[str],
    cancer_types: list[str],
) -> pd.DataFrame:
    """Spearman r between activation and NES across GO terms, per cancer type."""
    results = []
    for ct in cancer_types:
        act_vec = act_df.loc[ct, overlap_terms].values.astype(float)
        nes_vec = nes_df.loc[ct, overlap_terms].values.astype(float)
        valid = np.isfinite(act_vec) & np.isfinite(nes_vec)
        if valid.sum() < 5:
            continue
        r, p = spearmanr(act_vec[valid], nes_vec[valid])
        results.append({"cancer_type": ct, "spearman_r": r, "pval": p, "n": valid.sum()})
    return pd.DataFrame(results).sort_values("spearman_r", ascending=False)


def plot_per_cancer_type_barplot(
    corr_ct: pd.DataFrame,
    output_path: Path,
    title: str | None = None,
    term_label: str = "GO terms",
) -> None:
    df = corr_ct.sort_values("spearman_r")
    colors = ["#d62728" if r > 0 else "#1f77b4" for r in df["spearman_r"]]

    fig, ax = plt.subplots(figsize=(6, max(4, len(df) * 0.25 + 1)))
    ax.barh(range(len(df)), df["spearman_r"], color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["cancer_type"], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Spearman r  (activation vs -log10(p), across {term_label})", fontsize=8)
    if title is None:
        title = "Per-cancer-type correlation:\nGONNECT activation vs GSEA -log10(NOM p-val)"
    ax.set_title(title, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved per-cancer-type barplot to {output_path}")


def plot_overall_scatter(
    act_df: pd.DataFrame,
    nes_df: pd.DataFrame,
    overlap_terms: list[str],
    cancer_types: list[str],
    corr_term: pd.DataFrame,
    corr_ct: pd.DataFrame,
    output_path: Path,
    title: str | None = None,
    term_label: str = "GO terms",
) -> None:
    """All (activation, NES) pairs in one scatter with marginal r distributions."""
    act_vals = act_df.loc[cancer_types, overlap_terms].values.astype(float).ravel()
    nes_vals = nes_df.loc[cancer_types, overlap_terms].fillna(np.nan).values.astype(float).ravel()
    valid = np.isfinite(act_vals) & np.isfinite(nes_vals)
    act_v = act_vals[valid]
    nes_v = nes_vals[valid]

    r_all, p_all = spearmanr(act_v, nes_v)

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.4)

    # ── Panel A: overall scatter ───────────────────────────────────────────
    ax_s = fig.add_subplot(gs[0])
    ax_s.scatter(act_v, nes_v, s=2, alpha=0.3, color="#555555", linewidths=0, rasterized=True)
    # regression line
    m, b = np.polyfit(act_v, nes_v, 1)
    xr = np.linspace(act_v.min(), act_v.max(), 100)
    ax_s.plot(xr, m * xr + b, color="#d62728", linewidth=1.5)
    ax_s.set_xlabel("Mean |activation| (z-scored)", fontsize=8)
    ax_s.set_ylabel("-log10(NOM p-val)", fontsize=8)
    ax_s.set_title(
        f"All {term_label} × cancer types\n"
        f"Spearman r = {r_all:.3f},  p = {p_all:.2g}  (n = {valid.sum()})",
        fontsize=8,
    )
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)

    # ── Panel B: per-term r distribution ──────────────────────────────────
    ax_t = fig.add_subplot(gs[1])
    r_terms = corr_term["spearman_r"].values
    ax_t.hist(r_terms, bins=20, color="#4878d0", edgecolor="white", linewidth=0.4)
    ax_t.axvline(np.median(r_terms), color="#d62728", linewidth=1.5, linestyle="--",
                 label=f"median = {np.median(r_terms):.2f}")
    ax_t.axvline(0, color="black", linewidth=0.8)
    ax_t.set_xlabel("Spearman r", fontsize=8)
    ax_t.set_ylabel(f"Number of {term_label}", fontsize=8)
    ax_t.set_title(f"Per-term correlation\n({len(r_terms)} terms)", fontsize=8)
    ax_t.legend(fontsize=7, frameon=False)
    ax_t.spines["top"].set_visible(False)
    ax_t.spines["right"].set_visible(False)

    # ── Panel C: per-cancer-type r distribution ────────────────────────────
    ax_c = fig.add_subplot(gs[2])
    r_cts = corr_ct["spearman_r"].values
    ax_c.hist(r_cts, bins=15, color="#6acc65", edgecolor="white", linewidth=0.4)
    ax_c.axvline(np.median(r_cts), color="#d62728", linewidth=1.5, linestyle="--",
                 label=f"median = {np.median(r_cts):.2f}")
    ax_c.axvline(0, color="black", linewidth=0.8)
    ax_c.set_xlabel("Spearman r", fontsize=8)
    ax_c.set_ylabel("Number of cancer types", fontsize=8)
    ax_c.set_title(f"Per-cancer-type correlation\n({len(r_cts)} cancer types)", fontsize=8)
    ax_c.legend(fontsize=7, frameon=False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    if title is None:
        title = "GONNECT bottleneck activation vs GSEA -log10(NOM p-val): summary"
    fig.suptitle(title, fontsize=10)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overall scatter to {output_path}")


def run_permutation_null(
    act_df: pd.DataFrame,
    enr_df: pd.DataFrame,
    overlap_terms: list[str],
    cancer_types: list[str],
    n_perms: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Permute cancer-type labels of the activation matrix n_perms times.
    For each permutation compute:
      - global Spearman r (all term×ct pairs)
      - median per-term Spearman r
      - median per-cancer-type Spearman r
    Returns dict of null arrays for each metric.
    """
    rng = np.random.default_rng(seed)
    act = act_df.loc[cancer_types, overlap_terms].values.astype(float)
    enr = enr_df.loc[cancer_types, overlap_terms].fillna(np.nan).values.astype(float)
    n_ct, n_terms = act.shape

    null_global, null_per_term, null_per_ct = [], [], []

    for _ in range(n_perms):
        perm = rng.permutation(n_ct)
        act_p = act[perm]

        # Global
        a_flat = act_p.ravel()
        e_flat = enr.ravel()
        valid = np.isfinite(a_flat) & np.isfinite(e_flat)
        r_global, _ = spearmanr(a_flat[valid], e_flat[valid])
        null_global.append(r_global)

        # Per-term
        r_terms = []
        for j in range(n_terms):
            valid_j = np.isfinite(act_p[:, j]) & np.isfinite(enr[:, j])
            if valid_j.sum() < 5:
                continue
            r, _ = spearmanr(act_p[valid_j, j], enr[valid_j, j])
            r_terms.append(r)
        null_per_term.append(np.median(r_terms))

        # Per-cancer-type
        r_cts = []
        for i in range(n_ct):
            valid_i = np.isfinite(act_p[i]) & np.isfinite(enr[i])
            if valid_i.sum() < 5:
                continue
            r, _ = spearmanr(act_p[i, valid_i], enr[i, valid_i])
            r_cts.append(r)
        null_per_ct.append(np.median(r_cts))

    return {
        "global":    np.array(null_global),
        "per_term":  np.array(null_per_term),
        "per_ct":    np.array(null_per_ct),
    }


def plot_permutation_test(
    null: dict,
    obs_global: float,
    obs_per_term: float,
    obs_per_ct: float,
    n_perms: int,
    output_path: Path,
) -> None:
    def perm_pval(obs, null_arr):
        return (np.sum(null_arr >= obs) + 1) / (len(null_arr) + 1)

    p_global  = perm_pval(obs_global,   null["global"])
    p_term    = perm_pval(obs_per_term, null["per_term"])
    p_ct      = perm_pval(obs_per_ct,   null["per_ct"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw={"wspace": 0.4})

    for ax, null_arr, obs, p, title, xlabel in [
        (axes[0], null["global"],   obs_global,   p_global,
         "Global Spearman r", "Spearman r (all term × ct pairs)"),
        (axes[1], null["per_term"], obs_per_term, p_term,
         "Median per-term r", "Median Spearman r across GO terms"),
        (axes[2], null["per_ct"],   obs_per_ct,   p_ct,
         "Median per-cancer-type r", "Median Spearman r across cancer types"),
    ]:
        ax.hist(null_arr, bins=40, color="#aaaaaa", edgecolor="white",
                linewidth=0.3, label=f"Null (n={n_perms})")
        ax.axvline(obs, color="#d62728", linewidth=2.0,
                   label=f"Observed = {obs:.3f}\np = {p:.3g}")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7.5, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Permutation test: observed activation-enrichment correlation vs null\n"
        "(cancer-type labels of activation matrix shuffled)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved permutation test plot to {output_path}")
    print(f"  Permutation p-values:  global={p_global:.3g},  "
          f"per-term={p_term:.3g},  per-ct={p_ct:.3g}")


def plot_scatter_grid(
    act_df: pd.DataFrame,
    nes_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    overlap_terms: list[str],
    cancer_types: list[str],
    term_names: dict,
    output_path: Path,
) -> None:
    """One scatter per GO term: activation vs -log10(p) across cancer types."""
    terms = corr_df["go_term_id"].tolist()
    n = len(terms)
    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.5))
    axes_flat = axes.flatten()

    for idx, term in enumerate(terms):
        ax = axes_flat[idx]
        act_vec = act_df.loc[cancer_types, term].values
        nes_vec = nes_df.loc[cancer_types, term].values
        valid = np.isfinite(act_vec) & np.isfinite(nes_vec)
        row = corr_df[corr_df["go_term_id"] == term].iloc[0]

        ax.scatter(act_vec[valid], nes_vec[valid], s=18, alpha=0.7,
                   c="#d62728" if row["spearman_r"] > 0 else "#1f77b4",
                   linewidths=0)
        for i, ct in enumerate(cancer_types):
            if valid[i]:
                ax.annotate(ct, (act_vec[i], nes_vec[i]),
                            fontsize=3.5, ha="center", va="bottom", alpha=0.7)
        name = term_names.get(term, "")
        ax.set_title(f"{term}\n{name}", fontsize=5.5, pad=2)
        ax.set_xlabel("|activation|", fontsize=6)
        ax.set_ylabel("-log10(NOM p-val)", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.text(0.97, 0.03, f"r={row['spearman_r']:.2f}\np={row['pval']:.2g}",
                transform=ax.transAxes, fontsize=5, ha="right", va="bottom",
                color="#555555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("GONNECT activation vs GSEA -log10(NOM p-val) per GO term (across cancer types)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter grid to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=Path, default=Path("data"))
    parser.add_argument("--emb-dir",    type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--gsea-csv",   type=Path, default=Path("fig_deg/gsea_bottleneck/gsea_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("fig_deg/activation_vs_gsea_bottleneck"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Bottleneck index ───────────────────────────────────────────────────
    print("Loading bottleneck index …")
    bottleneck = load_bottleneck_index(args.data_dir / "hard_links.csv")
    term_names = dict(zip(bottleneck["go_term_id"], bottleneck["go_term_name"]))
    print(f"  {len(bottleneck)} bottleneck GO terms")

    # ── TCGA metadata ──────────────────────────────────────────────────────
    print("Loading TCGA metadata …")
    with gzip.open(args.data_dir / "TCGA_complete_bp_top1k.csv.gz", "rt") as fh:
        meta = pd.read_csv(fh, usecols=["patient_id", "cancer_type", "sample_type"])

    # ── Mean activations per cancer type ───────────────────────────────────
    print("Loading embeddings and computing mean activations …")
    act_matrix, cancer_types = load_mean_activations(
        args.emb_dir, meta, MODEL_VERSION, MODULE, SEEDS,
        sample_type="Primary Tumor",
    )
    # act_matrix: (n_cancer_types, 109)
    act_df = pd.DataFrame(
        act_matrix,
        index=cancer_types,
        columns=bottleneck["go_term_id"].tolist(),
    )
    print(f"  {len(cancer_types)} cancer types × {act_matrix.shape[1]} GO terms")

    # ── GSEA results ───────────────────────────────────────────────────────
    print("Loading GSEA results …")
    gsea = pd.read_csv(args.gsea_csv)

    # Use -log10(NOM p-val) as enrichment strength (avoids NES bimodality near ±1
    # caused by separate normalization of positive/negative scores).
    # Clip p-val to [1/n_perm, 1] to avoid inf; 1000 permutations -> min p = 0.001.
    gsea["enr_score"] = -np.log10(np.clip(gsea["NOM p-val"], 1e-3, 1.0))

    enr_pivot = gsea.pivot_table(index="cancer_type", columns="Term",
                                 values="enr_score", aggfunc="first")
    present_cts = [ct for ct in cancer_types if ct in enr_pivot.index]
    enr_pivot = enr_pivot.reindex(index=present_cts)
    print(f"  {len(enr_pivot)} cancer types x {len(enr_pivot.columns)} GSEA terms")
    print(f"  Enrichment score = -log10(NOM p-val), range: "
          f"{enr_pivot.values[np.isfinite(enr_pivot.values)].min():.2f} – "
          f"{enr_pivot.values[np.isfinite(enr_pivot.values)].max():.2f}")

    # ── Overlap ────────────────────────────────────────────────────────────
    overlap_terms = sorted(set(bottleneck["go_term_id"]) & set(enr_pivot.columns))
    print(f"  Terms with both activation and GSEA results: {len(overlap_terms)}\n")

    nes_df = enr_pivot[overlap_terms].reindex(index=cancer_types)
    act_df_ov = act_df[overlap_terms].reindex(index=cancer_types)

    # ── Correlations ───────────────────────────────────────────────────────
    print("Computing Spearman correlations …")
    corr_df = compute_correlations(act_df_ov, nes_df, overlap_terms, cancer_types)
    corr_df["go_term_name"] = corr_df["go_term_id"].map(term_names).fillna("")

    csv_path = args.output_dir / "correlation_results.csv"
    corr_df.to_csv(csv_path, index=False)
    print(f"  Saved correlation results to {csv_path}")
    print(f"  Median Spearman r: {corr_df['spearman_r'].median():.3f}")
    print(f"  Terms with r>0.3: {(corr_df['spearman_r']>0.3).sum()}")
    print(f"  Terms with r<-0.3: {(corr_df['spearman_r']<-0.3).sum()}\n")

    # ── Per-cancer-type correlations ───────────────────────────────────────
    corr_ct = compute_per_cancer_type_correlations(act_df_ov, nes_df, overlap_terms, cancer_types)
    corr_ct.to_csv(args.output_dir / "correlation_results_per_cancer_type.csv", index=False)
    print(f"  Median per-cancer-type Spearman r: {corr_ct['spearman_r'].median():.3f}")
    print(f"  Cancer types with r>0.3: {(corr_ct['spearman_r']>0.3).sum()}")
    print(f"  Cancer types with r<-0.3: {(corr_ct['spearman_r']<-0.3).sum()}\n")

    # ── Permutation test ───────────────────────────────────────────────────
    n_perms = 1000
    print(f"Running permutation test ({n_perms} permutations) …")
    null = run_permutation_null(act_df_ov, nes_df, overlap_terms, cancer_types, n_perms=n_perms)

    # Observed summary stats
    act_flat = act_df_ov.loc[cancer_types, overlap_terms].values.astype(float).ravel()
    enr_flat = nes_df.loc[cancer_types, overlap_terms].fillna(np.nan).values.astype(float).ravel()
    valid = np.isfinite(act_flat) & np.isfinite(enr_flat)
    obs_global, _ = spearmanr(act_flat[valid], enr_flat[valid])
    obs_per_term   = corr_df["spearman_r"].median()
    obs_per_ct     = corr_ct["spearman_r"].median()

    # ── Plots ──────────────────────────────────────────────────────────────
    print("Plotting …")
    plot_correlation_barplot(
        corr_df, term_names,
        args.output_dir / "correlation_barplot_per_term.png",
    )
    plot_per_cancer_type_barplot(
        corr_ct,
        args.output_dir / "correlation_barplot_per_cancer_type.png",
    )
    plot_overall_scatter(
        act_df_ov, nes_df, overlap_terms, cancer_types, corr_df, corr_ct,
        args.output_dir / "correlation_summary.png",
    )
    plot_permutation_test(
        null, obs_global, obs_per_term, obs_per_ct, n_perms,
        args.output_dir / "permutation_test.png",
    )
    plot_side_by_side_clustermaps(
        act_df_ov, nes_df, overlap_terms, cancer_types, term_names,
        args.output_dir / "activation_vs_nes_heatmaps.png",
    )
    plot_scatter_grid(
        act_df_ov, nes_df, corr_df, overlap_terms, cancer_types, term_names,
        args.output_dir / "activation_vs_nes_scatter.png",
    )


if __name__ == "__main__":
    main()
