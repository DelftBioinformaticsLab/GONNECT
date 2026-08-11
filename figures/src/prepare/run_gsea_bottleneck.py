"""
GSEA using bottleneck-node receptive fields.

For each of the 109 bottleneck GO terms (encoder layer 3 sinks), the gene set
is the full "receptive field": all genes that can reach that node through any
path in the encoder graph (across all 4 layers).

Also produces:
  fig_deg/gsea_bottleneck/receptive_field_histogram.png  — gene-set size distribution
  fig_deg/gsea_bottleneck/gsea_results.csv
  fig_deg/gsea_bottleneck/gsea_sig_terms_per_cancer_type.csv
  fig_deg/gsea_bottleneck/gsea_nes_heatmap.png
  fig_deg/gsea_bottleneck/gsea_nes_heatmap_sigonly.png

Usage:
  pixi run python fig_deg/run_gsea_bottleneck.py
"""

import argparse
import gzip
import re
import warnings
from collections import defaultdict, deque
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from _paths import DATA_DIR, DEG_CSV, PREP_OUT_DIR

CANCER_TYPE_ORDER = [
    "BRCA", "LUAD", "UCEC", "LGG",  "KIRC", "HNSC", "THCA", "PRAD",
    "LUSC", "SKCM", "COAD", "OV",   "STAD", "BLCA", "LIHC", "CESC",
    "KIRP", "SARC", "ESCA", "PCPG", "PAAD", "READ", "TGCT", "LAML",
    "THYM", "MESO", "UVM",  "ACC",  "KICH", "UCS",  "DLBC", "CHOL",
]

MIN_GENES = 5
MAX_GENES = 1000
PERMUTATIONS = 1000
THREADS = 4


# ── Receptive field computation ───────────────────────────────────────────────

def build_receptive_fields(hard_links_path: Path) -> tuple[dict, dict, dict]:
    """
    Traverse the encoder graph backwards from each bottleneck node to find all
    genes in its receptive field.

    Returns:
      gene_sets  : {go_term_id: [gene_id, ...]}
      term_names : {go_term_id: human-readable name}
      rf_sizes   : {go_term_id: n_genes}
    """
    hl = pd.read_csv(hard_links_path)
    enc = hl[hl["component"] == "encoder"].copy()

    # ── Build reverse adjacency list: node -> set of predecessor nodes ────
    # Each edge is (source, sink); reverse means sink -> source
    reverse_graph: dict[str, set[str]] = defaultdict(set)
    for _, row in enc.iterrows():
        src = str(row["source_term_id"])
        snk = str(row["sink_term_id"])
        reverse_graph[snk].add(src)

    # ── Identify bottleneck nodes (sinks of layer 3) ──────────────────────
    last_layer = enc["layer"].max()
    bottleneck_rows = (
        enc[enc["layer"] == last_layer][["sink_term_id", "sink_term_name"]]
        .drop_duplicates()
    )
    bottleneck_nodes = bottleneck_rows["sink_term_id"].tolist()
    term_names = dict(zip(bottleneck_rows["sink_term_id"], bottleneck_rows["sink_term_name"]))

    # Also collect names from source columns for all GO terms
    for col_id, col_name in [("source_term_id", "source_term_name"),
                              ("sink_term_id",   "sink_term_name")]:
        for _, row in hl[hl[col_id].str.startswith("GO:", na=False)][[col_id, col_name]].drop_duplicates().iterrows():
            term_names.setdefault(row[col_id], row[col_name])

    # ── BFS backwards from each bottleneck node ───────────────────────────
    def is_gene(node_id: str) -> bool:
        """True if node is a raw gene (not a GO term or proxy node)."""
        return not (node_id.startswith("GO:") or node_id.startswith("Proxy:"))

    def extract_gene(node_id: str) -> str:
        """Strip Proxy:N_ prefix to get gene ID."""
        m = re.match(r"Proxy:\d+_(.*)", node_id)
        return m.group(1) if m else node_id

    gene_sets = {}
    for bn in bottleneck_nodes:
        visited = set()
        queue = deque([bn])
        genes_found = set()
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for pred in reverse_graph.get(node, []):
                if pred in visited:
                    continue
                if is_gene(pred):
                    genes_found.add(pred)
                else:
                    inner = extract_gene(pred)
                    if is_gene(inner):
                        genes_found.add(inner)
                    else:
                        queue.append(pred)
        gene_sets[bn] = sorted(genes_found)

    rf_sizes = {k: len(v) for k, v in gene_sets.items()}
    return gene_sets, term_names, rf_sizes


# ── Histogram ─────────────────────────────────────────────────────────────────

def plot_rf_histogram(rf_sizes: dict, term_names: dict, output_path: Path) -> None:
    sizes = sorted(rf_sizes.values())
    n_terms = len(sizes)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"wspace": 0.35})

    # Left: linear scale
    axes[0].hist(sizes, bins=30, color="#4878d0", edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Number of genes in receptive field", fontsize=9)
    axes[0].set_ylabel("Number of GO terms", fontsize=9)
    axes[0].set_title("Bottleneck receptive field sizes (linear)", fontsize=9)

    # Right: log x-scale
    log_bins = np.logspace(np.log10(max(1, min(sizes))), np.log10(max(sizes)), 25)
    axes[1].hist(sizes, bins=log_bins, color="#4878d0", edgecolor="white", linewidth=0.4)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Number of genes in receptive field (log scale)", fontsize=9)
    axes[1].set_ylabel("Number of GO terms", fontsize=9)
    axes[1].set_title("Bottleneck receptive field sizes (log scale)", fontsize=9)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(np.median(sizes), color="firebrick", linewidth=1.2,
                   linestyle="--", label=f"median = {int(np.median(sizes))}")
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        f"Gene receptive fields of {n_terms} GONNECT bottleneck GO terms\n"
        f"(min={min(sizes)}, median={int(np.median(sizes))}, max={max(sizes)})",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved histogram to {output_path}")


# ── GSEA ─────────────────────────────────────────────────────────────────────

def run_gsea_for_cancer_type(ct, deg_ct, gene_sets, permutations, threads):
    deg_ct = deg_ct.copy()
    deg_ct["rank_score"] = deg_ct["mean_diff"] * (-np.log10(np.clip(deg_ct["padj"], 1e-300, 1.0)))
    rnk = deg_ct.set_index("gene")["rank_score"].sort_values(ascending=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pre_res = gp.prerank(
            rnk=rnk,
            gene_sets=gene_sets,
            min_size=MIN_GENES,
            max_size=MAX_GENES,
            permutation_num=permutations,
            weight=1.0,
            ascending=False,
            threads=threads,
            no_plot=True,
            seed=42,
            verbose=False,
            outdir=None,
        )

    res = pre_res.res2d.copy()
    res["cancer_type"] = ct
    return res


# ── Clustermap ────────────────────────────────────────────────────────────────

def _safe_linkage(mat: np.ndarray):
    d = pdist(mat, metric="correlation")
    d = np.where(np.isfinite(d), d, 2.0)
    d = np.clip(d, 0, None)
    return linkage(d, method="average")


def plot_nes_clustermap(results, term_names, cancer_types, output_path,
                        fdr_threshold=0.25, min_sig_cts=1, sig_only=False):
    pivot_nes = results.pivot_table(index="Term", columns="cancer_type", values="NES", aggfunc="first")
    pivot_fdr = results.pivot_table(index="Term", columns="cancer_type", values="FDR q-val", aggfunc="first")

    ct_cols = [ct for ct in cancer_types if ct in pivot_nes.columns]
    pivot_nes = pivot_nes.reindex(columns=ct_cols).fillna(0).astype(float)
    pivot_fdr = pivot_fdr.reindex(columns=ct_cols).fillna(1).astype(float)

    keep = (pivot_fdr < fdr_threshold).sum(axis=1) >= min_sig_cts
    pivot_nes = pivot_nes[keep]
    pivot_fdr = pivot_fdr[keep]

    if len(pivot_nes) == 0:
        print(f"  No terms with FDR<{fdr_threshold} in ≥{min_sig_cts} cancer type(s). Skipping.")
        return

    print(f"  Clustermap ({'sig only' if sig_only else 'all'}): {len(pivot_nes)} terms × {len(ct_cols)} cancer types")

    pivot_nes.index = [
        f"{go}  {term_names.get(go, '')}" if term_names.get(go) else go
        for go in pivot_nes.index
    ]
    pivot_fdr.index = pivot_nes.index

    vmax = max(float(np.nanpercentile(np.abs(pivot_nes.values), 95)), 1.0)

    if sig_only:
        pivot_display = pivot_nes.where(pivot_fdr < fdr_threshold, other=np.nan)
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color="white")
        cluster_mat = pivot_display.fillna(0).values
        row_link = _safe_linkage(cluster_mat)
        col_link = _safe_linkage(cluster_mat.T)
    else:
        pivot_display = pivot_nes
        cmap = "RdBu_r"
        row_link, col_link = None, None

    n_rows, n_cols = pivot_nes.shape
    fig_h = max(8, n_rows * 0.22 + 3)
    fig_w = max(10, n_cols * 0.38 + 6)

    g = sns.clustermap(
        pivot_display, cmap=cmap, vmin=-vmax, vmax=vmax,
        figsize=(fig_w, fig_h), xticklabels=True, yticklabels=True,
        linewidths=0, dendrogram_ratio=(0.12, 0.06),
        cbar_pos=(1.02, 0.3, 0.015, 0.3),
        metric="correlation", method="average",
        row_linkage=row_link, col_linkage=col_link,
    )
    g.ax_heatmap.tick_params(axis="x", labelsize=7)
    g.ax_heatmap.tick_params(axis="y", labelsize=5.5)
    g.ax_cbar.set_title("NES", fontsize=8, pad=6)
    g.ax_cbar.tick_params(labelsize=7)

    if not sig_only:
        row_order = g.dendrogram_row.reordered_ind
        col_order = g.dendrogram_col.reordered_ind
        fdr_r = pivot_fdr.values[np.ix_(row_order, col_order)]
        sy, sx = np.where(fdr_r < fdr_threshold)
        g.ax_heatmap.scatter(sx + 0.5, sy + 0.5, marker=".", s=8,
                             c="black", linewidths=0, zorder=3)

    suffix = " (significant only)" if sig_only else f" (dots = FDR < {fdr_threshold})"
    g.fig.suptitle(
        f"Pre-ranked GSEA (bottleneck receptive fields): NES{suffix}\n"
        f"{n_rows} terms significant in ≥{min_sig_cts} cancer type",
        fontsize=9, y=1.01,
    )
    g.fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(g.fig)
    print(f"  Saved clustermap to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",     type=Path, default=DATA_DIR)
    parser.add_argument("--deg-csv",      type=Path, default=DEG_CSV)
    parser.add_argument("--output-dir",   type=Path, default=PREP_OUT_DIR / "gsea_gonnect_bottleneck")
    parser.add_argument("--permutations", type=int,  default=PERMUTATIONS)
    parser.add_argument("--threads",      type=int,  default=THREADS)
    parser.add_argument("--fdr",          type=float, default=0.25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Receptive fields ───────────────────────────────────────────────────
    print("Building bottleneck receptive fields …")
    gene_sets, term_names, rf_sizes = build_receptive_fields(args.data_dir / "hard_links.csv")

    sizes = list(rf_sizes.values())
    print(f"  {len(gene_sets)} bottleneck terms")
    print(f"  Receptive field sizes: min={min(sizes)}, median={int(np.median(sizes))}, "
          f"mean={np.mean(sizes):.1f}, max={max(sizes)}")
    usable = sum(1 for s in sizes if MIN_GENES <= s <= MAX_GENES)
    print(f"  Usable for GSEA (>={MIN_GENES} genes): {usable}\n")

    plot_rf_histogram(rf_sizes, term_names, args.output_dir / "receptive_field_histogram.png")

    # Print per-term breakdown
    rf_df = pd.DataFrame([
        {"go_term_id": k, "go_term_name": term_names.get(k, ""), "n_genes": v}
        for k, v in rf_sizes.items()
    ]).sort_values("n_genes", ascending=False)
    rf_df.to_csv(args.output_dir / "receptive_field_sizes.csv", index=False)
    print(f"  Saved receptive field sizes to {args.output_dir}/receptive_field_sizes.csv\n")

    # ── DEG results ────────────────────────────────────────────────────────
    print(f"Loading DEG results …")
    deg = pd.read_csv(args.deg_csv)
    present = set(deg["cancer_type"].unique())
    cancer_types = [ct for ct in CANCER_TYPE_ORDER if ct in present] + \
                   sorted(present - set(CANCER_TYPE_ORDER))
    print(f"  {len(cancer_types)} cancer types, {deg['gene'].nunique()} genes\n")

    # ── GSEA ───────────────────────────────────────────────────────────────
    all_results = []
    for ct in cancer_types:
        deg_ct = deg[deg["cancer_type"] == ct]
        print(f"  GSEA {ct} …", end=" ", flush=True)
        res = run_gsea_for_cancer_type(ct, deg_ct, gene_sets,
                                       args.permutations, args.threads)
        n_sig = (res["FDR q-val"] < args.fdr).sum()
        print(f"{len(res)} terms tested, {n_sig} FDR<{args.fdr}")
        all_results.append(res)

    combined = pd.concat(all_results, ignore_index=True)
    combined["term_name"] = combined["Term"].map(term_names).fillna("")

    csv_path = args.output_dir / "gsea_results.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved GSEA results to {csv_path}")

    # Per-cancer-type sig list
    sig = combined[combined["FDR q-val"] < args.fdr].copy()
    sig = sig.sort_values(["cancer_type", "NES"], ascending=[True, False])
    sig[["cancer_type", "Term", "term_name", "NES", "NOM p-val", "FDR q-val"]].to_csv(
        args.output_dir / "gsea_sig_terms_per_cancer_type.csv", index=False
    )
    print(f"Saved sig terms per cancer type ({len(sig)} entries)")

    # ── Clustermaps ────────────────────────────────────────────────────────
    print("\nPlotting clustermaps …")
    plot_nes_clustermap(combined, term_names, cancer_types,
                        args.output_dir / "gsea_nes_heatmap.png",
                        fdr_threshold=args.fdr, sig_only=False)
    plot_nes_clustermap(combined, term_names, cancer_types,
                        args.output_dir / "gsea_nes_heatmap_sigonly.png",
                        fdr_threshold=args.fdr, sig_only=True)


if __name__ == "__main__":
    main()
