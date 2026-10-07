"""
Pre-ranked GSEA against GO terms in the processed GONNECT ontology.

Ranking metric: mean_diff × −log10(padj)  (per cancer type, one-vs-rest)

Gene sets are taken from data/hard_links.csv (encoder component). --gene-sets
picks the definition (see gsea_gene_sets.py):

  receptive_field  (default) every gene with a path to the term, through proxies
                   and child terms alike: its own and its descendants'
                   annotations, as the Methods define it. This is what Figure 4
                   reads, from <data-dir>/gsea_gonnect_receptive_fields/; on the
                   bottleneck it equals run_gsea_bottleneck.py's sets.
  direct           the term's own annotations, traced through proxy chains of
                   any length.
  build_gene_sets  what the published figure used (gsea_gonnect_layers/):
                   build_gene_sets below, which reads layers 0 and 1 only and so
                   misses every gene whose proxy chain enters its term at layer
                   2 or 3.

Outputs, under --output-dir (default figures/out/prepare/gsea_gonnect_<definition>/):
  gsea_results.csv   — NES, pval, FDR per cancer type × GO term
  gsea_heatmap.png   — NES heatmap (GO terms × cancer types)

Usage:
  pixi run python figures/src/prepare/run_gsea.py [--gene-sets receptive_field] [--output-dir DIR]
"""

import argparse
import re
import warnings
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from _paths import DATA_DIR, DEG_CSV, PREP_OUT_DIR

CANCER_TYPE_ORDER = [
    "BRCA", "LUAD", "UCEC", "LGG",  "KIRC", "HNSC", "THCA", "PRAD",
    "LUSC", "SKCM", "COAD", "OV",   "STAD", "BLCA", "LIHC", "CESC",
    "KIRP", "SARC", "ESCA", "PCPG", "PAAD", "READ", "TGCT", "LAML",
    "THYM", "MESO", "UVM",  "ACC",  "KICH", "UCS",  "DLBC", "CHOL",
]

MIN_GENES = 3    # minimum gene set size (GONNECT terms are small)
MAX_GENES = 500
PERMUTATIONS = 1000
THREADS = 4


# ── Gene-set extraction ───────────────────────────────────────────────────────

def build_gene_sets(hard_links_path: Path) -> tuple[dict, dict]:
    """
    Parse hard_links.csv and return:
      gene_sets  : {go_term_id: [gene_id, ...]}
      term_names : {go_term_id: human-readable name}
    Only uses encoder edges (layer 0 and 1) to extract direct gene->GO membership.
    """
    hl = pd.read_csv(hard_links_path)
    enc = hl[hl["component"] == "encoder"].copy()

    def classify(s):
        s = str(s)
        if s.startswith("GO:"):    return "GO"
        if s.startswith("Proxy:") and "GO:" in s: return "Proxy_GO"
        if s.startswith("Proxy:"): return "Proxy_gene"
        return "gene"

    enc["src_type"] = enc["source_term_id"].apply(classify)
    enc["snk_type"] = enc["sink_term_id"].apply(classify)

    def gene_from_proxy(s):
        m = re.match(r"Proxy:\d+_(.*)", str(s))
        return m.group(1) if m else s

    def go_from_proxy(s):
        m = re.match(r"Proxy:\d+_(GO:\d+)", str(s))
        return m.group(1) if m else s

    # proxy_gene_id -> gene_id
    proxy_gene_map = {
        row["sink_term_id"]: row["source_term_id"]
        for _, row in enc[
            (enc["src_type"] == "gene") & (enc["snk_type"] == "Proxy_gene")
        ].iterrows()
    }

    # proxy_GO_id -> GO_id
    proxy_go_map = {p: go_from_proxy(p) for p in
                    enc["source_term_id"].tolist() + enc["sink_term_id"].tolist()
                    if str(p).startswith("Proxy:") and "GO:" in str(p)}

    gene_go_pairs = []

    # Layer 0: gene -> GO (direct)
    for _, row in enc[
        (enc["layer"] == 0) & (enc["src_type"] == "gene") & (enc["snk_type"] == "GO")
    ].iterrows():
        gene_go_pairs.append((row["source_term_id"], row["sink_term_id"]))

    # Layer 0: gene -> Proxy_GO
    for _, row in enc[
        (enc["layer"] == 0) & (enc["src_type"] == "gene") & (enc["snk_type"] == "Proxy_GO")
    ].iterrows():
        go = proxy_go_map.get(row["sink_term_id"], go_from_proxy(row["sink_term_id"]))
        gene_go_pairs.append((row["source_term_id"], go))

    # Layer 1: Proxy_gene -> GO
    for _, row in enc[
        (enc["layer"] == 1) & (enc["src_type"] == "Proxy_gene") & (enc["snk_type"] == "GO")
    ].iterrows():
        gene = proxy_gene_map.get(row["source_term_id"], gene_from_proxy(row["source_term_id"]))
        gene_go_pairs.append((gene, row["sink_term_id"]))

    # Layer 1: Proxy_gene -> Proxy_GO
    for _, row in enc[
        (enc["layer"] == 1) & (enc["src_type"] == "Proxy_gene") & (enc["snk_type"] == "Proxy_GO")
    ].iterrows():
        gene = proxy_gene_map.get(row["source_term_id"], gene_from_proxy(row["source_term_id"]))
        go   = proxy_go_map.get(row["sink_term_id"],   go_from_proxy(row["sink_term_id"]))
        gene_go_pairs.append((gene, go))

    pairs_df = pd.DataFrame(gene_go_pairs, columns=["gene", "go_term"]).drop_duplicates()

    gene_sets = (
        pairs_df.groupby("go_term")["gene"]
        .apply(list)
        .to_dict()
    )

    # Collect human-readable names from source/sink columns
    term_names = {}
    for col_id, col_name in [("source_term_id", "source_term_name"),
                              ("sink_term_id",   "sink_term_name")]:
        go_rows = hl[hl[col_id].str.startswith("GO:", na=False)][[col_id, col_name]].drop_duplicates()
        for _, row in go_rows.iterrows():
            term_names[row[col_id]] = row[col_name]

    return gene_sets, term_names


# ── GSEA ─────────────────────────────────────────────────────────────────────

def run_gsea_for_cancer_type(
    ct: str,
    deg_ct: pd.DataFrame,
    gene_sets: dict,
    outdir: Path,
    permutations: int,
    threads: int,
) -> pd.DataFrame | None:
    """Run prerank GSEA for one cancer type. Returns results DataFrame."""
    # Ranking metric: mean_diff * -log10(padj)
    deg_ct = deg_ct.copy()
    deg_ct["rank_score"] = deg_ct["mean_diff"] * (-np.log10(np.clip(deg_ct["padj"], 1e-300, 1.0)))

    rnk = deg_ct.set_index("gene")["rank_score"].sort_values(ascending=False)

    ct_outdir = outdir / ct
    ct_outdir.mkdir(parents=True, exist_ok=True)

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

def plot_nes_clustermap(
    results: pd.DataFrame,
    term_names: dict,
    cancer_types: list,
    output_path: Path,
    fdr_threshold: float = 0.25,
    min_sig_cts: int = 1,
    sig_only: bool = False,
) -> None:
    """
    NES clustermap: GO terms (rows) × cancer types (columns), both axes clustered.
    sig_only=True: non-significant cells are white; clustering still uses full NES matrix.
    sig_only=False: all cells coloured; significant cells marked with a dot.
    """
    import seaborn as sns
    import matplotlib.colors as mcolors

    pivot_nes = results.pivot_table(index="Term", columns="cancer_type", values="NES", aggfunc="first")
    pivot_fdr = results.pivot_table(index="Term", columns="cancer_type", values="FDR q-val", aggfunc="first")

    ct_cols = [ct for ct in cancer_types if ct in pivot_nes.columns]
    pivot_nes = pivot_nes.reindex(columns=ct_cols).fillna(0).astype(float)
    pivot_fdr = pivot_fdr.reindex(columns=ct_cols).fillna(1).astype(float)

    keep = (pivot_fdr < fdr_threshold).sum(axis=1) >= min_sig_cts
    pivot_nes = pivot_nes[keep]
    pivot_fdr = pivot_fdr[keep]

    if len(pivot_nes) == 0:
        print(f"  No GO terms with FDR<{fdr_threshold} in ≥{min_sig_cts} cancer type(s). Skipping clustermap.")
        return

    print(f"  Clustermap ({'sig only' if sig_only else 'all'}): {len(pivot_nes)} GO terms × {len(ct_cols)} cancer types")

    # Row labels: "GO:ID  name"
    pivot_nes.index = [
        f"{go}  {term_names[go]}" if go in term_names else go
        for go in pivot_nes.index
    ]
    pivot_fdr.index = pivot_nes.index

    vmax = float(np.nanpercentile(np.abs(pivot_nes.values), 95))
    vmax = max(vmax, 1.0)

    # For sig_only: mask non-significant cells to NaN in the display matrix,
    # but pass the full matrix for clustering so the dendrogram is identical.
    if sig_only:
        pivot_display = pivot_nes.where(pivot_fdr < fdr_threshold, other=np.nan)
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color="white")
    else:
        pivot_display = pivot_nes
        cmap = "RdBu_r"

    n_rows, n_cols = pivot_nes.shape
    fig_h = max(8, n_rows * 0.22 + 3)
    fig_w = max(10, n_cols * 0.38 + 6)

    # For sig_only, cluster on the masked matrix (NaN→0); otherwise cluster on full NES.
    if sig_only:
        cluster_mat = pivot_display.fillna(0).values
        row_linkage = _linkage_from_full(cluster_mat)
        col_linkage = _linkage_from_full(cluster_mat.T)
    else:
        row_linkage = None
        col_linkage = None

    g = sns.clustermap(
        pivot_display,
        cmap=cmap,
        vmin=-vmax, vmax=vmax,
        figsize=(fig_w, fig_h),
        xticklabels=True,
        yticklabels=True,
        linewidths=0,
        dendrogram_ratio=(0.12, 0.06),
        cbar_pos=(1.02, 0.3, 0.015, 0.3),
        metric="correlation",
        method="average",
        row_linkage=row_linkage,
        col_linkage=col_linkage,
    )

    g.ax_heatmap.tick_params(axis="x", labelsize=7)
    g.ax_heatmap.tick_params(axis="y", labelsize=5.5)
    g.ax_cbar.set_title("NES", fontsize=8, pad=6)
    g.ax_cbar.tick_params(labelsize=7)

    if not sig_only:
        # Dot overlay on significant cells
        row_order = g.dendrogram_row.reordered_ind
        col_order = g.dendrogram_col.reordered_ind
        fdr_reordered = pivot_fdr.values[np.ix_(row_order, col_order)]
        sig_y, sig_x = np.where(fdr_reordered < fdr_threshold)
        g.ax_heatmap.scatter(
            sig_x + 0.5, sig_y + 0.5,
            marker=".", s=8, c="black", linewidths=0, zorder=3,
        )

    suffix = " (significant only)" if sig_only else f" (dots = FDR < {fdr_threshold})"
    g.fig.suptitle(
        f"Pre-ranked GSEA: NES per GO term × cancer type{suffix}\n"
        f"{n_rows} terms significant in ≥{min_sig_cts} cancer type",
        fontsize=9, y=1.01,
    )

    g.fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(g.fig)
    print(f"  Saved clustermap to {output_path}")


def _linkage_from_full(mat: np.ndarray):
    """Compute average-linkage using correlation distance. NaN distances (all-zero rows) → max distance."""
    from scipy.spatial.distance import pdist
    condensed = pdist(mat, metric="correlation")
    condensed = np.clip(condensed, 0, None)
    condensed = np.where(np.isfinite(condensed), condensed, 2.0)  # 2 = max correlation distance
    return linkage(condensed, method="average")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    type=Path, default=DATA_DIR)
    parser.add_argument("--deg-csv",     type=Path, default=DEG_CSV)
    parser.add_argument("--gene-sets", choices=["receptive_field", "direct", "build_gene_sets"],
                        default="receptive_field", help="gene-set definition (see the module docstring)")
    parser.add_argument("--output-dir",  type=Path, default=None,
                        help="default: figures/out/prepare/gsea_gonnect_receptive_fields/ for receptive_field, "
                             "gsea_gonnect_direct_traced/ for direct, gsea_gonnect_layers/ for build_gene_sets")
    parser.add_argument("--permutations", type=int,  default=PERMUTATIONS)
    parser.add_argument("--threads",      type=int,  default=THREADS)
    parser.add_argument("--fdr",          type=float, default=0.25,
                        help="FDR threshold for heatmap significance dots")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = PREP_OUT_DIR / {"receptive_field": "gsea_gonnect_receptive_fields",
                                          "direct": "gsea_gonnect_direct_traced",
                                          "build_gene_sets": "gsea_gonnect_layers"}[args.gene_sets]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load gene sets ─────────────────────────────────────────────────────
    hl_path = args.data_dir / "hard_links.csv"
    print(f"Building {args.gene_sets} gene sets from {hl_path} …")
    gene_sets, term_names = build_gene_sets(hl_path)
    if args.gene_sets != "build_gene_sets":
        import gsea_gene_sets
        traced = gsea_gene_sets.all_sets(hl_path)[args.gene_sets]
        gene_sets = {t: sorted(g) for t, g in traced.items() if g}
    sizes = {k: len(v) for k, v in gene_sets.items()}
    usable = sum(1 for s in sizes.values() if MIN_GENES <= s <= MAX_GENES)
    print(f"  {len(gene_sets)} GO terms total, {usable} with {MIN_GENES}–{MAX_GENES} genes\n")

    # ── Load DEG results ───────────────────────────────────────────────────
    print(f"Loading DEG results from {args.deg_csv} …")
    deg = pd.read_csv(args.deg_csv)
    present = set(deg["cancer_type"].unique())
    cancer_types = [ct for ct in CANCER_TYPE_ORDER if ct in present] + \
                   sorted(present - set(CANCER_TYPE_ORDER))
    print(f"  {len(cancer_types)} cancer types, {deg['gene'].nunique()} genes\n")

    # ── Run GSEA per cancer type ───────────────────────────────────────────
    all_results = []
    for ct in cancer_types:
        deg_ct = deg[deg["cancer_type"] == ct]
        print(f"  GSEA {ct} …", end=" ", flush=True)
        res = run_gsea_for_cancer_type(
            ct, deg_ct, gene_sets, args.output_dir,
            permutations=args.permutations, threads=args.threads,
        )
        if res is not None:
            n_sig = (res["FDR q-val"] < args.fdr).sum()
            print(f"{len(res)} terms tested, {n_sig} FDR<{args.fdr}")
            all_results.append(res)

    # ── Save combined results ──────────────────────────────────────────────
    combined = pd.concat(all_results, ignore_index=True)

    # Add human-readable name column
    combined["term_name"] = combined["Term"].map(term_names).fillna("")

    csv_path = args.output_dir / "gsea_results.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved combined GSEA results to {csv_path}")

    # ── Per-cancer-type significant term lists ────────────────────────────
    sig = combined[combined["FDR q-val"] < args.fdr].copy()
    sig = sig.sort_values(["cancer_type", "NES"], ascending=[True, False])
    per_ct_path = args.output_dir / "gsea_sig_terms_per_cancer_type.csv"
    sig[["cancer_type", "Term", "term_name", "NES", "NOM p-val", "FDR q-val"]].to_csv(per_ct_path, index=False)
    print(f"Saved per-cancer-type significant terms to {per_ct_path}")

    # ── Clustermaps ────────────────────────────────────────────────────────
    print("Plotting NES clustermap (all values) …")
    plot_nes_clustermap(
        combined, term_names, cancer_types,
        output_path=args.output_dir / "gsea_nes_heatmap.png",
        fdr_threshold=args.fdr,
        min_sig_cts=1,
        sig_only=False,
    )
    print("Plotting NES clustermap (significant only) …")
    plot_nes_clustermap(
        combined, term_names, cancer_types,
        output_path=args.output_dir / "gsea_nes_heatmap_sigonly.png",
        fdr_threshold=args.fdr,
        min_sig_cts=1,
        sig_only=True,
    )


if __name__ == "__main__":
    main()
