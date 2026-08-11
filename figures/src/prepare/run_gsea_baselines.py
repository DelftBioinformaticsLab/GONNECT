"""
Pre-ranked GSEA against a GMT gene-set database, for use with VEGA / OntoVAE
baselines whose latent dimensions correspond 1:1 to named pathways.

Equivalent to run_gsea_bottleneck.py but takes the gene sets straight from a
GMT file instead of building receptive fields from data/hard_links.csv.

Usage:
  pixi run python fig_deg/run_gsea_baselines.py \\
      --gmt        figures/data/ontologies/hallmark_v2026_1_Hs_uniprot.gmt \\
      --output-csv figures/out/prepare/gsea_baselines/gsea_results_hallmark.csv
"""

import argparse
import warnings
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

from _paths import DEG_CSV

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


def load_gmt(gmt_path: Path) -> dict[str, list[str]]:
    gene_sets = {}
    with open(gmt_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, _desc, *genes = parts
            genes = [g for g in genes if g]
            if genes:
                gene_sets[name] = genes
    return gene_sets


def run_gsea_for_cancer_type(ct, deg_ct, gene_sets, permutations, threads):
    deg_ct = deg_ct.copy()
    deg_ct["rank_score"] = deg_ct["mean_diff"] * (
        -np.log10(np.clip(deg_ct["padj"], 1e-300, 1.0))
    )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deg-csv",      type=Path,
                        default=DEG_CSV)
    parser.add_argument("--gmt",          type=Path, required=True,
                        help="GMT file mapping pathway -> gene IDs (UniProt).")
    parser.add_argument("--output-csv",   type=Path, required=True)
    parser.add_argument("--permutations", type=int,  default=PERMUTATIONS)
    parser.add_argument("--threads",      type=int,  default=THREADS)
    parser.add_argument("--fdr",          type=float, default=0.25)
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading gene sets from {args.gmt} …")
    gene_sets = load_gmt(args.gmt)
    sizes = [len(v) for v in gene_sets.values()]
    print(f"  {len(gene_sets)} pathways, gene-set sizes: "
          f"min={min(sizes)}, median={int(np.median(sizes))}, max={max(sizes)}\n")

    print(f"Loading DEG results from {args.deg_csv} …")
    deg = pd.read_csv(args.deg_csv)
    present = set(deg["cancer_type"].unique())
    cancer_types = [ct for ct in CANCER_TYPE_ORDER if ct in present] + \
                   sorted(present - set(CANCER_TYPE_ORDER))
    print(f"  {len(cancer_types)} cancer types, {deg['gene'].nunique()} genes\n")

    all_results = []
    for ct in cancer_types:
        deg_ct = deg[deg["cancer_type"] == ct]
        print(f"  GSEA {ct} …", end=" ", flush=True)
        res = run_gsea_for_cancer_type(
            ct, deg_ct, gene_sets, args.permutations, args.threads,
        )
        n_sig = (res["FDR q-val"] < args.fdr).sum()
        print(f"{len(res)} terms tested, {n_sig} FDR<{args.fdr}")
        all_results.append(res)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(args.output_csv, index=False)
    print(f"\nSaved GSEA results to {args.output_csv}")


if __name__ == "__main__":
    main()
