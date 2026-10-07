"""
Figure 4 under one gene-set definition: rerun GSEA and rescore every GONNECT layer.

Figure 4 scores GONNECT's nodes against two GSEA references built from two
gene-set definitions. Panels a and b (every layer, the 109 bottleneck terms at
enc L3 / dec L5 included) read gsea_gonnect_layers/, whose sets come from
run_gsea.build_gene_sets: a term's direct annotations, read from encoder layers
0 and 1 only. Panel c reads gsea_gonnect_bottleneck/, whose sets are the
bottleneck terms' receptive fields (run_gsea_bottleneck.build_receptive_fields).

This step
  1. traces both definitions through every encoder layer (gsea_gene_sets.py)
     and audits build_gene_sets against the full trace;
  2. reruns GSEA exactly as run_gsea.py does (pre-ranked on deg_results.csv,
     mean_diff x -log10 padj, 1000 permutations, seed 42, set size 3-500) on
       - build_gene_sets, unchanged   (must reproduce gsea_gonnect_layers/),
       - the traced direct sets       -> <prep-out>/gsea_direct_traced/,
       - the receptive fields         -> <prep-out>/gsea_receptive_fields/
         (on the bottleneck it must reproduce gsea_gonnect_bottleneck/);
  3. scores enc L0-L3 and dec L5-L8 of GONNECT and GONNECT-SL against every
     reference with Figure 4's per-seed statistic (fig4 --pool seeds: the mean
     of the five per-seed median-per-CT Spearman r, with its seed-wise
     column-shuffle null), on each reference's own terms and on the terms all
     references at that layer share.

The AUCs are fig4's own (gonnect_layer_full_auc), cached in --cache-dir.

Usage:
  python gsea_consistency.py --data-dir MAIN/figures/data \
      --activations-dir MAIN/figures/out/prepare/decoder_reextracted/go_term_activations \
      [--cache-dir DIR] [--output-dir DIR] [--prep-out DIR] [--n-perms 1000] [--threads 4]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fig4
import gsea_gene_sets as gs
import run_gsea
from _paths import DATA_DIR, PREP_OUT_DIR

MODELS = {"2.0": "GONNECT", "2.1": "GONNECT-SL"}
MODULE_LAYERS = {"encoder": [0, 1, 2, 3], "decoder": [5, 6, 7, 8]}
GSEA_DIRS = {"build_gene_sets": "gsea_build_gene_sets_rerun",
             "direct": "gsea_direct_traced",
             "receptive_field": "gsea_receptive_fields"}
#: reference name -> (GSEA csv, gene-set definition it was built from)
REF_ORDER = ["alllayer", "bottleneck", "direct_traced", "receptive_field"]


def run_definition(sets: dict[str, set], deg: pd.DataFrame, out_csv: Path,
                   permutations: int, threads: int) -> pd.DataFrame:
    """run_gsea.main's GSEA loop, on the given sets."""
    gene_sets = {t: sorted(g) for t, g in sets.items() if g}
    present = set(deg["cancer_type"].unique())
    cts = [ct for ct in run_gsea.CANCER_TYPE_ORDER if ct in present] + sorted(present - set(run_gsea.CANCER_TYPE_ORDER))
    res = []
    for ct in cts:
        res.append(run_gsea.run_gsea_for_cancer_type(ct, deg[deg["cancer_type"] == ct], gene_sets,
                                                     out_csv.parent, permutations, threads))
    combined = pd.concat(res, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    return combined


def compare_nom_p(new: pd.DataFrame, deposited: pd.DataFrame) -> dict:
    m = new.merge(deposited, on=["cancer_type", "Term"], suffixes=("", "_dep"))
    d = (m["NOM p-val"].astype(float) - m["NOM p-val_dep"].astype(float)).abs()
    return {"rows_compared": len(m), "rows_deposited": len(deposited), "max_abs_diff_nom_p": float(d.max())}


def covered_terms(enr: pd.DataFrame, terms: list[str]) -> list[str]:
    return sorted(set(enr.columns) & set(terms))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--activations-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=PREP_OUT_DIR / "gsea_consistency" / "auc_cache")
    p.add_argument("--output-dir", type=Path, default=PREP_OUT_DIR / "gsea_consistency")
    p.add_argument("--prep-out", type=Path, default=PREP_OUT_DIR,
                   help="where gsea_receptive_fields/ and gsea_direct_traced/ go")
    p.add_argument("--n-perms", type=int, default=1000, help="column-shuffle permutations per statistic")
    p.add_argument("--gsea-permutations", type=int, default=run_gsea.PERMUTATIONS)
    p.add_argument("--threads", type=int, default=run_gsea.THREADS)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--rerun-gsea", action="store_true",
                   help="recompute GSEA results that already exist (default: reuse them)")
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    hl_path = args.data_dir / "hard_links.csv"

    # ── 1. gene sets ──────────────────────────────────────────────────────
    audit, extra = gs.audit(hl_path)
    audit.to_csv(out / "build_gene_sets_audit.tsv", sep="\t", index=False)
    sets_df = gs.write_sets(hl_path, out / "gene_sets_per_term.tsv")
    sets = gs.all_sets(hl_path)
    enc_layer = gs.term_layers(gs.encoder_edges(hl_path))
    layer_maps = {m: fig4.build_layer_map(hl_path, m) for m in MODULE_LAYERS}
    if any(layer_maps["decoder"].get(t) != 8 - L for t, L in enc_layer.items()):
        raise ValueError("decoder layer of a GO term is not 8 - its encoder layer")
    print(audit.T.to_string())
    print("missed pairs by entry layer:", extra["missed_pairs_by_entry_layer"])

    # ── 2. GSEA ───────────────────────────────────────────────────────────
    deg = pd.read_csv(args.data_dir / "deg_results.csv")
    gsea_csv = {}
    checks = {}
    for definition, sub in GSEA_DIRS.items():
        base = out if definition == "build_gene_sets" else args.prep_out
        path = base / sub / "gsea_results.csv"
        t0 = time.time()
        if path.exists() and not args.rerun_gsea:   # resumable: each definition is written whole
            res = pd.read_csv(path)
        else:
            res = run_definition(sets[definition], deg, path, args.gsea_permutations, args.threads)
        print(f"GSEA {definition}: {res['Term'].nunique()} terms x {res['cancer_type'].nunique()} CTs "
              f"in {time.time() - t0:.0f}s -> {path}")
        gsea_csv[definition] = path
    dep_layers = pd.read_csv(args.data_dir / "gsea_gonnect_layers" / "gsea_results.csv")
    dep_bn = pd.read_csv(args.data_dir / "gsea_gonnect_bottleneck" / "gsea_results.csv")
    checks["build_gene_sets rerun vs gsea_gonnect_layers"] = compare_nom_p(
        pd.read_csv(gsea_csv["build_gene_sets"]), dep_layers)
    checks["receptive_field rerun vs gsea_gonnect_bottleneck"] = compare_nom_p(
        pd.read_csv(gsea_csv["receptive_field"]), dep_bn)
    for k, v in checks.items():
        print(f"  check {k}: {v}")

    refs = {
        "alllayer": fig4.load_enrichment_pivot(args.data_dir / "gsea_gonnect_layers" / "gsea_results.csv",
                                               fig4.CANCER_TYPE_ORDER),
        "bottleneck": fig4.load_enrichment_pivot(args.data_dir / "gsea_gonnect_bottleneck" / "gsea_results.csv",
                                                 fig4.CANCER_TYPE_ORDER),
        "direct_traced": fig4.load_enrichment_pivot(gsea_csv["direct"], fig4.CANCER_TYPE_ORDER),
        "receptive_field": fig4.load_enrichment_pivot(gsea_csv["receptive_field"], fig4.CANCER_TYPE_ORDER),
    }
    ref_definition = {"alllayer": "build_gene_sets", "bottleneck": "receptive_field",
                      "direct_traced": "direct", "receptive_field": "receptive_field"}

    # ── coverage and set sizes per layer ─────────────────────────────────
    cov_rows = []
    for L in range(4):
        terms = [t for t, l in enc_layer.items() if l == L]
        for ref in REF_ORDER:
            cov = covered_terms(refs[ref], terms)
            sizes = [len(sets[ref_definition[ref]][t]) for t in cov]
            all_sizes = [len(sets[ref_definition[ref]].get(t, ())) for t in terms]
            cov_rows.append({"encoder_layer": L, "decoder_layer": 8 - L, "reference": ref,
                             "definition": ref_definition[ref], "go_terms": len(terms),
                             "gsea_covered": len(cov),
                             "median_size_covered": float(np.median(sizes)) if sizes else np.nan,
                             "median_size_all_terms": float(np.median(all_sizes))})
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(out / "coverage_per_layer.tsv", sep="\t", index=False)
    print(cov_df.to_string())

    # ── 3. scoring ────────────────────────────────────────────────────────
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for version, model in MODELS.items():
        for module, layers in MODULE_LAYERS.items():
            for L in layers:
                frames = fig4.gonnect_layer_frames(args.activations_dir, version, module, L,
                                                   layer_maps[module], args.cache_dir, True)
                layer_terms = list(frames[0].columns)
                avail = [r for r in REF_ORDER if covered_terms(refs[r], layer_terms)]
                common = sorted(set.intersection(*[set(covered_terms(refs[r], layer_terms)) for r in avail]))
                for ref in avail:
                    for scope, terms in [("own", covered_terms(refs[ref], layer_terms)), ("common", common)]:
                        enr = refs[ref][terms]
                        res = fig4.seed_mean_entry(frames, enr, args.n_perms, args.rng_seed)
                        rows.append({
                            "model": model, "module": module, "layer": f"{module[:3]} L{L}",
                            "reference": ref, "scope": scope, "n_terms": len(terms),
                            "stat_mean_over_seeds": res["obs_pooled"],
                            "perm_p": fig4.perm_pval(res["obs_pooled"], res["null"]),
                            "null_mean": float(np.mean(res["null"])),
                            "per_seed": ";".join(f"{v:.4f}" for v in res["obs_per_seed"]),
                        })
                        print(f"  {model} {module} L{L} {ref:16s} {scope:6s} n={len(terms):3d} "
                              f"r={res['obs_pooled']:+.3f}", flush=True)
    stats = pd.DataFrame(rows)
    stats.to_csv(out / "stats_per_layer.tsv", sep="\t", index=False)
    wide = stats.pivot_table(index=["model", "layer", "scope"], columns="reference",
                             values="stat_mean_over_seeds", sort=False)
    wide = wide.reindex(columns=[r for r in REF_ORDER if r in wide.columns])
    wide.to_csv(out / "stats_wide.tsv", sep="\t")
    pd.Series({k: str(v) for k, v in checks.items()}).to_csv(out / "reproduction_checks.tsv", sep="\t",
                                                            header=False)
    print(wide.round(3).to_string())


if __name__ == "__main__":
    main()
