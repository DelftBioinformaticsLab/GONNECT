"""Figure 4's statistic on untrained OntoVAE and VEGA: their structural baselines.

The counterpart of untrained_control.py for the baseline panels (d and e). It
scores the untrained models that baselines/ontovae/untrained_ontovae.py and
baselines/vega/untrained_vega.py export: 10 initializations on each split seed,
50 in all, never trained. That is on the true gene-set graph for OntoVAE, and
on all three of VEGA's graphs (true, DPR, FR). It scores the five shipped
trained seeds the same way, so the two sides of every comparison are computed
by one code path.

VEGA masks only its decoder, so an untrained VEGA's latent (the posterior mean
of a fully connected encoder) is the same on every graph: its DPR and FR
references equal its true one, initialization by initialization.

The statistic is Figure 4's per-seed dot for a baseline (fig4.baseline_auc_one_seed
and fig4.per_ct_median_r). Per (cancer type, pathway), the one-vs-rest ROC-AUC
over the seed's test samples, symmetrized as max(AUC, 1 - AUC). Per cancer type,
the Spearman r across pathways against the GSEA -log10(NOM p) of the same gene
sets. Then the median over cancer types. For OntoVAE this is per layer, L0-L10
as Figure 4 shows them, each term being the mean of its three neurons. The AUC
here is computed on ranks for every pathway at once, which equals
roc_auc_score. The step checks that against fig4's own loop on a trained seed
before it scores anything.

Both exports take the untrained latent at its posterior mean rather than a
sample (see their docstrings), so the baseline is the wiring's best case.

Writes, to figures/out/prepare/untrained_baselines/:

  untrained_baselines_values.tsv    one row per (row, label, state, seed, init)
  untrained_baselines_summary.tsv   per violin: trained against untrained, with
                                    a two-sided Mann-Whitney p and its
                                    Benjamini-Hochberg q across the table
  fig4_untrained_baselines.tsv      the untrained reference for fig4.py --untrained

Run from the repository root, after the two exports:
    pixi run python figures/src/prepare/untrained_baselines.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu, rankdata

PREPARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARE_DIR))
sys.path.insert(0, str(PREPARE_DIR.parent))
from _paths import DATA_DIR, PREP_OUT_DIR
import fig4
from _common import CANCER_TYPE_ORDER

REPO_ROOT = PREPARE_DIR.parents[2]
UNTRAINED_DIR = REPO_ROOT / "out" / "baselines" / "untrained"
OUT_DIR = PREP_OUT_DIR / "untrained_baselines"

SEEDS = [2, 3, 4, 5, 6]
ONTOVAE_LAYERS = range(0, 11)   # the layers Figure 4 shows
VEGA_DBS = {"hallmark": "hallmark_v2026_1_Hs_uniprot", "reactomes": "reactomes_uniprot"}
VEGA_VARIANTS = ["true", "degree_preserving", "random"]   # the graphs Figure 4 shows, in its order
TRAINED = "trained"
UNTRAINED = "untrained"


def auc_frame(values: np.ndarray, cancer_types_of_rows: np.ndarray, columns) -> pd.DataFrame:
    """fig4.baseline_auc_one_seed on ranks: every pathway at once, float32 as fig4 stores it."""
    present = [ct for ct in pd.unique(cancer_types_of_rows) if ct is not None]
    cancer_types = ([ct for ct in CANCER_TYPE_ORDER if ct in present]
                    + sorted(set(present) - set(CANCER_TYPE_ORDER)))
    ranks = rankdata(values, axis=0)
    out = np.zeros((len(cancer_types), values.shape[1]), dtype=np.float32)
    for i, cancer_type in enumerate(cancer_types):
        positive = cancer_types_of_rows == cancer_type
        n_pos = positive.sum()
        n_neg = len(positive) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        auc = (ranks[positive].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        out[i] = np.maximum(auc, 1 - auc)
    return pd.DataFrame(out, index=cancer_types, columns=list(columns))


def statistic(auc: pd.DataFrame, enrichment: pd.DataFrame, n_perms: int, rng_seed: int) -> tuple[float, float, int]:
    """(Figure 4's median per-cancer-type r, its column-shuffle p, the number of terms scored)."""
    cancer_types = [ct for ct in auc.index if ct in enrichment.index]
    terms = sorted(set(auc.columns) & set(enrichment.columns))
    act = auc.loc[cancer_types, terms].to_numpy(dtype=float)
    enr = enrichment.loc[cancer_types, terms].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # constant pathways leave some per-term r undefined; unused here
        observed = fig4._summary_stats(act, enr)[2]
    null = fig4.run_null(act, enr, "col", n_perms, rng_seed)["per_ct"]
    return observed, fig4.perm_pval(observed, null), len(terms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--untrained-dir", type=Path, default=UNTRAINED_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--inits-per-seed", type=int, default=10)
    parser.add_argument("--n-perms", type=int, default=1000)
    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.data_dir / "TCGA_complete_bp_top1k.csv.gz", usecols=["cancer_type"])
    cancer_type_of_row = meta["cancer_type"].to_numpy()
    method_dbs, ontovae_cfg = fig4.baseline_cfgs(args.data_dir)
    enrichment = {("VEGA", db): fig4.load_enrichment_pivot(method_dbs[("VEGA", db)]["gsea_csv"], CANCER_TYPE_ORDER)
                  for db in VEGA_DBS}
    enrichment["OntoVAE"] = fig4.load_enrichment_pivot(ontovae_cfg["gsea_csv"], CANCER_TYPE_ORDER)
    shipped = args.data_dir / "baseline_activations"

    # The published OntoVAE term -> layer map, read off the shipped activation columns (term, layer, neuron)
    columns = pd.read_parquet(shipped / "ontovae" / f"run_seed-{SEEDS[0]}" / "pathway_activities_test_true.parquet"
                              ).columns
    layer_of = {term: layer for term, layer, _ in columns}

    # The rank AUC against fig4's roc_auc_score loop, on a trained seed
    reference = fig4.baseline_auc_one_seed(method_dbs[("VEGA", "hallmark")], "true",
                                           dict(enumerate(cancer_type_of_row)), SEEDS[0])
    acts, rows, pathways = fig4.load_vega_csv(
        shipped / "vega" / f"run_seed-{SEEDS[0]}" / f"z_test_true_{VEGA_DBS['hallmark']}.csv",
        seed=SEEDS[0], cfg=None, variant="true")
    difference = np.abs(auc_frame(acts, cancer_type_of_row[rows], pathways).to_numpy() - reference.to_numpy()).max()
    if difference > 1e-6:
        raise AssertionError(f"rank AUC differs from fig4.baseline_auc_one_seed by {difference:.2e}")
    print(f"rank AUC matches fig4.baseline_auc_one_seed (max |diff| {difference:.1e})", flush=True)

    rows_out = []

    def record(row, label, state, seed, init, result):
        observed, p, n_terms = result
        rows_out.append({"row": row, "label": label, "state": state, "seed": seed, "init": init,
                         "n_terms": n_terms, "median_r": observed, "col_shuffle_p": p})

    def score_vega(path, state, seed, init):
        for db, stem in VEGA_DBS.items():
            for variant in VEGA_VARIANTS:
                acts, rows, pathways = fig4.load_vega_csv(path(variant, stem), seed=seed, cfg=None, variant=variant)
                label = f"{fig4.VEGA_DB_DISPLAY[db]} {fig4.RANDOMIZATION_DISPLAY[variant]}"
                record("VEGA", label, state, seed, init,
                       statistic(auc_frame(acts, cancer_type_of_row[rows], pathways), enrichment[("VEGA", db)],
                                 args.n_perms, args.rng_seed))

    def score_ontovae(frame: pd.DataFrame, rows: np.ndarray, state, seed, init):
        auc = auc_frame(frame.to_numpy(dtype=np.float64), cancer_type_of_row[rows], frame.columns)
        for layer in ONTOVAE_LAYERS:
            record("OntoVAE", f"L{layer}", state, seed, init,
                   statistic(auc[[t for t in auc.columns if layer_of.get(t) == layer]], enrichment["OntoVAE"],
                             args.n_perms, args.rng_seed))

    for seed in SEEDS:
        # Trained: the shipped activations, loaded the way fig4 loads them
        score_vega(lambda variant, stem: shipped / "vega" / f"run_seed-{seed}" / f"z_test_{variant}_{stem}.csv",
                   TRAINED, seed, None)
        # fig4.load_ontovae_layer itself, one layer at a time: it averages each term's three neurons in float64,
        # and averaging in float32 instead flips enough near-ties in the AUC ranks to move the statistic by ~1e-3
        path = shipped / "ontovae" / f"run_seed-{seed}" / "pathway_activities_test_true.parquet"
        for layer in ONTOVAE_LAYERS:
            acts, rows, terms = fig4.load_ontovae_layer(path, seed=seed, cfg={**ontovae_cfg, "layer": layer},
                                                        variant="true")
            record("OntoVAE", f"L{layer}", TRAINED, seed, None,
                   statistic(auc_frame(acts, cancer_type_of_row[rows], terms), enrichment["OntoVAE"],
                             args.n_perms, args.rng_seed))

        # Untrained: the exports, one directory per initialization
        for j in range(args.inits_per_seed):
            init = SEEDS.index(seed) * args.inits_per_seed + j
            init_dir = args.untrained_dir
            score_vega(lambda variant, stem: (init_dir / "vega" / f"run_seed-{seed}" / f"init-{init}"
                                              / f"z_test_{variant}_{stem}.csv"),
                       UNTRAINED, seed, init)
            frame = pd.read_parquet(init_dir / "ontovae" / f"run_seed-{seed}" / f"init-{init}"
                                    / "pathway_activities_test_true.parquet")
            score_ontovae(frame, frame.index.to_numpy(), UNTRAINED, seed, init)
        print(f"seed {seed}: trained and {args.inits_per_seed} untrained models scored", flush=True)

    values = pd.DataFrame(rows_out)
    values.to_csv(args.out_dir / "untrained_baselines_values.tsv", sep="\t", index=False, float_format="%.6g")

    summary = []
    for (row, label), group in values.groupby(["row", "label"], sort=False):
        trained = group[group.state == TRAINED].median_r
        untrained = group[group.state == UNTRAINED]
        summary.append({
            "row": row, "label": label, "n_terms": int(group.n_terms.iloc[0]),
            "trained_mean": trained.mean(), "trained_sd": trained.std(),
            "untrained_mean": untrained.median_r.mean(), "untrained_sd": untrained.median_r.std(),
            "untrained_q025": untrained.median_r.quantile(0.025), "untrained_q975": untrained.median_r.quantile(0.975),
            "difference": trained.mean() - untrained.median_r.mean(),
            "mannwhitney_p": mannwhitneyu(trained, untrained.median_r, alternative="two-sided").pvalue,
            "untrained_colshuffle_sig": (untrained.col_shuffle_p < 0.05).mean(),
        })
    summary = pd.DataFrame(summary)
    summary.insert(summary.columns.get_loc("mannwhitney_p") + 1, "mannwhitney_q",
                   false_discovery_control(summary.mannwhitney_p, method="bh"))
    summary.to_csv(args.out_dir / "untrained_baselines_summary.tsv", sep="\t", index=False, float_format="%.6g")
    (values[values.state == UNTRAINED][["row", "label", "init", "n_terms", "median_r"]]
     .to_csv(args.out_dir / "fig4_untrained_baselines.tsv", sep="\t", index=False, float_format="%.6g"))

    # The trained side must be what Figure 4 plots as dots
    published = pd.read_csv(PREP_OUT_DIR.parent / "fig4.csv").set_index(["row", "label"])
    for (row, label), group in values[values.state == TRAINED].groupby(["row", "label"], sort=False):
        dots = published.loc[(row, label), [f"seed_{i}" for i in range(len(SEEDS))]].to_numpy(dtype=float)
        if not np.allclose(group.sort_values("seed").median_r.to_numpy(), dots, atol=1e-9, equal_nan=True):
            print(f"WARNING: trained {row} {label} differs from fig4.csv's per-seed dots")

    print(f"\nwrote {args.out_dir}\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
