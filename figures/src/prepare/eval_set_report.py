"""Figure 4 on all samples against the test split, tabulated into figures/out/prepare/eval_set/.

Reads two fig4.py tables, one per --eval-set (both run with --pool seeds and the
same untrained references, each with that eval set), and writes

  violins.tsv            one row per violin: the red line (mean over seeds),
                         its permutation p and the untrained median, under
                         each eval set, and the change
  test_split_counts.tsv  per cancer type and split seed: the primary tumours in
                         that seed's test split (what --eval-set test scores),
                         next to the all-primary-tumour count (what 'all' scores
                         for GONNECT) and the baselines' published test rows of
                         every sample type
  baseline_rows.tsv      per seed: the baselines' test files against split_data's
                         test split (their id-based reload adds every row of a
                         test patient, so they hold more rows than the split)
  activation_rows.txt    the check that every go_term_activations file lists the
                         TCGA rows in their positional order, which is what
                         lets fig4 select a split by position

Run from the repository root:
    pixi run python figures/src/prepare/eval_set_report.py \
        --all figures/out/prepare/eval_set/all/fig4.csv \
        --test figures/out/prepare/eval_set/test/fig4.csv
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATA_DIR, PREP_OUT_DIR
import fig4
from _common import CANCER_TYPE_ORDER

OUT_DIR = PREP_OUT_DIR / "eval_set"
MIN_POSITIVES = 5


def violins(all_csv: Path, test_csv: Path) -> pd.DataFrame:
    keep = ["row", "label", "obs_mean_over_seeds", "perm_p", "stars", "untrained_median"]
    before = pd.read_csv(all_csv)[keep]
    after = pd.read_csv(test_csv)[keep]
    table = before.merge(after, on=["row", "label"], suffixes=("_all", "_test"), validate="one_to_one")
    table["delta"] = table.obs_mean_over_seeds_test - table.obs_mean_over_seeds_all
    table["above_untrained_all"] = table.obs_mean_over_seeds_all - table.untrained_median_all
    table["above_untrained_test"] = table.obs_mean_over_seeds_test - table.untrained_median_test
    return table


def counts(meta: pd.DataFrame, test_split: fig4.TestSplit) -> pd.DataFrame:
    primary = meta.sample_type.to_numpy() == fig4.EVAL_SAMPLE_TYPE
    out = {"all_primary": meta.cancer_type[primary].value_counts()}
    for seed in sorted(test_split.rows):
        rows = test_split.of(None, seed)
        out[f"test_any_type_s{seed}"] = meta.cancer_type.iloc[rows].value_counts()
        out[f"test_primary_s{seed}"] = meta.cancer_type.iloc[test_split.primary(None, seed)].value_counts()
    table = pd.DataFrame(out).fillna(0).astype(int)
    order = [ct for ct in CANCER_TYPE_ORDER if ct in table.index] + sorted(set(table.index) - set(CANCER_TYPE_ORDER))
    table = table.loc[order]
    primary_cols = [c for c in table.columns if c.startswith("test_primary_")]
    table["test_primary_min"] = table[primary_cols].min(axis=1)
    table["test_primary_mean"] = table[primary_cols].mean(axis=1)
    table["flag_lt5"] = table["test_primary_min"] < MIN_POSITIVES
    return table.rename_axis("cancer_type")


def baseline_rows(data_dir: Path, meta: pd.DataFrame, test_split: fig4.TestSplit) -> pd.DataFrame:
    """Per seed: what the baselines' published test files hold against split_data's test split."""
    method_dbs, _ = fig4.baseline_cfgs(data_dir)
    cfg = method_dbs[("VEGA", "hallmark")]
    out = []
    for seed in fig4.BASELINE_SEEDS:
        path = cfg["dir"] / f"run_seed-{seed}" / cfg["file_template"].format(variant="true")
        rows = pd.read_csv(path, usecols=[0]).iloc[:, 0].to_numpy(dtype=int)
        expected = test_split.of(None, seed)
        extra = np.setdiff1d(rows, expected)
        primary = meta.sample_type.to_numpy() == fig4.EVAL_SAMPLE_TYPE
        out.append({
            "seed": seed,
            "file_rows": len(rows),
            "split_test_rows": len(expected),
            "split_rows_missing_from_file": int(np.setdiff1d(expected, rows).size),
            "extra_rows_in_file": int(extra.size),
            "extra_rows_of_a_test_patient": int(np.isin(meta.patient_id.to_numpy()[extra],
                                                        meta.patient_id.to_numpy()[expected]).sum()),
            "scored_all (file rows, any type)": len(rows),
            "scored_test (split rows, primary)": int(primary[expected].sum()),
        })
    return pd.DataFrame(out)


def check_activation_rows(activations_dir: Path, meta: pd.DataFrame) -> list[str]:
    lines = []
    for path in sorted(activations_dir.glob("AE_2.[01].*_activations.csv.gz")):
        with gzip.open(path, "rt") as fh:
            ids = pd.read_csv(fh, usecols=["patient_id", "sample_type", "cancer_type"])
        same = (len(ids) == len(meta)
                and (ids.patient_id.to_numpy() == meta.patient_id.to_numpy()).all()
                and (ids.sample_type.to_numpy() == meta.sample_type.to_numpy()).all()
                and (ids.cancer_type.to_numpy() == meta.cancer_type.to_numpy()).all())
        lines.append(f"{path.name}: {len(ids)} rows, {'same order as TCGA csv' if same else 'ORDER DIFFERS'}")
        if not same:
            raise AssertionError(lines[-1])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--activations-dir", type=Path, default=None)
    parser.add_argument("--all", dest="all_csv", type=Path, required=True)
    parser.add_argument("--test", dest="test_csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(args.data_dir / "TCGA_complete_bp_top1k.csv.gz", "rt") as fh:
        meta = pd.read_csv(fh, usecols=["patient_id", "cancer_type", "sample_type"])
    test_split = fig4.TestSplit.build(meta, fig4.GONNECT_SEEDS)

    table = violins(args.all_csv, args.test_csv)
    table.to_csv(args.out_dir / "violins.tsv", sep="\t", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    ct = counts(meta, test_split)
    ct.to_csv(args.out_dir / "test_split_counts.tsv", sep="\t")
    print()
    print(ct.to_string())
    print(f"\ncancer types with fewer than {MIN_POSITIVES} primary tumours in some test split: "
          f"{', '.join(ct.index[ct.flag_lt5]) or 'none'}")

    accounting = baseline_rows(args.data_dir, meta, test_split)
    accounting.to_csv(args.out_dir / "baseline_rows.tsv", sep="\t", index=False)
    print()
    print(accounting.to_string(index=False))

    activations_dir = args.activations_dir or args.data_dir / "go_term_activations"
    lines = check_activation_rows(activations_dir, meta)
    (args.out_dir / "activation_rows.txt").write_text("\n".join(lines) + "\n")
    print(f"\n{len(lines)} activation files list the TCGA rows in positional order")


if __name__ == "__main__":
    main()
