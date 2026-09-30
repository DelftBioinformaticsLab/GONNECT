"""All four metrics of the runs whose embeddings were written on the cluster, on held-out samples.

Two presets, matching `embed_cluster_runs.py`, which runs next to the checkpoints
(never deposited): it checks each run against its training log, then writes the
test rows' embeddings to one .npz per preset, with a report beside it.

  s1    Supplementary Figure S1's ct=5, ct=10 and 2,000-gene runs (AE_5.x, AE_4.x,
        AE_6.x). The main configuration's runs are Figure 2's, which
        `test_split_metrics.py` scores.
  fig3  Figure 3's fully random arm (AE_10.2) and its two randomized soft-link arms
        (AE_11.1 degree-preserving, AE_10.1 fully random). The degree-preserving
        fixed-link arm (AE_2.2) has deposited embeddings, so `test_split_metrics.py`
        scores it.

This step scores those embeddings with Figure 2's metric function
(`rescore_baselines.clustering_metrics`). Only runs that passed both of the report's
checks are kept. The split check is that the run's reconstruction MSE on the held-out
rows equals its log. The workbook check is that the log holds the MSE the figure's
workbook has for that setting, model and seed. That also keeps each figure's set of
bars as it was: a run the workbook has no value for is left out.

MSE is written as well, taken from the log the way Figure 2 takes it: the masked test
loss for fixed runs, the plain MSE for soft-link runs. The workbooks mostly did the
same, but for S1's ct=10 SL-enc and SL-dec they took the test loss including the
soft-link penalty, which puts those bars 0.008-0.012 too high. Each preset prints
which runs' workbook MSE differs from the log's.

Writes --out-dir/sweep_metrics.csv (s1) or randomized_metrics.csv (fig3), long format
(setting, metric, method, repeat, value, n_rows), and prints each setting's means next
to the workbook's. Deterministic.

Run from the repository root:
    pixi run python figures/src/prepare/cluster_test_metrics.py s1
    pixi run python figures/src/prepare/cluster_test_metrics.py fig3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATA_DIR, PREP_OUT_DIR
from _common import read_xlsx_metrics
from rescore_baselines import clustering_metrics

OUT_DIR = PREP_OUT_DIR / "cluster_test_metrics"
EMBEDDINGS = DATA_DIR / "latent_embeddings" / "cluster_test_split"
METRICS = {"Silhouette": "SS", "ARI": "ARI", "NMI": "NMI"}   # clustering_metrics key -> figure name
PRESETS = {  # preset: (output file, {setting: workbook})
    "s1": ("sweep_metrics.csv", {"ct=5": "metric_data_TCGA_1000_5.xlsx",
                                 "ct=10": "metric_data_TCGA_1000_10.xlsx",
                                 "2k genes": "metric_data_TCGA_2000_30.xlsx"}),
    "fig3": ("randomized_metrics.csv", {"ct=30": "metric_data_TCGA_1000_30_new.xlsx"}),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("preset", choices=sorted(PRESETS))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS,
                        help="Folder holding embed_cluster_runs.py's .npz and report")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_name, workbooks = PRESETS[args.preset]

    report = json.loads((args.embeddings / f"{args.preset}_report.json").read_text())
    arrays = np.load(args.embeddings / f"{args.preset}_test_embeddings.npz")
    kept = [r for r in report if r["status"] == "ok" and r["split_ok"] and r["workbook_ok"]]
    dropped = sorted({r["run"] for r in report} - {r["run"] for r in kept})
    print(f"{len(kept)} runs pass both checks; left out: {', '.join(dropped) or 'none'}")
    off = [r["run"] for r in kept if abs(r["mse"] - r["workbook"]) > 1e-12]
    print(f"workbook MSE differs from the log's: {', '.join(off) or 'none'}")

    records = []
    for r in kept:
        run = r["run"]
        scores = clustering_metrics(arrays[run].astype(np.float64), arrays[f"{run}.labels"])
        values = {"MSE": r["mse"], **{name: scores[key] for key, name in METRICS.items()}}
        records += [{"setting": r["setting"], "metric": name, "method": r["method"],
                     "repeat": f"run-{r['seed']}", "value": value, "n_rows": r["n_test"]}
                    for name, value in values.items()]
    table = pd.DataFrame(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / out_name, index=False)

    for setting, workbook in workbooks.items():
        old = read_xlsx_metrics(args.data_dir / "metrics" / workbook, strict_repeats=False)
        new = table[table["setting"] == setting]
        means = pd.concat({
            "all samples": old.groupby(["metric", "method"]).value.mean(),
            "test split": new.groupby(["metric", "method"]).value.mean(),   # MSE: from the logs
        }, axis=1).dropna()
        print(f"\n{setting}, mean over seeds:")
        print(means.round(3).to_string())
    print(f"\nWritten to {args.out_dir / out_name}")


if __name__ == "__main__":
    main()
