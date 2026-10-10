"""Clustering metrics of the MLP and GONNECT models on held-out samples, for figures 2, 3 and S3.

The workbooks took SS, ARI and NMI and the per-cancer-type SS and k-NN purity
over all 9,797 samples, training data included, while every MSE and every
baseline metric come from the test split. This scores the saved embeddings so
that each of these metrics evaluates held-out samples only:

  SS, ARI, NMI    over the test split (Figure 2b-d, Figure 3 / S4). The metrics
                  are `rescore_baselines.clustering_metrics`, i.e. exactly what
                  the baselines are scored with: silhouette against the true
                  cancer types, k-means at k = number of classes for ARI and NMI.
  SS per type     per-sample silhouette over the same test split, averaged per
                  cancer type (Figure 2j, S3d): the per-type breakdown of SS, as
                  per-type MSE is of MSE.
  purity per type k-NN purity with everything held out, for k = 10, 20 and 30
                  (Figure 2k shows k = 10, S3a-c all three): each test sample's k
                  nearest neighbours are searched among the other samples of its
                  own test split, none of which were trained on. A type with fewer
                  than k test samples cannot fill k neighbours with its own kind, so
                  its purity would measure its size rather than its separation;
                  such a type is left blank (NaN) when that happens in any of the
                  five seeds, so every value shown averages the same five splits.
                  The larger k, the more types are blank: half of them at k = 30,
                  which is why Figure 2 shows k = 10. A `Random` column holds chance
                  level, the type's share of the test split with the sample itself
                  excluded.

The split is `gonnect.train.train.split_data` at the seed the run trained on;
its validation part is used by neither per-type metric. For AE_2.0 and AE_2.1
that is the run's own seed. The randomized AE_2.2 arm is numbered 22-26 but
trained on splits 2-6: each run's logged test loss reproduces on that split and
no other. The embeddings are the `latent_embeddings/` files, which cover all
9,797 rows, so split rows are selected by position. Per-type values are
averaged over the five seeds.

Writes to --out-dir:
  gonnect_clustering.csv       SS / ARI / NMI, long format (metric, method, repeat,
                               value, n_rows)
  per_type_ss.csv              cancer types x methods
  per_type_purity_within_test_k{k}.csv
                               cancer types x methods, plus `Random`, for k = 10, 20, 30;
                               NaN for blanked types
and prints SS / ARI / NMI and per-type SS against the all-sample values they replace.
Deterministic, and reads only shipped inputs.

Run from the repository root:
    pixi run python figures/src/prepare/test_split_metrics.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATA_DIR, EMBEDDINGS_DIR, PREP_OUT_DIR, TCGA_CSV
from _common import (EMBEDDING_MODELS, PURITY_K_VALUES, SEEDS_BY_VERSION, load_cancer_types,
                     load_embedding, read_per_cluster_workbook, read_xlsx_metrics)
from rescore_baselines import clustering_metrics
from gonnect.train.train import split_data

OUT_DIR = PREP_OUT_DIR / "test_split_metrics"
TRAIN_FRACTION = 0.7   # as every GONNECT run was trained; see Methods
METRICS = {"Silhouette": "SS", "ARI": "ARI", "NMI": "NMI"}
RANDOM = "Random"
# Run number minus split seed. AE_2.2.22-26 trained on splits 2-6; see the docstring.
SPLIT_OFFSET = {"2.0": 0, "2.1": 0, "2.2": 20}


def split_positions(labels: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of the (train, test) splits `split_data` makes at `seed`."""
    train, _, test = split_data(labels.to_frame(), 0, split=TRAIN_FRACTION, seed=seed)
    return train.index.to_numpy(), test.index.to_numpy()


def per_type(values: np.ndarray, labels: np.ndarray) -> pd.Series:
    return pd.Series(values).groupby(labels).mean()


def within_test_same(X: np.ndarray, y: np.ndarray, k_max: int) -> np.ndarray:
    """Whether each row's k_max nearest other rows share its label, nearest first.

    Neighbours are searched among the rows themselves, the row itself excluded.
    """
    _, ind = NearestNeighbors(n_neighbors=k_max + 1).fit(X).kneighbors(X)
    keep = ind != np.arange(len(X))[:, None]
    # With an exact duplicate the sample itself may fall outside its own k + 1
    # neighbours; drop the farthest instead, so every row keeps k_max.
    keep[keep.all(axis=1), -1] = False
    nbrs = ind[keep].reshape(len(X), k_max)
    return y[nbrs] == y[:, None]


def blank_small(series: pd.Series, counts: pd.Series, k: int) -> pd.Series:
    """NaN for every type with fewer than k samples in this test split."""
    return series.where(counts.reindex(series.index).fillna(0) >= k)


def compare(name: str, new: pd.DataFrame, old: pd.DataFrame) -> None:
    cols = [m for m in new.columns if m in old.columns]
    diff = new[cols] - old.loc[new.index, cols]
    rho = [new[m].corr(old.loc[new.index, m], method="spearman") for m in cols]
    print(f"{name}: mean |change| {diff.abs().values.mean():.3f}, max {diff.abs().values.max():.3f}"
          f" ({diff.abs().max().idxmax()}, {diff.abs().max(axis=1).idxmax()}),"
          f" Spearman over types {min(rho):.3f}-{max(rho):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    labels = load_cancer_types(args.data_dir / TCGA_CSV.name).astype(str)
    y = labels.to_numpy()
    ks = list(PURITY_K_VALUES)
    records, ss_per_type = [], {}
    purity_per_type: dict[int, dict[str, pd.Series]] = {k: {} for k in ks}
    chance: dict[int, dict[int, pd.Series]] = {k: {} for k in ks}
    for version, module, method in EMBEDDING_MODELS:
        ss_seeds, purity_seeds = [], {k: [] for k in ks}
        for run in SEEDS_BY_VERSION[version]:
            seed = run - SPLIT_OFFSET[version]
            X = load_embedding(args.data_dir / EMBEDDINGS_DIR.name, version, run,
                               module).astype(np.float64)
            if len(X) != len(labels):
                raise SystemExit(f"AE_{version}.{run}_{module}: {len(X)} rows, "
                                 f"{len(labels)} labels")
            train, test = split_positions(labels, seed)

            scores = clustering_metrics(X[test], y[test])
            records += [{"metric": name, "method": method, "repeat": f"run-{seed}",
                         "value": scores[key], "n_rows": len(test)}
                        for key, name in METRICS.items()]

            ss_seeds.append(per_type(silhouette_samples(X[test], y[test]), y[test]))
            same = within_test_same(X[test], y[test], max(ks))
            counts = pd.Series(y[test]).value_counts()
            for k in ks:
                purity_seeds[k].append(blank_small(per_type(same[:, :k].mean(axis=1), y[test]),
                                                   counts, k))
                chance[k].setdefault(seed, blank_small((counts - 1) / (len(test) - 1), counts, k))

            print(f"  {method:<16} run {run} (split {seed})  " + "  ".join(
                f"{name} {scores[key]:.3f}" for key, name in METRICS.items()), flush=True)
        ss_per_type[method] = pd.concat(ss_seeds, axis=1).mean(axis=1)
        # skipna=False: a type blanked in any seed stays blank.
        for k in ks:
            purity_per_type[k][method] = pd.concat(purity_seeds[k], axis=1).mean(axis=1, skipna=False)

    table = pd.DataFrame(records)
    ss_table = pd.DataFrame(ss_per_type)
    purity_tables = {k: pd.DataFrame(purity_per_type[k]).assign(
        **{RANDOM: pd.concat(chance[k].values(), axis=1).mean(axis=1, skipna=False)}) for k in ks}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "gonnect_clustering.csv", index=False)
    ss_table.rename_axis("cancer_type").to_csv(args.out_dir / "per_type_ss.csv")
    for k, frame in purity_tables.items():
        frame.rename_axis("cancer_type").to_csv(
            args.out_dir / f"per_type_purity_within_test_k{k}.csv")

    # What these replace: the workbooks' SS / ARI / NMI and per-type SS. The old
    # all-sample purity was computed on the fly and never deposited, so it has
    # no counterpart to print; preprint v3's panels are in the git history.
    workbook = read_xlsx_metrics(args.data_dir / "metrics" / "metric_data_TCGA_1000_30_new.xlsx")
    means = pd.concat({
        "all samples": workbook.groupby(["metric", "method"]).value.mean(),
        "test split": table.groupby(["metric", "method"]).value.mean(),
    }, axis=1).dropna()
    methods = [m for _, _, m in EMBEDDING_MODELS]
    print("\nMean over seeds, all samples against test split:")
    for metric in METRICS.values():
        print(f"\n{metric}")
        print(means.loc[metric].reindex([m for m in methods if m in means.loc[metric].index])
              .round(3).to_string())

    print()
    compare("SS per type", ss_table, read_per_cluster_workbook(
        args.data_dir / "metrics" / "mse_per_cluster_TCGA_1000_30.xlsx", verbose=False)["SS"].astype(float))
    print()
    for k, frame in purity_tables.items():
        blank = frame.index[frame.isna().all(axis=1)].tolist()
        print(f"purity within the test split, k={k}: {len(blank)} type(s) blank "
              f"(fewer than {k} test samples in some seed): {', '.join(blank) or 'none'}")
    print(f"\nWritten to {args.out_dir}")


if __name__ == "__main__":
    main()
