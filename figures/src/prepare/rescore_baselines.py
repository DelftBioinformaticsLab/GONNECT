"""Rescore the OntoVAE and VEGA clustering metrics from their deposited latents.

The deposited baseline metrics took the silhouette against the k-means
clusters, where GONNECT's SS takes it against the true cancer types. The two
are not the same quantity -- on this data the k-means one runs higher -- so
Figures 2 and 3 compared baselines and GONNECT on different scales. The
runners now default to the label definition (`--silhouette-against`), but
retraining would not reproduce the deposited models: their initialisation was
never seeded. This scores the latents those models already produced instead.

The latents are the test-split embeddings in `baseline_activations/`:

  VEGA     z_test_<arm>_<gmt>.csv; the first column is the TCGA row position.
  OntoVAE  pathway_activities_test_<arm>.parquet, columns (term, depth, neuron).
           Upstream OntoVAE puts the latent in front of the decoder activations,
           so it is the depth-0 block -- all three neurons per term, as
           `get_embedding` returns it, not the per-term mean. The file carries
           no row positions; they are borrowed from the matching VEGA file, as
           `plot_activation_vs_gsea_baselines.load_ontovae_layer` does.

NMI and ARI are recomputed from the same latents rather than carried over, so
all three clustering metrics describe one embedding. That matters because the
latents cannot be tied to one deposited run for certain: both models sample
their embedding (the reparameterisation trick), and the NMI of the deposited
latents falls within k-means noise of both the flat and the nested metrics
files. k-means is the runners' -- k = number of classes, seed 42 -- with
n_init=10 spelled out, which is what scikit-learn 0.23 and 1.3 (the baseline
environments) default to and 1.4+ no longer does.

MSE needs reconstructions, which were not deposited, so it is carried over
from the nested metrics file unchanged.

Rows are scored as deposited, i.e. including the rows the baselines' ID-based
split pulled into both train and test; see baselines/make_splits.py.

Writes to --out-dir:
  ontovae_rand.txt, vega_rand.txt   test split only, in the shape of
                                    data/metrics/*_rand.txt, so the figure
                                    readers take them as they are
  rescored_metrics.csv              one row per (method, arm, seed): every
                                    rescored value next to the deposited one,
                                    plus the k-means silhouette of the same
                                    latents as a check on the definition

Deterministic, and reads only shipped inputs.

Run from the repository root:
    pixi run python figures/src/prepare/rescore_baselines.py
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_DIR, PREP_OUT_DIR, TCGA_CSV

OUT_DIR = PREP_OUT_DIR / "rescore_baselines"

SEEDS = [2, 3, 4, 5, 6]
ARMS = ("true", "degree_preserving", "random")
# GMT stem -> the method key the figures use for its true arm.
VEGA_GMTS = {"hallmark_v2026_1_Hs_uniprot": "vega_hallmark",
             "reactomes_uniprot": "vega_reactome674"}
METRIC_SEED = 42
KMEANS_N_INIT = 10


def read_arm_file(path: Path) -> dict:
    """``run-N: {...}`` lines of a nested metrics file, as {run-N: dict}."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            run_id, payload = line.split(":", 1)
            out[run_id.strip()] = ast.literal_eval(payload.strip())
    return out


def vega_latent(data_dir: Path, seed: int, arm: str, gmt: str) -> tuple[np.ndarray, np.ndarray]:
    """(latent, TCGA row positions) for one VEGA run."""
    z = pd.read_csv(data_dir / "baseline_activations" / "vega" / f"run_seed-{seed}"
                    / f"z_test_{arm}_{gmt}.csv",
                    index_col=0)
    return z.to_numpy(dtype=np.float64), z.index.to_numpy(dtype=int)


def ontovae_latent(data_dir: Path, seed: int, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """(latent, TCGA row positions) for one OntoVAE run; rows borrowed from VEGA."""
    acts = pd.read_parquet(data_dir / "baseline_activations" / "ontovae" / f"run_seed-{seed}"
                           / f"pathway_activities_test_{arm}.parquet")
    latent = acts.loc[:, [c for c in acts.columns if c[1] == 0]].to_numpy(dtype=np.float64)
    _, rows = vega_latent(data_dir, seed, arm, "hallmark_v2026_1_Hs_uniprot")
    if len(rows) != len(latent):
        raise SystemExit(f"OntoVAE seed {seed} {arm}: {len(latent)} rows, but the VEGA file "
                         f"it borrows row positions from has {len(rows)}")
    return latent, rows


def clustering_metrics(latent: np.ndarray, labels: np.ndarray) -> dict:
    """NMI and ARI of k-means against the labels; silhouette against both."""
    assignments = KMeans(n_clusters=len(np.unique(labels)), random_state=METRIC_SEED,
                         n_init=KMEANS_N_INIT).fit_predict(latent)
    return {
        "NMI": float(normalized_mutual_info_score(labels, assignments)),
        "ARI": float(adjusted_rand_score(labels, assignments)),
        "Silhouette": float(silhouette_score(latent, labels)),
        "Silhouette_kmeans": float(silhouette_score(latent, assignments)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    cancer_types = pd.read_csv(args.data_dir / TCGA_CSV.name,
                               usecols=["cancer_type"])["cancer_type"].to_numpy()
    deposited_ontovae = read_arm_file(args.data_dir / "metrics" / "ontovae_rand.txt")
    deposited_vega = read_arm_file(args.data_dir / "metrics" / "vega_rand.txt")

    ontovae_out, vega_out, rows = {}, {}, []

    def score(method, seed, arm, latent, positions, deposited):
        rescored = clustering_metrics(latent, cancer_types[positions])
        test = {k: rescored[k] for k in ("NMI", "ARI", "Silhouette")}
        test["mse"] = deposited["mse"]
        rows.append({"method": method, "arm": arm, "seed": seed, "n_rows": len(positions),
                     **{f"{k}_rescored": v for k, v in rescored.items()},
                     **{f"{k}_deposited": deposited[k] for k in ("NMI", "ARI", "Silhouette")},
                     "mse": deposited["mse"]})
        print(f"  {method:<17} {arm:<17} seed {seed}  SS {deposited['Silhouette']:.3f} -> "
              f"{rescored['Silhouette']:.3f}  (k-means SS of these latents "
              f"{rescored['Silhouette_kmeans']:.3f})", flush=True)
        return {"test": test}

    for seed in SEEDS:
        run = f"run-{seed}"
        ontovae_out[run] = {
            arm: score("ontovae", seed, arm, *ontovae_latent(args.data_dir, seed, arm),
                       deposited_ontovae[run][arm]["test"])
            for arm in ARMS
        }
        vega_out[run] = {
            gmt: {arm: score(method, seed, arm, *vega_latent(args.data_dir, seed, arm, gmt),
                             deposited_vega[run][gmt][arm]["test"])
                  for arm in ARMS}
            for gmt, method in VEGA_GMTS.items()
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, nested in (("ontovae_rand.txt", ontovae_out), ("vega_rand.txt", vega_out)):
        (args.out_dir / name).write_text(
            "".join(f"{run}: {payload}\n" for run, payload in nested.items()), encoding="utf-8")
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "rescored_metrics.csv", index=False)

    summary = (table.groupby(["method", "arm"], sort=False)
               [["Silhouette_deposited", "Silhouette_rescored", "Silhouette_kmeans_rescored",
                 "NMI_deposited", "NMI_rescored", "ARI_deposited", "ARI_rescored"]].mean())
    print("\nMean over seeds:")
    print(summary.round(3).to_string())
    print(f"\nWritten to {args.out_dir}")


if __name__ == "__main__":
    main()
