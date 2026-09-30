"""
Train OntoVAE on the TCGA splits, across the three GO-graph arms (true, fully random, and degree_preserving).

Writes, per seed:

  metrics.txt                                  run-<seed>: {arm: {split: {metric: ...}}}
  run_seed-<seed>/pathway_activities_test_<arm>.parquet
  run_seed-<seed>/models/best_model_<arm>.pt

Upstream OntoVAE is used unpatched -- see `ontovae_compat.py` for the two adaptations
layered on top, and for why the fork's `mask_list` change is not among them.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Appended, not prepended: goatools pulls in a top-level module named `compat`,
# which a prepended path here would shadow.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(os.path.dirname(_HERE))  # baselines/, for the shared modules
from ontovae_compat import (  # noqa: E402
    ensure_onto_vae_importable,
    force_cpu,
    gene_index_map,
    install_drop_last_loader,
    install_memory_efficient_adamw,
)
from mask_randomization import randomize_mask_stack  # noqa: E402

ARMS = ("true", "random", "degree_preserving")
METRIC_SEED = 42  # KMeans, matching the original runs and figures/src/_common

# Genes absent from GONNECT's processed ontology. The published runs dropped
# these from the test split only; see `load_split`.
DROP_GENES = (
    "A6NC42", "O60635", "O95857", "P00450", "P02814", "P05976", "P11686", "P12882",
    "P18283", "P22792", "P30408", "P48230", "Q4ZG55", "Q53G44", "Q5T7N2", "Q5TH69",
    "Q685J3", "Q7Z7J9", "Q86XP6", "Q8IWL1", "Q8IWL2", "Q8IZW8", "Q8WXD2", "Q969L2",
    "Q96KN4", "Q9BYZ8", "Q9NR99", "Q9NRC9", "Q9UKX2",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expr-data", default="figures/data/TCGA_complete_bp_top1k.csv.gz")
    p.add_argument("--split-dir", default="out/baselines/splits",
                   help="Where make_splits.py wrote the sample IDs; read-only")
    p.add_argument("--out-dir", default="out/baselines/ontovae",
                   help="Where this run's metrics, models and activations go")
    p.add_argument("--obo", default="data/go-basic.obo")
    p.add_argument("--gene-annot", required=True, help="OntoVAE gene->GO annotation table")
    p.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--top-thresh", type=int, default=1000)
    p.add_argument("--bottom-thresh", type=int, default=30)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)  # float, not int -- see module docstring
    p.add_argument("--kl-coeff", type=float, default=1e-4)
    p.add_argument("--id-col", default="patient_id")
    p.add_argument("--label-col", default="cancer_type")
    p.add_argument("--n-nan-cols", type=int, default=5,
                   help="Leading metadata columns in the expression matrix")
    p.add_argument("--mse-gene-mask", choices=("published", "corrected"), default="published",
                   help="Gene selection for the reconstruction MSE; see masked_mse")
    p.add_argument("--silhouette-against", choices=("labels", "kmeans"), default="labels",
                   help="What the silhouette scores; see clustering_metrics")
    p.add_argument("--device", choices=("auto", "cpu"), default="auto",
                   help="'cpu' pins to CPU; the GPU path needs ~10 GB free")
    p.add_argument("--optimizer-memory", choices=("auto", "on", "off"), default="auto",
                   help="Use the low-memory AdamW path; see install_memory_efficient_adamw")
    p.add_argument("--ontology-cache", default=None,
                   help="Path to cached_ontology.pkl (default: <out-dir>/cached_ontology.pkl)")
    p.add_argument("--onto-vae-path", default=None,
                   help="Import onto_vae from this checkout instead of the installed package")
    return p.parse_args()


# ---------------------------------------------------------------- data

def load_split(expr, ids_path, id_col, label_col, n_nan_cols, drop_genes=None):
    """Return (genes x samples frame, labels) for the samples named in `ids_path`.

    Transposed because `match_dataset` wants genes on the index. `drop_genes` is
    passed for the test split only, as the published runs had it -- test carries
    971 genes against train's 1000.
    """
    ids = np.load(ids_path, allow_pickle=True)
    sub = expr[expr[id_col].isin(ids)].dropna(subset=[label_col])
    labels = sub[label_col].copy()
    frame = sub.drop(columns=list(expr.columns[:n_nan_cols])).set_index(sub[id_col])
    if drop_genes:
        frame = frame.drop(columns=list(drop_genes))
    frame.index.name = None
    labels.index = frame.index
    return frame.T, labels


def build_ontology(obo, gene_annot, top, bottom, cache_path):
    """Build the trimmed GO DAG and its decoder masks, caching the result (~20 min).

    The cache is validated on the trimming thresholds *and* the presence of
    decoder masks, so a cache from a different package version is rebuilt rather
    than silently reused.
    """
    from onto_vae.ontobj import Ontobj

    key = f"{top}_{bottom}"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            ont = pickle.load(fh)
        if key in ont.masks and "decoder" in ont.masks[key]:
            print(f"Loaded cached ontology from {cache_path}")
            return _compact_masks(ont, key)
        print("Cached ontology has no decoder masks for these thresholds; rebuilding")

    # Each stage is minutes-long and silent, so mark them or a build is
    # indistinguishable from a hang.
    ont = Ontobj(description="GO-basic")
    for stage, call in (
        (f"initialize_dag ({os.path.basename(obo)}, {os.path.basename(gene_annot)})",
         lambda: ont.initialize_dag(obo=obo, gene_annot=gene_annot)),
        (f"trim_dag (top={top}, bottom={bottom})",
         lambda: ont.trim_dag(top_thresh=top, bottom_thresh=bottom)),
        ("create_masks",
         lambda: ont.create_masks(top_thresh=top, bottom_thresh=bottom)),
    ):
        started = time.monotonic()
        print(f"  {stage} ...", flush=True)
        call()
        print(f"  {stage} done in {time.monotonic() - started:.0f}s", flush=True)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(ont, fh)
    print(f"Built and cached ontology at {cache_path}")
    return _compact_masks(ont, key)


def _compact_masks(ont, key):
    """Store the decoder masks as uint8 rather than pandas' int64.

    The values are binary, and the stack is held three times over during a run,
    so this saves ~2 GB. OntoVAE converts each mask to float32 regardless.
    """
    masks = ont.masks[key]["decoder"]
    if masks and masks[0].dtype != np.uint8:
        ont.masks[key]["decoder"] = [m.astype(np.uint8) for m in masks]
    return ont


# ---------------------------------------------------------------- metrics

def masked_mse(model, ont, dataset, gene_map, variant="published"):
    """Reconstruction MSE over the genes OntoVAE and GONNECT share.

    `gene_map` indexes this split's genes into the ontology's gene ordering,
    with -1 for genes it does not carry.

    `published` selects on `gene_map != 1`, as the original runs did. The
    sentinel is -1, not 1, so that rule drops the gene at ontology position 1
    and keeps the unmatched ones, whose -1 indexes the last column. It is the
    default because the deposited MSE values were computed under it.
    `corrected` uses `!= -1`, and is not comparable with them. On this data the
    two differ by under ~1%.
    """
    key = f"{model.top}_{model.bottom}"
    rec = torch.as_tensor(model.get_reconstructed_values(ont, dataset), dtype=torch.float32)
    obs = torch.as_tensor(ont.data[key][dataset], dtype=torch.float32)

    sentinel = 1 if variant == "published" else -1
    selected = gene_map[gene_map != sentinel]
    return F.mse_loss(rec[:, selected], obs[:, selected], reduction="mean").item()


def clustering_metrics(latent, labels, silhouette_against="labels"):
    """NMI and ARI for k-means at k = number of true classes, plus the silhouette.

    `silhouette_against` picks what the silhouette scores. `labels` takes the
    true classes, as GONNECT's SS does. `kmeans` takes the k-means clusters, as
    the original runs did -- a definition that runs higher than the label one,
    so the deposited values are not comparable with GONNECT's. Unlike the MSE
    gene mask, the comparable definition is the default. Both are returned as
    well, so a run reports the two without a second pass.
    """
    n_clusters = int(labels.nunique())
    assignments = KMeans(n_clusters=n_clusters, random_state=METRIC_SEED).fit_predict(latent)
    silhouettes = {
        name: float(silhouette_score(latent, against)) if n_clusters > 1 else float("nan")
        for name, against in (("labels", labels), ("kmeans", assignments))
    }
    metrics = {
        "NMI": float(normalized_mutual_info_score(labels, assignments)),
        "ARI": float(adjusted_rand_score(labels, assignments)),
        "Silhouette": silhouettes[silhouette_against],
    }
    return metrics, silhouettes


def latent_embedding(model, ont, dataset):
    key = f"{model.top}_{model.bottom}"
    x = torch.tensor(ont.data[key][dataset], dtype=torch.float32, device=model.device)
    model.eval()
    with torch.no_grad():
        return model.get_embedding(x).cpu().numpy()


def export_pathway_activities(model, ont, dataset, sample_ids, out_path):
    """Write per-sample activations for every GO term, as fig4 reads them.

    `get_pathway_activities` columns line up row-for-row with `extract_annot`,
    which is what makes the GO term IDs recoverable.
    """
    activities = model.get_pathway_activities(ont, dataset)
    annot = ont.extract_annot(top_thresh=model.top, bottom_thresh=model.bottom)
    if activities.shape[1] != len(annot):
        raise RuntimeError(
            f"{activities.shape[1]} activation columns vs {len(annot)} annotated terms -- "
            "the term ordering assumption no longer holds"
        )

    frame = pd.DataFrame(activities, index=sample_ids, columns=annot["ID"].values)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frame.to_parquet(out_path)
    print(f"  wrote {out_path}  ({frame.shape[0]} samples x {frame.shape[1]} terms)")
    return frame


# ---------------------------------------------------------------- run

def run_arm(ont, arm, seed, names, labels, gene_maps, run_dir, args, true_masks):
    """Train and evaluate one graph arm for one seed."""
    from onto_vae.vae_model import OntoVAE

    key = f"{args.top_thresh}_{args.bottom_thresh}"
    ont.masks[key]["decoder"] = randomize_mask_stack(true_masks, arm, seed)

    # The published runs left torch unseeded, so results here are statistically
    # equivalent to the deposited ones rather than identical.
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = OntoVAE(ontobj=ont, dataset=names["train"],
                    top_thresh=args.top_thresh, bottom_thresh=args.bottom_thresh)
    model.to(model.device)

    model_path = os.path.join(run_dir, "models", f"best_model_{arm}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.train_model(model_path, lr=args.lr, kl_coeff=args.kl_coeff,
                      batch_size=args.batch_size, epochs=args.epochs)

    out = {}
    for split in ("train", "test"):
        metrics, silhouettes = clustering_metrics(latent_embedding(model, ont, names[split]),
                                                  labels[split], args.silhouette_against)
        metrics["mse"] = masked_mse(model, ont, names[split], gene_maps[split],
                                    variant=args.mse_gene_mask)
        # Reported, but kept out of metrics.txt, whose shape the figure readers
        # depend on.
        other = "corrected" if args.mse_gene_mask == "published" else "published"
        print(f"    {split} mse[{args.mse_gene_mask}]={metrics['mse']:.6f}  "
              f"mse[{other}]={masked_mse(model, ont, names[split], gene_maps[split], other):.6f}")
        print(f"    {split} silhouette[labels]={silhouettes['labels']:.6f}  "
              f"silhouette[kmeans]={silhouettes['kmeans']:.6f}")
        out[split] = metrics
    print(f"  {arm}: test NMI {out['test']['NMI']:.4f}  MSE {out['test']['mse']:.4f}")

    export_pathway_activities(
        model, ont, names["test"], labels["test"].index,
        os.path.join(run_dir, f"pathway_activities_test_{arm}.parquet"),
    )

    del model
    torch.cuda.empty_cache()
    return out


def main():
    args = parse_args()
    # Reduces allocator fragmentation, which matters because the decoder's
    # large layer leaves little room for a contiguous block. Read when the
    # allocator initialises, so it must precede the first device allocation.
    # Ignored on Windows -- PyTorch warns and carries on -- but effective on
    # the Linux cluster where the real runs happen.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    ensure_onto_vae_importable(args.onto_vae_path)
    if args.device == "cpu":
        force_cpu()
    install_drop_last_loader()
    if install_memory_efficient_adamw(args.optimizer_memory):
        print("Using the low-memory AdamW path (foreach=False)")

    expr = pd.read_csv(args.expr_data)
    cache_path = args.ontology_cache or os.path.join(args.out_dir, "cached_ontology.pkl")
    ont = build_ontology(args.obo, args.gene_annot, args.top_thresh, args.bottom_thresh,
                         cache_path)

    key = f"{args.top_thresh}_{args.bottom_thresh}"

    # Each arm randomizes from the true graph, not the previous arm's output.
    true_masks = [m.copy() for m in ont.masks[key]["decoder"]]

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "metrics.txt")
    with open(metrics_path, "w") as out_fh:
        for seed in args.seeds:
            print(f"\n===== seed {seed} =====")
            split_run_dir = os.path.join(args.split_dir, f"run_seed-{seed}")
            run_dir = os.path.join(args.out_dir, f"run_seed-{seed}")
            os.makedirs(run_dir, exist_ok=True)

            frames, labels, names, gene_maps = {}, {}, {}, {}
            for split, files, drop in (("train", ["train_ids.npy", "val_ids.npy"], None),
                                       ("test", ["test_ids.npy"], DROP_GENES)):
                parts = [load_split(expr, os.path.join(split_run_dir, f), args.id_col,
                                    args.label_col, args.n_nan_cols, drop_genes=drop)
                         for f in files]
                frames[split] = pd.concat([f for f, _ in parts], axis=1)
                labels[split] = pd.concat([lab for _, lab in parts], axis=0)
                names[split] = f"TCGA_run_seed-{seed}-{split}"
                ont.match_dataset(expr_data=frames[split], name=names[split],
                                  top_thresh=args.top_thresh, bottom_thresh=args.bottom_thresh)

                gene_maps[split] = gene_index_map(frames[split].index, ont.genes[key])
                absent = int((gene_maps[split] == -1).sum())
                print(f"  {split}: {len(gene_maps[split])} genes, {absent} absent from the "
                      f"trimmed ontology")

            seed_metrics = {arm: run_arm(ont, arm, seed, names, labels, gene_maps,
                                         run_dir, args, true_masks)
                            for arm in args.arms}
            out_fh.write(f"run-{seed}: {seed_metrics}\n")
            out_fh.flush()

    print(f"\nMetrics written to {metrics_path}")


if __name__ == "__main__":
    main()
