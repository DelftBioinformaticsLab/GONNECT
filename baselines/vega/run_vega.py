"""
Train VEGA on the TCGA splits, across the three gene-set-graph arms (true, fully random, and degree_preserving).

Writes, per seed:

  metrics.txt                       run-<seed>: {gmt: {arm: {split: {metric: ...}}}}
  run_seed-<seed>/z_{train,test}_<arm>_<gmt>.csv      latent embeddings
  run_seed-<seed>/best_model_<arm>_<gmt>/             saved model

`figures/src/_common._VEGA_ANNOTATIONS` keys on the GMT file *stem*, so the two
GMTs must keep the names `reactomes_uniprot.gmt` and
`hallmark_v2026_1_Hs_uniprot.gmt`.

Upstream VEGA is used unmodified; see `../environments/vega.yml`.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mask_randomization import randomize_mask_stack  # noqa: E402

ARMS = ("true", "random", "degree_preserving")
METRIC_SEED = 42  # KMeans, matching the original runs and figures/src/_common

# Genes absent from GONNECT's processed ontology. VEGA keeps them in the model
# and excludes them only when scoring MSE.
DROP_GENES = (
    "A6NC42", "O60635", "O95857", "P00450", "P02814", "P05976", "P11686", "P12882",
    "P18283", "P22792", "P30408", "P48230", "Q4ZG55", "Q53G44", "Q5T7N2", "Q5TH69",
    "Q685J3", "Q7Z7J9", "Q86XP6", "Q8IWL1", "Q8IWL2", "Q8IZW8", "Q8WXD2", "Q969L2",
    "Q96KN4", "Q9BYZ8", "Q9NR99", "Q9NRC9", "Q9UKX2",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expr-data", default="data/TCGA_complete_bp_top1k.csv")
    p.add_argument("--split-dir", default="out/baselines/splits",
                   help="Where make_splits.py wrote the sample IDs; read-only")
    p.add_argument("--out-dir", default="out/baselines/vega",
                   help="Where this run's metrics, models and embeddings go")
    p.add_argument("--gmt-files", nargs="+",
                   default=["data/reactomes_uniprot.gmt",
                            "data/hallmark_v2026_1_Hs_uniprot.gmt"])
    p.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-size", type=float, default=0.82,
                   help="Fraction of the training data VEGA holds back internally")
    p.add_argument("--add-nodes", type=int, default=1,
                   help="Fully-connected UNANNOTATED latent nodes appended to the mask. "
                        "They carry the genes no gene set annotates: 602 of 1000 for "
                        "Reactome, 552 for Hallmark.")
    p.add_argument("--max-eval-cells", type=int, default=10_000)
    p.add_argument("--silhouette-against", choices=("labels", "kmeans"), default="labels",
                   help="What the silhouette scores; see evaluate")
    p.add_argument("--id-col", default="patient_id")
    p.add_argument("--label-col", default="cancer_type")
    p.add_argument("--n-nan-cols", type=int, default=5)
    return p.parse_args()


# ---------------------------------------------------------------- data

def load_split(expr, ids_path, id_col, label_col, n_nan_cols):
    """Return (samples x genes frame, labels) for the samples named in `ids_path`."""
    ids = np.load(ids_path, allow_pickle=True)
    sub = expr[expr[id_col].isin(ids)].dropna(subset=[label_col])
    frame = sub.drop(columns=list(expr.columns[:n_nan_cols]))
    frame.index = sub[id_col].values
    labels = pd.Series(sub[label_col].values, index=frame.index)
    return frame, labels


def build_adata(frame, labels, label_col):
    """Wrap an expression frame as the AnnData VEGA's setup expects."""
    import anndata as ad

    var = pd.DataFrame(index=pd.Index(frame.columns, name=None))
    obs = pd.DataFrame({label_col: labels.values}, index=frame.index.astype(str))
    adata = ad.AnnData(X=frame.values.astype(np.float32), obs=obs, var=var)
    adata.obs_names_make_unique()
    return adata


# ---------------------------------------------------------------- metrics

def evaluate(model, frame, labels, device, max_cells, silhouette_against="labels"):
    """NMI, ARI, silhouette on the latent space, plus MSE over the shared genes.

    `silhouette_against` picks what the silhouette scores: `labels`, the true
    classes, as GONNECT's SS does, or `kmeans`, the k-means clusters, as the
    original runs did -- a definition that runs higher, so the deposited values
    are not comparable with GONNECT's. Both are returned alongside the metrics.
    """
    if len(frame) > max_cells:
        frame, labels = _stratified_subsample(frame, labels, max_cells)

    latent, recon, observed = _encode_decode(model, frame, device)

    keep = ~frame.columns.isin(DROP_GENES)
    mse = F.mse_loss(recon[:, keep], observed[:, keep], reduction="mean").item()

    n_clusters = int(labels.nunique())
    if n_clusters <= 1:
        return ({"NMI": float("nan"), "ARI": float("nan"),
                 "Silhouette": float("nan"), "mse": float(mse)},
                {"labels": float("nan"), "kmeans": float("nan")})

    z = latent.numpy()
    assignments = KMeans(n_clusters=n_clusters, random_state=METRIC_SEED).fit_predict(z)
    silhouettes = {"labels": float(silhouette_score(z, labels)),
                   "kmeans": float(silhouette_score(z, assignments))}
    metrics = {
        "NMI": float(normalized_mutual_info_score(labels, assignments)),
        "ARI": float(adjusted_rand_score(labels, assignments)),
        "Silhouette": silhouettes[silhouette_against],
        "mse": float(mse),
    }
    return metrics, silhouettes


def _stratified_subsample(frame, labels, max_cells):
    """Sample `max_cells` rows keeping each label's share of the whole."""
    rng = np.random.default_rng(METRIC_SEED)
    arr = labels.values
    classes, counts = np.unique(arr, return_counts=True)
    per_class = np.maximum(1, np.round(max_cells * counts / len(arr)).astype(int))
    picked = np.concatenate([
        rng.choice(np.where(arr == c)[0], min(n, int((arr == c).sum())), replace=False)
        for c, n in zip(classes, per_class)
    ])
    return frame.iloc[picked], labels.iloc[picked]


def _encode_decode(model, frame, device, chunk=1024):
    """Run the model over `frame` in chunks; return (latent, recon, observed) on CPU."""
    latents, recons, observeds = [], [], []
    with torch.no_grad():
        for i in range(0, len(frame), chunk):
            x = torch.tensor(frame.values[i:i + chunk], device=device, dtype=torch.float32)
            z, _, _ = model.encode(x, batch_index=None)
            latents.append(z.cpu())
            recons.append(model.decode(z, batch_index=None).cpu())
            observeds.append(x.cpu())
    return torch.cat(latents), torch.cat(recons), torch.cat(observeds)


def save_embeddings(model, frame, gmv_names, path, device):
    """Write the latent embedding as CSV -- the file fig4 reads."""
    latent, _, _ = _encode_decode(model, frame, device)
    out = pd.DataFrame(latent.numpy(), index=frame.index, columns=gmv_names)
    out.to_csv(path)
    print(f"    wrote {os.path.basename(path)}  ({out.shape[0]} x {out.shape[1]})")


# ---------------------------------------------------------------- run

def run_gmt(adata_train, frames, labels, gmt_path, seed, run_dir, args, device):
    """Train and evaluate all arms for one gene-set database."""
    import vega

    gmt_name = os.path.splitext(os.path.basename(gmt_path))[0]
    print(f"\n  === {gmt_name} ===")

    # Re-run per GMT: create_mask refuses to overwrite an existing mask.
    vega.utils.setup_anndata(adata_train)
    vega.utils.create_mask(adata_train, gmt_path, add_nodes=args.add_nodes)
    true_mask = adata_train.uns["_vega"]["mask"].copy()
    gmv_names = list(adata_train.uns["_vega"]["gmv_names"])
    print(f"  mask {true_mask.shape}  edges={int(true_mask.sum())}  GMVs={len(gmv_names)}")

    np.save(os.path.join(run_dir, f"gene_names_{gmt_name}.npy"), list(adata_train.var.index))

    out = {}
    for arm in args.arms:
        print(f"  --- {arm} ---")
        # randomize_mask_stack takes a list of layers; VEGA has exactly one.
        mask = randomize_mask_stack([true_mask], arm, seed)[0]
        np.save(os.path.join(run_dir, f"mask_{arm}_{gmt_name}.npy"), mask)
        adata_train.uns["_vega"]["mask"] = mask

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = vega.VEGA(adata_train, positive_decoder=True,
                          use_cuda=(device.type == "cuda"))
        model.train_vega(n_epochs=args.n_epochs, batch_size=args.batch_size,
                         train_size=args.train_size, use_gpu=(device.type == "cuda"))
        model.save(path=os.path.join(run_dir, f"best_model_{arm}_{gmt_name}"),
                   save_adata=True, save_history=True, overwrite=True)
        model.eval()

        out[arm] = {}
        for split in ("train", "test"):
            out[arm][split], silhouettes = evaluate(model, frames[split], labels[split], device,
                                                    args.max_eval_cells, args.silhouette_against)
            print(f"    {split} silhouette[labels]={silhouettes['labels']:.6f}  "
                  f"silhouette[kmeans]={silhouettes['kmeans']:.6f}")
        print(f"    test NMI {out[arm]['test']['NMI']:.4f}  MSE {out[arm]['test']['mse']:.4f}")

        for split in ("train", "test"):
            save_embeddings(model, frames[split], gmv_names,
                            os.path.join(run_dir, f"z_{split}_{arm}_{gmt_name}.csv"), device)

        del model
        torch.cuda.empty_cache()

    return gmt_name, out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    expr = pd.read_csv(args.expr_data)
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "metrics.txt")

    with open(metrics_path, "w") as out_fh:
        for seed in args.seeds:
            print(f"\n===== seed {seed} =====")
            split_run_dir = os.path.join(args.split_dir, f"run_seed-{seed}")
            run_dir = os.path.join(args.out_dir, f"run_seed-{seed}")
            os.makedirs(run_dir, exist_ok=True)

            frames, labels = {}, {}
            for split, files in (("train", ["train_ids.npy", "val_ids.npy"]),
                                 ("test", ["test_ids.npy"])):
                parts = [load_split(expr, os.path.join(split_run_dir, f), args.id_col,
                                    args.label_col, args.n_nan_cols) for f in files]
                frames[split] = pd.concat([f for f, _ in parts], axis=0)
                labels[split] = pd.concat([lab for _, lab in parts], axis=0)
                print(f"  {split}: {frames[split].shape[0]} samples, "
                      f"{frames[split].shape[1]} genes")

            adata_train = build_adata(frames["train"], labels["train"], args.label_col)

            seed_metrics = {}
            for gmt_path in args.gmt_files:
                name, result = run_gmt(adata_train, frames, labels, gmt_path,
                                       seed, run_dir, args, device)
                seed_metrics[name] = result

            out_fh.write(f"run-{seed}: {seed_metrics}\n")
            out_fh.flush()

    print(f"\nMetrics written to {metrics_path}")


if __name__ == "__main__":
    main()
