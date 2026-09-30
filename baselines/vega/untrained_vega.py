"""
Untrained VEGA: the latent embeddings of freshly initialized, never-trained models.

The structural baseline for Figure 4's VEGA panel: how well a latent node's
activation lines up with the GSEA enrichment of its gene set before any
training. For every split seed and every --arms graph it builds --inits-per-seed
models exactly as run_vega.py does, skips training, and writes the test split's
latent embedding in the shape of the trained runs' z_test files. The first
column is each sample's row in the expression matrix.

The randomized arms (degree_preserving, random) use the mask run_vega.py draws
for that split seed, randomize_mask_stack(true mask, arm, seed). Every arm gets
the same initialization seeds, as run_vega.py gives every arm the same seed.

It writes the posterior *mean*, where run_vega.py writes a sample. An untrained
model's posterior variance is of order one, so a sample would add noise that
buries whatever the wiring alone carries, and the baseline would come out lower
than it is.

Writes out/baselines/untrained/vega/run_seed-<seed>/init-<k>/z_test_<arm>_<gmt>.csv,
numbering the initializations k = 0, 1, ... across seeds; k is also the torch seed.

Run from the repository root, in the VEGA environment (../environments/vega.yml):
    python baselines/vega/untrained_vega.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_vega import ARMS, build_adata, randomize_mask_stack  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expr-data", default="data/TCGA_complete_bp_top1k.csv")
    p.add_argument("--split-dir", default="out/baselines/splits")
    p.add_argument("--out-dir", default="out/baselines/untrained/vega")
    p.add_argument("--gmt-files", nargs="+",
                   default=["data/reactomes_uniprot.gmt", "data/hallmark_v2026_1_Hs_uniprot.gmt"])
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--inits-per-seed", type=int, default=10)
    p.add_argument("--add-nodes", type=int, default=1)
    p.add_argument("--id-col", default="patient_id")
    p.add_argument("--label-col", default="cancer_type")
    p.add_argument("--n-nan-cols", type=int, default=5)
    return p.parse_args()


def load_rows(expr, ids_path, args):
    """The samples named in `ids_path` as run_vega.load_split selects them, plus their rows in `expr`."""
    ids = np.load(ids_path, allow_pickle=True)
    sub = expr[expr[args.id_col].isin(ids)].dropna(subset=[args.label_col])
    frame = sub.drop(columns=list(expr.columns[:args.n_nan_cols]))
    rows = sub.index.to_numpy()
    frame.index = sub[args.id_col].values
    labels = pd.Series(sub[args.label_col].values, index=frame.index)
    return frame, labels, rows


def main():
    import vega

    args = parse_args()
    # CPU throughout: VEGA only moves a model to the GPU when training starts, and a forward pass is cheap
    device = torch.device("cpu")
    expr = pd.read_csv(args.expr_data)

    for s, seed in enumerate(args.seeds):
        split_dir = os.path.join(args.split_dir, f"run_seed-{seed}")
        train = [load_rows(expr, os.path.join(split_dir, f), args) for f in ("train_ids.npy", "val_ids.npy")]
        adata_train = build_adata(pd.concat([f for f, _, _ in train]), pd.concat([lab for _, lab, _ in train]),
                                  args.label_col)
        test, _, test_rows = load_rows(expr, os.path.join(split_dir, "test_ids.npy"), args)
        x = torch.tensor(test.values, dtype=torch.float32, device=device)
        print(f"\n===== seed {seed}: {adata_train.n_obs} train, {len(test_rows)} test samples =====")

        for gmt_path in args.gmt_files:
            gmt_name = os.path.splitext(os.path.basename(gmt_path))[0]
            # As in run_vega.run_gmt: set up afresh per GMT, since create_mask will not overwrite a mask
            vega.utils.setup_anndata(adata_train)
            vega.utils.create_mask(adata_train, gmt_path, add_nodes=args.add_nodes)
            true_mask = adata_train.uns["_vega"]["mask"].copy()
            gmv_names = list(adata_train.uns["_vega"]["gmv_names"])

            for arm in args.arms:
                # As in run_vega.run_gmt: VEGA has one masked layer, and the arm's mask depends on the split seed only
                adata_train.uns["_vega"]["mask"] = randomize_mask_stack([true_mask], arm, seed)[0]
                for j in range(args.inits_per_seed):
                    init = s * args.inits_per_seed + j
                    torch.manual_seed(init)
                    np.random.seed(init)
                    model = vega.VEGA(adata_train, positive_decoder=True, use_cuda=False)
                    model.eval()
                    with torch.no_grad():
                        _, mean, _ = model.encode(x, batch_index=None)

                    out_dir = os.path.join(args.out_dir, f"run_seed-{seed}", f"init-{init}")
                    os.makedirs(out_dir, exist_ok=True)
                    pd.DataFrame(mean.cpu().numpy(), index=test_rows, columns=gmv_names).to_csv(
                        os.path.join(out_dir, f"z_test_{arm}_{gmt_name}.csv"))
                    del model
                print(f"  {gmt_name} {arm}: {args.inits_per_seed} untrained models written "
                      f"({len(gmv_names)} latent nodes)", flush=True)
            adata_train.uns["_vega"]["mask"] = true_mask


if __name__ == "__main__":
    main()
