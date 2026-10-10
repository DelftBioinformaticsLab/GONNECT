"""
Untrained OntoVAE: the pathway activities of freshly initialized, never-trained models.

The structural baseline for Figure 4's OntoVAE panel: how well a GO term's
activity lines up with the GSEA enrichment of its gene set before any training.
For every split seed it builds --inits-per-seed models on the true GO decoder
masks exactly as run_ontovae.py does, from the same cached ontology, skips
training, and writes the test split's pathway activities: one column per GO
term, the mean of its neurons, in the ontology's term order.

That ontology is the one behind preprint v3. Its 4,131 terms, their order and each
term's layer all match the deposited activation files.

The activities are taken at the posterior *mean*. OntoVAE's
get_pathway_activities samples the latent (reparameterize), and an untrained
model's posterior variance is of order one, so a sample would add noise that
buries whatever the wiring alone carries. The model's reparameterize is swapped
for one returning the mean, so the package's own code computes everything else.

Writes out/baselines/untrained/ontovae/run_seed-<seed>/init-<k>/pathway_activities_test_true.parquet,
indexed by each sample's row in the expression matrix; the initializations are
numbered k = 0, 1, ... across seeds, and k is also the torch seed.

Run from the repository root, in the OntoVAE environment (../environments/ontovae.yml):
    python baselines/ontovae/untrained_ontovae.py --gene-annot data/gene_annot_ontovae.txt
"""

import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(os.path.dirname(_HERE))
from ontovae_compat import ensure_onto_vae_importable, force_cpu  # noqa: E402
from run_ontovae import DROP_GENES, build_ontology, load_split  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expr-data", default="figures/data/TCGA_complete_bp_top1k.csv.gz")
    p.add_argument("--split-dir", default="out/baselines/splits")
    p.add_argument("--out-dir", default="out/baselines/untrained/ontovae")
    p.add_argument("--obo", default="data/go-basic.obo")
    p.add_argument("--gene-annot", required=True)
    p.add_argument("--ontology-cache", default="out/baselines/ontovae/cached_ontology.pkl")
    p.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--inits-per-seed", type=int, default=10)
    p.add_argument("--top-thresh", type=int, default=1000)
    p.add_argument("--bottom-thresh", type=int, default=30)
    p.add_argument("--id-col", default="patient_id")
    p.add_argument("--label-col", default="cancer_type")
    p.add_argument("--n-nan-cols", type=int, default=5)
    p.add_argument("--device", choices=("auto", "cpu"), default="auto")
    return p.parse_args()


def rows_of(expr, ids_path, args):
    """The expression-matrix rows run_ontovae.load_split selects, in its order."""
    ids = np.load(ids_path, allow_pickle=True)
    return expr[expr[args.id_col].isin(ids)].dropna(subset=[args.label_col]).index.to_numpy()


def main():
    args = parse_args()
    # As in run_ontovae.main: building fifty of these models in one process fragments the CUDA allocator
    # until a 1 GiB layer no longer fits, so let it grow segments. Read at the first allocation.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ensure_onto_vae_importable(None)
    if args.device == "cpu":
        force_cpu()
    from onto_vae.vae_model import OntoVAE

    expr = pd.read_csv(args.expr_data)
    ont = build_ontology(args.obo, args.gene_annot, args.top_thresh, args.bottom_thresh, args.ontology_cache)
    terms = ont.extract_annot(top_thresh=args.top_thresh, bottom_thresh=args.bottom_thresh)["ID"].values

    key = f"{args.top_thresh}_{args.bottom_thresh}"
    for s, seed in enumerate(args.seeds):
        inits = [s * args.inits_per_seed + j for j in range(args.inits_per_seed)]
        paths = {init: os.path.join(args.out_dir, f"run_seed-{seed}", f"init-{init}",
                                    "pathway_activities_test_true.parquet") for init in inits}
        todo = [init for init in inits if not os.path.exists(paths[init])]
        if not todo:   # every initialization is seeded, so a rerun only makes what is missing
            print(f"\n===== seed {seed}: all {len(inits)} initializations already written =====", flush=True)
            continue

        split_dir = os.path.join(args.split_dir, f"run_seed-{seed}")
        names = {}
        for split, files, drop in (("train", ["train_ids.npy", "val_ids.npy"], None),
                                   ("test", ["test_ids.npy"], DROP_GENES)):
            frame = pd.concat([load_split(expr, os.path.join(split_dir, f), args.id_col, args.label_col,
                                          args.n_nan_cols, drop_genes=drop)[0] for f in files], axis=1)
            names[split] = f"TCGA_run_seed-{seed}-{split}"
            ont.match_dataset(expr_data=frame, name=names[split],
                              top_thresh=args.top_thresh, bottom_thresh=args.bottom_thresh)
        test_rows = rows_of(expr, os.path.join(split_dir, "test_ids.npy"), args)
        print(f"\n===== seed {seed}: {len(test_rows)} test samples =====", flush=True)

        for init in todo:
            out_path = paths[init]
            torch.manual_seed(init)
            np.random.seed(init)
            model = OntoVAE(ontobj=ont, dataset=names["train"],
                            top_thresh=args.top_thresh, bottom_thresh=args.bottom_thresh)
            model.to(model.device)
            model.reparameterize = lambda mu, log_var: mu   # the posterior mean, not a sample
            activities = model.get_pathway_activities(ont, names["test"])
            if activities.shape != (len(test_rows), len(terms)):
                raise RuntimeError(f"activities {activities.shape}, expected ({len(test_rows)}, {len(terms)})")

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pd.DataFrame(activities, index=test_rows, columns=terms).to_parquet(out_path)
            del model, activities
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  init {init} written", flush=True)

        # match_dataset stores each split expanded to the ontology's ~22k genes, ~1.7 GB a seed. Kept for
        # every seed, it grows until (under WSL) CUDA can no longer allocate the decoder's 1 GiB layer.
        for name in names.values():
            ont.data[key].pop(name, None)
        gc.collect()


if __name__ == "__main__":
    main()
