"""Training-time benchmark for the Supplementary Table S3.

Trains each model variant to convergence with the hyperparameters of preprint v3 and records
epochs to convergence, wall-clock time and seconds per epoch. Gradient clipping is left off,
matching the setting the results in preprint v3 were produced with.

Results are appended to out/benchmark/training_time.tsv, one row per run, and per-epoch 
validation losses to out/benchmark/progress/. Runs already present in the TSV are skipped,
so the benchmark can be interrupted and restarted without repeating work.

Run from the repository root:
    python src/benchmark_training_time.py
"""
import contextlib
import io
import os
import sys
import time

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from gonnect.model.build_model import build_model
from gonnect.train.loss import MSE, MSE_Masked, MSE_Soft_Link_Sum
from gonnect.train.train import make_data_splits

# Hyperparameters, matching the AE_2.0.* / AE_2.1.* experiment scripts
DATASET = "TCGA_complete_bp_top1k"
MERGE_CONDITIONS = (1, 30, 50)
DTYPE, DEVICE = torch.float64, "cuda"
N_NAN_COLS, N_SAMPLES, BATCH_SIZE = 5, 9797, 100
MAX_EPOCHS, PATIENCE, LEARNING_RATE, MOMENTUM = 10000, 10, 0.01, 0.9
DATA_SPLIT, N_GO_LAYERS, SOFT_LINK_ALPHA = 0.7, 5, 100
SEEDS = [2, 3, 4, 5, 6]

# Ordered shortest-first so that a usable table exists early in a long run
VARIANTS = [
    ("MLP", "none", False, "mse"),
    ("GONNECT-enc", "encoder", False, "mse masked"),
    ("GONNECT-SL-enc", "encoder", True, "soft links"),
    ("GONNECT-SL-dec", "decoder", True, "soft links"),
    ("GONNECT-SL-both", "both", True, "soft links"),
    ("GONNECT-dec", "decoder", False, "mse masked"),
    ("GONNECT-both", "both", False, "mse masked"),
]

OUT_DIR = os.path.join(ROOT, "out", "benchmark")
OUT = os.path.join(OUT_DIR, "training_time.tsv")
PROGRESS_DIR = os.path.join(OUT_DIR, "progress")
os.makedirs(PROGRESS_DIR, exist_ok=True)


def find_dataset():
    """The expression matrix ships separately (see README); accept either location."""
    for candidate in (os.path.join(ROOT, "data", f"{DATASET}.csv.gz"),
                      os.path.join(ROOT, "figures", "data", f"{DATASET}.csv.gz")):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{DATASET}.csv.gz not found in data/ or figures/data/")


data = pd.read_csv(find_dataset(), nrows=N_SAMPLES, usecols=range(N_NAN_COLS + 1000),
                   compression="gzip")
genes = list(data.columns[N_NAN_COLS:])
gene_mask = torch.load(
    os.path.join(ROOT, "out", "masks", "genes", str(MERGE_CONDITIONS), f"{DATASET}_gene_mask.pt"),
    weights_only=True)
MASKS_DIR = os.path.join(ROOT, "out", "masks")

completed = set()
if os.path.exists(OUT):
    completed = {tuple(line.split("\t")[:2]) for line in open(OUT)
                 if "\t" in line and not line.startswith("variant")}
else:
    with open(OUT, "w") as f:
        f.write("variant\tseed\tepochs\ttrain_s\ts_per_epoch\tval\ttest_mse\tstopped\tsetup_s\n")

remaining = sum(1 for s in SEEDS for v in VARIANTS if (v[0], str(s)) not in completed)
print(f"[{time.strftime('%H:%M:%S')}] {remaining} runs to go, {len(completed)} already recorded",
      flush=True)
start_all = time.time()

for seed in SEEDS:
    if all((v[0], str(seed)) in completed for v in VARIANTS):
        continue
    _, trainloader, validationloader, testloader = make_data_splits(
        data, N_NAN_COLS, N_SAMPLES, BATCH_SIZE, DATA_SPLIT, seed)

    for name, bi_module, soft_links, loss_name in VARIANTS:
        if (name, str(seed)) in completed:
            continue

        quiet = io.StringIO()
        t_setup = time.time()
        with contextlib.redirect_stdout(quiet):
            model = build_model("dense", bi_module, soft_links, DATASET, False, MERGE_CONDITIONS,
                                N_GO_LAYERS, torch.nn.ReLU, DTYPE, genes, cluster=False,
                                random_version=None, masks_dir=MASKS_DIR)
        model.to(DEVICE)
        model.masks_to(DEVICE)
        if loss_name == "soft links":
            loss_fn = MSE_Soft_Link_Sum(model, alpha=SOFT_LINK_ALPHA)
        elif loss_name == "mse masked":
            loss_fn = MSE_Masked(gene_mask, DEVICE)
        else:
            loss_fn = MSE()
        mse_fn = MSE()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        setup_s = time.time() - t_setup

        print(f"[{time.strftime('%H:%M:%S')}] -> seed {seed} {name} started", flush=True)
        progress = open(os.path.join(PROGRESS_DIR, f"{name}_{seed}.log"), "w", buffering=1)
        best, waited, t_start = float("inf"), 0, time.time()

        for epoch in range(MAX_EPOCHS):
            for (batch,) in trainloader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                loss_fn(model(batch), batch).backward()
                optimizer.step()          # no gradient clipping, see module docstring
                model.mask_weights()
            with torch.no_grad():
                validation = sum(float(loss_fn(model(b.to(DEVICE)), b.to(DEVICE)))
                                 for (b,) in validationloader) / len(validationloader)
            progress.write(f"epoch {epoch + 1} val {validation:.8f}\n")
            if validation < best:
                best, waited = validation, 0
            else:
                waited += 1
                if waited >= PATIENCE:
                    break
        progress.close()

        elapsed = time.time() - t_start
        with torch.no_grad():
            test_mse = sum(float(mse_fn(model(b.to(DEVICE)), b.to(DEVICE)))
                           for (b,) in testloader) / len(testloader)
        stopped = waited >= PATIENCE
        with open(OUT, "a") as f:
            f.write(f"{name}\t{seed}\t{epoch + 1}\t{elapsed:.1f}\t{elapsed / (epoch + 1):.4f}\t"
                    f"{validation:.6f}\t{test_mse:.6f}\t{stopped}\t{setup_s:.1f}\n")
        print(f"[{time.strftime('%H:%M:%S')}] seed {seed} {name:16s} {epoch + 1:6d} epochs  "
              f"{'converged' if stopped else 'HIT CAP'}  {elapsed / 60:7.1f} min  "
              f"{elapsed / (epoch + 1):.3f} s/epoch  test MSE {test_mse:.4f}", flush=True)

print(f"\n[{time.strftime('%H:%M:%S')}] benchmark complete in "
      f"{(time.time() - start_all) / 3600:.2f} h -> {OUT}", flush=True)
