"""Embed the held-out test rows of training runs that exist only on the cluster, checking each first.

Runs on the cluster, not here: the checkpoints it reads were never deposited.
`cluster_test_metrics.py` scores what it writes. Two presets:
  s1    Supplementary Figure S1's sweep: ct=5 (AE_5.0 fixed, AE_5.1 soft links),
        ct=10 (AE_4.0, AE_4.1) and 2,000 genes (AE_6.0, AE_6.1), ~60 GB
  fig3  Figure 3's randomized arms not in latent_embeddings/: fully random (AE_10.2),
        degree-preserving with soft links (AE_11.1) and fully random with soft
        links (AE_10.1), ~2.4 GB
Each preset covers modules encoder / decoder / both, seeds 2-6. Every run's seed is
its version number, as in its training script. Later runs carry their slurm job id in
their file names; where a rerun left two, the one whose log matches the workbook is
kept, and the report names it.

Nothing is rebuilt. Each checkpoint's encoder and decoder weights are applied directly
(linear layers, ReLU between them), which reproduces the deposited embeddings to ~1e-13
and needs neither the masks nor the settings. Two checks per run:
  split     plain MSE over the test rows `gonnect.train.train.split_data` holds out at
            the run's seed, batch-averaged as training logged it, must equal the log's
            MSE column
  workbook  one of the log's last two columns must equal the value the preset's
            workbook holds for that setting / model / seed, where it has one. The
            workbooks mostly took column 3 (the test loss) for fixed runs and
            column 4 (the MSE) for soft-link runs, but not always: for S1's ct=10
            SL-enc and SL-dec they took column 3, which carries the soft-link
            penalty. The report records which column matched
The report also records `mse`, the test error Figure 2 reports for the run: the
masked test loss for fixed runs, the plain MSE for soft-link runs. The test rows
are encoded and saved as float32 with their row positions and cancer types, to
<output dir>/<preset>_test_embeddings.npz plus <preset>_report.json.

Usage, from a GONNECT checkout holding data/ and out/trained_models/, with any
environment that has torch, pandas and scikit-learn:
    python figures/src/prepare/embed_cluster_runs.py <preset> <checkout> <output dir>
On DAIC it ran as a CPU job (4 cores, 16 GB) in the Thesis_BINN container,
container_pixi_0.2.1.sif, calling /opt/app/.pixi/envs/default/bin/python directly.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PRESETS = {  # preset: {experiment: (setting, dataset, soft links, method prefix)}
    "s1": {
        "AE_5.0": ("ct=5", "top1k", False, "GONNECT-"), "AE_5.1": ("ct=5", "top1k", True, "GONNECT-SL-"),
        "AE_4.0": ("ct=10", "top1k", False, "GONNECT-"), "AE_4.1": ("ct=10", "top1k", True, "GONNECT-SL-"),
        "AE_6.0": ("2k genes", "top2k", False, "GONNECT-"), "AE_6.1": ("2k genes", "top2k", True, "GONNECT-SL-"),
    },
    "fig3": {
        "AE_10.2": ("ct=30", "top1k", False, "GONNECT-RR-"),
        "AE_11.1": ("ct=30", "top1k", True, "GONNECT-R-SL-"),
        "AE_10.1": ("ct=30", "top1k", True, "GONNECT-RR-SL-"),
    },
}
MODULES = {"encoder": "enc", "decoder": "dec", "both": "both"}
SEEDS = [2, 3, 4, 5, 6]
N_META = 5
# Each preset's workbook MSE: S1's sweep workbooks, and Figure 3's metric_data_TCGA_1000_30_new.xlsx.
WORKBOOK_MSE = {
    "s1": {
        'ct=5|GONNECT-SL-enc|2': 0.4338363957207062,
        'ct=5|GONNECT-SL-enc|3': 0.3816415586337533,
        'ct=5|GONNECT-SL-enc|4': 0.3468149749443626,
        'ct=5|GONNECT-SL-enc|5': 0.3893941281858812,
        'ct=5|GONNECT-SL-enc|6': 0.3782084488987141,
        'ct=5|GONNECT-SL-dec|2': 0.3425324313572854,
        'ct=5|GONNECT-SL-dec|3': 0.3866129165017662,
        'ct=5|GONNECT-SL-dec|4': 0.3380363242068639,
        'ct=5|GONNECT-SL-dec|5': 0.383532460144747,
        'ct=5|GONNECT-SL-dec|6': 0.4225782899576932,
        'ct=5|GONNECT-SL-both|2': 0.2038909990207161,
        'ct=5|GONNECT-SL-both|3': 0.2070732982772081,
        'ct=5|GONNECT-SL-both|4': 0.1957697526208968,
        'ct=5|GONNECT-SL-both|5': 0.1940624224563319,
        'ct=5|GONNECT-SL-both|6': 0.2028804976530233,
        'ct=5|GONNECT-enc|2': 0.2829890594363538,
        'ct=5|GONNECT-enc|3': 0.2734811593429354,
        'ct=5|GONNECT-enc|4': 0.9885495958478336,
        'ct=5|GONNECT-enc|5': 0.2834410766487649,
        'ct=5|GONNECT-enc|6': 0.9969657736695092,
        'ct=5|GONNECT-dec|2': 0.9906801138749146,
        'ct=5|GONNECT-dec|3': 0.9931090850863392,
        'ct=5|GONNECT-dec|4': 0.9885503130705734,
        'ct=5|GONNECT-dec|5': 0.9847854453884912,
        'ct=5|GONNECT-dec|6': 0.9969529844665151,
        'ct=10|GONNECT-SL-enc|2': 0.3798892721932609,
        'ct=10|GONNECT-SL-enc|3': 0.3502606486706993,
        'ct=10|GONNECT-SL-enc|4': 0.3539300513886353,
        'ct=10|GONNECT-SL-enc|5': 0.3534118081032058,
        'ct=10|GONNECT-SL-enc|6': 0.3483745604893157,
        'ct=10|GONNECT-SL-dec|2': 0.32399226733774,
        'ct=10|GONNECT-SL-dec|3': 0.340232155689214,
        'ct=10|GONNECT-SL-dec|4': 0.315920264282024,
        'ct=10|GONNECT-SL-dec|5': 0.3285882156151712,
        'ct=10|GONNECT-SL-dec|6': 0.3684805139695875,
        'ct=10|GONNECT-SL-both|2': 0.1877259855461639,
        'ct=10|GONNECT-SL-both|3': 0.1940326644083713,
        'ct=10|GONNECT-SL-both|4': 0.191516212498133,
        'ct=10|GONNECT-SL-both|5': 0.1955280037245977,
        'ct=10|GONNECT-SL-both|6': 0.1990496249188009,
        'ct=10|GONNECT-enc|2': 0.2733983361017608,
        'ct=10|GONNECT-enc|3': 0.2710131930477643,
        'ct=10|GONNECT-enc|4': 0.2926839456113783,
        'ct=10|GONNECT-enc|5': 0.281229943293172,
        'ct=10|GONNECT-enc|6': 0.2928747053372304,
        'ct=10|GONNECT-dec|2': 0.988644937647898,
        'ct=10|GONNECT-dec|3': 0.9910371408941842,
        'ct=10|GONNECT-dec|4': 0.9865785396578844,
        'ct=10|GONNECT-dec|5': 0.9828378041909324,
        'ct=10|GONNECT-dec|6': 0.9949284107805012,
        '2k genes|GONNECT-SL-enc|2': 0.2095713782796484,
        '2k genes|GONNECT-SL-enc|3': 0.20232369387215,
        '2k genes|GONNECT-SL-enc|4': 0.2120169157329802,
        '2k genes|GONNECT-SL-enc|5': 0.2094676272851141,
        '2k genes|GONNECT-SL-enc|6': 0.2067052643374749,
        '2k genes|GONNECT-SL-dec|2': 0.2336129479684205,
        '2k genes|GONNECT-SL-dec|3': 0.2343249892404691,
        '2k genes|GONNECT-SL-dec|4': 0.2339692885502628,
        '2k genes|GONNECT-SL-dec|5': 0.23233692346506,
        '2k genes|GONNECT-SL-dec|6': 0.2388628192711576,
        '2k genes|GONNECT-enc|2': 0.2509175519312709,
        '2k genes|GONNECT-enc|3': 0.2593237290773225,
        '2k genes|GONNECT-enc|4': 0.2601973545873535,
        '2k genes|GONNECT-enc|5': 0.2584244536831086,
        '2k genes|GONNECT-enc|6': 0.2566934584607107,
        '2k genes|GONNECT-dec|2': 0.648924731274805,
        '2k genes|GONNECT-dec|3': 0.6541033131307594,
        '2k genes|GONNECT-dec|4': 0.6272554875673387,
        '2k genes|GONNECT-dec|5': 0.647450346278641,
        '2k genes|GONNECT-dec|6': 0.6215137881824615,
    },
    "fig3": {
        'ct=30|GONNECT-R-SL-enc|2': 0.2307314783899798,
        'ct=30|GONNECT-R-SL-enc|3': 0.2427977054444094,
        'ct=30|GONNECT-R-SL-enc|4': 0.2329188586333049,
        'ct=30|GONNECT-R-SL-enc|5': 0.2305715178986984,
        'ct=30|GONNECT-R-SL-enc|6': 0.2434842114301676,
        'ct=30|GONNECT-R-SL-dec|2': 0.2352017579723928,
        'ct=30|GONNECT-R-SL-dec|3': 0.2331524699563037,
        'ct=30|GONNECT-R-SL-dec|4': 0.2349747504340598,
        'ct=30|GONNECT-R-SL-dec|5': 0.249352893483979,
        'ct=30|GONNECT-R-SL-dec|6': 0.2487713633130718,
        'ct=30|GONNECT-R-SL-both|2': 0.2150721309717404,
        'ct=30|GONNECT-R-SL-both|3': 0.2464070525065522,
        'ct=30|GONNECT-R-SL-both|4': 0.2339261268810266,
        'ct=30|GONNECT-R-SL-both|5': 0.2150793758576353,
        'ct=30|GONNECT-R-SL-both|6': 0.2412288068013,
        'ct=30|GONNECT-RR-SL-enc|2': 0.1951788658549245,
        'ct=30|GONNECT-RR-SL-enc|3': 0.194666084460756,
        'ct=30|GONNECT-RR-SL-enc|4': 0.1938664377318912,
        'ct=30|GONNECT-RR-SL-enc|5': 0.1918966108804957,
        'ct=30|GONNECT-RR-SL-enc|6': 0.1994323187360187,
        'ct=30|GONNECT-RR-SL-dec|2': 0.1994866239594099,
        'ct=30|GONNECT-RR-SL-dec|3': 0.1922654899509336,
        'ct=30|GONNECT-RR-SL-dec|4': 0.2072266399545448,
        'ct=30|GONNECT-RR-SL-dec|5': 0.2068168665933031,
        'ct=30|GONNECT-RR-SL-dec|6': 0.1955865251296454,
        'ct=30|GONNECT-RR-SL-both|2': 0.2054152992336899,
        'ct=30|GONNECT-RR-SL-both|3': 0.2100638687757196,
        'ct=30|GONNECT-RR-SL-both|4': 0.1865564475400448,
        'ct=30|GONNECT-RR-SL-both|5': 0.1972937757291887,
        'ct=30|GONNECT-RR-SL-both|6': 0.2108517970442007,
        'ct=30|GONNECT-RR-enc|2': 0.2167668819899293,
        'ct=30|GONNECT-RR-enc|3': 0.2114264288084505,
        'ct=30|GONNECT-RR-enc|4': 0.2107777857544947,
        'ct=30|GONNECT-RR-enc|5': 0.2186799123794524,
        'ct=30|GONNECT-RR-enc|6': 0.2302504507745867,
        'ct=30|GONNECT-RR-dec|2': 0.6287355316557854,
        'ct=30|GONNECT-RR-dec|3': 0.6547084442691563,
        'ct=30|GONNECT-RR-dec|4': 0.6282366144780676,
        'ct=30|GONNECT-RR-dec|5': 0.6285097425355215,
        'ct=30|GONNECT-RR-dec|6': 0.6502008210427201,
        'ct=30|GONNECT-RR-both|2': 0.730374951283722,
        'ct=30|GONNECT-RR-both|3': 0.7642295689144402,
        'ct=30|GONNECT-RR-both|4': 0.739318126245745,
        'ct=30|GONNECT-RR-both|5': 0.7343196524534698,
        'ct=30|GONNECT-RR-both|6': 0.7490931567197344,
    },
}


def layers(state, part):
    keys = sorted({k.rsplit(".", 1)[0] for k in state if k.startswith(part + ".") and k.endswith(".weight")},
                  key=lambda k: int(k.split(".")[-1]))
    return [(state[k + ".weight"], state[k + ".bias"]) for k in keys]


def forward(stack, x):
    for i, (w, b) in enumerate(stack):
        x = x @ w.T + b
        if i < len(stack) - 1:
            x = torch.relu(x)
    return x


def test_rows(expr, seed):
    _, remaining = train_test_split(expr, train_size=0.7, random_state=seed)
    _, test = train_test_split(remaining, train_size=0.5, random_state=seed)
    return test.index.to_numpy()


def checkpoints(folder, run):
    """A run's (checkpoint, log) pairs. Later runs carry their slurm job id in the
    name (AE_10.1.2_encoder_12438341_model.pt), and a rerun leaves two of them."""
    names = sorted(folder.glob(f"{run}_model.pt")) + sorted(folder.glob(f"{run}_[0-9]*_model.pt"))
    pairs = [(c, c.with_name(c.name[:-len("model.pt")] + "results.txt")) for c in names]
    return [(c, log) for c, log in pairs if log.exists()]


def evaluate(ckpt, log, data, seed, soft, workbook):
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    enc, dec = layers(state, "encoder"), layers(state, "decoder")
    n_genes = enc[0][0].shape[1]
    expr = data.iloc[:, N_META:N_META + n_genes]
    rows = test_rows(expr, seed)
    x = torch.tensor(expr.to_numpy()[rows], dtype=enc[0][0].dtype)
    with torch.no_grad():
        z = forward(enc, x)
        rec = forward(dec, z)
    recomputed = float(np.mean([((rec[i:i + 100] - x[i:i + 100]) ** 2).mean().item()
                                for i in range(0, len(x), 100)]))
    logged = [float(v) for v in log.read_text().strip().splitlines()[-1].split()]
    column = None if workbook is None else next(
        (c + 1 for c in (2, 3) if abs(logged[c] - workbook) < 1e-12), None)
    row = dict(status="ok", checkpoint=ckpt.name, n_test=len(rows), dim=int(z.shape[1]),
               n_genes=int(n_genes), logged_test_loss=logged[2], logged_mse=logged[3],
               recomputed_mse=recomputed, split_ok=abs(recomputed - logged[3]) < 1e-8,
               mse=logged[3] if soft else logged[2],
               workbook=workbook, workbook_column=column,
               workbook_ok=None if workbook is None else column is not None)
    return row, z.numpy().astype(np.float32), rows


preset, ROOT, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
experiments, workbook_mse = PRESETS[preset], WORKBOOK_MSE[preset]
datasets = {name: pd.read_csv(ROOT / "data" / f"TCGA_complete_bp_{name}.csv.gz")
            for name in sorted({dataset for _, dataset, _, _ in experiments.values()})}
arrays, report = {}, []
for exp, (setting, dataset, soft, prefix) in experiments.items():
    data = datasets[dataset]
    for module, short in MODULES.items():
        method = f"{prefix}{short}"
        for seed in SEEDS:
            run = f"{exp}.{seed}_{module}"
            row = {"run": run, "setting": setting, "method": method, "seed": seed}
            candidates = checkpoints(ROOT / "out" / "trained_models" / exp, run)
            if not candidates:
                report.append({**row, "status": "missing"})
                print(f"{run}: missing", flush=True)
                continue
            workbook = workbook_mse.get(f"{setting}|{method}|{seed}")
            # With a rerun on disk, keep the one whose log the workbook took its value from.
            results = [evaluate(c, log, data, seed, soft, workbook) for c, log in candidates]
            result, z, rows = next((r for r in results if r[0]["workbook_ok"]), results[0])
            row.update(result, n_candidates=len(candidates))
            report.append(row)
            arrays[run] = z
            arrays[f"{run}.rows"] = rows
            arrays[f"{run}.labels"] = data["cancer_type"].to_numpy()[rows].astype(str)
            print(f"{run}: dim {row['dim']}, split_ok {row['split_ok']}, "
                  f"workbook_ok {row['workbook_ok']} (column {row['workbook_column']}), "
                  f"{row['checkpoint']} of {len(candidates)}", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / f"{preset}_test_embeddings.npz", **arrays)
(OUT / f"{preset}_report.json").write_text(json.dumps(report, indent=1))
bad = [r["run"] for r in report if r["status"] == "ok" and (not r["split_ok"] or r["workbook_ok"] is False)]
print(f"done: {sum(r['status'] == 'ok' for r in report)} runs embedded, "
      f"{sum(r['status'] == 'missing' for r in report)} missing, failed checks: {bad or 'none'}")
