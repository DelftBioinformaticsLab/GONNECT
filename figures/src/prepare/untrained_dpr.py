"""Figure 4's panel c statistic on an untrained GONNECT-DPR: the randomized wiring's baseline.

The counterpart of untrained_control.py for panel c. GONNECT-DPR (AE_2.2) is
GONNECT with degree-preserving randomized GO links in the encoder, the decoder
or both. Each trained split seed has its own randomized mask, shared by the
three modules: AE_2.2.<22..26> loaded out/masks/*/<merge>/..._random<8..12>.pt
(the version minus 14, as gonnect.analysis.latent_space has it). This step
builds --inits-per-mask models on each of those five masks, 50 in all, leaves
them at their initialization and scores their bottleneck as panel c does.

The statistic is panel c's per-seed dot: per (cancer type, bottleneck node) the
one-vs-rest ROC-AUC over primary tumours, symmetrized as max(AUC, 1 - AUC); per
cancer type the Spearman r across the nodes against the bottleneck GSEA
-log10(NOM p); the median over cancer types. The AUC is computed on ranks,
which equals fig4's roc_auc_score loop; the step checks that on a trained seed,
and checks that the trained seeds reproduce the dots fig4 draws.

Two identities follow from the architecture, not from the data. The `decoder`
model's bottleneck comes from GONNECT's plain dense encoder, so its untrained
values do not depend on the randomized mask at all. And the `encoder` and
`both` models build the same masked encoder from the same seed, so their
untrained bottlenecks are equal.

When the AE_2.2 checkpoints are present under out/trained_models/, the step
first rebuilds every trained model with this builder and checks that it
reproduces the deposited latent embedding. The _random8-12 masks are not in
the repository or the deposit yet (see 4TU_TODO.md).

Writes, to figures/out/prepare/untrained_dpr/:

  untrained_dpr_values.tsv    one row per (module, state, seed or init)
  fig4_untrained_dpr.tsv      the untrained reference for fig4.py --untrained

Run from the repository root:
    pixi run python figures/src/prepare/untrained_dpr.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PREPARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARE_DIR))
sys.path.insert(0, str(PREPARE_DIR.parent))
from _paths import DATA_DIR, PREP_OUT_DIR
import fig4
from _common import CANCER_TYPE_ORDER
from untrained_baselines import auc_frame

from gonnect.model.build_model import build_model

REPO_ROOT = PREPARE_DIR.parents[2]
MASKS_DIR = REPO_ROOT / "out" / "masks"
CHECKPOINT_DIR = REPO_ROOT / "out" / "trained_models" / "AE_2.2"
OUT_DIR = PREP_OUT_DIR / "untrained_dpr"

# The architecture AE_2.2 was trained with (src/AE_2.2.py)
DATASET = "TCGA_complete_bp_top1k"
N_META_COLS = 5
MERGE_CONDITIONS = (1, 30, 50)
N_GO_LAYERS_USED = 5
VERSION = "2.2"
ROW = fig4.EMB_VERSION_DISPLAY[VERSION]
TRAINED_SEEDS = fig4.RAND_SEEDS                          # 22..26
MASK_OF_SEED = {seed: seed - 14 for seed in TRAINED_SEEDS}   # 8..12
MODULES = ["encoder", "decoder", "both"]
TRAINED = "trained"
UNTRAINED = "untrained"


def build(module: str, random_version: int, masks_dir: Path) -> torch.nn.Module:
    with contextlib.redirect_stdout(io.StringIO()):   # build_model narrates the whole architecture
        model = build_model("dense", module, False, DATASET, False, MERGE_CONDITIONS, N_GO_LAYERS_USED,
                            torch.nn.ReLU, torch.float64, random_version=random_version, masks_dir=masks_dir)
    return model.eval()


def bottleneck(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return model.encoder(x).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--masks-dir", type=Path, default=MASKS_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--inits-per-mask", type=int, default=10)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    expression = pd.read_csv(args.data_dir / f"{DATASET}.csv.gz")
    meta = expression.iloc[:, :N_META_COLS]
    x = torch.tensor(expression.iloc[:, N_META_COLS:].to_numpy(dtype=np.float64))
    primary = (meta["sample_type"] == "Primary Tumor").to_numpy()
    cancer_types = meta["cancer_type"].to_numpy()[primary]
    columns = fig4.load_bottleneck_columns(args.data_dir / "hard_links.csv")
    enrichment = fig4.load_enrichment_pivot(args.data_dir / "gsea_gonnect_bottleneck" / "gsea_results.csv",
                                            CANCER_TYPE_ORDER)
    emb_dir = args.data_dir / "latent_embeddings"

    def score(z: np.ndarray) -> float:
        # In float32, as fig4.load_embedding hands the trained latents to the AUC
        auc = auc_frame(z[primary].astype(np.float32), cancer_types, columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # a constant node leaves its per-term r undefined; unused here
            return fig4.per_ct_median_r(auc, enrichment)

    # The rank AUC against fig4's own loop, on a trained seed
    reference, reference_cts = fig4.gonnect_auc_one_seed(emb_dir, meta, VERSION, "encoder", TRAINED_SEEDS[0])
    z = fig4.load_embedding(emb_dir, VERSION, TRAINED_SEEDS[0], "encoder")
    ours = auc_frame(z[primary], cancer_types, columns)
    difference = np.abs(ours.loc[reference_cts].to_numpy() - reference).max()
    if difference > 1e-6:
        raise AssertionError(f"rank AUC differs from fig4.gonnect_auc_one_seed by {difference:.2e}")
    print(f"rank AUC matches fig4.gonnect_auc_one_seed (max |diff| {difference:.1e})", flush=True)

    rows = []
    for module in MODULES:
        for seed in TRAINED_SEEDS:
            z = fig4.load_embedding(emb_dir, VERSION, seed, module)
            checkpoint = args.checkpoint_dir / f"AE_{VERSION}.{seed}_{module}_model.pt"
            if checkpoint.exists():
                model = build(module, MASK_OF_SEED[seed], args.masks_dir)
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                rebuilt = bottleneck(model, x)
                # Against the file itself, in float64: load_embedding rounds it to float32
                deposited = torch.load(emb_dir / f"AE_{VERSION}" / f"AE_{VERSION}.{seed}_{module}_full_dataset.pt",
                                       map_location="cpu", weights_only=True).numpy()
                if not np.allclose(rebuilt, deposited, rtol=1e-9, atol=1e-12 * np.abs(deposited).max()):
                    raise AssertionError(f"AE_{VERSION}.{seed}_{module}: the builder does not reproduce the "
                                         f"deposited latent (max |diff| {np.abs(rebuilt - deposited).max():.2e})")
            rows.append({"row": ROW, "label": fig4.MODULE_ABBREV[module], "state": TRAINED, "seed": seed,
                         "mask": MASK_OF_SEED[seed], "init": None, "rebuilt": checkpoint.exists(),
                         "median_r": score(z)})
        n_rebuilt = sum(r["rebuilt"] for r in rows if r["label"] == fig4.MODULE_ABBREV[module])
        print(f"{module}: trained seeds scored; {n_rebuilt} of {len(TRAINED_SEEDS)} rebuilt from their "
              f"checkpoint and matched the deposited latent", flush=True)

        for s, seed in enumerate(TRAINED_SEEDS):
            for j in range(args.inits_per_mask):
                init = s * args.inits_per_mask + j
                torch.manual_seed(init)
                model = build(module, MASK_OF_SEED[seed], args.masks_dir)
                rows.append({"row": ROW, "label": fig4.MODULE_ABBREV[module], "state": UNTRAINED, "seed": seed,
                             "mask": MASK_OF_SEED[seed], "init": init, "rebuilt": False,
                             "median_r": score(bottleneck(model, x))})
        print(f"{module}: {len(TRAINED_SEEDS) * args.inits_per_mask} untrained models scored", flush=True)

    values = pd.DataFrame(rows)
    values["n_terms"] = len(set(columns) & set(enrichment.columns))
    values.to_csv(args.out_dir / "untrained_dpr_values.tsv", sep="\t", index=False, float_format="%.6g")
    (values[values.state == UNTRAINED][["row", "label", "init", "n_terms", "median_r"]]
     .to_csv(args.out_dir / "fig4_untrained_dpr.tsv", sep="\t", index=False, float_format="%.6g"))

    # The trained side must be what Figure 4 plots as dots
    drawn = pd.read_csv(PREP_OUT_DIR.parent / "fig4.csv").set_index(["row", "label"])
    for label, group in values[values.state == TRAINED].groupby("label", sort=False):
        dots = drawn.loc[(ROW, label), [f"seed_{i}" for i in range(len(TRAINED_SEEDS))]].to_numpy(dtype=float)
        if not np.allclose(group.median_r.to_numpy(), dots, atol=1e-9):
            print(f"WARNING: trained {ROW} {label} differs from fig4.csv's per-seed dots")

    print(f"\nwrote {args.out_dir}\n")
    summary = values.groupby(["label", "state"], sort=False).median_r.describe(percentiles=[0.025, 0.5, 0.975])
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
