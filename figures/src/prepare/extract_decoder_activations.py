"""Decoder activations re-extracted from the checkpoints, every column holding the node it names.

The decoder files in go_term_activations/ were written by the old
activations_per_term, which labelled each decoder layer's outputs with the
terms of its input layer (see untrained_control.py). This rebuilds them from
the decoder-only checkpoints of the cluster runs with the corrected labelling,
so every decoder layer is complete -- unlike relabel_decoder_activations.py,
which can only rename what the shipped files hold and so keeps 109 of decoder
layer 6's 245 GO terms.

Every checkpoint is checked against the shipped files before it is used:
labelled the old way, its outputs must reproduce the shipped decoder file, and
its encoder output the saved latent embedding. That shows these are the models
the shipped files came from, not later retrains. Where the encoder-only
checkpoint of the same run is present too, it must reproduce the shipped
encoder file, which confirms that those files need no correction.

Needs, from the cluster checkout's out/trained_models/ (not deposited), in the
same place here:
    AE_2.0/AE_2.0.{2..6}_decoder_model.pt
    AE_2.1/AE_2.1.{2..6}_decoder_model.pt
and, only for the encoder check, AE_2.{0,1}/AE_2.{0,1}.{2..6}_encoder_model.pt.

Writes figures/out/prepare/decoder_reextracted/go_term_activations/, a drop-in
for go_term_activations/: the corrected decoder files beside the shipped encoder
files, which are hard-linked in. The headers are the shipped ones; only the
values under them change.

Run from the repository root:
    pixi run python figures/src/prepare/extract_decoder_activations.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_DIR, PREP_OUT_DIR
from relabel_decoder_activations import link
from untrained_control import (DATASET, DECODER_INPUTS, DECODER_OUTPUTS, ENCODER_OUTPUTS, MASKS_DIR,
                               MERGE_CONDITIONS, MODELS, N_GO_LAYERS_USED, N_META_COLS, REPO_ROOT, TRAINED_SEEDS,
                               go_columns, holds_latent, latent, node_layers)

from gonnect.model.build_model import build_model

CHECKPOINTS_DIR = REPO_ROOT / "out" / "trained_models"
OUT_DIR = PREP_OUT_DIR / "decoder_reextracted" / "go_term_activations"


def run_checkpoint(path: Path, module: str, soft_links: bool, x: torch.Tensor,
                   masks_dir: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """The latent z and the linear layers' outputs of the module a checkpoint constrains."""
    with contextlib.redirect_stdout(io.StringIO()):   # build_model narrates the whole architecture
        model = build_model("dense", module, soft_links, DATASET, False, MERGE_CONDITIONS, N_GO_LAYERS_USED,
                            torch.nn.ReLU, torch.float64, masks_dir=masks_dir)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    model.set_store_activations(True)
    with torch.no_grad():
        z = model.encoder(x)
        model.decoder(z)
    model.set_store_activations(False)
    coder = model.encoder if module == "encoder" else model.decoder
    return z.numpy(), [a.numpy() for a in list(coder.activations.values())[::2]]


def reproduces(frame: pd.DataFrame, shipped: pd.DataFrame) -> float:
    """Largest difference from the shipped values, relative to their scale; raises if the headers differ."""
    if list(frame.columns) != list(shipped.columns):
        raise ValueError("the rebuilt columns are not the shipped ones")
    values, reference = frame.to_numpy(), shipped.to_numpy()
    return float(np.abs(values - reference).max() / np.abs(reference).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--masks-dir", type=Path, default=MASKS_DIR)
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="largest relative difference from the shipped values that still counts as reproduced")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    expression = pd.read_csv(args.data_dir / f"{DATASET}.csv.gz")
    x = torch.tensor(expression.iloc[:, N_META_COLS:].to_numpy(dtype=np.float64))
    hard_links = pd.read_csv(args.data_dir / "hard_links.csv", usecols=["component", "layer", "source_index",
                                                                        "source_term_id", "sink_index",
                                                                        "sink_term_id"])
    layers = node_layers(hard_links, args.masks_dir)
    source_dir = args.data_dir / "go_term_activations"

    for version, soft_links in MODELS.values():
        for seed in TRAINED_SEEDS:
            run = f"AE_{version}.{seed}"
            checkpoints = args.checkpoints_dir / f"AE_{version}"

            encoder = f"{run}_encoder_activations.csv.gz"
            if (checkpoints / f"{run}_encoder_model.pt").exists():
                shipped = pd.read_csv(source_dir / encoder)
                _, outputs = run_checkpoint(checkpoints / f"{run}_encoder_model.pt", "encoder", soft_links, x,
                                            args.masks_dir)
                rebuilt = pd.concat([go_columns(out, layers[name]) for out, name in zip(outputs, ENCODER_OUTPUTS)],
                                    axis=1)
                difference = reproduces(rebuilt, shipped.iloc[:, N_META_COLS:])
                if difference > args.tolerance:
                    raise ValueError(f"{run} encoder checkpoint does not reproduce {encoder} ({difference:.1e})")
                print(f"{run} encoder: checkpoint reproduces the shipped file ({difference:.1e}), linked", flush=True)
            link(source_dir / encoder, args.out_dir / encoder)

            decoder = f"{run}_decoder_activations.csv.gz"
            shipped = pd.read_csv(source_dir / decoder)
            z, outputs = run_checkpoint(checkpoints / f"{run}_decoder_model.pt", "decoder", soft_links, x,
                                        args.masks_dir)
            if not holds_latent(z, latent(args.data_dir, version, seed, "decoder")):
                raise ValueError(f"{run} decoder checkpoint does not reproduce its saved latent embedding")
            # Labelled the old way -- every output under its input layer's terms -- it must give the shipped file
            old = pd.concat([go_columns(out, layers[name]) for out, name in zip(outputs, DECODER_INPUTS)], axis=1)
            difference = reproduces(old, shipped.iloc[:, N_META_COLS:])
            if difference > args.tolerance:
                raise ValueError(f"{run} decoder checkpoint does not reproduce {decoder} ({difference:.1e})")

            corrected = pd.concat([go_columns(z, layers["bottleneck"])]
                                  + [go_columns(out, layers[name]) for out, name in zip(outputs, DECODER_OUTPUTS)],
                                  axis=1)
            corrected = corrected[shipped.columns[N_META_COLS:]]   # the shipped header, in the shipped order
            pd.concat([shipped.iloc[:, :N_META_COLS], corrected], axis=1).to_csv(
                args.out_dir / decoder, index=False, compression={"method": "gzip", "compresslevel": 1})
            print(f"{run} decoder: checkpoint reproduces the shipped file ({difference:.1e}) and its latent; "
                  f"wrote {corrected.shape[1]} corrected columns", flush=True)

    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
