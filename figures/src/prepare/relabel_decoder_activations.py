"""Stand-in decoder activations, every column named after the node it holds.

The decoder files in go_term_activations/ are labelled one layer off: each
column holds the node at the same position one layer closer to the genes (see
untrained_control.py, and tests/test_activations.py for the function that wrote
them, since fixed). Until they are regenerated from the checkpoints, this
writes the same values under the right names. The bottleneck comes from
latent_embeddings/, and the columns named after the last GO layer, which hold
reconstructed genes, are dropped. Decoder layer 6 was only written as far as the
bottleneck is wide, so it keeps 109 of its 245 GO terms (51 of the 92 with a
GSEA result); every other layer is complete.

The output is a drop-in for go_term_activations/: the encoder files are
hard-linked in unchanged, so plot_perm_nulls_layers.py --activations-dir and
fig4.py --activations-dir read it as they read the shipped directory. A decoder
file already in the corrected layout is linked as it is.

Writes figures/out/prepare/decoder_relabelled/go_term_activations/.

Run from the repository root:
    pixi run python figures/src/prepare/relabel_decoder_activations.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_DIR, PREP_OUT_DIR
from untrained_control import (MASKS_DIR, MODELS, N_META_COLS, TRAINED_SEEDS, decoder_layout, latent,
                               node_layers, relabelled_decoder)

OUT_DIR = PREP_OUT_DIR / "decoder_relabelled" / "go_term_activations"


def link(source: Path, target: Path) -> None:
    """Put an unchanged input in place without copying it (a copy where links are not possible)."""
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--masks-dir", type=Path, default=MASKS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hard_links = pd.read_csv(args.data_dir / "hard_links.csv", usecols=["component", "layer", "source_index",
                                                                        "source_term_id", "sink_index",
                                                                        "sink_term_id"])
    layers = node_layers(hard_links, args.masks_dir)
    source_dir = args.data_dir / "go_term_activations"

    for version, _ in MODELS.values():
        for seed in TRAINED_SEEDS:
            encoder = f"AE_{version}.{seed}_encoder_activations.csv.gz"
            link(source_dir / encoder, args.out_dir / encoder)

            decoder = f"AE_{version}.{seed}_decoder_activations.csv.gz"
            shipped = pd.read_csv(source_dir / decoder)
            z = latent(args.data_dir, version, seed, "decoder")
            if decoder_layout(shipped, z, layers["bottleneck"]) == "corrected":
                link(source_dir / decoder, args.out_dir / decoder)
                print(f"{decoder}: already in the corrected layout, linked")
                continue
            relabelled = pd.concat([shipped.iloc[:, :N_META_COLS], relabelled_decoder(shipped, z, layers)], axis=1)
            relabelled.to_csv(args.out_dir / decoder, index=False,
                              compression={"method": "gzip", "compresslevel": 1})
            print(f"{decoder}: {shipped.shape[1] - N_META_COLS} shipped GO columns -> "
                  f"{relabelled.shape[1] - N_META_COLS} relabelled", flush=True)

    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
