"""Pull the weight matrices figures S7/S8 need out of the alpha-sweep checkpoints.

    out/trained_models/AE_3.<v>/AE_3.<v>.<r>_<module>_model.pt
      |- extract_alpha_sweep_weights.py --> data/alpha_sweep_weights/
                                              AE_3.<v>.<r>_<module>_weights.pt

This docstring is argparse's --help text, and a Windows console encodes it as
cp1252, so it stays ASCII: a Greek alpha or a box-drawing arrow here crashes
--help with UnicodeEncodeError before it prints anything.

Each output is a plain dict keyed by the source state-dict key, so a reader can
still see which layer a matrix came from:

    {"encoder.net_layers.0.weight": tensor, ... , "encoder.net_layers.6.weight": tensor}

Run from the repository root, with the training output in place:
    pixi run python figures/src/prepare/extract_alpha_sweep_weights.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINED_MODELS = REPO_ROOT / "out" / "trained_models"
OUT_DIR = DATA_DIR / "alpha_sweep_weights"

# nn.Sequential indices carrying a weight matrix (ReLU sits on the odd ones).
WEIGHTED = [0, 2, 4, 6]

# (subdirectory, run stem, module). The module named here is the one the run
# constrains, and the only one anything reads -- except for the fully connected
# reference, where the "encoder" half of an unconstrained autoencoder is what
# figS8 draws as its MLP series.
RUNS = [("AE_3.-1", "AE_3.-1.2", "none",    "encoder"),
        ("AE_3.-1", "AE_3.-1.2", "encoder", "encoder"),
        ("AE_3.-1", "AE_3.-1.2", "decoder", "decoder"),
        ("AE_3.0",  "AE_3.0.3",  "encoder", "encoder"),
        ("AE_3.0",  "AE_3.0.3",  "decoder", "decoder"),
        ("AE_3.1",  "AE_3.1.3",  "encoder", "encoder"),
        ("AE_3.1",  "AE_3.1.3",  "decoder", "decoder"),
        ("AE_3.2",  "AE_3.2.3",  "encoder", "encoder"),
        ("AE_3.2",  "AE_3.2.3",  "decoder", "decoder"),
        ("AE_3.3",  "AE_3.3.3",  "encoder", "encoder"),
        ("AE_3.3",  "AE_3.3.3",  "decoder", "decoder")]


def main() -> None:
    # Raw, as run_all.py does: the docstring's chain diagram and its indented
    # example are laid out on purpose, and the default formatter reflows both
    # into one paragraph.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trained-models", type=Path, default=TRAINED_MODELS,
                        help=f"training output root (default: {TRAINED_MODELS})")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help=f"where the extracts go (default: {OUT_DIR})")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_in = total_out = 0
    for subdir, stem, run_module, kept in RUNS:
        src = args.trained_models / subdir / f"{stem}_{run_module}_model.pt"
        if not src.exists():
            raise SystemExit(
                f"missing {src}\nThe alpha-sweep checkpoints are training output, not "
                f"repository content. See figures/README.md, 'Figures S7 and S8'.")
        state = torch.load(src, map_location="cpu", weights_only=False)
        keys = [f"{kept}.net_layers.{i}.weight" for i in WEIGHTED]
        missing = [k for k in keys if k not in state]
        if missing:
            raise SystemExit(f"{src} has no {missing[0]} -- unexpected architecture")

        dst = args.out_dir / f"{stem}_{run_module}_weights.pt"
        torch.save({k: state[k].clone() for k in keys}, dst)
        total_in += src.stat().st_size
        total_out += dst.stat().st_size
        print(f"  {src.name:34s} -> {dst.name:34s} "
              f"{src.stat().st_size / 1e6:6.1f} -> {dst.stat().st_size / 1e6:5.1f} MB")

    print(f"\n{len(RUNS)} runs, {total_in / 1e6:.0f} MB -> {total_out / 1e6:.0f} MB "
          f"({100 * (1 - total_out / total_in):.0f}% smaller) in {args.out_dir}")


if __name__ == "__main__":
    main()
