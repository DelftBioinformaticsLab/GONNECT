"""Every number the alpha sweep is quoted for, in one table.

Supplementary Figures S7 and S8 show the sweep; this is what they show, written
out so the rebuttal can cite values instead of asking a reader to measure a
curve. Four blocks, one row per (alpha, module):

  final MSE          the last epoch of the fixed 1,000-epoch schedule
  epochs to plateau  the first epoch within PLATEAU_TOL of the run's total MSE
                     reduction, on a smoothed curve. Not a convergence
                     criterion -- these runs had early stopping disabled --
                     just where the curve stops moving.
  active soft links  |w| > ACTIVE_THRESH, the threshold the paper uses
  |w| percentiles    biological GO edges against soft links, and the ratio of
                     the two, which is the "order of magnitude below" claim

A note on which weights count as GO. The fixed-link checkpoint pins every non-GO
weight at exactly 0 and every proxy weight at exactly 1, so those two constants
identify all three groups without a mask file. Proxy edges are excluded from the
percentile block: they are held at 1 by construction and make up 6,423 of the
9,561 GO positions, so leaving them in would put the median GO weight at 1.0 and
say nothing about what the model learned. They are still counted in the
`n_proxy` column so the three groups visibly add up.

Both blocks read from --data-dir: `loss_traces/` for the MSE columns and
`alpha_sweep_weights/` for the weight ones, i.e. the same inputs figS7 and figS8
draw from. See figS7's docstring for why the two fixed-link baselines carry a masked
MSE and are flagged here rather than silently mixed in.

Run from the repository root:
    pixi run python figures/src/prepare/alpha_sweep_numbers.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_DIR, PREP_OUT_DIR

OUT_DIR = PREP_OUT_DIR / "alpha_sweep"

MODULES = ["encoder", "decoder"]
WEIGHTED = [0, 2, 4, 6]          # nn.Sequential indices carrying a weight matrix

# (label, subdir, stem, module-dependent?) -- the same sweep figS7 plots.
SWEEP = [
    ("MLP",          "AE_3.-1", "AE_3.-1.2", False),
    ("alpha=1e2",    "AE_3.0",  "AE_3.0.3",  True),
    ("alpha=1e3",    "AE_3.1",  "AE_3.1.3",  True),
    ("alpha=1e4",    "AE_3.2",  "AE_3.2.3",  True),
    ("alpha=1e5",    "AE_3.3",  "AE_3.3.3",  True),
    ("Fixed links",  "AE_3.-1", "AE_3.-1.2", True),
]
SOFT_LINK_RUNS = {"alpha=1e2", "alpha=1e3", "alpha=1e4", "alpha=1e5"}
MASKED_RUNS = {"AE_3.-1.2_encoder", "AE_3.-1.2_decoder"}

FIXED_LINKS = "AE_3.-1.2"
ACTIVE_THRESH = 0.1
PLATEAU_TOL = 0.01               # within 1% of the total MSE reduction
SMOOTH_WINDOW = 11               # epochs, centred rolling median before the crossing
PERCENTILES = [50, 90, 99, 100]


def trace_path(data_dir: Path, subdir: str, stem: str, module: str) -> Path:
    return data_dir / "loss_traces" / subdir / f"{stem}_{module}_results.txt"


def weights_path(data_dir: Path, stem: str, module: str) -> Path:
    """The per-module weight extract for one run.

    These hold only the four matrices of the module a run constrains, cut out of
    the full autoencoder checkpoint by prepare/extract_alpha_sweep_weights.py --
    the other half is never read here and would double the deposit.
    """
    return data_dir / "alpha_sweep_weights" / f"{stem}_{module}_weights.pt"


def epochs_to_plateau(mse: np.ndarray) -> int:
    """First epoch reaching within PLATEAU_TOL of the run's total MSE reduction.

    The band is a fraction of the descent (first epoch minus last), not of the
    final value. Measured against the final value instead, every run scores
    950-990: the plateau carries batch noise of order 1% of its own height, so
    that version reports when the noise last happened to exceed the band, which
    is a property of the noise and not of convergence.

    The curve is smoothed over SMOOTH_WINDOW epochs before the crossing is
    taken, for the same reason -- a single noisy epoch dipping into the band
    early should not count as having arrived.
    """
    smooth = pd.Series(mse).rolling(SMOOTH_WINDOW, center=True,
                                    min_periods=1).median().to_numpy()
    target = mse[-1] + PLATEAU_TOL * (smooth[0] - mse[-1])
    reached = np.flatnonzero(smooth <= target)
    return int(reached[0] + 1) if len(reached) else len(mse)


def weight_groups(data_dir: Path, stem: str, module: str) -> dict:
    """|w| for the three disjoint groups of one checkpoint's module.

    'bio' is the GO edges the model actually learns, 'proxy' the ones held at 1,
    'soft' every position GO does not connect.
    """
    fixed = torch.load(weights_path(data_dir, FIXED_LINKS, module),
                       map_location="cpu", weights_only=True)
    state = torch.load(weights_path(data_dir, stem, module),
                       map_location="cpu", weights_only=True)
    groups = {"bio": [], "proxy": [], "soft": []}
    for i in WEIGHTED:
        key = f"{module}.net_layers.{i}.weight"
        w_fixed = fixed[key].numpy()
        w = np.abs(state[key].numpy())
        proxy = w_fixed == 1.0
        bio = (w_fixed != 0) & ~proxy
        groups["bio"].append(w[bio])
        groups["proxy"].append(w[proxy])
        groups["soft"].append(w[~(w_fixed != 0)])
    return {k: np.concatenate(v) for k, v in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, subdir, stem, module_dependent in SWEEP:
        for module in MODULES:
            run_module = module if module_dependent else "none"
            name = f"{stem}_{run_module}"
            frame = pd.read_csv(trace_path(args.data_dir, subdir, stem, run_module),
                                sep="\t")
            mse = frame[frame.columns[-1]].to_numpy()

            row = {
                "run": label,
                "module": module,
                "run_file": name,
                "epochs": len(mse),
                "final_mse": mse[-1],
                "mse_is_masked": name in MASKED_RUNS,
                "epochs_to_plateau": epochs_to_plateau(mse),
            }

            groups = weight_groups(args.data_dir, stem, run_module) \
                if module_dependent else None
            if groups is not None:
                row["n_go_bio"] = len(groups["bio"])
                row["n_proxy"] = len(groups["proxy"])
                row["n_soft"] = len(groups["soft"])
                row["active_soft_links"] = int((groups["soft"] > ACTIVE_THRESH).sum())
                for p in PERCENTILES:
                    row[f"go_p{p}"] = np.percentile(groups["bio"], p)
                    row[f"soft_p{p}"] = np.percentile(groups["soft"], p)
                # The "order of magnitude below" claim, made checkable: the bulk
                # of the soft links against the strongest GO edges.
                row["log10_ratio_p90"] = np.log10(row["go_p90"] / max(row["soft_p90"],
                                                                     1e-300))
            rows.append(row)

    table = pd.DataFrame(rows)
    out = args.out_dir / "alpha_sweep_numbers.tsv"
    table.to_csv(out, sep="\t", index=False, float_format="%.6g")
    print(f"wrote {out}\n")

    for col in ("n_go_bio", "n_proxy", "n_soft", "active_soft_links"):
        table[col] = table[col].astype("Int64")
    sl = table[table.run.isin(SOFT_LINK_RUNS)]
    print("Final test MSE and epochs to plateau (1,000-epoch schedule, seed 1)")
    print(table[["run", "module", "final_mse", "epochs_to_plateau",
                 "mse_is_masked"]].to_string(index=False,
                                             float_format=lambda v: f"{v:.4f}"))
    print("\nActive soft links (|w| > 0.1)")
    print(sl[["run", "module", "active_soft_links", "n_soft"]]
          .to_string(index=False))
    print("\n|w| percentiles: biological GO edges vs soft links "
          "(proxy edges, held at 1, excluded)")
    cols = ["run", "module"] + [f"{g}_p{p}" for p in PERCENTILES
                                for g in ("go", "soft")] + ["log10_ratio_p90"]
    print(sl[cols].to_string(index=False,
                             float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
