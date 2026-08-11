"""
Activation preservation: GONNECT (fixed link) vs GONNECT-SL.

For each seed in {2..6}, module in {encoder, decoder}, layer (encoder 0..3 /
decoder 5..8), this compares AE_2.0.s vs AE_2.1.s test-set activations
per GO neuron:

  1. Per-node signed Pearson r   -- strictest, penalises sign flips
  2. Per-node |r|                -- the interpretability-preservation metric:
                                    allows the meaningless per-neuron sign-
                                    convention freedom (AE hidden units have
                                    no built-in "high = up" convention) but
                                    forbids cross-neuron mixing, which would
                                    scramble the GO labelling.

Note: linear CKA was removed -- it is too generous for GONNECT, where
neurons are labelled with specific GO terms. CKA's rotation invariance
would treat two layers as equivalent even if the GO labelling is
scrambled. See the all_vs_all_abs_r.py script for the multi-pair
heatmap version using the same |r| metric.

Outputs:
  out/prepare/activation_preservation/activation_preservation_per_node.csv  raw per-node stats
  fig_sl_preserve/activation_preservation.png           per-layer boxplot
  fig_sl_preserve/activation_preservation_summary.csv   summary table
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _paths import DATA_DIR, PREP_OUT_DIR

SEEDS = [2, 3, 4, 5, 6]
MODULES = ["encoder", "decoder"]


def build_layer_map(hl: pd.DataFrame, module: str) -> dict[str, int]:
    sub = hl[hl["component"] == module]
    col = "sink_term_id" if module == "encoder" else "source_term_id"
    pairs = sub[[col, "layer"]].drop_duplicates()
    pairs = pairs[pairs[col].str.startswith("GO:")]
    return dict(zip(pairs[col], pairs["layer"].astype(int)))


def per_node_corr(
    activ_dir: Path,
    seed: int,
    module: str,
    layer_map: dict[str, int],
    spearman_subsample: int = 4000,
) -> pd.DataFrame:
    p_fixed = activ_dir / f"AE_2.0.{seed}_{module}_activations.csv.gz"
    p_soft = activ_dir / f"AE_2.1.{seed}_{module}_activations.csv.gz"
    df0 = pd.read_csv(p_fixed)
    df1 = pd.read_csv(p_soft)
    if not df0["patient_id"].equals(df1["patient_id"]):
        raise RuntimeError(f"patient_id order mismatch seed={seed} module={module}")

    go_cols = [c for c in df0.columns if c.startswith("GO:")]
    A0 = df0[go_cols].to_numpy(dtype=np.float64)
    A1 = df1[go_cols].to_numpy(dtype=np.float64)

    # Pearson, vectorized over nodes
    A0c = A0 - A0.mean(axis=0)
    A1c = A1 - A1.mean(axis=0)
    s0 = np.sqrt((A0c**2).sum(axis=0))
    s1 = np.sqrt((A1c**2).sum(axis=0))
    denom = s0 * s1
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = (A0c * A1c).sum(axis=0) / denom
    pearson[denom == 0] = np.nan

    # Spearman, subsampled for speed
    rng = np.random.default_rng(seed)
    n = len(df0)
    k = min(n, spearman_subsample)
    idx = rng.choice(n, size=k, replace=False)
    A0s = A0[idx]
    A1s = A1[idx]
    rho = np.empty(A0.shape[1])
    for j in range(A0.shape[1]):
        if A0s[:, j].std() == 0 or A1s[:, j].std() == 0:
            rho[j] = np.nan
        else:
            rho[j] = stats.spearmanr(A0s[:, j], A1s[:, j]).statistic

    return pd.DataFrame({
        "seed": seed,
        "module": module,
        "go_term": go_cols,
        "layer": [layer_map.get(c, -1) for c in go_cols],
        "pearson_r": pearson,
        "abs_pearson_r": np.abs(pearson),
        "spearman_rho": rho,
        "var_fixed": A0.var(axis=0),
        "var_soft": A1.var(axis=0),
    })


def plot_per_layer(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, module in zip(axes, MODULES):
        sub = df[df["module"] == module]
        layers = sorted(sub["layer"].unique())
        data = [sub.loc[sub["layer"] == L, "abs_pearson_r"].dropna().values for L in layers]
        bp = ax.boxplot(
            data, positions=range(len(layers)), widths=0.55,
            patch_artist=True, showfliers=True,
            flierprops=dict(marker="o", markersize=2, alpha=0.3),
            medianprops=dict(color="black"),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#5b8def")
            patch.set_alpha(0.6)

        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.axhline(1, color="grey", lw=0.6, ls=":")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(L) for L in layers])
        ax.set_xlabel("layer")
        ax.set_title(module)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("|Pearson r| per GO node (AE_2.0.s vs AE_2.1.s)")

    fig.suptitle(
        "Activation preservation: GONNECT (fixed) vs GONNECT-SL, same seed\n"
        "Per-GO-node |Pearson r| across 9797 test samples; boxplot pools 5 seeds",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved {out_path}")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for module in MODULES:
        for layer in sorted(df.loc[df["module"] == module, "layer"].unique()):
            if layer < 0:
                continue
            sub = df[(df["module"] == module) & (df["layer"] == layer)]
            per_seed_signed = sub.groupby("seed")["pearson_r"].median()
            per_seed_abs = sub.groupby("seed")["abs_pearson_r"].median()
            n_nodes = (sub["seed"] == sub["seed"].iloc[0]).sum() if len(sub) else 0
            rows.append({
                "module": module,
                "layer": int(layer),
                "n_nodes_per_seed": int(n_nodes),
                "median_r_signed_mean":  per_seed_signed.mean(),
                "median_r_signed_std":   per_seed_signed.std(ddof=1),
                "median_abs_r_mean":     per_seed_abs.mean(),
                "median_abs_r_std":      per_seed_abs.std(ddof=1),
                "frac_abs_r_gt_0.7":     (sub["abs_pearson_r"] > 0.7).mean(),
                "frac_abs_r_gt_0.5":     (sub["abs_pearson_r"] > 0.5).mean(),
                "frac_abs_r_lt_0.2":     (sub["abs_pearson_r"] < 0.2).mean(),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PREP_OUT_DIR / "activation_preservation")
    args = parser.parse_args()

    activ_dir = args.data_dir / "go_term_activations"
    hard_links_path = args.data_dir / "hard_links.csv"

    print(f"Loading hard_links from {hard_links_path}")
    hl = pd.read_csv(hard_links_path, usecols=["component", "layer", "source_term_id", "sink_term_id"])
    layer_maps = {m: build_layer_map(hl, m) for m in MODULES}
    for m, lm in layer_maps.items():
        print(f"  {m}: {len(lm)} GO term -> layer entries")

    per_node_rows = []
    for module in MODULES:
        for seed in SEEDS:
            print(f"  {module} seed={seed} ...", flush=True)
            per_node = per_node_corr(activ_dir, seed, module, layer_maps[module])
            n_unmapped = (per_node["layer"] == -1).sum()
            if n_unmapped:
                print(f"    {n_unmapped} GO terms had no layer assignment "
                      f"(treated as layer = -1)")
            per_node_rows.append(per_node)
    out = pd.concat(per_node_rows, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_node_csv = args.output_dir / "activation_preservation_per_node.csv"
    out.to_csv(per_node_csv, index=False)
    print(f"Saved {per_node_csv}: {len(out):,} rows")

    summary = summarize(out)
    summary_csv = args.output_dir / "activation_preservation_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved {summary_csv}")
    print("\n=== Summary (per module x layer) ===")
    print(summary.to_string(index=False))

    plot_per_layer(out, args.output_dir / "activation_preservation.png")


if __name__ == "__main__":
    main()
