"""
Per-layer permutation null analysis for GONNECT activations.

Loads the all-layer activation files in ``data/go_term_activations/`` (one CSV.gz per
seed × module × model variant) and runs the three permutation nulls
(row / column / row+column) at every layer of the encoder and decoder.

Produces:
  - one perm-null figure per (variant, module, layer)
  - one trajectory figure per module showing how the activation–enrichment
    correlation evolves across layers, comparing model variants. This is the
    GONNECT analogue of the OntoVAE per-layer comparison in
    plot_activation_vs_gsea_baselines.py.
  - a unified summary CSV with one row per (variant, module, layer, null, metric)

Usage:
  pixi run python fig_deg/perm_nulls_layers/plot_perm_nulls_layers.py
"""

import argparse
import gzip
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# Reuse perm-null helpers from the bottleneck-only script
from gonnect_names import EMB_VERSION_DISPLAY  # noqa: E402
from plot_perm_nulls import (  # noqa: E402

    _summary_stats,
    run_null,
    perm_pval,
    plot_combined,
)

from _paths import ACTIVATIONS_DIR, DATA_DIR, GSEA_LAYERS_CSV, PREP_OUT_DIR


CANCER_TYPE_ORDER = [
    "BRCA", "LUAD", "UCEC", "LGG",  "KIRC", "HNSC", "THCA", "PRAD",
    "LUSC", "SKCM", "COAD", "OV",   "STAD", "BLCA", "LIHC", "CESC",
    "KIRP", "SARC", "ESCA", "PCPG", "PAAD", "READ", "TGCT", "LAML",
    "THYM", "MESO", "UVM",  "ACC",  "KICH", "UCS",  "DLBC", "CHOL",
]


# ── Layer mapping ─────────────────────────────────────────────────────────────

def build_layer_map(hard_links_path: Path, module: str) -> dict[str, int]:
    """
    Return {go_term_id: layer_num} for the chosen module.

    Encoder: term → its sink layer (output of that layer in the encoder).
    Decoder: term → its source layer (input of that layer in the decoder).
             That puts the bottleneck at decoder layer 5, then 6/7/8 going
             toward gene reconstruction. Symmetric to the encoder's 0..3.
    """
    hl = pd.read_csv(hard_links_path)
    if module == "encoder":
        sub = hl[hl["component"] == "encoder"]
        pairs = sub[["sink_term_id", "layer"]].drop_duplicates()
        return dict(zip(pairs["sink_term_id"], pairs["layer"].astype(int)))
    elif module == "decoder":
        sub = hl[hl["component"] == "decoder"]
        pairs = sub[["source_term_id", "layer"]].drop_duplicates()
        return dict(zip(pairs["source_term_id"], pairs["layer"].astype(int)))
    raise ValueError(module)


# ── Activation loading ────────────────────────────────────────────────────────

def load_activations_avg(
    activations_dir: Path,
    version: str,
    seeds: list[int],
    module: str,
    sample_type: str = "Primary Tumor",
    *,
    metric: str = "mean_abs",
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Load all-seed activations for one (version, module) combo.

    metric="mean_abs": take |·|, average across seeds, return mean |activation|
    metric="auc":      average raw values across seeds (no abs), return that.
                       AUC per (cancer_type, GO_term) is computed downstream
                       in aggregate_per_cancer_type.

    Returns:
      meta : (n_samples, metadata_cols) after sample-type filter
      mat  : (n_samples, n_go_terms) ndarray
      go_cols : list of GO term column names in matrix order
    """
    agg_sum: np.ndarray | None = None
    meta: pd.DataFrame | None = None
    go_cols: list[str] | None = None

    for s in tqdm(seeds, desc=f"  reading seeds (AE_{version}, {module}, {metric})", ncols=80):
        path = activations_dir / f"AE_{version}.{s}_{module}_activations.csv.gz"
        t0 = time.time()
        with gzip.open(path, "rt") as fh:
            df = pd.read_csv(fh)
        if go_cols is None:
            go_cols = [c for c in df.columns if c.startswith("GO:")]
            meta_cols = [c for c in df.columns if c not in go_cols]
            meta = df[meta_cols].copy()
        raw = df[go_cols].to_numpy(dtype=np.float32)
        vals = np.abs(raw) if metric == "mean_abs" else raw
        agg_sum = vals if agg_sum is None else agg_sum + vals
        tqdm.write(f"    seed {s}: {df.shape} loaded in {time.time()-t0:.1f}s")

    mat = agg_sum / len(seeds)

    if sample_type != "all":
        mask = (meta["sample_type"] == sample_type).to_numpy()
        mat = mat[mask]
        meta = meta.loc[mask].reset_index(drop=True)

    return meta, mat, go_cols


def aggregate_per_cancer_type(
    mat: np.ndarray,
    meta: pd.DataFrame,
    go_cols: list[str],
    cancer_types: list[str],
    *,
    metric: str = "mean_abs",
) -> pd.DataFrame:
    """
    Build a (n_cancer_types, n_terms) score matrix.

    metric="mean_abs": per-CT mean of |activation| values across samples.
    metric="auc":      per-(CT, term) one-vs-rest ROC-AUC of the per-sample raw
                       activation, symmetrized with max(AUC, 1-AUC). Mirrors
                       fig_graph/compute_node_cancer_specificity.py.
    """
    n_terms = mat.shape[1]
    out = np.zeros((len(cancer_types), n_terms), dtype=np.float32)
    if metric == "mean_abs":
        for i, ct in enumerate(cancer_types):
            mask = (meta["cancer_type"] == ct).to_numpy()
            out[i] = mat[mask].mean(axis=0) if mask.any() else np.nan
    elif metric == "auc":
        labels = meta["cancer_type"].to_numpy()
        for i, ct in enumerate(cancer_types):
            y = (labels == ct).astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                out[i] = np.nan
                continue
            for j in range(n_terms):
                a = roc_auc_score(y, mat[:, j])
                out[i, j] = max(a, 1.0 - a)
    else:
        raise ValueError(metric)
    return pd.DataFrame(out, index=cancer_types, columns=go_cols)


# ── Per-combo runner ──────────────────────────────────────────────────────────

def run_combo(
    *,
    version: str,
    seeds: list[int],
    module: str,
    label: str,
    metric: str,
    null_modes: tuple[str, ...],
    activations_dir: Path,
    hard_links_path: Path,
    enr_pivot: pd.DataFrame,
    output_root: Path,
    n_perms: int,
    rng_seed: int,
    pool: str = "activations",
) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"{label}  (AE_{version}, module={module}, metric={metric}, seeds={seeds})")
    print(f"{'='*60}")

    # pool="auc" averages the per-seed AUC matrices, as compute_baseline_nulls.py does for the baselines,
    # instead of computing one AUC on the seed-averaged activations. A node's sign is arbitrary per seed,
    # so averaging activations can partly cancel.
    per_seed_auc = metric == "auc" and pool == "auc"
    print("Loading activations …")
    if per_seed_auc:
        per_seed = [load_activations_avg(activations_dir, version, [s], module, metric=metric) for s in seeds]
        meta, mat, go_cols = per_seed[0]
    else:
        meta, mat, go_cols = load_activations_avg(
            activations_dir, version, seeds, module, metric=metric,
        )

    layer_map = build_layer_map(hard_links_path, module)
    layers_present = sorted({layer_map[t] for t in go_cols if t in layer_map})
    print(f"  {mat.shape[0]} samples × {len(go_cols)} GO terms across layers {layers_present}")

    # Cancer types present in both meta and enrichment matrix
    present_in_meta = set(meta["cancer_type"].unique())
    cancer_types = [
        ct for ct in CANCER_TYPE_ORDER
        if ct in present_in_meta and ct in enr_pivot.index
    ]
    print(f"  {len(cancer_types)} cancer types")

    # Aggregate to (n_ct, n_terms) once — mean |abs| or per-(ct, term) AUC
    print(f"  aggregating per cancer type (metric={metric}{', averaged over seeds' if per_seed_auc else ''}) …")
    if per_seed_auc:
        act_per_ct = sum(aggregate_per_cancer_type(m, me, cols, cancer_types, metric=metric).astype(np.float64)
                         for me, m, cols in per_seed) / len(per_seed)
    else:
        act_per_ct = aggregate_per_cancer_type(mat, meta, go_cols, cancer_types, metric=metric)

    out_dir = output_root / f"AE_{version}_{module}_{metric}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_label = "AUC (one-vs-rest)" if metric == "auc" else "mean |activation|"

    summary_rows: list[dict] = []
    for L in layers_present:
        layer_terms = [t for t in go_cols if layer_map.get(t) == L]
        layer_terms_gsea = [t for t in layer_terms if t in enr_pivot.columns]
        n_terms = len(layer_terms_gsea)
        if n_terms < 5:
            print(f"  layer {L}: only {n_terms} terms with GSEA — skipping")
            continue

        print(f"  layer {L}: {n_terms}/{len(layer_terms)} GSEA-covered terms — running {n_perms} perms × 3 nulls")

        act = act_per_ct.loc[cancer_types, layer_terms_gsea].to_numpy(dtype=np.float64)
        enr = enr_pivot.loc[cancer_types, layer_terms_gsea].to_numpy(dtype=np.float64)

        obs_g, obs_t, obs_c = _summary_stats(act, enr)
        observed = {"global": obs_g, "per_term": obs_t, "per_ct": obs_c}
        print(f"    observed:  global={obs_g:+.3f}  per-term={obs_t:+.3f}  per-ct={obs_c:+.3f}")

        nulls = {}
        for mode in null_modes:
            nulls[mode] = run_null(
                act, enr, mode=mode, n_perms=n_perms, seed=rng_seed,
                desc=f"    {mode:>6} null  L{L}",
            )
            ps = {m: perm_pval(observed[m], nulls[mode][m])
                  for m in ("global", "per_term", "per_ct")}
            print(f"    {mode:>6} null:  p_global={ps['global']:.3g}  "
                  f"p_per-term={ps['per_term']:.3g}  p_per-ct={ps['per_ct']:.3g}")

        # Skip the 9-panel summary plot when we only ran one null mode.
        if len(null_modes) == 3:
            plot_combined(
                nulls, observed, n_perms,
                out_dir / f"perm_layer_{L}.png",
                label=f"{label}  layer {L}  ({n_terms} terms)",
                metric_label=metric_label,
            )

        # Save the raw null arrays so the trajectory-violin script can render
        # the actual null shape, not a normal approximation.
        np.savez(
            out_dir / f"perm_layer_{L}_arrays.npz",
            metric_label=np.array(metric_label),
            layer=np.int32(L),
            n_terms=np.int32(n_terms),
            **{f"{mode}__{mkey}": nulls[mode][mkey]
               for mode in null_modes
               for mkey in ("global", "per_term", "per_ct")},
            **{f"observed__{mkey}": np.float64(observed[mkey])
               for mkey in ("global", "per_term", "per_ct")},
        )

        for null_mode in null_modes:
            for mkey in ("global", "per_term", "per_ct"):
                p = perm_pval(observed[mkey], nulls[null_mode][mkey])
                summary_rows.append({
                    "version": version,
                    "module": module,
                    "label": label,
                    "layer": L,
                    "n_terms": n_terms,
                    "null_mode": null_mode,
                    "metric": mkey,
                    "observed": observed[mkey],
                    "null_mean": float(nulls[null_mode][mkey].mean()),
                    "null_std":  float(nulls[null_mode][mkey].std()),
                    "p_value":   p,
                })

    return summary_rows


# ── Trajectory figure ─────────────────────────────────────────────────────────

VARIANT_STYLES = {
    "2.0": {"color": "#1f77b4", "marker": "o",
            "label": f"{EMB_VERSION_DISPLAY['2.0']} (AE_2.0)"},
    "2.1": {"color": "#2ca02c", "marker": "s",
            "label": f"{EMB_VERSION_DISPLAY['2.1']} (AE_2.1)"},
}


def plot_trajectory(
    summary_df: pd.DataFrame,
    module: str,
    null_mode: str,
    output_path: Path,
) -> None:
    """OntoVAE-style two-panel layer trajectory comparing model variants."""
    sub = summary_df[(summary_df["module"] == module) & (summary_df["null_mode"] == null_mode)]
    if sub.empty:
        print(f"  no rows for module={module}, null={null_mode} — skipping trajectory")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.32})

    for ax, metric, ylabel, title_prefix in [
        (axes[0], "per_term", "Median per-term Spearman r", "Per-term"),
        (axes[1], "global",   "Global Spearman r",          "Global"),
    ]:
        for version, sty in VARIANT_STYLES.items():
            v = (sub[(sub["version"] == version) & (sub["metric"] == metric)]
                 .sort_values("layer"))
            if v.empty:
                continue
            ax.plot(v["layer"], v["observed"], "-", linewidth=1.4,
                    color=sty["color"], marker=None, label=sty["label"])
            sig = (v["p_value"] < 0.05).to_numpy()
            ax.scatter(v.loc[sig, "layer"], v.loc[sig, "observed"],
                       s=55, color=sty["color"], marker=sty["marker"],
                       edgecolor=sty["color"], linewidth=1.2, zorder=3)
            ax.scatter(v.loc[~sig, "layer"], v.loc[~sig, "observed"],
                       s=55, color="white", marker=sty["marker"],
                       edgecolor=sty["color"], linewidth=1.2, zorder=3)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"{module} layer", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{title_prefix} correlation vs {module} layer\n"
                     "(filled = perm p<0.05, hollow = ns)", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Activation–enrichment correlation across {module} layers"
        f" (null = {null_mode})",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",        type=Path, default=DATA_DIR)
    parser.add_argument("--activations-dir", type=Path, default=ACTIVATIONS_DIR)
    parser.add_argument("--gsea-csv",        type=Path, default=GSEA_LAYERS_CSV,
                        help="GSEA results across all layer terms (default: fig_deg/gsea/gsea_results.csv)")
    parser.add_argument("--output-dir",      type=Path, default=PREP_OUT_DIR / "perm_nulls_gonnect")
    parser.add_argument("--seeds",           type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--n-perms",         type=int, default=1000)
    parser.add_argument("--rng-seed",        type=int, default=42)
    parser.add_argument("--combos",          type=str, nargs="+",
                        default=["2.0:encoder", "2.0:decoder", "2.1:encoder", "2.1:decoder"],
                        help="version:module pairs to run")
    parser.add_argument("--metrics",         type=str, nargs="+",
                        default=["mean_abs", "auc"],
                        choices=["mean_abs", "auc"],
                        help="Per-(ct, term) score(s) to compute.")
    parser.add_argument("--null-modes",      type=str, nargs="+",
                        default=["col"],
                        choices=["row", "col", "rowcol"],
                        help="Which shuffle nulls to compute. Default 'col' "
                             "(only what the headline figure needs).")
    parser.add_argument("--pool",            type=str, default="activations",
                        choices=["activations", "auc"],
                        help="How --metrics auc pools the seeds: one AUC on the seed-averaged "
                             "activations (the preprint v3 figure), or the per-seed AUC averaged, "
                             "as the baseline nulls do.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build a single enrichment pivot up-front (shared across all combos)
    print(f"Loading GSEA results from {args.gsea_csv} …")
    gsea = pd.read_csv(args.gsea_csv)
    gsea["enr_score"] = -np.log10(np.clip(gsea["NOM p-val"], 1e-3, 1.0))
    enr_pivot = gsea.pivot_table(index="cancer_type", columns="Term",
                                 values="enr_score", aggfunc="first")
    print(f"  {enr_pivot.shape[0]} cancer types × {enr_pivot.shape[1]} GO terms")

    labels = {v: f"{EMB_VERSION_DISPLAY[v]} (AE_{v})" for v in ("2.0", "2.1")}

    all_rows: list[dict] = []
    t_overall = time.time()
    todo = [(combo, metric) for combo in args.combos for metric in args.metrics]
    for i, (combo, metric) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] elapsed {time.time()-t_overall:.0f}s")
        version, module = combo.split(":")
        rows = run_combo(
            version=version,
            seeds=args.seeds,
            module=module,
            label=labels.get(version, f"AE_{version}"),
            metric=metric,
            null_modes=tuple(args.null_modes),
            activations_dir=args.activations_dir,
            hard_links_path=args.data_dir / "hard_links.csv",
            enr_pivot=enr_pivot,
            output_root=args.output_dir,
            n_perms=args.n_perms,
            rng_seed=args.rng_seed,
            pool=args.pool,
        )
        for r in rows:
            r["metric_score"] = metric
        all_rows.extend(rows)

    summary_df = pd.DataFrame(all_rows)
    summary_path = args.output_dir / "summary_all.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved unified summary to {summary_path}")

    # Trajectory plots — strictest null (rowcol) by default; also save row/col variants
    for null_mode in ("rowcol", "row", "col"):
        for module in sorted(summary_df["module"].unique()):
            plot_trajectory(
                summary_df, module=module, null_mode=null_mode,
                output_path=args.output_dir / f"trajectory_{module}_{null_mode}.png",
            )


if __name__ == "__main__":
    main()
