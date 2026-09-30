"""The figures that read decoder activations, twice: as shipped, and corrected.

The shipped decoder activations are labelled one layer off (see
untrained_control.py). Three figures read them: Figure 4 (decoder violins of
panels a and b), Figure S6 (the decoder heatmaps) and Figure 5 (panel c, through
activation_preservation_per_node.csv). This draws each of them twice, into
as_published/ and decoder_fixed/. The corrected activations come from
--fixed-dir: by default extract_decoder_activations.py's, re-extracted from the
checkpoints and complete; without the checkpoints,
relabel_decoder_activations.py's (--fixed-dir .../decoder_relabelled), which
keep 51 of decoder layer 6's 92 GSEA-covered terms.

  fig4    as published: the shipped activations and perm_nulls_gonnect/, with
          the untrained decoder labelled the way the old activations_per_term
          labelled it (fig4_untrained_as_shipped.tsv). Fixed: the corrected
          activations, the nulls plot_perm_nulls_layers.py rebuilt from them,
          and the untrained decoder labelled correctly
          (fig4_untrained_corrected.tsv). The fixed run starts its per-seed AUC
          cache from the shipped one minus the GONNECT decoder entries, which
          it recomputes.
  figS6   the shipped activations, then the corrected ones.
  fig5    the shipped activation_preservation_per_node.csv, then the one
          activation_preservation.py builds from the corrected activations.
          Only panel c reads decoder activations. Its table compares two
          rebuilds by the same code, the shipped activations against the
          corrected ones, so the encoder rows come out identical.

Every other input is the same shipped file in both runs, so whatever does not
read the decoder activations must come out identical; the fig4 table checks
panels c-e for it.

Needs, from the steps it compares (run them first; see prepare/README.md), with
<fixed> the --fixed-dir:

  extract_decoder_activations.py  <fixed>/go_term_activations/        (all)
  plot_perm_nulls_layers.py       <fixed>/perm_nulls_gonnect/         (fig4)
  activation_preservation.py      <fixed>/activation_preservation/    (fig5)
    and on the shipped files      activation_preservation/            (fig5)
  untrained_control.py            untrained_control/fig4_untrained_*.tsv, run
                                  with --activations-dir <fixed>/go_term_activations (fig4)

Writes, to figures/out/prepare/fig4_decoder_comparison/:

  as_published/, decoder_fixed/   fig4, figS6 and fig5 (.png, .pdf; fig4 also .csv)
  comparison_fig4.tsv    every panel a and b violin under both labellings
  comparison_figS6.tsv   per GO term: which node the shipped column held, and
                         the agreement of the three instances' per-cancer-type
                         means, signed and absolute (the figure's claim)
  comparison_fig5.tsv    per module and layer: the activation-preservation summary

Run from the repository root:
    pixi run python figures/src/prepare/fig4_decoder_comparison.py [--figures figS6 fig5]
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PREPARE_DIR = Path(__file__).resolve().parent
FIGURES_SRC = PREPARE_DIR.parent
sys.path.insert(0, str(PREPARE_DIR))
from _paths import DATA_DIR, PREP_OUT_DIR

OUT_DIR = PREP_OUT_DIR / "fig4_decoder_comparison"
UNTRAINED_DIR = PREP_OUT_DIR / "untrained_control"
UNTRAINED_BASELINES = PREP_OUT_DIR / "untrained_baselines" / "fig4_untrained_baselines.tsv"
FIXED_DIR = PREP_OUT_DIR / "decoder_reextracted"
REBUILT_PRESERVATION = PREP_OUT_DIR / "activation_preservation" / "activation_preservation_per_node.csv"
FIGURES = ["fig4", "figS6", "fig5"]
GONNECT_ROWS = ["GONNECT", "GONNECT-SL"]
VARIANTS = ["as_published", "decoder_fixed"]


def run(script: str, out_dir: Path, data_dir: Path, extra: list[str]) -> None:
    command = [sys.executable, str(FIGURES_SRC / script), "--data-dir", str(data_dir),
               "--out-dir", str(out_dir), *extra]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


# ── Figure 4 ─────────────────────────────────────────────────────────────────

def seed_cache(shipped: Path, target: Path) -> None:
    """A fresh per-seed AUC cache: the shipped one without the GONNECT decoder entries, which the corrected files
    replace. Rebuilt on every run, since fig4.py does not check an entry against the files it came from, so one
    left by an earlier --fixed-dir would be reused silently."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for entry in shipped.glob("*.npz"):
        if not (entry.name.startswith("AE_") and "_decoder_" in entry.name):
            shutil.copy2(entry, target / entry.name)


def compare_fig4(published: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    """Panels a and b side by side; panels c-e checked for being unchanged."""
    keys = ["row", "label"]
    others_p = published[~published.row.isin(GONNECT_ROWS)].set_index(keys)
    others_f = fixed[~fixed.row.isin(GONNECT_ROWS)].set_index(keys)
    columns = ["obs_pooled", "perm_p"]
    if not np.allclose(others_p[columns].to_numpy(), others_f.loc[others_p.index, columns].to_numpy(),
                       rtol=0, atol=1e-12):
        print("WARNING: panels c-e differ between the two runs, though neither reads the decoder files")

    table = []
    seeds = [c for c in published.columns if c.startswith("seed_")]
    for (row, label), p in published[published.row.isin(GONNECT_ROWS)].set_index(keys).iterrows():
        f = fixed.set_index(keys).loc[(row, label)]
        record = {"row": row, "label": label}
        for name, result in (("published", p), ("fixed", f)):
            record[f"obs_pooled_{name}"] = result.obs_pooled
            record[f"perm_p_{name}"] = result.perm_p
            record[f"seed_mean_{name}"] = result[seeds].astype(float).mean()
            record[f"untrained_median_{name}"] = result.untrained_median
            record[f"untrained_q975_{name}"] = result.untrained_q975
        table.append(record)
    return pd.DataFrame(table)


def fig4(args) -> None:
    # The baselines' untrained reference does not depend on the decoder labels, so both runs take it when present
    baselines = [str(UNTRAINED_BASELINES)] if UNTRAINED_BASELINES.exists() else []
    for variant, extra in (
            ("as_published", ["--untrained", str(args.untrained_dir / "fig4_untrained_as_shipped.tsv"), *baselines]),
            ("decoder_fixed", ["--untrained", str(args.untrained_dir / "fig4_untrained_corrected.tsv"), *baselines,
                               "--activations-dir", str(args.fixed_dir / "go_term_activations"),
                               "--gonnect-nulls-dir", str(args.fixed_dir / "perm_nulls_gonnect"),
                               "--cache-dir", str(args.out_dir / "decoder_fixed" / "cache")])):
        if variant == "decoder_fixed":
            seed_cache(args.data_dir / "cache" / "auc_4row", args.out_dir / variant / "cache")
        run("fig4.py", args.out_dir / variant, args.data_dir, extra)

    published, fixed = (pd.read_csv(args.out_dir / v / "fig4.csv") for v in VARIANTS)
    table = compare_fig4(published, fixed)
    table.to_csv(args.out_dir / "comparison_fig4.tsv", sep="\t", index=False, float_format="%.6g")
    print(table[["row", "label", "obs_pooled_published", "obs_pooled_fixed", "perm_p_published", "perm_p_fixed",
                 "seed_mean_published", "seed_mean_fixed", "untrained_median_published", "untrained_median_fixed"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ── Figure S6 ────────────────────────────────────────────────────────────────

def shipped_column_holds(data_dir: Path, masks_dir: Path) -> dict[str, str]:
    """{decoder GO term: the node its shipped column actually holds}."""
    from untrained_control import DECODER_INPUTS, DECODER_OUTPUTS, node_layers

    hard_links = pd.read_csv(data_dir / "hard_links.csv", usecols=["component", "layer", "source_index",
                                                                   "source_term_id", "sink_index", "sink_term_id"])
    layers = node_layers(hard_links, masks_dir)
    # The decoder's last linear layer outputs the genes, in the expression matrix's column order
    genes = pd.read_csv(data_dir / "TCGA_complete_bp_top1k.csv.gz", nrows=0).columns[5:]
    layers["genes"] = pd.Series([f"gene {g}" for g in genes])
    holds = {}
    for read, written in zip(DECODER_INPUTS, DECODER_OUTPUTS + ["genes"]):
        for pos, label in layers[read].items():
            if label.startswith("GO:"):
                holds[label] = layers[written][pos]
    return holds


def compare_figS6(args) -> pd.DataFrame:
    """Per term, what Figure S6 claims: instances agree on |mean| per cancer type but not on its sign."""
    sys.path.insert(0, str(FIGURES_SRC))
    import figS5
    from _common import CANCER_TYPE_ORDER, load_cancer_types
    from untrained_control import MASKS_DIR

    hard_links = args.data_dir / "hard_links.csv"
    tcga = args.data_dir / "TCGA_complete_bp_top1k.csv.gz"
    allowed = figS5.eligible_go_terms(hard_links, "decoder")
    labels = load_cancer_types(tcga)
    patient_ids = pd.read_csv(tcga, usecols=["patient_id"])["patient_id"]
    cancer_types = [ct for ct in CANCER_TYPE_ORDER if ct in set(labels)]
    names = figS5.go_term_names(hard_links)
    holds = shipped_column_holds(args.data_dir, MASKS_DIR)

    def mean_r(profiles):
        rs = [np.corrcoef(a, b)[0, 1] for a, b in itertools.combinations(profiles, 2)
              if np.ptp(a) > 0 and np.ptp(b) > 0]
        return float(np.mean(rs)) if rs else np.nan

    table = {t: {"term": t, "name": names.get(t, ""), "shipped_column_holds": holds.get(t, "")}
             for t in figS5.GO_TERMS}
    for variant, directory in (("published", args.data_dir / "go_term_activations"),
                               ("fixed", args.fixed_dir / "go_term_activations")):
        means = [figS5.mean_per_cancer_type(figS5._activation_path(directory, "decoder", seed), allowed, labels,
                                            patient_ids, cancer_types) for seed in figS5.DEFAULT_SEEDS]
        for term in figS5.GO_TERMS:
            profiles = [m[term].to_numpy() for m in means]
            table[term][f"signed_r_{variant}"] = mean_r(profiles)
            # Panel b is the absolute value of panel a, which centres each column first (figS5.normalize_terms)
            table[term][f"abs_r_{variant}"] = mean_r([np.abs(p - p.mean()) for p in profiles])
            table[term][f"max_abs_mean_{variant}"] = max(float(np.abs(p).max()) for p in profiles)
            table[term][f"range_{variant}"] = min(float(np.ptp(p)) for p in profiles)
    return pd.DataFrame(table.values())


def figS6(args) -> None:
    run("figS6.py", args.out_dir / "as_published", args.data_dir, [])
    run("figS6.py", args.out_dir / "decoder_fixed", args.data_dir,
        ["--activations-dir", str(args.fixed_dir / "go_term_activations")])
    table = compare_figS6(args)
    table.to_csv(args.out_dir / "comparison_figS6.tsv", sep="\t", index=False, float_format="%.6g")
    print(table[["term", "shipped_column_holds", "signed_r_published", "signed_r_fixed", "abs_r_published",
                 "abs_r_fixed", "max_abs_mean_published", "max_abs_mean_fixed"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ── Figure 5 ─────────────────────────────────────────────────────────────────

def fig5(args) -> None:
    from activation_preservation import summarize

    fixed_csv = args.fixed_dir / "activation_preservation" / "activation_preservation_per_node.csv"
    run("fig5.py", args.out_dir / "as_published", args.data_dir, [])
    run("fig5.py", args.out_dir / "decoder_fixed", args.data_dir, ["--preservation-csv", str(fixed_csv)])

    # The table compares two tables built by the same code: the deposited one matches a rebuild from the shipped
    # activations to 1e-12, except for nodes constant in the fixed-link model, whose correlation comes out NaN
    # or +-1e-17 depending on summation order.
    keys = ["module", "layer"]
    published = summarize(pd.read_csv(REBUILT_PRESERVATION)).set_index(keys)
    fixed = summarize(pd.read_csv(fixed_csv)).set_index(keys)
    table = published.join(fixed, lsuffix="_published", rsuffix="_fixed").reset_index()
    table.to_csv(args.out_dir / "comparison_fig5.tsv", sep="\t", index=False, float_format="%.6g")
    print(table[["module", "layer", "n_nodes_per_seed_published", "n_nodes_per_seed_fixed",
                 "median_abs_r_mean_published", "median_abs_r_mean_fixed",
                 "frac_abs_r_gt_0.7_published", "frac_abs_r_gt_0.7_fixed"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--untrained-dir", type=Path, default=UNTRAINED_DIR)
    parser.add_argument("--fixed-dir", type=Path, default=FIXED_DIR)
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=FIGURES)
    args = parser.parse_args()

    needed = {
        "fig4": [args.untrained_dir / "fig4_untrained_as_shipped.tsv",
                 args.untrained_dir / "fig4_untrained_corrected.tsv",
                 args.fixed_dir / "go_term_activations",
                 args.fixed_dir / "perm_nulls_gonnect"],
        "figS6": [args.fixed_dir / "go_term_activations"],
        "fig5": [args.fixed_dir / "activation_preservation" / "activation_preservation_per_node.csv",
                 REBUILT_PRESERVATION],
    }
    missing = sorted({str(p) for f in args.figures for p in needed[f] if not p.exists()})
    if missing:
        sys.exit("run the steps this compares first; missing:\n  " + "\n  ".join(missing))

    steps = {"fig4": fig4, "figS6": figS6, "fig5": fig5}
    for figure in args.figures:
        print(f"\n== {figure} ==", flush=True)
        steps[figure](args)
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
