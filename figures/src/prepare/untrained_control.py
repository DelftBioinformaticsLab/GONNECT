"""Figure 4's statistic on an untrained GONNECT: what the wiring gives for free.

Figure 4 asks, per cancer type, whether the GO nodes that separate that cancer
type best are the ones whose gene sets GSEA finds most enriched in it. A GONNECT
encoder node only sees the genes of its own term, so random weights already make
it separate the cancer types in which those genes shift -- and the column-shuffle
test, which breaks exactly that node-to-gene-set pairing, cannot tell the wiring
apart from what training added. This step measures the wiring's share directly:
it builds GONNECT from the shipped masks, leaves it at its initialization, and
runs the same statistic on it.

The statistic is Figure 4's per-seed dot: per (cancer type, node) the one-vs-rest
ROC-AUC over primary tumours, symmetrized as max(AUC, 1 - AUC); per cancer type
the Spearman r across one layer's nodes against the GSEA -log10(NOM p) of the
same terms; the median over cancer types. Every trained seed and every
initialization gives one value per layer, and each also gets its own
column-shuffle p.

  untrained   --n-inits initializations each of GONNECT and GONNECT-SL, seeded
              0, 1, ...; GONNECT-SL starts from the same GO weights plus soft
              links drawn from N(0, 1e-3), as its training did
  trained     the five shipped seeds of AE_2.0 (GONNECT) and AE_2.1 (GONNECT-SL)

The trained runs were never seeded, so no initialization here is the start of a
trained seed: the comparison is between distributions, not pairs.

The shipped decoder activations
-------------------------------
The files in go_term_activations/ were written by a version of
gonnect.analysis.activations.activations_per_term that labelled every decoder
layer's outputs with the terms of that layer's *input*. Each decoder column
therefore holds the node at the same position one layer closer to the genes;
the columns named after bottleneck terms hold nodes of decoder layer 6. This
step detects that layout -- those columns are then an affine function of the
latent z, but not z itself -- and relabels: the bottleneck is read from
latent_embeddings/, and every other column is named after the node it holds.
Decoder layer 6 was only written as far as the bottleneck is wide, so 51 of its
92 GSEA-covered terms are recovered; the other decoder layers are complete. The
untrained decoder is scored on the same terms, so the two sides of a comparison
always see one term set. Files regenerated with the corrected function are
recognized and used as they are: --activations-dir reads the trained models
from such files, like extract_decoder_activations.py's complete ones, while the
`as shipped` states below still come from --data-dir.

While the shipped files are in the shifted layout, two more states keep the
comparison with what Figure 4 plots: `trained, as shipped` scores the files
under their own labels, and `untrained, as shipped` labels the untrained decoder
the way the old function would have, on the same terms. Together they are the
whole analysis as it stood, with the untrained reference added.

Writes, to figures/out/prepare/untrained_control/:

  untrained_control_values.tsv    one row per (model, layer, state, seed)
  untrained_control_summary.tsv   one row per (model, layer): trained against
                                  untrained, with a two-sided Mann-Whitney p and
                                  its Benjamini-Hochberg q across the table
  untrained_control.png / .pdf
  fig4_untrained_corrected.tsv    the untrained reference for fig4.py --untrained,
  fig4_untrained_as_shipped.tsv   with the decoder labelled correctly, or as the
                                  shipped files label it (the encoder is the
                                  same in both)

Deterministic: the initializations and the column shuffles are seeded. Besides
--data-dir it needs the masks under out/masks/, which are in the repository.

Run from the repository root:
    pixi run python figures/src/prepare/untrained_control.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy.stats import false_discovery_control, mannwhitneyu, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_DIR, PREP_OUT_DIR
from plot_perm_nulls import _summary_stats, perm_pval, run_null
from plot_perm_nulls_layers import CANCER_TYPE_ORDER, aggregate_per_cancer_type, build_layer_map

from gonnect.model.build_model import build_model

REPO_ROOT = Path(__file__).resolve().parents[3]
MASKS_DIR = REPO_ROOT / "out" / "masks"
OUT_DIR = PREP_OUT_DIR / "untrained_control"

# The architecture AE_2.0 and AE_2.1 were trained with (src/AE_2.*.py)
DATASET = "TCGA_complete_bp_top1k"
N_META_COLS = 5
MERGE_CONDITIONS = (1, 30, 50)
N_GO_LAYERS_USED = 5

MODELS = {"GONNECT": ("2.0", False), "GONNECT-SL": ("2.1", True)}   # name: (version, soft links)
TRAINED_SEEDS = [2, 3, 4, 5, 6]
MODULES = ["encoder", "decoder"]
LAYERS = {"encoder": [0, 1, 2, 3], "decoder": [5, 6, 7, 8]}          # Figure 4's (and hard_links.csv's) numbering
MODULE_ABBREV = {"encoder": "enc", "decoder": "dec"}

TRAINED = "trained"
AS_SHIPPED = "trained, as shipped"
UNTRAINED = "untrained"
UNTRAINED_AS_SHIPPED = "untrained, as shipped"

# Every hard_links.csv view of each node layer, as (component, layer, side). A layer is the sink of one
# linear layer and the source of the next, in the encoder and the decoder alike, and all views must agree.
NODE_LAYERS = {
    "A": [("encoder", 0, "sink"), ("encoder", 1, "source"), ("decoder", 7, "sink"), ("decoder", 8, "source")],
    "B": [("encoder", 1, "sink"), ("encoder", 2, "source"), ("decoder", 6, "sink"), ("decoder", 7, "source")],
    "C": [("encoder", 2, "sink"), ("encoder", 3, "source"), ("decoder", 5, "sink"), ("decoder", 6, "source")],
    "bottleneck": [("encoder", 3, "sink"), ("decoder", 5, "source")],
}
ENCODER_OUTPUTS = ["A", "B", "C", "bottleneck"]   # what encoder linear layer i outputs
DECODER_INPUTS = ["bottleneck", "C", "B", "A"]    # what decoder linear layer i reads ...
DECODER_OUTPUTS = ["C", "B", "A"]                 # ... and outputs; its last one outputs the genes

# Figure 4's model colours, which pass the categorical CVD checks against white (deutan dE 23.5).
# The untrained runs are context, so they stay grey.
MODEL_COLORS = {"GONNECT": "#1f77b4", "GONNECT-SL": "#2ca02c"}
GREY = "#a3a3a3"
GREY_DARK = "#5f5f5f"
INK = "#1f1f1f"
RULE = "#8c8c8c"
GRID = "#e8e8e8"


@dataclass
class Scoring:
    """What every set of activations is scored against."""
    primary: np.ndarray          # which samples are primary tumours
    labels: np.ndarray           # their cancer types
    cancer_types: list[str]      # the cancer types with both samples and GSEA results
    enrichment: pd.DataFrame     # cancer type x term, -log10(NOM p)
    layer_maps: dict             # module -> {term: Figure 4 layer}
    n_perms: int
    rng_seed: int


# ── Which term every position holds ──────────────────────────────────────────

def node_layers(hard_links: pd.DataFrame, masks_dir: Path) -> dict[str, pd.Series]:
    """{layer name: Series mapping position -> term id}.

    hard_links.csv is checked against the masks the untrained models are built
    from first: a mask's rows are the sinks, and its columns the sources, of
    one hard_links.csv layer.
    """
    for module in MODULES:
        masks = torch.load(masks_dir / module / str(MERGE_CONDITIONS) / f"{DATASET}_dense_edge_masks.pt",
                           weights_only=True)
        # The encoder's last mask and the decoder's first connect the GO root, which the models leave out
        masks = masks[:-1] if module == "encoder" else masks[1:]
        for mask, layer in zip(masks, LAYERS[module]):
            rows = hard_links[(hard_links.component == module) & (hard_links.layer == layer)]
            if set(zip(rows.sink_index, rows.source_index)) != set(map(tuple, mask.nonzero().tolist())):
                raise ValueError(f"hard_links.csv {module} layer {layer} does not match the masks in {masks_dir}")

    layers = {}
    for name, views in NODE_LAYERS.items():
        pairs = pd.concat([
            hard_links[(hard_links.component == component) & (hard_links.layer == layer)]
            [[f"{side}_index", f"{side}_term_id"]].set_axis(["position", "term"], axis=1)
            for component, layer, side in views
        ]).drop_duplicates()
        if pairs.position.duplicated().any():
            raise ValueError(f"hard_links.csv disagrees about which term sits where in layer {name}")
        ids = pairs.set_index("position").term.sort_index()
        if not ids.index.equals(pd.RangeIndex(len(ids))):
            raise ValueError(f"hard_links.csv leaves positions of layer {name} unaccounted for")
        layers[name] = ids
    return layers


def go_columns(values: np.ndarray, ids: pd.Series) -> pd.DataFrame:
    """One layer's values as a frame of its GO-term nodes; proxies and genes are dropped."""
    go = ids[ids.str.startswith("GO:")]
    return pd.DataFrame(values[:, go.index.to_numpy()], columns=go.to_numpy())


# ── Activations ──────────────────────────────────────────────────────────────

def untrained_activations(module: str, soft_links: bool, seed: int, x: torch.Tensor,
                          layers: dict[str, pd.Series], masks_dir: Path) -> dict[str, pd.DataFrame]:
    """Every GO node's pre-activation in a freshly initialized GONNECT.

    Returns {UNTRAINED: frame}, each column named after the node it holds, and
    for the decoder also {UNTRAINED_AS_SHIPPED: frame} with every linear
    layer's outputs named after the terms of its input layer, which is how the
    shipped decoder files were labelled.
    """
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):   # build_model narrates the whole architecture
        model = build_model("dense", module, soft_links, DATASET, False, MERGE_CONDITIONS, N_GO_LAYERS_USED,
                            torch.nn.ReLU, torch.float64, masks_dir=masks_dir)
    model.eval()
    model.set_store_activations(True)
    with torch.no_grad():
        z = model.encoder(x)
        model.decoder(z)
    model.set_store_activations(False)

    coder = model.encoder if module == "encoder" else model.decoder
    outputs = [a.numpy() for a in list(coder.activations.values())[::2]]   # the linear layers
    if module == "encoder":
        frames = [go_columns(out, layers[name]) for out, name in zip(outputs, ENCODER_OUTPUTS)]
        return {UNTRAINED: pd.concat(frames, axis=1)}
    frames = [go_columns(z.numpy(), layers["bottleneck"])]
    frames += [go_columns(out, layers[name]) for out, name in zip(outputs, DECODER_OUTPUTS)]
    as_shipped = [go_columns(out, layers[name]) for out, name in zip(outputs, DECODER_INPUTS)]
    return {UNTRAINED: pd.concat(frames, axis=1), UNTRAINED_AS_SHIPPED: pd.concat(as_shipped, axis=1)}


def read_activations(activations_dir: Path, version: str, seed: int, module: str) -> pd.DataFrame:
    frame = pd.read_csv(activations_dir / f"AE_{version}.{seed}_{module}_activations.csv.gz")
    return frame[[c for c in frame.columns if c.startswith("GO:")]]


def latent(data_dir: Path, version: str, seed: int, module: str) -> np.ndarray:
    path = data_dir / "latent_embeddings" / f"AE_{version}" / f"AE_{version}.{seed}_{module}_full_dataset.pt"
    return torch.load(path, map_location="cpu", weights_only=True).numpy()


def holds_latent(values: np.ndarray, z: np.ndarray) -> bool:
    return np.allclose(values, z, rtol=1e-9, atol=1e-12 * np.abs(z).max())


def decoder_layout(shipped: pd.DataFrame, z: np.ndarray, bottleneck: pd.Series) -> str:
    """'corrected' when the bottleneck columns hold z, 'shifted' when they hold decoder layer 6."""
    held = shipped[bottleneck.to_numpy()].to_numpy()
    if holds_latent(held, z):
        return "corrected"
    # Decoder layer 6 is W z + b, so the shifted columns are an exact affine function of z
    design = np.c_[z, np.ones(len(z))]
    coef, *_ = np.linalg.lstsq(design, held, rcond=None)
    if np.abs(held - design @ coef).max() <= 1e-8 * np.abs(held).max():
        return "shifted"
    raise ValueError("decoder activations match neither the shifted nor the corrected layout")


def relabelled_decoder(shipped: pd.DataFrame, z: np.ndarray, layers: dict[str, pd.Series]) -> pd.DataFrame:
    """Name every shipped decoder column after the node it holds, and take the bottleneck from z.

    The column named after input position j of decoder linear layer i holds that
    layer's output at position j. The columns named after the terms of the last
    GO layer hold reconstructed genes, and are dropped.
    """
    frames = [go_columns(z, layers["bottleneck"])]
    for read, written in zip(DECODER_INPUTS, DECODER_OUTPUTS):
        labels, held = layers[read], layers[written]
        pairs = [(label, held[pos]) for pos, label in labels.items()
                 if label.startswith("GO:") and pos in held.index and held[pos].startswith("GO:")]
        frames.append(pd.DataFrame({node: shipped[label].to_numpy() for label, node in pairs}))
    return pd.concat(frames, axis=1)


# ── Figure 4's statistic ─────────────────────────────────────────────────────

def auc_matrix(frame: pd.DataFrame, scoring: Scoring) -> pd.DataFrame:
    """Per (cancer type, node) one-vs-rest ROC-AUC over primary tumours, symmetrized as max(AUC, 1 - AUC).

    Mann-Whitney U on midranks, which is what roc_auc_score computes (a tie
    counts one half), for every node at once. The values pass through float32
    first, as fig4.py reads them, so ties fall where they fall there.
    """
    values = frame.to_numpy(dtype=np.float32)[scoring.primary]
    ranks = rankdata(values, axis=0)
    out = np.empty((len(scoring.cancer_types), values.shape[1]), dtype=np.float32)
    for i, cancer_type in enumerate(scoring.cancer_types):
        positive = scoring.labels == cancer_type
        n_pos = positive.sum()
        n_neg = len(positive) - n_pos
        auc = (ranks[positive].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        out[i] = np.maximum(auc, 1 - auc)
    return pd.DataFrame(out, index=scoring.cancer_types, columns=frame.columns)


def check_auc(frame: pd.DataFrame, scoring: Scoring, meta_primary: pd.DataFrame) -> None:
    """The vectorized AUC against fig4's own roc_auc_score loop, on a few nodes."""
    cols = list(frame.columns[:40])
    reference = aggregate_per_cancer_type(frame[cols].to_numpy(dtype=np.float32)[scoring.primary], meta_primary,
                                          cols, scoring.cancer_types, metric="auc")
    difference = np.abs(reference.to_numpy() - auc_matrix(frame[cols], scoring).to_numpy()).max()
    if difference > 1e-6:
        raise AssertionError(f"vectorized AUC differs from roc_auc_score by {difference:.2e}")


def layer_terms(columns, module: str, scoring: Scoring) -> dict[int, list[str]]:
    """The terms Figure 4 scores in each layer: that layer's nodes with a GSEA result."""
    return {layer: [t for t in columns if scoring.layer_maps[module].get(t) == layer
                    and t in scoring.enrichment.columns]
            for layer in LAYERS[module]}


def score(frame: pd.DataFrame, terms: dict[int, list[str]], scoring: Scoring) -> dict[int, tuple[float, float]]:
    """{layer: (median per-cancer-type Spearman r, column-shuffle p)}."""
    auc = auc_matrix(frame, scoring)
    out = {}
    for layer, cols in terms.items():
        act = auc[cols].to_numpy(dtype=float)
        enr = scoring.enrichment.loc[scoring.cancer_types, cols].to_numpy(dtype=float)
        with warnings.catch_warnings():
            # A node constant over all samples leaves its per-term correlation undefined; that statistic is unused
            warnings.simplefilter("ignore")
            statistic = _summary_stats(act, enr)[2]
        null = run_null(act, enr, "col", scoring.n_perms, scoring.rng_seed)["per_ct"]
        out[layer] = (statistic, perm_pval(statistic, null))
    return out


# ── Summary and figure ───────────────────────────────────────────────────────

def summarize(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, layer), group in values.groupby(["model", "layer"], sort=False):
        trained = group[group.state == TRAINED]
        untrained = group[group.state == UNTRAINED]
        shipped = group[group.state == AS_SHIPPED]
        untrained_shipped = group[group.state == UNTRAINED_AS_SHIPPED]
        row = {
            "model": model,
            "layer": layer,
            "n_terms": int(trained.n_terms.iloc[0]),
            "trained_mean": trained.median_r.mean(),
            "trained_sd": trained.median_r.std(),
            "untrained_mean": untrained.median_r.mean(),
            "untrained_sd": untrained.median_r.std(),
            "untrained_q025": untrained.median_r.quantile(0.025),
            "untrained_q975": untrained.median_r.quantile(0.975),
            "difference": trained.median_r.mean() - untrained.median_r.mean(),
            "mannwhitney_p": mannwhitneyu(trained.median_r, untrained.median_r, alternative="two-sided").pvalue,
            # Share of runs whose own column-shuffle test comes out significant
            "trained_colshuffle_sig": (trained.col_shuffle_p < 0.05).mean(),
            "untrained_colshuffle_sig": (untrained.col_shuffle_p < 0.05).mean(),
            "as_shipped_mean": shipped.median_r.mean() if len(shipped) else np.nan,
            "untrained_as_shipped_mean": untrained_shipped.median_r.mean() if len(untrained_shipped) else np.nan,
            "as_shipped_n_terms": int(shipped.n_terms.iloc[0]) if len(shipped) else np.nan,
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.insert(summary.columns.get_loc("mannwhitney_p") + 1, "mannwhitney_q",
                   false_discovery_control(summary.mannwhitney_p, method="bh"))
    return summary


def write_fig4_references(values: pd.DataFrame, out_dir: Path) -> None:
    """The untrained runs in the shape fig4.py --untrained reads: one row per (row, label, init).

    The as-shipped file takes the decoder from the `untrained, as shipped`
    state wherever there is one; the encoder is the same in both files.
    """
    corrected = values[values.state == UNTRAINED]
    relabelled = values[values.state == UNTRAINED_AS_SHIPPED]
    replaced = set(zip(relabelled.model, relabelled.module))
    as_shipped = pd.concat([corrected[[key not in replaced for key in zip(corrected.model, corrected.module)]],
                            relabelled])
    for name, frame in (("corrected", corrected), ("as_shipped", as_shipped)):
        (frame.rename(columns={"model": "row", "layer": "label", "seed": "init"})
         [["row", "label", "init", "n_terms", "median_r"]]
         .to_csv(out_dir / f"fig4_untrained_{name}.tsv", sep="\t", index=False, float_format="%.6g"))


def plot(values: pd.DataFrame, summary: pd.DataFrame, n_inits: int, out_dir: Path) -> None:
    """Per layer: the untrained runs in grey, the trained seeds in the model's colour."""
    layer_names = [f"{MODULE_ABBREV[m]} L{layer}" for m in MODULES for layer in LAYERS[m]]
    # A gap between encoder and decoder, as in Figure 4
    x_of = {name: i + (0.6 if name.startswith("dec") else 0.0) for i, name in enumerate(layer_names)}
    offsets = {UNTRAINED: -0.2, TRAINED: 0.06, AS_SHIPPED: 0.29}

    fig, axes = plt.subplots(len(MODELS), 1, figsize=(9.0, 7.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, model in zip(axes, MODELS):
        color = MODEL_COLORS[model]
        for name, x in x_of.items():
            cell = values[(values.model == model) & (values.layer == name)]
            untrained = cell[cell.state == UNTRAINED].median_r.to_numpy()
            trained = cell[cell.state == TRAINED].median_r.to_numpy()
            shipped = cell[cell.state == AS_SHIPPED].median_r.to_numpy()

            xu = x + offsets[UNTRAINED]
            ax.scatter(xu + rng.uniform(-0.07, 0.07, len(untrained)), untrained, s=9, color=GREY,
                       linewidths=0, alpha=0.75, zorder=2)
            ax.hlines(np.median(untrained), xu - 0.11, xu + 0.11, color=GREY_DARK, linewidth=2.0, zorder=3)

            xt = x + offsets[TRAINED]
            ax.hlines(trained.mean(), xt - 0.13, xt + 0.13, color=color, linewidth=2.0, zorder=3)
            ax.scatter(xt + rng.uniform(-0.035, 0.035, len(trained)), trained, s=34, color=color,
                       edgecolors="white", linewidths=1.0, zorder=4)
            if len(shipped):
                ax.scatter(x + offsets[AS_SHIPPED] + rng.uniform(-0.03, 0.03, len(shipped)), shipped, s=26,
                           facecolors="white", edgecolors=color, linewidths=1.1, zorder=4)

        ax.axhline(0, color=RULE, linewidth=0.7, zorder=1)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)
            ax.spines[side].set_linewidth(0.7)
        ax.tick_params(colors=INK, labelsize=8, length=3, width=0.7, color=RULE)

        ticks = []
        for name in layer_names:
            row = summary[(summary.model == model) & (summary.layer == name)].iloc[0]
            n_shipped = row.as_shipped_n_terms
            partial = pd.notna(n_shipped) and n_shipped != row.n_terms
            ticks.append(f"{name}\n{row.n_terms} of {int(n_shipped)} terms" if partial
                         else f"{name}\n{row.n_terms} terms")
        ax.set_xticks(list(x_of.values()), ticks)
        ax.set_ylabel("median per-cancer-type\nSpearman r", fontsize=9, color=INK)
        ax.set_title(model, loc="left", fontsize=10, fontweight="bold", color=INK)

        handles = [
            Line2D([], [], linestyle="", marker="o", markersize=4, markerfacecolor=GREY, markeredgecolor="none",
                   label=f"untrained, {n_inits} initializations (bar: median)"),
            Line2D([], [], linestyle="", marker="o", markersize=6.5, markerfacecolor=color,
                   markeredgecolor="white", label="trained, 5 seeds (bar: mean)"),
            Line2D([], [], linestyle="", marker="o", markersize=5.5, markerfacecolor="white",
                   markeredgecolor=color, label="trained decoder, labels as shipped (Figure 4)"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8, labelcolor=INK,
                  handletextpad=0.3, borderaxespad=0.2)

    fig.tight_layout(h_pad=2.0)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"untrained_control.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--masks-dir", type=Path, default=MASKS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--activations-dir", type=Path, default=None,
                        help="trained activations (default: <data-dir>/go_term_activations, whose decoder is "
                             "relabelled), e.g. extract_decoder_activations.py's complete ones")
    parser.add_argument("--n-inits", type=int, default=50,
                        help="untrained initializations per model and module")
    parser.add_argument("--n-perms", type=int, default=1000, help="column shuffles behind every p")
    parser.add_argument("--rng-seed", type=int, default=42, help="seed of the column shuffles, as in Figure 4")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    expression = pd.read_csv(args.data_dir / f"{DATASET}.csv.gz")
    meta = expression.iloc[:, :N_META_COLS]
    x = torch.tensor(expression.iloc[:, N_META_COLS:].to_numpy(dtype=np.float64))
    primary = (meta["sample_type"] == "Primary Tumor").to_numpy()
    meta_primary = meta.loc[primary].reset_index(drop=True)
    labels = meta_primary["cancer_type"].to_numpy()

    gsea = pd.read_csv(args.data_dir / "gsea_gonnect_layers" / "gsea_results.csv")
    gsea["enr_score"] = -np.log10(np.clip(gsea["NOM p-val"], 1e-3, 1.0))
    enrichment = gsea.pivot_table(index="cancer_type", columns="Term", values="enr_score", aggfunc="first")

    hard_links_path = args.data_dir / "hard_links.csv"
    hard_links = pd.read_csv(hard_links_path, usecols=["component", "layer", "source_index", "source_term_id",
                                                       "sink_index", "sink_term_id"])
    layers = node_layers(hard_links, args.masks_dir)
    scoring = Scoring(
        primary=primary,
        labels=labels,
        cancer_types=[ct for ct in CANCER_TYPE_ORDER if ct in set(labels) and ct in enrichment.index],
        enrichment=enrichment,
        layer_maps={module: build_layer_map(hard_links_path, module) for module in MODULES},
        n_perms=args.n_perms,
        rng_seed=args.rng_seed,
    )
    print(f"{primary.sum()} primary tumours, {len(scoring.cancer_types)} cancer types; "
          f"hard_links.csv matches the masks in {args.masks_dir}", flush=True)

    rows = []

    def record(model, module, state, seed, terms, scores):
        for layer, (statistic, p) in scores.items():
            rows.append({"model": model, "module": module, "layer": f"{MODULE_ABBREV[module]} L{layer}",
                         "state": state, "seed": seed, "n_terms": len(terms[layer]),
                         "median_r": statistic, "col_shuffle_p": p})

    shipped_dir = args.data_dir / "go_term_activations"
    activations_dir = args.activations_dir or shipped_dir
    for model, (version, soft_links) in MODELS.items():
        for module in MODULES:
            terms = shipped_terms = None
            for seed in TRAINED_SEEDS:
                source = read_activations(activations_dir, version, seed, module)
                z = latent(args.data_dir, version, seed, module)
                if module == "encoder":
                    if not holds_latent(source[layers["bottleneck"].to_numpy()].to_numpy(), z):
                        raise ValueError(f"AE_{version}.{seed} encoder: bottleneck columns do not hold the latent")
                    layout, trained = "corrected", source
                    if terms is None:
                        check_auc(source, scoring, meta_primary)
                else:
                    layout = decoder_layout(source, z, layers["bottleneck"])
                    trained = source if layout == "corrected" else relabelled_decoder(source, z, layers)
                    # What Figure 4 scored: the shipped files, whichever files the trained models are read from
                    shipped = source if activations_dir == shipped_dir else read_activations(shipped_dir, version,
                                                                                               seed, module)
                    if decoder_layout(shipped, z, layers["bottleneck"]) == "shifted":
                        if shipped_terms is None:
                            shipped_terms = layer_terms(shipped.columns, module, scoring)
                        record(model, module, AS_SHIPPED, seed, shipped_terms, score(shipped, shipped_terms, scoring))
                if terms is None:
                    terms = layer_terms(trained.columns, module, scoring)
                record(model, module, TRAINED, seed, terms, score(trained, terms, scoring))
            coverage = ", ".join(f"L{layer} {len(cols)}" for layer, cols in terms.items())
            print(f"[{time.time() - start:5.0f}s] {model} {module}: trained seeds scored "
                  f"({layout} layout; GSEA-covered terms {coverage})", flush=True)

            for init in range(args.n_inits):
                frames = untrained_activations(module, soft_links, init, x, layers, args.masks_dir)
                record(model, module, UNTRAINED, init, terms, score(frames[UNTRAINED], terms, scoring))
                if shipped_terms is not None:
                    record(model, module, UNTRAINED_AS_SHIPPED, init, shipped_terms,
                           score(frames[UNTRAINED_AS_SHIPPED], shipped_terms, scoring))
            print(f"[{time.time() - start:5.0f}s] {model} {module}: {args.n_inits} initializations scored",
                  flush=True)

    values = pd.DataFrame(rows)
    values.to_csv(args.out_dir / "untrained_control_values.tsv", sep="\t", index=False, float_format="%.6g")
    write_fig4_references(values, args.out_dir)
    summary = summarize(values)
    summary.to_csv(args.out_dir / "untrained_control_summary.tsv", sep="\t", index=False, float_format="%.6g")
    plot(values, summary, args.n_inits, args.out_dir)
    print(f"\nwrote {args.out_dir}\n")

    shown = summary.drop(columns=["as_shipped_n_terms"])
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
