"""Supplementary Figure S2 — t-SNE of the soft-link (GONNECT-SL) embedding spaces.

Panels (model type selected by --version, default 2.1 = soft link):
  a  Input space (raw gene expression, PCA-50 then t-SNE)
  b  MLP encoder + MLP decoder          AE_2.0.<seed>_none
  c  MLP encoder + GONNECT decoder      AE_<version>.<seed>_decoder
  d  GONNECT encoder + MLP decoder      AE_<version>.<seed>_encoder
  e  GONNECT encoder + GONNECT decoder  AE_<version>.<seed>_both

The MLP-MLP reference always comes from AE_2.0, the only experiment that ships
a `_none` variant. --version 2.0 reproduces the Figure 2 embedding panels and
2.2 the randomized ones; the output file is named figS2 either way.

Layout: a 2 x 3 grid on the paper's standard canvas width — panels a-c on the
top row, d and e on the bottom row, and the 32-entry cancer-type legend in the
sixth slot next to panel e.

Inputs, relative to --data-dir:
  TCGA_complete_bp_top1k.csv.gz   expression matrix + cancer-type labels
  latent_embeddings/AE_<version>/AE_<version>.<seed>_<module>_full_dataset.pt
  cache/tsne/                     t-SNE cache (shared with fig2.py)
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from _common import (
    CANCER_COLORS,
    CANCER_TYPE_ORDER,
    EMB_VERSION_DISPLAY as VERSION_LABEL,
    PT_BODY,
    PT_PANEL,
    PT_SMALL,
    PT_TITLE,
    TCGA_META_COLS,
    TSNE_RANDOM_STATE,
    add_io_args,
    display,
    figsize,
    load_embedding,
    pt,
    report_text_overlaps,
    save_figure,
    tsne,
    tsne_cached,
)

# ── Configuration ────────────────────────────────────────────────────────────
PCA_COMPONENTS_INPUT = 50   # pre-reduce input space before t-SNE

PANEL_LABELS = ["a", "b", "c", "d", "e"]

# ── Layout ───────────────────────────────────────────────────────────────────
# The canvas width is fixed by the paper's shared standard, so only the height
# is free. This one is chosen to leave the five scatter panels near-square
# (~5.0 x 4.3 in each) once the margins and the row gap are taken out.
FIG_HEIGHT_IN = 10.5

# Margins are set explicitly rather than left at the matplotlib defaults: the
# figure has to fill the full standard width, since save_figure keeps it and
# trims only vertically. left/bottom hold the y/x labels and the panel letters.
GRID = dict(left=0.024, right=0.990, top=0.947, bottom=0.032,
            wspace=0.09, hspace=0.24)

# Marker area in pt^2, doubled from the 2.0 that suited the old geometry: sizes
# in points shrink with the figure when it is placed on the page, so the same
# number would now print at half the area it used to. 4.0 puts a point at ~0.9
# pt across on the page, which still resolves inside a cluster; going further
# (6.0 and up) floods the dense cores into flat blobs.
MARKER_SIZE = 4.0

# 32 cancer types. Three columns of 11 fit the legend slot comfortably at the
# paper's type scale; two columns of 16 would be taller than the panels beside
# it, and the extra column also spreads the block over more of the slot width.
LEGEND_NCOL = 3


def _panel_titles(version: str) -> list[str]:
    """Panel titles use the same method names as panels a-d of fig2."""
    name = VERSION_LABEL.get(version, f"AE_{version}")
    return [
        "Input space",
        display("MLP"),
        f"{name}-dec",
        f"{name}-enc",
        f"{name}-both",
    ]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _tsne_input(X: np.ndarray) -> np.ndarray:
    """t-SNE of the raw expression matrix, PCA-reduced first."""
    if X.shape[1] > PCA_COMPONENTS_INPUT:
        X = PCA(n_components=PCA_COMPONENTS_INPUT,
                random_state=TSNE_RANDOM_STATE).fit_transform(X)
    return tsne(X)


def _scatter(ax, xy: np.ndarray, labels: np.ndarray, color_map: dict) -> None:
    for ct in CANCER_TYPE_ORDER:
        mask = labels == ct
        if not mask.any():
            continue
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=[color_map[ct]],
            s=MARKER_SIZE,
            linewidths=0,
            rasterized=True,
        )
    ax.set_xticks([])
    ax.set_yticks([])


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Figure S2: t-SNE of GONNECT embeddings.")
    add_io_args(parser)
    parser.add_argument("--seed", type=int, default=2,
                        help="Model seed index (2-6 for 2.0/2.1, 22-26 for 2.2; default: 2)")
    parser.add_argument("--version", type=str, default="2.1", choices=sorted(VERSION_LABEL),
                        help="Model type: 2.0 fixed link, 2.1 soft link (S2), 2.2 randomized")
    args = parser.parse_args()

    tcga_file = args.data_dir / "TCGA_complete_bp_top1k.csv.gz"
    emb_dir = args.data_dir / "latent_embeddings"
    cache_dir = args.data_dir / "cache" / "tsne"
    panel_titles = _panel_titles(args.version)

    # ── Load TCGA data ───────────────────────────────────────────────────────
    print("Loading TCGA data …")
    tcga = pd.read_csv(tcga_file)
    labels = tcga["cancer_type"].values

    gene_cols = [c for c in tcga.columns if c not in TCGA_META_COLS]
    X_input = tcga[gene_cols].values.astype(np.float32)

    # Color map
    color_map = {ct: CANCER_COLORS[i] for i, ct in enumerate(CANCER_TYPE_ORDER)}

    # ── t-SNE per panel (cached under data/cache/tsne) ───────────────────────
    datasets = {}
    datasets["input"] = tsne_cached(cache_dir, "input_space",
                                    lambda: _tsne_input(X_input))

    # `_none` only exists for the fixed-link experiment; the randomized run uses
    # seeds 22-26, so map those back onto the matching AE_2.0 seed (22 -> 2).
    none_seed = args.seed if args.seed < 20 else args.seed - 20

    for version, seed, module in [
        ("2.0", none_seed, "none"),
        (args.version, args.seed, "decoder"),
        (args.version, args.seed, "encoder"),
        (args.version, args.seed, "both"),
    ]:
        datasets[module] = tsne_cached(
            cache_dir,
            f"AE_{version}.{seed}_{module}",
            lambda v=version, s=seed, m=module: tsne(load_embedding(emb_dir, v, s, m)),
        )

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize(FIG_HEIGHT_IN))

    # 2-row layout: 3 panels top, 2 panels + legend bottom
    gs = fig.add_gridspec(2, 3, **GRID)

    panel_data = [
        ("input",   0, 0),
        ("none",    0, 1),
        ("decoder", 0, 2),
        ("encoder", 1, 0),
        ("both",    1, 1),
    ]

    for (key, row, col), label, title in zip(panel_data, PANEL_LABELS, panel_titles):
        ax = fig.add_subplot(gs[row, col])
        _scatter(ax, datasets[key], labels, color_map)
        ax.set_title(title, fontsize=PT_TITLE, pad=pt(2.5))
        # panel letter in top-left corner
        ax.text(-0.035, 1.015, label, transform=ax.transAxes,
                fontsize=PT_PANEL, fontweight="bold", va="bottom", ha="left")
        # axis labels (tsne dims)
        ax.set_xlabel("tSNE-1", fontsize=PT_SMALL, labelpad=pt(1.5))
        ax.set_ylabel("tSNE-2", fontsize=PT_SMALL, labelpad=pt(1.5))

    # ── Legend in bottom-right cell ──────────────────────────────────────────
    ax_legend = fig.add_subplot(gs[1, 2])
    ax_legend.axis("off")
    handles = [
        mpatches.Patch(color=color_map[ct], label=ct)
        for ct in CANCER_TYPE_ORDER
        if ct in np.unique(labels)
    ]
    legend = ax_legend.legend(
        handles=handles,
        ncol=LEGEND_NCOL,
        loc="center",
        fontsize=PT_SMALL,
        frameon=False,
        title="Cancer Type",
        title_fontproperties={"weight": "bold", "size": PT_BODY},
        handlelength=1.3,
        handleheight=0.9,
        columnspacing=1.6,
        labelspacing=0.55,
    )
    # The paper's keys set their heading bold and flush with the swatches under
    # it; matplotlib centres a legend title over the whole block by default.
    legend.set_alignment("left")

    report_text_overlaps(fig, "figS2")
    save_figure(fig, args.out_dir, "figS2")


if __name__ == "__main__":
    main()
