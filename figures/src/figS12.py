"""Supplementary Figure S12: soft-link weight vs. GO graph edit counts.

One figure, two blocks:
  a  Encoder  -- layers 0-3 as columns
  b  Decoder  -- layers 5-8 as columns
Rows are the 8 GO-edit types (4 source-node, 4 sink-node), written out in full.
Each cell is a hexbin of edit count (x) vs. log10 soft-link weight (y) for the
edges whose *relevant endpoint is a real GO term* (source_* rows use GO sources,
sink_* rows use GO sinks; proxy/gene endpoints carry no DAG edits and are
dropped). Panel border colour = signed Spearman rho (coolwarm, white = no
effect); red line = running median; x capped at the 99th percentile for display.

Inputs (relative to --data-dir)
------------------------------
    soft_link_weights/AE_2.1.<seed>_encoder_soft_links.csv.gz
    soft_link_weights/AE_2.1.<seed>_decoder_soft_links.csv.gz
        one soft-link model per module, seed from --seed (default 2)
    cache/sl_edits/AE_2.1.<seed>_<module>_cells.npz
        per-cell (x, y) arrays, so layout iterations do not have to re-stream
        the ~3M-row weight files (minutes per module). Rebuilt automatically
        when missing or stale.

Usage
-----
    python figS12.py [--data-dir figures/data] [--out-dir figures/out] [--seed 2]
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as spstats

from _common import (DEC_LAYERS, ENC_LAYERS, FIG_WIDTH_IN, LAYERS_BY_MODULE,
                     PT_BODY, PT_PANEL, PT_SMALL, PT_TITLE, add_io_args,
                     figsize, pt, report_text_overlaps, save_figure)

# ── edit-type rows (order = top→bottom), with fully written-out labels ─────────
# Paired source/sink for each edit type so the two share a row-pair and an
# x-range (their counts span roughly the same scale).
EDIT_ROWS: List[Tuple[str, str]] = [
    ("source_super_merges", "Source node\nNo. merge events\nin supertree"),
    ("sink_super_merges",   "Sink node\nNo. merge events\nin supertree"),
    ("source_super_prunes", "Source node\nNo. prune events\nin supertree"),
    ("sink_super_prunes",   "Sink node\nNo. prune events\nin supertree"),
    ("source_sub_merges",   "Source node\nNo. merge events\nin subtree"),
    ("sink_sub_merges",     "Sink node\nNo. merge events\nin subtree"),
    ("source_sub_prunes",   "Source node\nNo. prune events\nin subtree"),
    ("sink_sub_prunes",     "Sink node\nNo. prune events\nin subtree"),
]
EDIT_COLS = [c for c, _ in EDIT_ROWS]
LOAD_COLS = ["layer_index", "source_term_id", "sink_term_id", "weight_magnitude"] + EDIT_COLS

_RNG = np.random.default_rng(42)
_CORR_N = 50_000
RHO_MAX = 0.30
BORDER_CMAP = plt.cm.coolwarm
X_DISPLAY_PCT = 99
MIN_PTS = 10

# ── layout ────────────────────────────────────────────────────────────────────
# All of the geometry below is in canvas inches and divided by the figure size
# where matplotlib wants a fraction. The width is fixed at the paper's standard
# 16.5 in, so 8 layer columns get ~1.68 in each and the height is what has to
# give: at the paper's type scale the rotated row labels ("No. merge events" is
# 2.0 in long at PT_BODY) set the minimum row pitch, and eight of them stacked
# is what makes this a full-page figure.
ROW_H = 1.92         # hexbin cell height
ROW_GAP = 0.30       # room for each row's x tick labels
MARGIN_L = 1.72      # shared y label + rotated row labels + col-0 y tick labels
MARGIN_R = 0.08
MARGIN_T = 0.85      # a)/b) headings + the "Layer N" panel titles
MARGIN_B = 1.55      # x tick labels + "edit count" + the rho colourbar
FIG_HEIGHT_IN = MARGIN_T + len(EDIT_ROWS) * ROW_H \
    + (len(EDIT_ROWS) - 1) * ROW_GAP + MARGIN_B

BLOCK_GAP = 0.22     # spacer column between the encoder and decoder blocks
WSPACE = 0.07

X_YLABEL = 0.21      # centre of the rotated "log10 |soft-link weight|"
X_ROWLABEL = 0.80    # centre of the rotated per-row labels
HEAD_DROP = 0.50     # a)/b) heading baseline, below the top of the canvas

CBAR_W, CBAR_H, CBAR_Y = 6.30, 0.10, 0.66

# The per-cell rho annotation is set over two lines: on one line it is 2.1 in
# wide, which does not fit a 1.68 in panel at any size in the paper's scale.
RHO_TEMPLATE = "ρ={rho:+.2f}\n(p={p})"
# Three x ticks per cell at most; the default locator puts 5-6 three-digit
# labels in 1.68 in, which collide with each other and across the column gap.
X_TICK_BINS = 3
# A tick this close to either end of the view gets its label anchored inward.
EDGE_TICK_FRAC = 0.12


def _edit_end(col: str) -> str:
    return "source" if col.startswith("source") else "sink"


def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan, np.nan
    if len(x) > _CORR_N:
        idx = _RNG.choice(len(x), _CORR_N, replace=False)
        x, y = x[idx], y[idx]
    sr = spstats.spearmanr(x, y)
    return float(sr.statistic), float(sr.pvalue)


def _running_median(x, y, n_bins=18, min_pts=MIN_PTS):
    if len(x) < min_pts * 2 or x.max() == x.min():
        return np.array([]), np.array([])
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    cx, cy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum() < min_pts:
            continue
        cx.append((lo + hi) / 2)
        cy.append(np.median(y[sel]))
    return np.array(cx), np.array(cy)


def _fmt_p(p):
    if np.isnan(p):
        return "nan"
    return f"{p:.0e}" if p < 0.001 else f"{p:.3f}"


def compute_cells(module: str, sl_dir: Path, seed: int, cache_dir: Path):
    """Return {(edit_col, layer): (x, log10|w|)} for GO-term-endpoint edges.

    Cached to <cache_dir>/AE_2.1.<seed>_<module>_cells.npz so layout iterations
    don't re-stream the 3M-row weight files. A cache written for a different set
    of edit types / layers is missing keys, so a KeyError falls through to a
    rebuild; deleting the cache forces one too.
    """
    cache = cache_dir / f"AE_2.1.{seed}_{module}_cells.npz"
    if cache.exists():
        print(f"[{module}] loading cache {cache.name}", flush=True)
        try:
            npz = np.load(cache)
            cells = {}
            for col in EDIT_COLS:
                for layer in LAYERS_BY_MODULE[module]:
                    cells[(col, layer)] = (npz[f"{col}|{layer}|x"], npz[f"{col}|{layer}|y"])
            return cells
        except KeyError as exc:
            print(f"[{module}] cache {cache.name} is stale ({exc}); rebuilding",
                  flush=True)

    path = sl_dir / f"AE_2.1.{seed}_{module}_soft_links.csv.gz"
    print(f"[{module}] loading {path.name} ...", flush=True)
    df = pd.read_csv(path, usecols=LOAD_COLS)
    print(f"[{module}] {len(df):,} rows", flush=True)

    wlog = np.log10(df["weight_magnitude"].clip(lower=1e-12).values)
    src_go = df["source_term_id"].str.startswith("GO:").values
    snk_go = df["sink_term_id"].str.startswith("GO:").values
    layer_vals = df["layer_index"].values

    cells, flat = {}, {}
    for col in EDIT_COLS:
        go = src_go if _edit_end(col) == "source" else snk_go
        counts = df[col].values.astype(float)
        for layer in LAYERS_BY_MODULE[module]:
            m = (layer_vals == layer) & go
            cells[(col, layer)] = (counts[m], wlog[m])
            flat[f"{col}|{layer}|x"] = counts[m]
            flat[f"{col}|{layer}|y"] = wlog[m]

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **flat)
    print(f"[{module}] cached -> {cache}", flush=True)
    return cells


def _pair_xlim(arrays):
    """Shared (xmin, xcap) for a source/sink row pair; None if no data."""
    xs = [a for a in arrays if len(a) >= MIN_PTS]
    if not xs:
        return None
    allx = np.concatenate(xs)
    xmin = float(allx.min())
    xcap = float(np.percentile(allx, X_DISPLAY_PCT))
    if xcap <= xmin:
        xcap = float(allx.max())
    if xcap <= xmin:
        xcap = xmin + 1.0
    return (xmin, xcap)


def _fit_edge_xticklabels(ax, xmin, xcap):
    """Keep the outermost x tick labels inside their own panel.

    Columns sit 1.68 in apart with a 0.11 in gap, so a centred three-digit
    label on a tick at the very end of the view hangs ~0.07 in past the panel
    and meets the "0" of the next column coming the other way. Anchoring the
    edge labels inward is the fix that costs neither type size nor panel width.
    Also drops ticks the locator placed outside the view.
    """
    locs = [t for t in ax.get_xticks() if xmin <= t <= xcap]
    ax.set_xticks(locs)
    span = xcap - xmin
    if span <= 0:
        return
    for loc, label in zip(locs, ax.get_xticklabels()):
        frac = (loc - xmin) / span
        if frac < EDGE_TICK_FRAC:
            label.set_ha("left")
        elif frac > 1.0 - EDGE_TICK_FRAC:
            label.set_ha("right")


def _draw_cell(ax, x, y, ylim, xlim, n_tests, is_first_row, is_last_row, layer,
               hide_yticklabels):
    n_pts = len(x)
    if is_first_row:
        ax.set_title(f"Layer {layer}", fontsize=PT_TITLE, fontweight="bold",
                     pad=pt(2.0))

    if n_pts < MIN_PTS:
        ax.text(0.5, 0.5, "no GO\nedges", transform=ax.transAxes,
                ha="center", va="center", fontsize=PT_SMALL, color="#999999")
        ax.set_xticks([])
        for name, sp in ax.spines.items():
            sp.set_visible(False)
        if hide_yticklabels:
            # fully blank cell (e.g. decoder L8 sink rows)
            ax.set_yticks([])
        else:
            # y-scale-bearing column: keep the left edge + y ticks for the row
            ax.set_ylim(*ylim)
            ax.spines["left"].set_visible(True)
            ax.tick_params(axis="y", labelsize=PT_SMALL, pad=pt(1.5))
            ax.tick_params(axis="x", length=0)
        return

    r_s, p_s = _spearman(x, y)
    p_corr = min(p_s * n_tests, 1.0) if not np.isnan(p_s) else np.nan

    xmin, xcap = xlim
    disp = x <= xcap
    ax.hexbin(x[disp], y[disp], gridsize=30, mincnt=1, cmap="Blues", linewidths=0.1)
    cx, cy = _running_median(x[disp], y[disp])
    if len(cx):
        ax.plot(cx, cy, color="tomato", lw=1.8)
    ax.set_xlim(xmin, xcap)
    ax.set_ylim(*ylim)
    ax.locator_params(axis="x", nbins=X_TICK_BINS)

    ax.text(0.04, 0.97, RHO_TEMPLATE.format(rho=r_s, p=_fmt_p(p_corr)),
            transform=ax.transAxes, ha="left", va="top", fontsize=PT_SMALL,
            linespacing=1.0,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.8))

    # Border weight scales with |rho|. The rho = 0 weight is the paper's shared
    # axis line width, so an uncorrelated panel's frame matches every other
    # frame in the paper and the strongest border is 4.75x that.
    t = max(-1.0, min((r_s if not np.isnan(r_s) else 0.0) / RHO_MAX, 1.0))
    lw0 = plt.rcParams["axes.linewidth"]
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER_CMAP(0.5 + 0.5 * t))
        sp.set_linewidth(lw0 * (1.0 + 3.75 * abs(t)))

    if hide_yticklabels:
        ax.set_yticklabels([])
    if is_last_row:
        ax.set_xlabel("edit count", fontsize=PT_BODY, labelpad=pt(1.5))
    ax.tick_params(labelsize=PT_SMALL, pad=pt(1.5))
    _fit_edge_xticklabels(ax, xmin, xcap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_args(parser)
    parser.add_argument("--seed", type=int, default=2,
                        help="AE_2.1.<seed> soft-link model to plot (default: 2)")
    args = parser.parse_args()

    sl_dir = args.data_dir / "soft_link_weights"
    cache_dir = args.data_dir / "cache" / "sl_edits"
    cells = {m: compute_cells(m, sl_dir, args.seed, cache_dir)
             for m in ("encoder", "decoder")}

    # shared y-limit across all populated cells
    los, his = [], []
    for mc in cells.values():
        for x, y in mc.values():
            if len(y) >= MIN_PTS:
                los.append(np.percentile(y, 0.5))
                his.append(np.percentile(y, 99.5))
    ylim = (min(los), max(his))
    print(f"shared ylim (log10|w|): {ylim}")

    n_rows = len(EDIT_ROWS)
    n_lay = len(ENC_LAYERS)
    n_tests = n_rows * n_lay * 2

    # Shared x-range per source/sink row pair, per module and layer.
    xlims = {}
    for module in ("encoder", "decoder"):
        for k in range(n_rows // 2):
            for layer in LAYERS_BY_MODULE[module]:
                arrs = [cells[module][(EDIT_COLS[2 * k + d], layer)][0] for d in (0, 1)]
                xlims[(module, k, layer)] = _pair_xlim(arrs) or (0.0, 1.0)

    # One explicit gridspec: [4 encoder cols | spacer | 4 decoder cols].
    # Narrower panels, tighter columns; between-block spacer kept clearly larger
    # than the within-block gap to separate a) from b).
    fig = plt.figure(figsize=figsize(FIG_HEIGHT_IN))
    gs = fig.add_gridspec(
        n_rows, n_lay * 2 + 1,
        width_ratios=[1] * n_lay + [BLOCK_GAP] + [1] * n_lay,
        left=MARGIN_L / FIG_WIDTH_IN, right=1.0 - MARGIN_R / FIG_WIDTH_IN,
        top=1.0 - MARGIN_T / FIG_HEIGHT_IN, bottom=MARGIN_B / FIG_HEIGHT_IN,
        wspace=WSPACE, hspace=ROW_GAP / ROW_H,
    )
    enc_cols = list(range(n_lay))
    dec_cols = list(range(n_lay + 1, n_lay * 2 + 1))

    row0 = {}            # top-row corner axes for heading placement
    enc_col0 = {}        # encoder col-0 axes per row, for aligned row titles
    for i, (col, _) in enumerate(EDIT_ROWS):
        pair = i // 2
        for jj, layer in enumerate(ENC_LAYERS):
            ax = fig.add_subplot(gs[i, enc_cols[jj]])
            _draw_cell(ax, *cells["encoder"][(col, layer)], ylim,
                       xlims[("encoder", pair, layer)], n_tests,
                       is_first_row=(i == 0), is_last_row=(i == n_rows - 1),
                       layer=layer, hide_yticklabels=(jj != 0))
            if jj == 0:
                enc_col0[i] = ax
            if i == 0 and jj in (0, n_lay - 1):
                row0[("enc", jj)] = ax
        for jj, layer in enumerate(DEC_LAYERS):
            ax = fig.add_subplot(gs[i, dec_cols[jj]])
            _draw_cell(ax, *cells["decoder"][(col, layer)], ylim,
                       xlims[("decoder", pair, layer)], n_tests,
                       is_first_row=(i == 0), is_last_row=(i == n_rows - 1),
                       layer=layer, hide_yticklabels=True)
            if i == 0 and jj in (0, n_lay - 1):
                row0[("dec", jj)] = ax

    # Aligned row titles (describe the x quantity per row), close to the panels.
    # They are what sets ROW_H: turned on their side, the longest line is nearly
    # as long as a row is tall, so shorter rows would run them into each other.
    for i, (_, row_label) in enumerate(EDIT_ROWS):
        pos = enc_col0[i].get_position()
        fig.text(X_ROWLABEL / FIG_WIDTH_IN, (pos.y0 + pos.y1) / 2, row_label,
                 rotation=90, ha="center", va="center", fontsize=PT_BODY,
                 linespacing=1.25)

    # Shared y-axis label (every panel is log10 |soft-link weight|), far left.
    top_y = enc_col0[0].get_position().y1
    bot_y = enc_col0[n_rows - 1].get_position().y0
    fig.text(X_YLABEL / FIG_WIDTH_IN, (top_y + bot_y) / 2,
             "log$_{10}$ | soft-link weight |",
             rotation=90, ha="center", va="center", fontsize=PT_TITLE)

    # Centered bold "a) Encoder" / "b) Decoder" headings, with space underneath.
    enc_l = row0[("enc", 0)].get_position()
    enc_r = row0[("enc", n_lay - 1)].get_position()
    dec_l = row0[("dec", 0)].get_position()
    dec_r = row0[("dec", n_lay - 1)].get_position()
    y_head = 1.0 - HEAD_DROP / FIG_HEIGHT_IN
    fig.text((enc_l.x0 + enc_r.x1) / 2, y_head, "a)  Encoder",
             ha="center", va="bottom", fontsize=PT_PANEL, fontweight="bold")
    fig.text((dec_l.x0 + dec_r.x1) / 2, y_head, "b)  Decoder",
             ha="center", va="bottom", fontsize=PT_PANEL, fontweight="bold")

    # Colorbar (plain, no extend), a touch further beneath the grid.
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(-RHO_MAX, RHO_MAX), cmap=BORDER_CMAP)
    sm.set_array([])
    cax = fig.add_axes(((FIG_WIDTH_IN - CBAR_W) / 2 / FIG_WIDTH_IN,
                        CBAR_Y / FIG_HEIGHT_IN,
                        CBAR_W / FIG_WIDTH_IN, CBAR_H / FIG_HEIGHT_IN))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("panel border = Spearman ρ  (− blue · 0 none · + red)",
                 fontsize=PT_BODY, labelpad=pt(1.5))
    cb.ax.tick_params(labelsize=PT_SMALL, pad=pt(1.5), length=pt(2.0),
                      width=plt.rcParams["xtick.major.width"])
    cb.outline.set_linewidth(plt.rcParams["axes.linewidth"])

    # No bbox cropping: the gridspec margins and the fig.text labels above are
    # placed in absolute figure coordinates, which tight cropping would shift.
    report_text_overlaps(fig, "figS12")
    save_figure(fig, args.out_dir, "figS12", dpi=150, tight=False)


if __name__ == "__main__":
    main()
