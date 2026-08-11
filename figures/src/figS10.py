"""Supplementary Figure S10: soft-link recovery trajectory, encoder module.

After removing 10% of the GO links (AE_9.1), do the soft links that sit on a
removed edge grow back to larger weights than the remaining soft links? Source
nodes are binned by their GO hard-link degree (log-spaced bins on n_hard, the
same edges for every layer) and each encoder layer (0-3) gets one column:

  top row     mean log10 median |w| per bin, removed edges vs. all other soft
              links, with half-violin distributions of log10 |w| behind them and
              Stouffer-combined Mann-Whitney U stars (combined over the seeds)
  bottom row  how many removed edges fall in each bin (mean per seed)

figS11.py is the decoder counterpart: it imports main() from here and runs it
with side="decoder".

Inputs (relative to --data-dir)
------------------------------
    soft_link_weights/AE_2.1.3_encoder_soft_links.csv.gz
        unperturbed baseline, seed 3, used only to derive the GO hard-link
        degree per source node
    AE_9.1_encoder_10perc_removed.csv.gz
        the GO edges that were removed (data-dir root, not soft_link_weights/)
    soft_link_weights/AE_9.1.<seed>_encoder_soft_links.csv.gz  for seeds 3, 4, 5, 6

Usage
-----
    python figS10.py [--data-dir figures/data] [--out-dir figures/out]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from _common import (FIG_WIDTH_IN, LAYERS_BY_MODULE, PT_BODY, PT_SMALL,
                     PT_SUPTITLE, PT_TITLE, add_io_args, figsize, pt,
                     report_text_overlaps, save_figure, stars)

# ── config ────────────────────────────────────────────────────────────────────
SEEDS = [3, 4, 5, 6]         # AE_9.1 seeds; seed 2 is deliberately excluded
# Global log-spaced bin edges on n_hard (powers of 2). Same edges for every layer
# so bins are directly comparable across layers.
LOG_BIN_EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
N_BINS = len(LOG_BIN_EDGES) - 1
SUBSAMPLE_OTH = 1000

C_REM = "tomato"
C_OTH = "steelblue"

# ── layout ────────────────────────────────────────────────────────────────────
# The canvas width is fixed by the paper's shared standard, so the four layer
# columns divide 16.5 in between them however many layers there are -- the width
# no longer grows with len(layers). Everything below is in canvas inches and
# converted to the figure fractions the gridspec wants, which keeps the margins
# readable as "how much room does that block of text need".
FIG_HEIGHT_IN = 7.7

# Column 0 carries two rotated y axis labels; the wider one is the bar row's
# two-line "# removed (per seed)", which needs ~0.5 in on its own, and it sits
# outside four-digit removed-edge counts.
MARGIN_L = 1.18
MARGIN_R = 0.08
MARGIN_T = 0.76     # suptitle + the "Layer N" panel titles
# The nine bin labels are set vertically (see below), so they are as deep as
# "256-512" is long, and the shared x axis label sits under them.
MARGIN_B = 1.27

# Column gap: every column carries its own y tick labels, since the layers do
# not share a weight scale.
WSPACE = 0.15
HSPACE = 0.08

# Bin labels run bottom-to-top. Nine of them share a column ~3.5 in wide, i.e.
# 0.38 in per bin, while "256-512" set at PT_SMALL is 0.81 in long -- at the
# original 40 degrees they would overlap two labels deep, and no rotation
# shallower than ~80 degrees clears them.
BIN_LABEL_ROTATION = 90

# Clear space a star marker must keep from its neighbours, as a multiple of the
# marker's own height. Small, but enough that "***" next to "***" reads as two
# groups rather than one run of six.
STAR_GAP = 0.6


def stouffer(ps, rbs):
    z = np.array([np.sign(rb) * abs(stats.norm.ppf(p / 2)) for p, rb in zip(ps, rbs)])
    if len(z) == 0 or np.any(~np.isfinite(z)):
        return np.nan, np.nan
    Z = np.mean(z) * np.sqrt(len(z))
    return Z, 2 * (1 - stats.norm.cdf(abs(Z)))


def half_violin(ax, data, x, side, color, alpha=0.22, width=0.95, min_n=5):
    """Draw a half-violin (one-sided KDE) at x on `side` in {'left','right'}."""
    if len(data) < min_n:
        return
    parts = ax.violinplot([data], positions=[x], widths=width,
                          showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        verts = body.get_paths()[0].vertices
        if side == "left":
            verts[:, 0] = np.minimum(verts[:, 0], x)
        else:
            verts[:, 0] = np.maximum(verts[:, 0], x)
        body.set_facecolor(color)
        body.set_alpha(alpha)
        body.set_edgecolor("none")
        body.set_zorder(0.5)


def _compute(side: str, data_dir: Path):
    """Per (layer, bin) median-|w| stats + half-violin sample pools for one side."""
    layers = LAYERS_BY_MODULE[side]
    sl_dir = data_dir / "soft_link_weights"
    baseline = sl_dir / f"AE_2.1.3_{side}_soft_links.csv.gz"
    removed_file = data_dir / f"AE_9.1_{side}_10perc_removed.csv.gz"

    # GO degree per source node from the unperturbed baseline.
    print(f"[{side}] loading baseline {baseline.name} for GO degree...", flush=True)
    sl_base = pd.read_csv(baseline, usecols=["layer_index", "source_index", "sink_index"])
    total_sinks = sl_base.groupby("layer_index")["sink_index"].nunique().rename("n_sinks")
    node_sl = (sl_base.groupby(["layer_index", "source_index"])
                      .size().reset_index(name="n_soft"))
    node_sl = node_sl.merge(total_sinks, on="layer_index")
    node_sl["n_hard"] = node_sl["n_sinks"] - node_sl["n_soft"]
    del sl_base
    node_sl["bin"] = pd.cut(node_sl["n_hard"].clip(lower=1),
                            bins=LOG_BIN_EDGES, labels=False,
                            right=False, include_lowest=True)

    removed = pd.read_csv(removed_file)
    removed_key = removed[["layer_index", "source_index", "sink_index"]].assign(is_removed=True)
    print(f"[{side}] removed edges: {len(removed):,}")

    rows = []
    violin_data = {}  # (layer, bin) -> {"removed": [...], "other": [...]}
    rng = np.random.default_rng(42)
    for seed in SEEDS:
        fn = sl_dir / f"AE_9.1.{seed}_{side}_soft_links.csv.gz"
        print(f"[{side}] loading seed {seed}...", flush=True)
        sl = pd.read_csv(fn, usecols=["layer_index", "source_index", "sink_index", "weight_magnitude"])
        sl = sl.merge(removed_key, on=["layer_index", "source_index", "sink_index"], how="left")
        sl["is_removed"] = sl["is_removed"].fillna(False).astype(bool)
        sl = sl.merge(node_sl[["layer_index", "source_index", "bin", "n_hard"]],
                      on=["layer_index", "source_index"], how="left")

        for layer in layers:
            for b in range(N_BINS):
                sub = sl[(sl["layer_index"] == layer) & (sl["bin"] == b)]
                w_rem = sub.loc[sub["is_removed"], "weight_magnitude"].values
                w_oth = sub.loc[~sub["is_removed"], "weight_magnitude"].values
                n_rem, n_oth = len(w_rem), len(w_oth)
                if n_rem == 0 or n_oth == 0:
                    continue
                U1, p = stats.mannwhitneyu(w_rem, w_oth, alternative="two-sided")
                r_rb = 2 * U1 / (n_rem * n_oth) - 1
                rows.append(dict(
                    seed=seed, layer=layer, bin=b, n_rem=n_rem,
                    med_rem=float(np.median(w_rem)), med_oth=float(np.median(w_oth)),
                    p=p, r_rb=r_rb,
                ))
                key = (int(layer), int(b))
                if key not in violin_data:
                    violin_data[key] = {"removed": [], "other": []}
                violin_data[key]["removed"].extend(np.log10(w_rem).tolist())
                w_oth_s = (rng.choice(w_oth, size=SUBSAMPLE_OTH, replace=False)
                           if n_oth > SUBSAMPLE_OTH else w_oth)
                violin_data[key]["other"].extend(np.log10(w_oth_s).tolist())
        del sl

    df = pd.DataFrame(rows)
    agg_rows = []
    for (layer, b), g in df.groupby(["layer", "bin"]):
        _, p_c = stouffer(g["p"].values, g["r_rb"].values)
        agg_rows.append(dict(
            layer=layer, bin=int(b),
            mean_log_med_rem=float(np.log10(g["med_rem"]).mean()),
            mean_log_med_oth=float(np.log10(g["med_oth"]).mean()),
            n_rem_mean=g["n_rem"].mean(),
            p_combined=p_c,
        ))
    agg = pd.DataFrame(agg_rows).sort_values(["layer", "bin"])
    return agg, violin_data, layers


def _text_size_data(ax, text):
    """Size of a star marker in this axes' data units."""
    fig = ax.figure
    artist = ax.text(0, 0, text, fontsize=PT_BODY, fontweight="bold")
    box = artist.get_window_extent(fig.canvas.get_renderer())
    artist.remove()
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(
        [(box.x0, box.y0), (box.x1, box.y1)])
    return abs(x1 - x0), abs(y1 - y0)


def _stack_marks(ax, marks):
    """Lift each star group above the ones to its left where they would touch.

    Bins sit one x-unit apart and at the paper's type scale "***" is almost
    exactly that wide, so two significant neighbours run together into one
    six-star blob and stop reading as two results. Lifting the right-hand group
    onto its own line keeps every marker centred over its own bin without
    setting the stars smaller than the rest of the figure. (The same collision
    and the same fix appear in fig3.)
    """
    sizes = {t: _text_size_data(ax, t) for _, t, _, _ in marks}
    placed, out, highest = [], [], -np.inf
    for x, text, y_base, color in sorted(marks, key=lambda m: m[0]):
        w, h = sizes[text]
        y = y_base
        for _ in range(len(placed) + 1):
            lifted = False
            for px, pw, py, ph in placed:
                if (abs(x - px) < (w + pw) / 2.0 + STAR_GAP * h
                        and y < py + ph + STAR_GAP * h
                        and y + h > py - STAR_GAP * h):
                    y = py + ph + STAR_GAP * h
                    lifted = True
            if not lifted:
                break
        placed.append((x, w, y, h))
        out.append((x, text, y, color))
        highest = max(highest, y + h)
    return out, highest


def _draw_significance_marks(ax, marks):
    """Place the Stouffer stars and open up the headroom they turn out to need.

    Stacking is measured in display space, so the y limit has to be settled
    before the marks are laid out -- and raising it to fit them changes the
    scale again. A couple of passes converge.
    """
    lo, hi = ax.get_ylim()
    span = hi - lo
    top = hi + 0.06 * span
    ax.set_ylim(lo, top)
    if not marks:
        return
    ax.figure.canvas.draw()
    for _ in range(6):
        placed, highest = _stack_marks(ax, marks)
        needed = highest + 0.03 * span
        if needed <= top:
            break
        top = needed
        ax.set_ylim(lo, top)
    for x, text, y, color in placed:
        ax.text(x, y, text, ha="center", va="bottom", fontsize=PT_BODY,
                color=color, fontweight="bold")


def _plot_trajectory(side, agg, violin_data, layers, out_dir, out_name):
    max_bin = int(agg["bin"].max())
    bin_lim = (-0.6, max_bin + 0.6)
    bin_ticks = list(range(max_bin + 1))

    fig = plt.figure(figsize=figsize(FIG_HEIGHT_IN))
    gs = fig.add_gridspec(
        2, len(layers), height_ratios=[3, 1],
        left=MARGIN_L / FIG_WIDTH_IN, right=1.0 - MARGIN_R / FIG_WIDTH_IN,
        top=1.0 - MARGIN_T / FIG_HEIGHT_IN, bottom=MARGIN_B / FIG_HEIGHT_IN,
        wspace=WSPACE, hspace=HSPACE)

    for ci, layer in enumerate(layers):
        sub = agg[agg["layer"] == layer].sort_values("bin")
        ax_top = fig.add_subplot(gs[0, ci])
        ax_bot = fig.add_subplot(gs[1, ci], sharex=ax_top)
        ax_top.tick_params(labelbottom=False)

        for b in bin_ticks:
            key = (int(layer), int(b))
            if key not in violin_data:
                continue
            half_violin(ax_top, violin_data[key]["other"],   b, "left",  C_OTH)
            half_violin(ax_top, violin_data[key]["removed"], b, "right", C_REM)

        ax_top.plot(sub["bin"], sub["mean_log_med_oth"], "-o",
                    color=C_OTH, label="other soft links", ms=5, lw=1.4,
                    markeredgecolor="white", markeredgewidth=0.5)
        has_rem = sub[sub["n_rem_mean"] > 0]
        ax_top.plot(has_rem["bin"], has_rem["mean_log_med_rem"], "-o",
                    color=C_REM, label="removed", ms=5, lw=1.4,
                    markeredgecolor="white", markeredgewidth=0.5)

        marks = []
        for _, r in has_rem.iterrows():
            s = stars(r["p_combined"])
            if not s:
                continue
            y_high = max(r["mean_log_med_rem"], r["mean_log_med_oth"])
            star_c = C_REM if r["mean_log_med_rem"] > r["mean_log_med_oth"] else C_OTH
            marks.append((float(r["bin"]), s, y_high, star_c))
        _draw_significance_marks(ax_top, marks)

        ax_top.set_title(f"Layer {layer}", fontsize=PT_TITLE, fontweight="bold",
                         pad=pt(2.0))
        ax_top.tick_params(labelsize=PT_SMALL, pad=pt(1.5))
        ax_top.grid(axis="y", alpha=0.2, lw=0.5)
        if ci == 0:
            ax_top.set_ylabel(r"$\log_{10}$ median $|w|$", fontsize=PT_BODY,
                              labelpad=pt(2.0))
            ax_top.legend(fontsize=PT_SMALL, frameon=False, loc="best",
                          handlelength=1.6, handletextpad=0.6,
                          borderpad=0.2, labelspacing=0.4)

        ax_bot.bar(sub["bin"], sub["n_rem_mean"], color=C_REM, alpha=0.75, width=0.85)
        ax_bot.tick_params(labelsize=PT_SMALL, pad=pt(1.5))
        if ci == 0:
            ax_bot.set_ylabel("# removed\n(per seed)", fontsize=PT_BODY,
                              labelpad=pt(2.0))

        range_labels = []
        for b in bin_ticks:
            if (sub["bin"] == b).any():
                range_labels.append(f"{LOG_BIN_EDGES[b]}–{LOG_BIN_EDGES[b+1]}")
            else:
                range_labels.append("")
        ax_bot.set_xticks(bin_ticks)
        ax_bot.set_xticklabels(range_labels, fontsize=PT_SMALL,
                               rotation=BIN_LABEL_ROTATION, ha="center",
                               va="top")
        ax_bot.set_xlim(bin_lim)
        ax_bot.grid(axis="y", alpha=0.2, lw=0.5)

    # One shared x axis label instead of one per column: at PT_BODY the string
    # is 4.4 in long and a column is 3.5 in, so four copies would collide with
    # each other. Centred on the grid, under the vertical bin labels.
    fig.text((MARGIN_L + (FIG_WIDTH_IN - MARGIN_L - MARGIN_R) / 2) / FIG_WIDTH_IN,
             0.07 / FIG_HEIGHT_IN, "GO hard-link degree (sparse → dense)",
             fontsize=PT_BODY, ha="center", va="bottom")

    fig.suptitle(f"Median soft-link weights across GO-degree bins, by layer ({side})",
                 fontsize=PT_SUPTITLE, y=1.0 - 0.06 / FIG_HEIGHT_IN, va="top")
    report_text_overlaps(fig, out_name)
    save_figure(fig, out_dir, out_name, dpi=150)
    plt.close(fig)


def main(side: str = "encoder", out_name: str = "figS10") -> None:
    """Build the trajectory figure for one module; figS11.py calls this too."""
    parser = argparse.ArgumentParser(
        description=f"Soft-link recovery trajectory ({side}) -> {out_name}")
    add_io_args(parser)
    args = parser.parse_args()

    agg, violin_data, layers = _compute(side, args.data_dir)
    _plot_trajectory(side, agg, violin_data, layers, args.out_dir, out_name)


if __name__ == "__main__":
    main()
