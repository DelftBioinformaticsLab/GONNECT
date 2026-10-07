"""
Gene sets of the GONNECT GO terms under two definitions, traced through every
encoder layer of hard_links.csv.

direct
    The term's own annotations: every gene that reaches the term through a
    chain of proxies only. A gene enters through its leaf proxies
    (``Proxy:k_<gene>``, a one-to-one copy carried up one layer at a time) and
    may land on the term itself or on one of the term's balancing proxies
    (``Proxy:k_GO:<term>``, which feed only that term). Genes of child terms
    are not included.

receptive_field
    Every gene that can reach the term through any path: its own annotations
    plus those of all its descendants. For the bottleneck terms this is exactly
    ``run_gsea_bottleneck.build_receptive_fields``.

The decoder is the transposed encoder graph, so a decoder node reaches the same
genes and takes the same sets.

``run_gsea.build_gene_sets`` (the sets behind the deposited
gsea_gonnect_layers/) reads gene -> term pairs from encoder layers 0 and 1
only. A gene whose leaf-proxy chain is two or more hops long enters its term at
layer 2 or 3, so that function drops it. ``audit()`` measures what it drops.

Usage:
  python gsea_gene_sets.py [--hard-links PATH] [--output-dir DIR]

writes
  <output-dir>/gene_sets_per_term.tsv   one row per (definition, term): layer,
                                        size, ';'-joined UniProt ids
  <output-dir>/build_gene_sets_audit.tsv
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import HARD_LINKS_CSV, PREP_OUT_DIR

DEFINITIONS = ("build_gene_sets", "direct", "receptive_field")


def node_kind(node: str) -> str:
    if node.startswith("GO:"):
        return "GO"
    if node.startswith("Proxy:"):
        return "proxy_GO" if "GO:" in node else "proxy_gene"
    return "gene"


def proxy_target(node: str) -> str:
    """The gene a leaf proxy copies, or the term a balancing proxy feeds."""
    m = re.match(r"Proxy:\d+_(.*)", node)
    return m.group(1) if m else node


def encoder_edges(hard_links: pd.DataFrame | Path) -> pd.DataFrame:
    hl = (pd.read_csv(hard_links, usecols=["component", "layer", "source_term_id", "sink_term_id"])
          if isinstance(hard_links, Path) else hard_links)
    enc = hl[hl["component"] == "encoder"][["layer", "source_term_id", "sink_term_id"]].astype(
        {"source_term_id": str, "sink_term_id": str})
    # The two naming assumptions everything below rests on: a balancing proxy
    # feeds only the term it is named after (or its next proxy), and a leaf
    # proxy copies only the gene it is named after.
    src_k = enc["source_term_id"].map(node_kind)
    snk_k = enc["sink_term_id"].map(node_kind)
    pgo = src_k == "proxy_GO"
    if not (enc.loc[pgo, "source_term_id"].map(proxy_target)
            == enc.loc[pgo, "sink_term_id"].map(proxy_target)).all():
        raise ValueError("a balancing proxy feeds a node other than its own term")
    pg = snk_k == "proxy_gene"
    if not (enc.loc[pg, "source_term_id"].map(proxy_target)
            == enc.loc[pg, "sink_term_id"].map(proxy_target)).all():
        raise ValueError("a leaf proxy is fed by something other than its own gene")
    return enc


def term_layers(enc: pd.DataFrame) -> dict[str, int]:
    """{GO term: encoder layer whose output it is} (fig4.build_layer_map)."""
    rows = enc[enc["sink_term_id"].str.startswith("GO:")][["sink_term_id", "layer"]].drop_duplicates()
    if rows["sink_term_id"].duplicated().any():
        raise ValueError("a GO term is the sink of more than one encoder layer")
    return dict(zip(rows["sink_term_id"], rows["layer"].astype(int)))


def trace_gene_sets(enc: pd.DataFrame) -> tuple[dict[str, set], dict[str, set]]:
    """({term: direct genes}, {term: receptive-field genes}) over all encoder layers."""
    direct: dict[str, set] = defaultdict(set)     # genes reaching a node through proxies only
    field: dict[str, set] = defaultdict(set)      # genes reaching a node through anything
    for layer in sorted(enc["layer"].unique()):   # sources of layer L are sinks of L-1: topological
        for src, snk in enc.loc[enc["layer"] == layer, ["source_term_id", "sink_term_id"]].itertuples(index=False):
            kind = node_kind(src)
            src_direct = {src} if kind == "gene" else direct[src]
            src_field = {src} if kind == "gene" else field[src]
            field[snk] |= src_field
            if kind != "GO":                      # a child term's genes are not the sink's own
                direct[snk] |= src_direct
    terms = term_layers(enc)
    return ({t: set(direct[t]) for t in terms}, {t: set(field[t]) for t in terms})


def build_gene_sets_reference(hard_links_path: Path) -> dict[str, set]:
    """run_gsea.build_gene_sets, unchanged, as sets."""
    from run_gsea import build_gene_sets
    sets, _ = build_gene_sets(hard_links_path)
    return {t: set(g) for t, g in sets.items() if t != "GO:0000000"}


def entry_layer_of_missed(enc: pd.DataFrame, missed: dict[str, set]) -> pd.Series:
    """Per missed (gene, term) pair, the encoder layer at which the gene's leaf proxy enters the term."""
    rows = []
    e = enc[enc["source_term_id"].map(node_kind) == "proxy_gene"]
    e = e.assign(gene=e["source_term_id"].map(proxy_target), term=e["sink_term_id"].map(proxy_target))
    lookup = e.groupby(["term", "gene"])["layer"].min()
    for t, genes in missed.items():
        for g in genes:
            rows.append(int(lookup.get((t, g), -1)))
    return pd.Series(rows, dtype=int)


def audit(hard_links_path: Path, min_size: int = 3, max_size: int = 500) -> tuple[pd.DataFrame, dict]:
    """Per layer: how many terms build_gene_sets leaves incomplete, and how many genes it misses."""
    enc = encoder_edges(hard_links_path)
    layers = term_layers(enc)
    direct, field = trace_gene_sets(enc)
    bgs = build_gene_sets_reference(hard_links_path)
    # GO:0000000 (the root) has balancing proxies but is not a node of the
    # model, so build_gene_sets gives it a set that no layer scores.
    for t, genes in bgs.items():                  # it never adds a gene the trace lacks
        if t in layers and not genes <= direct[t]:
            raise ValueError(f"build_gene_sets has genes the full trace does not, for {t}")
    missed = {t: direct[t] - bgs.get(t, set()) for t in layers}
    entry = entry_layer_of_missed(enc, {t: m for t, m in missed.items() if m})
    rows = []
    for L in sorted(set(layers.values())):
        ts = [t for t in layers if layers[t] == L]
        n_b = np.array([len(bgs.get(t, ())) for t in ts])
        n_d = np.array([len(direct[t]) for t in ts])
        n_r = np.array([len(field[t]) for t in ts])
        hit = n_d > n_b
        cov = lambda n: int(((n >= min_size) & (n <= max_size)).sum())
        rows.append({
            "encoder_layer": L, "go_terms": len(ts),
            "terms_affected": int(hit.sum()),
            "genes_missed": int((n_d - n_b).sum()),
            "genes_in_full_direct_sets": int(n_d.sum()),
            "share_of_direct_pairs_missed": float((n_d - n_b).sum() / max(n_d.sum(), 1)),
            "terms_emptied": int(((n_b == 0) & (n_d > 0)).sum()),
            f"covered_{min_size}_{max_size}_build_gene_sets": cov(n_b),
            f"covered_{min_size}_{max_size}_direct": cov(n_d),
            f"covered_{min_size}_{max_size}_receptive_field": cov(n_r),
            "newly_covered": int((((n_d >= min_size) & (n_d <= max_size)) & ~((n_b >= min_size) & (n_b <= max_size))).sum()),
            "median_size_build_gene_sets": float(np.median(n_b)),
            "median_size_direct": float(np.median(n_d)),
            "median_size_receptive_field": float(np.median(n_r)),
            "terms_direct_equals_rf": int(sum(direct[t] == field[t] for t in ts)),
        })
    extra = {"missed_pairs_by_entry_layer": entry.value_counts().sort_index().to_dict()}
    return pd.DataFrame(rows), extra


def all_sets(hard_links_path: Path) -> dict[str, dict[str, set]]:
    enc = encoder_edges(hard_links_path)
    direct, field = trace_gene_sets(enc)
    return {"build_gene_sets": build_gene_sets_reference(hard_links_path),
            "direct": direct, "receptive_field": field}


def write_sets(hard_links_path: Path, path: Path) -> pd.DataFrame:
    enc = encoder_edges(hard_links_path)
    layers = term_layers(enc)
    sets = all_sets(hard_links_path)
    rows = [{"definition": d, "go_term": t, "encoder_layer": layers[t],
             "decoder_layer": 8 - layers[t], "n_genes": len(sets[d].get(t, ())),
             "genes": ";".join(sorted(sets[d].get(t, ())))}
            for d in DEFINITIONS for t in sorted(layers, key=lambda t: (layers[t], t))]
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return df


def read_sets(path: Path, definition: str) -> dict[str, list[str]]:
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    df = df[(df["definition"] == definition) & (df["n_genes"] > 0)]
    return {t: g.split(";") for t, g in zip(df["go_term"], df["genes"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hard-links", type=Path, default=HARD_LINKS_CSV)
    p.add_argument("--output-dir", type=Path, default=PREP_OUT_DIR / "gsea_consistency")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table, extra = audit(args.hard_links)
    table.to_csv(args.output_dir / "build_gene_sets_audit.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.T)
    print("missed (gene, term) pairs by the layer the gene enters:", extra["missed_pairs_by_entry_layer"])
    df = write_sets(args.hard_links, args.output_dir / "gene_sets_per_term.tsv")
    print(f"wrote {len(df)} rows to {args.output_dir / 'gene_sets_per_term.tsv'}")


if __name__ == "__main__":
    main()
