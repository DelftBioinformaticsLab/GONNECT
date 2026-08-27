"""Turn baseline run output into the metrics files `figures/` reads.

    out/baselines/ontovae/metrics.txt  ->  ontovae_rand.txt   (nested, as-is)
                                           ontovae.txt        (flat, 'true' arm)
    out/baselines/vega/metrics.txt     ->  vega_rand.txt      (nested, as-is)
                                           vega_hallmark.txt      (flat, 'true')
                                           vega_reactome674.txt   (flat, 'true')

The two shapes are described in `figures/README.md`.

Output goes to `out/baselines/metrics/`, never into `figures/data/` -- the same
rule `figures/src/prepare/` follows. Compare, then copy deliberately.

Note the flat files are derived from the `true` arm of the same run, so the two
agree by construction. In the deposited data they do not: they came from
separate training runs, and Figure 2 reads the nested files while Figure 3 reads
the flat ones.
"""

import argparse
import ast
import os

# GMT stem -> flat file name. `674` is the Reactome gene set count.
VEGA_FLAT_NAMES = {
    "reactomes_uniprot": "vega_reactome674.txt",
    "hallmark_v2026_1_Hs_uniprot": "vega_hallmark.txt",
}


def read_metrics(path):
    """Parse `run-N: {...}` lines into a list of (run_id, payload)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or ": " not in line:
                continue
            run_id, payload = line.split(": ", 1)
            rows.append((run_id, ast.literal_eval(payload)))
    return rows


def write_metrics(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for run_id, payload in rows:
            fh.write(f"{run_id}: {payload}\n")
    print(f"  wrote {path}  ({len(rows)} runs)")


def collect_ontovae(src, out_dir):
    rows = read_metrics(src)
    write_metrics(os.path.join(out_dir, "ontovae_rand.txt"), rows)
    flat = [(run_id, payload["true"]) for run_id, payload in rows if "true" in payload]
    if flat:
        write_metrics(os.path.join(out_dir, "ontovae.txt"), flat)
    else:
        print("  no 'true' arm present; skipping ontovae.txt")


def collect_vega(src, out_dir):
    rows = read_metrics(src)
    write_metrics(os.path.join(out_dir, "vega_rand.txt"), rows)

    for gmt_stem, filename in VEGA_FLAT_NAMES.items():
        flat = [(run_id, payload[gmt_stem]["true"]) for run_id, payload in rows
                if gmt_stem in payload and "true" in payload[gmt_stem]]
        if flat:
            write_metrics(os.path.join(out_dir, filename), flat)
        else:
            print(f"  no 'true' arm for {gmt_stem}; skipping {filename}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ontovae-metrics", default="out/baselines/ontovae/metrics.txt")
    p.add_argument("--vega-metrics", default="out/baselines/vega/metrics.txt")
    p.add_argument("--out-dir", default="out/baselines/metrics")
    args = p.parse_args()

    for label, src, fn in (("OntoVAE", args.ontovae_metrics, collect_ontovae),
                           ("VEGA", args.vega_metrics, collect_vega)):
        if not os.path.exists(src):
            print(f"{label}: {src} not found, skipping")
            continue
        print(f"{label}: reading {src}")
        fn(src, args.out_dir)

    print("\nCompare against figures/data/metrics/ before copying anything across.")


if __name__ == "__main__":
    main()
