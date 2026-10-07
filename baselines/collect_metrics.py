"""Turn baseline run output into the metrics files `figures/` reads.

    out/baselines/ontovae/metrics.txt  ->  ontovae_rand.txt   (nested by graph arm, as-is)
    out/baselines/vega/metrics.txt     ->  vega_rand.txt      (nested by annotation and arm)

Both carry every graph arm, the true graph included, in the shape of
`figures/data/metrics/test_split/{ontovae,vega}_rand.txt`. Figures 2, 3, S1 and
S4 read those files: Figure 2 the `true` arm, Figure 3 every arm. The deposited
ones were rescored from the published runs' latents by
`figures/src/prepare/rescore_baselines.py`; a rerun's runners write the same
metrics directly, the silhouette taken against the true cancer types by
default (`--silhouette-against`). The flat one-method-per-file layout of the
original deposit (`ontovae.txt`, `vega_*.txt`) is no longer read.

Output goes to `out/baselines/metrics/`, never into `figures/data/` -- the same
rule `figures/src/prepare/` follows. Compare, then copy deliberately.
"""

import argparse
import ast
import os


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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ontovae-metrics", default="out/baselines/ontovae/metrics.txt")
    p.add_argument("--vega-metrics", default="out/baselines/vega/metrics.txt")
    p.add_argument("--out-dir", default="out/baselines/metrics")
    args = p.parse_args()

    for label, src, name in (("OntoVAE", args.ontovae_metrics, "ontovae_rand.txt"),
                             ("VEGA", args.vega_metrics, "vega_rand.txt")):
        if not os.path.exists(src):
            print(f"{label}: {src} not found, skipping")
            continue
        print(f"{label}: reading {src}")
        write_metrics(os.path.join(args.out_dir, name), read_metrics(src))

    print("\nCompare against figures/data/metrics/test_split/ before copying anything across.")


if __name__ == "__main__":
    main()
