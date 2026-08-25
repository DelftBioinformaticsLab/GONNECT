"""Build every figure in order, reporting what succeeded.

    pixi run python figures/src/run_all.py
    pixi run python figures/src/run_all.py --only fig2 figS9
    pixi run python figures/src/run_all.py --skip fig4

Each figure runs in its own process, so one failure does not stop the rest;
the exit code is non-zero if any figure failed.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from _common import DATA_DIR, OUT_DIR

SRC = Path(__file__).resolve().parent

# Ordered cheapest-first, so problems surface early.
FIGURES = ["fig3", "figS4", "figS1", "figS7", "fig2", "figS2",
           "figS5", "figS6", "figS8", "figS9", "figS10", "figS11", "figS12",
           "figS3", "fig5", "fig4"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--only", nargs="+", metavar="FIG",
                   help="run only these (e.g. --only fig2 figS9)")
    p.add_argument("--skip", nargs="+", metavar="FIG", default=[],
                   help="skip these")
    args = p.parse_args()

    figures = [f for f in (args.only or FIGURES) if f not in args.skip]
    results = []
    for name in figures:
        script = SRC / f"{name}.py"
        if not script.exists():
            results.append((name, "MISSING", 0.0))
            continue
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        start = time.monotonic()
        rc = subprocess.run(
            [sys.executable, str(script),
             "--data-dir", str(args.data_dir), "--out-dir", str(args.out_dir)]
        ).returncode
        results.append((name, "ok" if rc == 0 else f"FAILED (rc={rc})",
                        time.monotonic() - start))

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for name, status, secs in results:
        print(f"  {name:<10} {status:<16} {secs:6.1f}s")
    failed = [n for n, s, _ in results if s != "ok"]
    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
