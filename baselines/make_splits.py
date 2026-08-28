"""Write the per-seed train/validation/test sample IDs both baselines train on.

Writes `<out>/run_seed-<seed>/{train,val,test}_ids.npy`.

`--verify` asserts the partition is identical to
`gonnect.train.train.split_data`, which is what makes the comparison
like-for-like. It imports `gonnect`, so run this from GONNECT's own
environment.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_FRACTION = 0.7   # remainder is halved into validation and test
VAL_TEST_SPLIT = 0.5


def split_ids(data, id_col, seed):
    """Return (train, val, test) ID arrays for one seed.

    Cast to fixed-width unicode: an `object` array makes `np.save` fall back to
    pickle, and a numpy 2.x pickle cannot be read by the numpy 1.x in the
    baseline environments.
    """
    train, remaining = train_test_split(data, train_size=TRAIN_FRACTION, random_state=seed)
    validation, test = train_test_split(remaining, train_size=VAL_TEST_SPLIT, random_state=seed)
    return tuple(part[id_col].values.astype(str) for part in (train, validation, test))


def verify_against_gonnect(data, id_col, n_nan_cols, seed):
    """Check the row partition matches `gonnect.train.train.split_data`."""
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from gonnect.train.train import split_data

    gonnect = split_data(data, n_nan_cols, split=TRAIN_FRACTION, seed=seed)
    ours = split_ids(data, id_col, seed)
    for name, theirs, mine in zip(("train", "val", "test"), gonnect, ours):
        expected = data.loc[theirs.index, id_col].values
        if not np.array_equal(expected, mine):
            raise AssertionError(f"seed {seed}: {name} split differs from GONNECT's")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expr-data", default="data/TCGA_complete_bp_top1k.csv")
    p.add_argument("--out", default="out/baselines/splits")
    p.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--id-col", default="patient_id")
    p.add_argument("--n-nan-cols", type=int, default=5)
    p.add_argument("--verify", action="store_true",
                   help="Assert the partition matches gonnect.train.train.split_data")
    args = p.parse_args()

    data = pd.read_csv(args.expr_data)
    print(f"{len(data)} samples, {len(data.columns) - args.n_nan_cols} genes")

    for seed in args.seeds:
        if args.verify:
            verify_against_gonnect(data, args.id_col, args.n_nan_cols, seed)

        train, val, test = split_ids(data, args.id_col, seed)
        run_dir = os.path.join(args.out, f"run_seed-{seed}")
        os.makedirs(run_dir, exist_ok=True)
        for name, ids in (("train", train), ("val", val), ("test", test)):
            np.save(os.path.join(run_dir, f"{name}_ids.npy"), ids)

        checked = " (matches GONNECT)" if args.verify else ""
        print(f"seed {seed}: train={len(train)} val={len(val)} test={len(test)}{checked}")

    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
