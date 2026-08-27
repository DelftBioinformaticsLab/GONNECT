"""Graph-randomization nulls, shared by both baselines.

Mirrors `thesis_binn.data_processing.generate_masks`, so the baseline nulls
match GONNECT's own AE_2.2 arm:

  degree_preserving  Espinoza (2012) edge swaps. Preserves every node's in- and
                     out-degree, so edge count and model capacity are fixed.
  random             Full reshuffle. Preserves only the total edge count.

Both operate per layer. Fully connected layers are returned unchanged.
"""

import numpy as np

Q_SWAPS = 100  # successful swaps per edge


def _is_fully_connected(mask):
    return bool(mask.all())


def degree_preserving_mask(mask, seed=None, Q=Q_SWAPS):
    """Randomize `mask` by edge swaps, preserving all row and column sums."""
    if _is_fully_connected(mask):
        return mask.copy()

    rng = np.random.default_rng(seed)
    m = mask.copy()
    rows, cols = np.where(m)
    edges = np.stack([rows, cols], axis=1)
    n_edges = len(edges)
    if n_edges < 2:
        return m

    successful = 0
    while successful < Q * n_edges:
        i, j = rng.choice(n_edges, size=2, replace=False)
        r_a, c_x = edges[i]
        r_b, c_y = edges[j]
        # Distinct rows and columns, and both destinations empty, or the swap
        # would drop an edge.
        if r_a == r_b or c_x == c_y or m[r_a, c_y] or m[r_b, c_x]:
            continue

        m[r_a, c_x] = 0
        m[r_a, c_y] = 1
        m[r_b, c_y] = 0
        m[r_b, c_x] = 1
        edges[i, 1] = c_y
        edges[j, 1] = c_x
        successful += 1

    return m


def random_mask(mask, seed=None):
    """Redistribute `mask`'s edges uniformly, preserving only the edge count."""
    if _is_fully_connected(mask):
        return mask.copy()

    rng = np.random.default_rng(seed)
    out = np.zeros_like(mask)
    out.flat[rng.choice(mask.size, size=int(mask.sum()), replace=False)] = 1
    return out


def randomize_mask_stack(mask_list, arm, seed):
    """Apply `arm` to every mask in a stack.

    Each layer gets a derived seed, so layers of identical shape do not receive
    identical randomizations.
    """
    if arm == "true":
        return [m.copy() for m in mask_list]
    fn = {"degree_preserving": degree_preserving_mask, "random": random_mask}[arm]
    return [fn(m, seed=seed * 1000 + i) for i, m in enumerate(mask_list)]
