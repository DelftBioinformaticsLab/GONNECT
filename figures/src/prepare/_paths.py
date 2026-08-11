"""Where these scripts read and write, resolved from this file's location.

The figure scripts in ``figures/src`` take their defaults from ``_common``; these
do the same, so a command works from any working directory rather than only
from the repository root, which is what the originals assumed.

Regeneration writes to ``figures/out/prepare/<step>/`` rather than straight into
``figures/data``. That is deliberate: the shipped ``data/`` reproduces the
published figures exactly, and the GSEA and permutation steps are stochastic,
so a rerun that landed on top of it would silently move the figures. Compare
first, then copy over the shipped file if you mean to replace it.
"""

from pathlib import Path

PREPARE_DIR = Path(__file__).resolve().parent
FINAL_DIR = PREPARE_DIR.parent.parent
DATA_DIR = FINAL_DIR / "data"
PREP_OUT_DIR = FINAL_DIR / "out" / "prepare"

# Inputs these steps share.
TCGA_CSV = DATA_DIR / "TCGA_complete_bp_top1k.csv.gz"
HARD_LINKS_CSV = DATA_DIR / "hard_links.csv"
DEG_CSV = DATA_DIR / "deg_results.csv"
ONTOLOGIES_DIR = DATA_DIR / "ontologies"
ACTIVATIONS_DIR = DATA_DIR / "go_term_activations"
EMBEDDINGS_DIR = DATA_DIR / "latent_embeddings"
GSEA_LAYERS_CSV = DATA_DIR / "gsea_gonnect_layers" / "gsea_results.csv"

#: Per-sample baseline activations. Only needed by the two steps that rebuild
#: the baseline arms; see prepare/README.md for what to do when it is absent.
BASELINE_ACTIVATIONS_DIR = DATA_DIR / "baseline_activations"
VEGA_ACTIVATIONS_DIR = BASELINE_ACTIVATIONS_DIR / "vega"
ONTOVAE_ACTIVATIONS_DIR = BASELINE_ACTIVATIONS_DIR / "ontovae"
