"""Canonical model display names for every GONNECT paper figure.

Single source of truth for how a model is *labelled* in a figure. Internal keys
are left exactly as they appear in the source data (xlsx columns, txt dict keys,
embedding filenames) — this module only maps those keys to paper names, so no
data-loading code has to change.

Randomization vocabulary
------------------------
Two distinct null ontologies exist, and the historical naming flipped the
meaning of "R" between the GONNECT models and the OntoVAE/VEGA baselines:

    concept                        legacy GONNECT   legacy baseline   PAPER
    ---------------------------------------------------------------------
    degree-preserving randomized   GONNECT-R-*      *_degree_preserving   DPR
    fully randomized               GONNECT-RR-*     *_random              FR

`GONNECT-R-*` therefore means degree-preserving (confirmed by the loader comment
in fig2/plot_metrics.py and by randomize_ontologies.py, whose edge-swap
randomization preserves both per-set and per-gene degree). Both legacy spellings
are mapped here so figures agree regardless of which convention their input file
uses.

Usage
-----
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
    from gonnect_names import MODEL_DISPLAY, display
"""

from typing import Dict

# ── Randomization variants ───────────────────────────────────────────────────
DPR = "DPR"   # degree-preserving randomized
FR = "FR"     # fully randomized

# Maps every legacy spelling of a randomization variant onto the paper token.
RANDOMIZATION_DISPLAY: Dict[str, str] = {
    "true": "true",
    "degree_preserving": DPR,
    "rand_degree": DPR,
    "R": DPR,
    "random": FR,
    "rand_size": FR,
    "RR": FR,
}

# Spelled-out variant names for legends, where there is room for the full phrase.
RANDOMIZATION_LEGEND: Dict[str, str] = {
    "true": "True graph",
    "degree_preserving": f"Degree-preserving random ({DPR})",
    "random": f"Fully random ({FR})",
}

# VEGA annotation database -> paper name. The internal keys vary across scripts
# ("reactomes", "reactome674", the raw gmt stem), so map all of them.
VEGA_DB_DISPLAY: Dict[str, str] = {
    "hallmark": "Hallmark",
    "hallmark_v2026_1_Hs_uniprot": "Hallmark",
    "reactomes": "Reactome",
    "reactome674": "Reactome",
    "reactomes_uniprot": "Reactome",
}

# ── Module suffixes (which autoencoder half carries the GO prior) ────────────
MODULES = ["enc", "dec", "both"]

# Long module name (as used in embedding filenames) -> the suffix in MODULES.
# Note "both"[:3] == "bot", so slicing is not a safe abbreviation.
MODULE_ABBREV: Dict[str, str] = {
    "encoder": "enc",
    "decoder": "dec",
    "both": "both",
    "none": "none",
}

# Legend/group label per module, as used in fig_main/fig2.png.
PRIOR_DISPLAY: Dict[str, str] = {
    "enc": "Enc prior",
    "dec": "Dec prior",
    "both": "Both priors",
}

# ── Whole-model names (no enc/dec split) ────────────────────────────────────
# Fig2a-d calls the fixed-link model plain "GONNECT", so "GONNECT-FL" — used by
# the AUROC and perm-null scripts — renders without the -FL suffix.
MODEL_FAMILY_DISPLAY: Dict[str, str] = {
    "GONNECT-FL": "GONNECT",
    "GONNECT": "GONNECT",
    "GONNECT-SL": "GONNECT-SL",
    "GONNECT-R": f"GONNECT-{DPR}",
    "GONNECT-Rand": f"GONNECT-{DPR}",
    "GONNECT-RR": f"GONNECT-{FR}",
}

# Embedding-file version digit -> model family (data/README GONNECT Files.md:
# AE_x.0 fixed link, AE_x.1 soft link, AE_x.2 randomized).
EMB_VERSION_DISPLAY: Dict[str, str] = {
    "2.0": "GONNECT",
    "2.1": "GONNECT-SL",
    "2.2": f"GONNECT-{DPR}",
}
EMB_VERSION_SUFFIX: Dict[str, str] = {"2.0": "", "2.1": "_SL", "2.2": "_DPR"}


def _gonnect_family() -> Dict[str, str]:
    """Build the 12 GONNECT entries: {fixed, SL} x {true, DPR, FR} x modules."""
    out: Dict[str, str] = {}
    # (legacy randomization infix, paper randomization infix)
    variants = [("", ""), ("R-", f"{DPR}-"), ("RR-", f"{FR}-")]
    for legacy_rand, paper_rand in variants:
        for legacy_link, paper_link in [("", ""), ("SL-", "SL-")]:
            for module in MODULES:
                key = f"GONNECT-{legacy_rand}{legacy_link}{module}"
                out[key] = f"GONNECT-{paper_rand}{paper_link}{module}"
    return out


# ── The mapping: data key -> paper display name ─────────────────────────────
MODEL_DISPLAY: Dict[str, str] = {
    # Non-GO reference
    "MLP": "MLP",
    # OntoVAE
    "ontovae": "OntoVAE",
    "ontovae_degree_preserving": f"OntoVAE ({DPR})",
    "ontovae_random": f"OntoVAE ({FR})",
    # VEGA
    "vega_hallmark": "VEGA (Hallmark)",
    "vega_hallmark_degree_preserving": f"VEGA (Hallmark, {DPR})",
    "vega_hallmark_random": f"VEGA (Hallmark, {FR})",
    "vega_reactome674": "VEGA (Reactome)",
    "vega_reactome_degree_preserving": f"VEGA (Reactome, {DPR})",
    "vega_reactome_random": f"VEGA (Reactome, {FR})",
    # GONNECT: 18 entries covering both link types x three ontologies x modules
    **_gonnect_family(),
}

# ── SeaAD figures use two-line labels ───────────────────────────────────────
SEAAD_DISPLAY: Dict[str, str] = {
    "seaad_true": "OntoVAE\n(True GO)",
    "seaad_degree_preserving": f"OntoVAE\n({DPR})",
    "seaad_random": f"OntoVAE\n({FR})",
    "seaad_vega_reactome_true": "VEGA-Reactome\n(True)",
    "seaad_vega_reactome_degree_preserving": f"VEGA-Reactome\n({DPR})",
    "seaad_vega_reactome_random": f"VEGA-Reactome\n({FR})",
    "seaad_vega_hallmark_true": "VEGA-Hallmark\n(True)",
    "seaad_vega_hallmark_degree_preserving": f"VEGA-Hallmark\n({DPR})",
    "seaad_vega_hallmark_random": f"VEGA-Hallmark\n({FR})",
}

# ── Short forms, for dense axes where the full name does not fit ────────────
SHORT_DISPLAY: Dict[str, str] = {
    "MLP": "MLP",
    "ontovae": "OntoVAE",
    "vega_hallmark": "VEGA-H",
    "vega_reactome674": "VEGA-R",
    **{
        f"GONNECT-{legacy_rand}{legacy_link}{m}": f"{paper}{link}{m}"
        for legacy_rand, paper in [("", ""), ("R-", f"{DPR}-"), ("RR-", f"{FR}-")]
        for legacy_link, link in [("", ""), ("SL-", "SL-")]
        for m in MODULES
    },
}


def display(key: str, default: str | None = None) -> str:
    """Paper display name for a model key, across all naming tables."""
    for table in (MODEL_DISPLAY, SEAAD_DISPLAY, MODEL_FAMILY_DISPLAY):
        if key in table:
            return table[key]
    return key if default is None else default


def short(key: str, default: str | None = None) -> str:
    """Compact display name, falling back to the full name."""
    if key in SHORT_DISPLAY:
        return SHORT_DISPLAY[key]
    return display(key, default)
