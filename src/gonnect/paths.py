"""Locating the GO input files and mask archives that the library reads at runtime.

Installed from PyPI, ``gonnect`` has no repository around it. The two GO inputs
are far too large to ship in a wheel (``go-basic.obo`` alone is 31 MB) and the
pre-generated masks under ``out/masks/`` are dataset-specific artefacts, not
library data. Callers therefore point the library at their own copies.

Resolution order for each location, first hit wins:

1. An explicit path argument (``obo_path``, ``gaf_path``, ``data_dir``, ``masks_dir``).
2. ``cluster=True`` -- the Apptainer layout used for the runs behind preprint v3. Deprecated.
3. ``package_call=True`` -- relative to a module three levels below the repository
   root, i.e. ``src/gonnect/<subpackage>/``. Deprecated.
4. The ``GONNECT_DATA_DIR`` / ``GONNECT_MASKS_DIR`` environment variable.
5. ``./data`` and ``./out/masks`` relative to the current working directory.

Steps 2 and 3 exist so the 43 ``src/AE_*.py`` experiment scripts and the SLURM
jobs that produced the results in preprint v3 keep running unchanged; both flags are
deprecated for new code in favour of passing paths explicitly.
"""

import os
from pathlib import Path

DATA_DIR_ENV_VAR = "GONNECT_DATA_DIR"
MASKS_DIR_ENV_VAR = "GONNECT_MASKS_DIR"

OBO_FILENAME = "go-basic.obo"
GAF_FILENAME = "goa_human.gaf"

# The container image bind-mounts the repository at /opt/app; `cluster=True`
# selects that layout. See container_pixi.def and slurm/.
_CLUSTER_ROOT = Path("/opt/app")
# `package_call=True` means "called from src/gonnect/<subpackage>/", which is
# three levels below the repository root.
_PACKAGE_CALL_ROOT = Path("../../..")

_DOWNLOAD_HINTS = {
    OBO_FILENAME: "https://purl.obolibrary.org/obo/go/go-basic.obo",
    GAF_FILENAME: "https://current.geneontology.org/annotations/goa_human.gaf.gz",
}


def _legacy_root(package_call, cluster):
    """The repository root implied by the deprecated flags, or None if neither is set."""
    if cluster:
        return _CLUSTER_ROOT
    if package_call:
        return _PACKAGE_CALL_ROOT
    return None


def resolve_data_dir(data_dir=None, *, package_call=False, cluster=False) -> Path:
    """Directory holding go-basic.obo and goa_human.gaf. See module docstring for the order."""
    if data_dir is not None:
        return Path(data_dir)
    root = _legacy_root(package_call, cluster)
    if root is not None:
        return root / "data"
    env_value = os.environ.get(DATA_DIR_ENV_VAR)
    if env_value:
        return Path(env_value)
    return Path("data")


def resolve_masks_dir(masks_dir=None, *, package_call=False, cluster=False, root_dir=None) -> Path:
    """Directory holding the pre-generated layer/edge/proxy masks.

    ``root_dir`` is the pre-1.0 spelling: a repository root whose ``out/masks``
    subdirectory is used. It is honoured when no explicit ``masks_dir`` is given.
    """
    if masks_dir is not None:
        return Path(masks_dir)
    if root_dir is not None:
        return Path(root_dir) / "out" / "masks"
    root = _legacy_root(package_call, cluster)
    if root is not None:
        return root / "out" / "masks"
    env_value = os.environ.get(MASKS_DIR_ENV_VAR)
    if env_value:
        return Path(env_value)
    return Path("out") / "masks"


def require_file(path, what) -> Path:
    """Check that a required input exists, failing with a message that says how to get it.

    goatools' own error for a missing .obo is a bare parser traceback, which is a
    poor first experience for someone who just ran `pip install gonnect`.
    """
    path = Path(path)
    if path.is_file():
        return path
    hint = _DOWNLOAD_HINTS.get(path.name)
    message = f"{what} not found at {path}."
    if hint:
        message += f" Download it from {hint}"
        if path.name == GAF_FILENAME:
            message += " and gunzip it"
        message += "."
    message += (
        f" Pass an explicit path, or set ${DATA_DIR_ENV_VAR} to the directory"
        f" containing {OBO_FILENAME} and {GAF_FILENAME}."
    )
    raise FileNotFoundError(message)


def resolve_obo_path(obo_path=None, data_dir=None, *, package_call=False, cluster=False) -> Path:
    """Full path to go-basic.obo, verified to exist."""
    if obo_path is None:
        obo_path = resolve_data_dir(data_dir, package_call=package_call, cluster=cluster) / OBO_FILENAME
    return require_file(obo_path, "The Gene Ontology file go-basic.obo")


def resolve_gaf_path(gaf_path=None, data_dir=None, *, package_call=False, cluster=False) -> Path:
    """Full path to goa_human.gaf, verified to exist."""
    if gaf_path is None:
        gaf_path = resolve_data_dir(data_dir, package_call=package_call, cluster=cluster) / GAF_FILENAME
    return require_file(gaf_path, "The GO annotation file goa_human.gaf")
