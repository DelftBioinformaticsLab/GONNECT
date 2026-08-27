"""
Two adaptations of OntoVAE code required for the GONNECT baseline runs.

`install_drop_last_loader`
    Upstream's FastTensorDataLoader emits a short final batch. The decoder uses
    `nn.BatchNorm1d` (modules.py:69), which raises on a batch of one in training
    mode, so a split whose size leaves a remainder of 1 kills the run partway
    through. Dropping the short batch avoids it.

`gene_index_map`
    Maps GONNECT's gene ordering onto the trimmed ontology's, so reconstruction
    error is scored on the genes the two models share rather than on OntoVAE's
    full output. willsketch/onto-vae patched this into `Ontobj.match_dataset`
    as `<dataset>_GONNECT_GENE_MAP`; it needs no access to Ontobj internals, so
    here it is a plain function.
"""

import importlib
import os
import sys
import types

import numpy as np


def ensure_onto_vae_importable(checkout=None):
    """Make `onto_vae` importable, optionally from an un-installed checkout.

    Normally the package comes from `../environments/ontovae.yml`; `checkout`
    reads it from a working copy instead. `colorcet` is stubbed when absent --
    `Ontobj` imports it at module scope but only uses it for plotting.
    """
    if checkout:
        checkout = os.path.abspath(checkout)
        if checkout not in sys.path:
            sys.path.append(checkout)

    try:
        importlib.import_module("colorcet")
    except ModuleNotFoundError:
        sys.modules["colorcet"] = types.ModuleType("colorcet")

    importlib.import_module("onto_vae")


LOW_MEMORY_GPU_THRESHOLD_GIB = 16


def force_cpu():
    """Pin OntoVAE to the CPU.

    `OntoVAE.__init__` picks its device from `torch.cuda.is_available()`, so
    that call is the only lever -- `CUDA_VISIBLE_DEVICES` is unreliable across
    platforms.
    """
    import torch

    torch.cuda.is_available = lambda: False


def install_memory_efficient_adamw(mode="auto"):
    """Put AdamW on its single-tensor path (`foreach=False`).

    The decoder's final layer carries ~275M weights, and the default
    multi-tensor path allocates a further ~1 GB temporary that does not fit on a
    12 GB card. `foreach=False` is slower but numerically identical.

    `mode` is "auto" (enable below `LOW_MEMORY_GPU_THRESHOLD_GIB` of device
    memory), "on", or "off". Returns True when the patch was installed.
    """
    import torch

    if mode == "off":
        return False
    if mode == "auto":
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return False
        total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        if total_gib >= LOW_MEMORY_GPU_THRESHOLD_GIB:
            return False

    if getattr(torch.optim.AdamW, "_gonnect_low_memory", False):
        return True

    base = torch.optim.AdamW

    class LowMemoryAdamW(base):
        _gonnect_low_memory = True

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("foreach", False)
            super().__init__(*args, **kwargs)

    torch.optim.AdamW = LowMemoryAdamW
    return True


def install_drop_last_loader():
    """Make OntoVAE's training loop drop its short final batch.

    `vae_model` binds the loader by name at import, so the rebind has to happen
    on that module. Idempotent; returns the installed class.
    """
    from onto_vae import vae_model

    current = vae_model.FastTensorDataLoader
    if getattr(current, "_gonnect_drop_last", False):
        return current

    class DropLastTensorDataLoader(current):
        """Upstream's loader, minus any final batch smaller than `batch_size`."""

        _gonnect_drop_last = True

        def __init__(self, *tensors, batch_size=32, shuffle=False):
            super().__init__(*tensors, batch_size=batch_size, shuffle=shuffle)
            self.n_batches = self.dataset_len // self.batch_size

        def __next__(self):
            if self.i + self.batch_size > self.dataset_len:
                raise StopIteration
            return super().__next__()

    vae_model.FastTensorDataLoader = DropLastTensorDataLoader
    return DropLastTensorDataLoader


def gene_index_map(expr_genes, ontology_genes):
    """Index of each gene of `expr_genes` within `ontology_genes`, -1 if absent.

    `ontology_genes` is `ontobj.genes[f"{top}_{bottom}"]`.
    """
    position = {g: i for i, g in enumerate(ontology_genes)}
    return np.array([position.get(g, -1) for g in expr_genes], dtype=np.int64)
