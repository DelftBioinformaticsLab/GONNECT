# GONNECT

Code for the paper **"A critical evaluation of Gene Ontology priors in biologically-informed neural networks"**.

**Authors**: T. Verlaan\*, M.A. Lieftinck\*, W. Mwine, and M.J.T. Reinders
(\* equal contribution)

**Affiliation**: Delft University of Technology, Computer Science Department, Delft Bioinformatics Lab (Delft, 2628XE, The Netherlands)

**Corresponding author**: T. Verlaan (t.verlaan@tudelft.nl)

GONNECT is a research framework for training biologically-informed autoencoders that leverage Gene Ontology (GO) structure to analyze gene expression data. This project implements various autoencoder architectures that incorporate biological knowledge through GO-based network constraints. The associated publication is currently in pre-print and available at: https://doi.org/10.1101/2025.11.06.686983



## Installation

### Pip

```bash
pip install gonnect
```

For CUDA support on Windows you need to install PyTorch from its own index
first, since its PyPI wheel is CPU-only. This is not necessary on Linux, where
the PyPI wheel is already CUDA-enabled. macOS has no CUDA and runs on CPU.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install gonnect
```

### Conda (currently under review, NOT WORKING YET)

```bash
conda install -c conda-forge gonnect
```

## Installation from source (for reproducibility)

Reproducing the paper needs the figure pipeline, the SLURM experiment scripts and
the baselines, none of which are in the published package. This project uses
[Pixi](https://pixi.sh) to reconstruct the exact environment:

```bash
# Install pixi if not already installed
curl -fsSL https://pixi.sh/install.sh | bash

git clone https://github.com/DelftBioinformaticsLab/GONNECT.git
cd GONNECT
pixi install
pixi shell
```

Pixi pins Python 3.12, PyTorch with **pytorch-cuda 12.4** and a fully resolved
`pixi.lock`. CUDA 12.4 is declared as a system requirement, so the environment
will not solve without it, and the platforms are `linux-64` and `win-64` only.

For the cluster runs, build the Apptainer container instead. The name encodes the
version in `pyproject.toml`, and the SLURM scripts expect exactly this file name:

```bash
apptainer build container_pixi_1.0.0.sif container_pixi.def
```

The image deliberately contains **only** the environment (`pyproject.toml` and
`pixi.lock`) — no source code and no data. Both are bind-mounted at run time, so
every invocation needs `--pwd /opt/app` (otherwise `pixi run` cannot find the
manifest) plus the three binds:

```bash
apptainer exec --nv --writable-tmpfs --pwd /opt/app --containall \
    --bind src/:/opt/app/src/ \
    --bind data/:/opt/app/data/ \
    --bind out/:/opt/app/out/ \
    ./container_pixi_1.0.0.sif pixi run python -u src/AE_2.0.0_both.py
```

`slurm/` has the exact invocations, all pinned to this image name. Note that the
published results were produced with the earlier `container_pixi_0.2.1.sif`,
before the image name was unified with the package version.

### Repository structure

```
GONNECT/
├── src/
│   ├── gonnect/                 # The published package (pip install gonnect)
│   │   ├── model/               # Autoencoder, Encoder, Decoder, SparseLinear
│   │   ├── train/               # Training loop and loss functions
│   │   ├── data_processing/     # GO preprocessing, mask generation
│   │   ├── paths.py             # Resolves GO inputs and mask directories
│   │   └── analysis/            # Evaluation/plotting -- repo only, not published
│   ├── AE_*.py                  # Experiment scripts (AE_2.x, AE_3.x)
│   ├── GO_preprocessing_local.py  # Build GO layers/masks outside the cluster
│   └── benchmark_training_time.py # Training-time benchmark
├── figures/                     # Reproduces every generated figure
│   ├── src/                     # One figN.py per figure + run_all.py
│   ├── out/                     # Generated figures (PDFs committed)
│   └── data/                    # Figure inputs -- not in git, ~12 GB
├── baselines/                   # OntoVAE and VEGA comparisons
├── slurm/                       # SLURM job submission scripts
├── tools/                       # One-off dataset preparation (needs scanpy)
├── tests/                       # Unit tests for the GO preprocessing pipeline
├── data/                        # go-basic.obo; other inputs not in git
├── out/masks/                   # Pre-generated GO edge masks used by build_model
├── .github/workflows/           # Tag-triggered PyPI release
├── pyproject.toml               # Package metadata + pixi configuration
├── pixi.lock                    # Fully resolved environment
├── container_pixi.def           # Apptainer container definition
└── experiment_log               # Run log: which jobs ran, where, how long
```

## Usage

The package exports the model, the training loop and the loss functions:

```python
from gonnect import (
    Autoencoder, build_model,
    MSE, MSE_L1, MSE_Masked, MSE_Soft_Link_Sum, MSE_Soft_Link_Proxyless,
    make_data_splits, split_data, train, test, train_with_validation,
)
```

### Gene Ontology inputs

Building a biologically-informed model needs the ontology and the human gene
annotations. Both can be downloaded as follows:

```bash
mkdir go_data && cd go_data
curl -O https://purl.obolibrary.org/obo/go/go-basic.obo
curl -O https://current.geneontology.org/annotations/goa_human.gaf.gz && gunzip goa_human.gaf.gz
```

Point the library at them per call, or by setting the `GONNECT_DATA_DIR` environment variable:

```python
model = build_model(..., go_preprocessing=True, data_dir="go_data")
```

### Training a model

A full training run, building the GO hierarchy from the ontology files:

```python
import pandas as pd
import torch

from gonnect import MSE, build_model, make_data_splits, train_with_validation

n_nan_cols = 5   # leading metadata columns, not gene expression
n_samples = 9797

# Your expression matrix: samples as rows, genes as columns, after n_nan_cols
# metadata columns.
data = pd.read_csv("expression.csv.gz", compression="gzip")
genes = list(data.columns[n_nan_cols:])

# Returns four DataLoaders: full, train, validation, test.
dataloader, trainloader, validationloader, testloader = make_data_splits(
    data, n_nan_cols, n_samples, batch_size=64, data_split=0.7, seed=6
)

model = build_model(
    model_type="dense",             # "dense" or "sparse"
    biologically_informed="both",   # "encoder", "decoder", "both", or "none"
    soft_links=False,
    dataset_name="my_dataset",
    go_preprocessing=True,          # build the GO hierarchy from go-basic.obo
    merge_conditions=(1, 30, 50),   # min parents, min children, min terms per layer
    n_go_layers_used=5,
    activation_fn=torch.nn.ReLU,
    dtype=torch.float64,
    genes=genes,
    data_dir="go_data",             # where go-basic.obo and goa_human.gaf live
)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# Note the argument order: testloader comes before validationloader.
epoch_losses = train_with_validation(
    100, trainloader, testloader, validationloader,
    model, optimizer, MSE(), patience=10, device="cpu",
)
```

GO preprocessing takes several minutes on the full ontology. To reuse a
hierarchy, generate the masks once and load them instead:

```python
model = build_model(
    ..., go_preprocessing=False, masks_dir="out/masks",
)
```

`plot_depth_distribution` in `gonnect.data_processing.dag_analysis` needs
matplotlib, which is not a dependency — install it separately if you want it.

## Reproducing the paper

All steps assume `pixi shell` and the inputs from the
[4TU deposit](https://doi.org/10.4121/0d78788b-6bd7-4941-a942-245309107b6d).

**Training:** Submit the SLURM jobs that produced the published models:

```bash
sbatch slurm/AE_2.0/AE_2.0.0_both.sh
```

Each `src/AE_*.py` is configured for the container (`project_folder = "/opt/app"`,
`cluster = True`, `device = "cuda"`), so it expects to run inside the Apptainer
image rather than on a workstation. AE_2.x are the masked-MSE models; AE_3.x add
soft-link regularization.

> **Filenames vs. configurations.** The `AE_*.py` scripts were edited in place
> between runs, so a filename does not always match the experiment inside it —
> `src/AE_2.0.0_both.py` sets `experiment_version = ".6"` and runs AE_2.0.6.
> Check `experiment_name` / `experiment_version` at the top of a script, and see
> `experiment_log` for what was actually run.

**Figures:** One script per figure under [`figures/src/`](figures/src), plus a
runner:

```bash
pixi run python figures/src/run_all.py
```

**Baselines:** OntoVAE and VEGA comparisons live in
[`baselines/`](baselines) with their own environments and README.


## Data Availability

All data underlying the paper is deposited at 4TU.ResearchData:
[10.4121/0d78788b-6bd7-4941-a942-245309107b6d](https://doi.org/10.4121/0d78788b-6bd7-4941-a942-245309107b6d).

That covers both the processed gene expression (TCGA) and Gene Ontology inputs
used to train the models, and the derived inputs behind the figures —
activations, latent embeddings, model checkpoints, GSEA results and permutation
nulls. See [`figures/README.md`](figures/README.md) for what each file is and
which script reads it.

## Contributing

This is a research project. For questions or collaboration please contact the corresponding author T. Verlaan (t.verlaan@tudelft.nl).

## License

Released under the MIT License — see [LICENSE](LICENSE).

## Citation

If you use this package or repository, please cite:

> Verlaan T.\*, Lieftinck M.A.\*, Mwine W., Reinders M.J.T. *A critical
> evaluation of Gene Ontology priors in biologically-informed neural networks.*
> bioRxiv (2026). <https://doi.org/10.1101/2025.11.06.686983>
