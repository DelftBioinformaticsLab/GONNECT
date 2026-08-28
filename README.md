
# GONNECT

Code for the paper **"A critical evaluation of Gene Ontology priors in
biologically-informed neural networks"**.

**Authors**: T. Verlaan\*, M.A. Lieftinck\*, W. Mwine, and M.J.T. Reinders
(\* equal contribution)

**Affiliation**: Delft University of Technology, Computer Science Department, Delft Bioinformatics Lab (Delft, 2628XE, The Netherlands)

**Corresponding author**: T. Verlaan (t.verlaan@tudelft.nl)

GONNECT is a research framework for training biologically-informed autoencoders that leverage Gene Ontology (GO) structure to analyze gene expression data. This project implements various autoencoder architectures that incorporate biological knowledge through GO-based network constraints. The associated publication is currently in pre-print and available at: https://doi.org/10.1101/2025.11.06.686983


## Project Structure

```
GONNECT/
├── src/                         # Source code
│   ├── gonnect/                 # Core library (imported as `gonnect`)
│   │   ├── model/               # Model architectures (Autoencoder, Encoder, Decoder)
│   │   ├── train/               # Training utilities and loss functions
│   │   ├── data_processing/     # GO preprocessing and data handling
│   │   └── analysis/            # Analysis tools for trained models
│   ├── AE_*.py                  # Experiment scripts (AE_2.x, AE_3.x)
│   ├── GO_preprocessing_local.py  # Build GO layers/masks outside the cluster
│   └── benchmark_training_time.py # Training-time benchmark
├── figures/                     # Everything needed to reproduce the paper's figures
│   ├── src/                     # One figN.py per figure (see figures/README.md)
│   ├── out/                     # Generated figures
│   └── data/                    # Figure inputs -- not in git, see figures/README.md
├── slurm/                       # SLURM job submission scripts
├── data/                        # GO ontology (go-basic.obo); other inputs not in git
├── out/masks/                   # Pre-generated GO edge masks used by build_model
├── pyproject.toml               # Project metadata + pixi configuration
├── pixi.lock                    # Fully resolved environment
├── container_pixi.def           # Apptainer container definition
└── experiment_log               # Experiment tracking log
```

## Installation

### Using Pixi (Recommended)

This project uses [Pixi](https://pixi.sh) for dependency management:

```bash
# Install pixi if not already installed
curl -fsSL https://pixi.sh/install.sh | bash

# Clone the repository
git clone https://github.com/DelftBioinformaticsLab/GONNECT.git
cd GONNECT

# Install dependencies
pixi install

# Activate the environment
pixi shell
```

### Using Apptainer

For HPC environments, use the provided Apptainer container:

```bash
# Build the container. The name encodes the version in pyproject.toml;
# the SLURM scripts expect exactly this file name.
apptainer build container_pixi_0.3.0.sif container_pixi.def
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
    ./container_pixi_0.3.0.sif pixi run python -u src/AE_2.0.0_both.py
```

See `slurm/` for the exact invocations used for the published runs (those pin
`container_pixi_0.2.1.sif`, the environment version in force at training time).

### Requirements

- **Python ≥3.12.** `data_processing/go_preprocessing.py` uses PEP 701 f-strings,
  which are a `SyntaxError` on 3.9–3.11.
- **CUDA 12.4.** Declared as a pixi *system requirement*, so the environment will
  not solve on a machine without it. GPU is not optional.
- **Platforms: `linux-64` and `win-64` only.** macOS is not supported.

### Dependencies

- **PyTorch** (>2.4.1) with **pytorch-cuda** 12.4
- **pandas** (≥2.2.3) for data manipulation
- **goatools** (≥1.4.12) for Gene Ontology processing
- **scikit-learn** (≥1.6.1) and **scipy** for machine learning utilities
- **scanpy** (≥1.10.4) for single-cell analysis
- **matplotlib** (≥3.10.8, <3.11), **seaborn** & **plotly** for visualization
- **pyarrow**, **gseapy**, **openpyxl**, **fastcluster**, **tqdm** for the figure
  and enrichment pipelines in `figures/`

See `pyproject.toml` for the complete specification and `pixi.lock` for the
fully resolved environment.

## Usage

### Basic Training

Train a biologically-informed autoencoder:

```python
import pandas as pd
import torch

from gonnect import MSE, build_model, make_data_splits, train_with_validation

dataset_name = "TCGA_complete_bp_top1k"
merge_conditions = (1, 30, 50)  # min parents, min children, min terms per layer
n_nan_cols = 5                  # leading metadata columns, not gene expression

# Data. `n_nan_cols` metadata columns are stripped before the split.
data = pd.read_csv(f"data/{dataset_name}.csv.gz", compression="gzip")
genes = list(data.columns[n_nan_cols:])
dataloader, trainloader, validationloader, testloader = make_data_splits(
    data, n_nan_cols, n_samples=9797, batch_size=64, data_split=0.7, seed=6
)

# Model
model = build_model(
    model_type="dense",             # "dense" or "sparse"
    biologically_informed="both",   # "encoder", "decoder", "both", or "none"
    soft_links=True,
    dataset_name=dataset_name,
    go_preprocessing=False,         # False -> load pre-generated masks from out/masks/
    merge_conditions=merge_conditions,
    n_go_layers_used=5,
    activation_fn=torch.nn.ReLU,
    dtype=torch.float64,
    genes=genes,
)

# Training. Note the argument order: testloader comes before validationloader.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
epoch_losses = train_with_validation(
    100, trainloader, testloader, validationloader,
    model, optimizer, MSE(), patience=10, device="cpu",
)
```

With `go_preprocessing=False`, `build_model` loads pre-generated masks from
`out/masks/`, resolved relative to the working directory — so run this from the
repository root.

### Running Experiments

The project includes pre-configured experiment scripts:

```bash
# Submit to a SLURM cluster (this is how the published runs were produced)
sbatch slurm/AE_2.0/AE_2.0.0_both.sh
```

Each `src/AE_*.py` is configured for the container: it sets
`project_folder = "/opt/app"`, `cluster = True` and `device = "cuda"`, so it
expects to be run inside the Apptainer image with the binds shown above rather
than directly on a workstation.

> **Filenames vs. configurations.** The `AE_*.py` scripts were edited in place
> between runs, so a filename does not always match the experiment inside it —
> e.g. `src/AE_2.0.0_both.py` sets `experiment_version = ".6"` and runs AE_2.0.6.
> Check the `experiment_name` / `experiment_version` at the top of a script, and
> see `experiment_log` for what was actually run.

The project includes multiple experiment series:

- **AE_2.x**: Models with masked MSE loss and various patience/seed settings
- **AE_3.x**: Experiments with soft-link regularization

`experiment_log` records which jobs were run, on which nodes, and how long they
took. It is a run log, not a results table.

## Reproducing the Figures

Every figure in the paper that was generated by code has a script under
[`figures/src/`](figures/src), one per figure, plus `run_all.py` to build them
all:

```bash
pixi run python figures/src/run_all.py
```

The inputs those scripts read (~12 GB) are not in this repository. See
[`figures/README.md`](figures/README.md) for where to obtain them, what each
script consumes, and which figures are hand-drawn schematics rather than
generated output.

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

Please cite the paper, not this repository:

> Verlaan T.\*, Lieftinck M.A.\*, Mwine W., Reinders M.J.T. *A critical
> evaluation of Gene Ontology priors in biologically-informed neural networks.*
> bioRxiv (2025). <https://doi.org/10.1101/2025.11.06.686983>

[CITATION.cff](CITATION.cff) declares this as the `preferred-citation`, so
GitHub's "Cite this repository" button and `cffconvert` both resolve to it.
