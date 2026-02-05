
# GONNECT: Coupling Biological Systems to Neural Networks for Improved Model Interpretability

**Authors**: M.A. Lieftinck, T. Verlaan, and M.J.T. Reinders

**Affiliation**: Delft University of Technology, Computer Science Department, Delft Bioinformatics Lab (Delft, 2628XE, The Netherlands)

**Corresponding author**: T. Verlaan (t.verlaan@tudelft.nl)

GONNECT is a research framework for training biologically-informed autoencoders that leverage Gene Ontology (GO) structure to analyze gene expression data. This project implements various autoencoder architectures that incorporate biological knowledge through GO-based network constraints. The associated publication is currently in pre-print and available at: [link to publication].

## Overview

GONNECT implements biologically-informed neural networks (BINNs) that use Gene Ontology biological process (BP) hierarchies to constrain autoencoder architectures.

Key features:

- **Multiple Architecture Types**: Dense and sparse biologically-informed encoders/decoders
- **Flexible Training**: Support for encoder-only, decoder-only, or full autoencoder training
- **GO Integration**: Automatic processing of Gene Ontology hierarchies for network structure
- **Loss Functions**: MSE, masked MSE, and soft-link regularization
- **HPC Support**: SLURM job scripts for high-performance computing environments
- **Containerization**: Singularity container support with Pixi package management

## Project Structure

```
GONNECT/
├── src/                          # Source code
│   ├── thesis_binn/             # Core library
│   │   ├── model/               # Model architectures (Autoencoder, Encoder, Decoder)
│   │   ├── train/               # Training utilities and loss functions
│   │   ├── data_processing/     # GO preprocessing and data handling
│   │   └── analysis/            # Analysis tools for trained models
│   ├── AE_*.py                  # Experiment scripts (various versions)
│   └── old_scripts/             # Legacy experimental code
├── slurm/                       # SLURM job submission scripts
├── data/                        # Data directory (GO ontology files)
├── pixi.toml                    # Package management configuration
├── container_pixi.def           # Singularity container definition
└── experiment_log               # Detailed experiment tracking log
```

## Installation

### Using Pixi (Recommended)

This project uses [Pixi](https://pixi.sh) for dependency management:

```bash
# Install pixi if not already installed
curl -fsSL https://pixi.sh/install.sh | bash

# Clone the repository
git clone https://github.com/mlieftinck/GONNECT.git
cd GONNECT

# Install dependencies
pixi install

# Activate the environment
pixi shell
```

### Using Apptainer

For HPC environments, use the provided Apptainer container:

```bash
# Build the container
apptainer build gonnect.sif container_pixi.def

# Run experiments in the container (add cuda flags if needed)
apptainer exec gonnect.sif pixi run python src/AE_2.0.0_both.py
```

### Dependencies

- **PyTorch** (≥2.4.1) with CUDA support
- **pandas** (≥2.2.3) for data manipulation
- **goatools** (≥1.4.12) for Gene Ontology processing
- **scikit-learn** (≥1.6.1) for machine learning utilities
- **scanpy** (≥1.10.4) for single-cell analysis
- **matplotlib** & **plotly** for visualization

See `pixi.toml` for complete dependency specifications.

## Usage

### Basic Training

Train a biologically-informed autoencoder:

```python
from thesis_binn.model.build_model import build_model
from thesis_binn.train.train import make_data_splits, train_with_validation

# Configure experiment
model = build_model(
    model_type="dense",
    biologically_informed="both",  # "encoder", "decoder", "both", or "none"
    soft_links=True,
    dataset_name="TCGA_complete_bp_top1k",
    go_preprocessing=False,
    merge_conditions=(1, 30, 50),
    n_go_layers_used=5,
    activation_fn=torch.nn.ReLU,
    dtype=torch.float64,
    genes=gene_list
)

# Train the model
train_with_validation(n_epochs, trainloader, testloader, validationloader, 
                     model, optimizer, loss_fn, patience, device)
```

### Running Experiments

The project includes pre-configured experiment scripts:

```bash
# Run a specific experiment
python src/AE_2.0.0_both.py

# Submit to SLURM cluster
sbatch slurm/AE_2.0/AE_2.0.0_both.sh
```

The project includes multiple experiment series:

- **AE_1.x**: Basic autoencoder models with different data splits
- **AE_2.x**: Models with masked MSE loss and various patience/seed settings
- **AE_3.x**: Experiments with soft-link regularization

See `experiment_log` for detailed experiment tracking and results.

## Data Availability

The analyses performed in the paper rely on the gene ontology and TCGA gene expression datasets. The versions we used after processing are hosted at: [link to datasets].


## Contributing

This is a research project. For questions or collaboration please contact the corresponding author T. Verlaan (t.verlaan@tudelft.nl).


## Citation

If you use this code in your research, please cite the associated paper:

[TODO: add citation here]

