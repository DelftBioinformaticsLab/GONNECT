"""Script for quickly storing all embeddings for future analysis."""
import pandas as pd
import torch
from gonnect.model.Autoencoder import Autoencoder
from gonnect.model.build_model import build_model
from gonnect.train.train import split_data


def save_embeddings(model: Autoencoder, dataset:pd.DataFrame, path):
    model.eval()
    with torch.no_grad():
        # Compute latent representation of data
        x = torch.tensor(dataset.values)
        latent_x = model.encoder(x)
    torch.save(latent_x, path)


def create_model(experiment_name: str, binn_module: str, version: int):
    """Helper function that only takes the variable parameters as argument and adds remaining constant variables outside of main script."""
    random_version = None
    soft_links = False
    if experiment_name == "AE_2.2":
        random_version = version - 14 # Hardcoded for last set of experiments
    if experiment_name == "AE_2.1":
        soft_links = True

    model_name = binn_module
    biologically_informed = model_name
    # Don't change these parameters
    project_folder = "../../.."
    dataset_name = "TCGA_complete_bp_top1k"
    n_nan_cols = 5
    model_type = "dense"
    go_preprocessing = False
    merge_conditions = (1, 30, 50)
    n_go_layers_used = 5
    activation_fn = torch.nn.ReLU
    dtype = torch.float64
    genes = None

    model = build_model(model_type, biologically_informed, soft_links, dataset_name, go_preprocessing,
                        merge_conditions, n_go_layers_used, activation_fn, dtype, genes,
                        random_version=random_version, package_call=True)

    model.load_state_dict(torch.load(f"{project_folder}/out/trained_models/{experiment_name}/{experiment_name}."
                                     f"{str(version)}_{model_name}_model.pt", weights_only=True))
    return model


if __name__ == '__main__':
    # Model parameters
    experiment_names = ["AE_2.0", "AE_2.1", "AE_2.2"]
    modules = ["none", "encoder", "decoder", "both"]
    seeds = [2, 3, 4, 5, 6]
    versions = seeds

    # Prepare data splits for each seed
    dataset = pd.read_csv(f"../../../data/TCGA_complete_bp_top1k.csv.gz", compression="gzip")
    fullset = dataset[dataset.columns[5:]]
    testsets = {}
    for seed in seeds:
        _, _, testset = split_data(dataset, n_nan_cols=5, seed=seed)
        testsets[seed] = testset

    # Store all embeddings
    for experiment_name in experiment_names:
        if experiment_name != "AE_2.0":
            modules = modules[1:] # Ignore "none" for other experiments
        if experiment_name == "AE_2.2":
            versions = [22, 23, 24, 25, 26] # Inconsistent naming for this experiment
        for module in modules:
            for seed, version in zip(seeds, versions):
                model = create_model(experiment_name, module, version)
                save_embeddings(model, fullset, f"../../../out/embeddings/{experiment_name}/{experiment_name}.{version}_{module}_full_dataset.pt")
                save_embeddings(model, testsets[seed],f"../../../out/embeddings/{experiment_name}/{experiment_name}.{version}_{module}_testset.pt")

