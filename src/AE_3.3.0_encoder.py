import os
import pandas as pd
import torch

from thesis_binn.model.build_model import build_model
from thesis_binn.train.loss import MSE_Soft_Link_Sum, MSE, MSE_Masked
from thesis_binn.train.train import make_data_splits, train_with_validation, save_training_losses

if __name__ == "__main__":
    experiment_name = "AE_3.3"
    experiment_version = ".1"
    model_name = "encoder"
    project_folder = "/opt/app"
    cluster = True
    device = "cuda"
    # Model params
    model_type = "dense"
    biologically_informed = model_name
    soft_links = True
    random_version = None
    activation_fn = torch.nn.ReLU
    # GO params
    go_preprocessing = False
    merge_conditions = (1, 30, 50)  # min parents, min children, min terms per layer
    n_go_layers_used = 5
    # Training params
    dataset_name = "TCGA_complete_bp_top1k"
    loss = "soft links"
    soft_link_alpha = 100000
    n_samples = 9797
    batch_size = 100
    n_epochs = 10000
    learning_rate = 0.01
    momentum = 0.9
    patience = 100
    # Storage params
    save_losses = True
    loss_path = experiment_name + experiment_version + "_" + model_name
    save_weights = True
    save_weights_path = experiment_name + experiment_version + "_" + model_name
    load_weights = False
    load_weights_path = experiment_name + experiment_version + "_" + model_name
    # Additional params
    data_split = 0.7
    seed = 1
    dtype = torch.float64
    n_nan_cols = 5

    # Data processing
    data = pd.read_csv(f"{project_folder}/data/{dataset_name}.csv.gz", nrows=min(n_samples, 9797),
                       usecols=range(n_nan_cols + 1000), compression="gzip")
    genes = list(data.columns[n_nan_cols:])
    dataloader, trainloader, validationloader, testloader = make_data_splits(data,
                                                                             n_nan_cols,
                                                                             n_samples,
                                                                             batch_size,
                                                                             data_split,
                                                                             seed)
    # Construct model
    model = build_model(model_type,
                        biologically_informed,
                        soft_links,
                        dataset_name,
                        go_preprocessing,
                        merge_conditions,
                        n_go_layers_used,
                        activation_fn,
                        dtype,
                        genes,
                        cluster=cluster,
                        random_version=random_version)
    if load_weights:
        model.load_state_dict(
            torch.load(f"{project_folder}/out/trained_models/{experiment_name}/{load_weights_path}_model.pt",
                       weights_only=True))
        print(f"\n----- Loaded weights from file ({save_weights_path}_model.pt) -----")

    # Training
    if loss == "mse":
        loss_fn = MSE()
    elif loss == "mse masked":
        input_mask = torch.load(f"{project_folder}/out/masks/genes/{merge_conditions}/{dataset_name}_gene_mask.pt",
                                weights_only=True)
        loss_fn = MSE_Masked(input_mask, device)
    elif loss == "soft links":
        loss_fn = MSE_Soft_Link_Sum(model, alpha=soft_link_alpha)
    else:
        raise Exception(f"Unknown loss function: {loss}")

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    epoch_losses = train_with_validation(n_epochs, trainloader, testloader, validationloader, model, optimizer, loss_fn,
                                         patience, device)
    if save_losses:
        loss_directory = f"{project_folder}/out/trained_models/{experiment_name}"
        os.makedirs(loss_directory, exist_ok=True)
        save_training_losses(epoch_losses, f"{loss_directory}/{loss_path}_results.txt")

    if save_weights:
        weights_directory = f"{project_folder}/out/trained_models/{experiment_name}"
        os.makedirs(weights_directory, exist_ok=True)
        torch.save(model.state_dict(), f"{weights_directory}/{save_weights_path}_model.pt")
        print(f"\n----- Saved weights to file ({save_weights_path}_model.pt) -----")
