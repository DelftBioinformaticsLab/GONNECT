import pandas as pd
import torch
import argparse
from gonnect.model.build_model import build_model
from gonnect.train.train import make_data_splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parents", help="minimal allowed number of parents", type=int, default=1)
    parser.add_argument("-c", "--children", help="minimal allowed number of children", type=int, default=30)
    parser.add_argument("-t", "--terms_per_layer", help="minimal allowed number of terms per layer", type=int, default=50)
    args = parser.parse_args()
    experiment_name = f"GO_{args.parents}_{args.children}_{args.terms_per_layer}"
    experiment_version = ""
    model_name = "encoder"
    project_folder = ".."
    cluster = False
    device = "cpu"
    # Model params
    model_type = "dense"
    biologically_informed = model_name
    soft_links = False
    random_version = None
    activation_fn = torch.nn.ReLU
    # GO params
    go_preprocessing = True
    merge_conditions = (args.parents, args.children, args.terms_per_layer)  # min parents, min children, min terms per layer
    n_go_layers_used = 6
    # Training params
    dataset_name = "TCGA_complete_bp_top2k"  # "SEAAD_A9_RNAseq_final-nuclei.2024-02-13_bp_top1k"
    n_genes = 1000
    loss = "mse masked"
    soft_link_alpha = 100
    n_samples = 1000 # 9797
    batch_size = 100
    n_epochs = 10000
    learning_rate = 0.01
    momentum = 0.9
    patience = 10
    # Storage params
    save_losses = False
    loss_path = experiment_name + experiment_version + "_" + model_name
    save_weights = False
    save_weights_path = experiment_name + experiment_version + "_" + model_name
    load_weights = False
    load_weights_path = experiment_name + experiment_version + "_" + model_name
    # Additional params
    data_split = 0.7
    seed = 6
    dtype = torch.float64
    n_nan_cols = 5  # 133

    # Data processing
    data = pd.read_csv(f"{project_folder}/data/{dataset_name}.csv.gz", nrows=min(n_samples, 9797),
                       # usecols=range(n_nan_cols + n_genes),
                       compression="gzip")
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
                        random_version=random_version,
                        data_dir=f"{project_folder}/data")
    if load_weights:
        model.load_state_dict(
            torch.load(f"{project_folder}/out/trained_models/{experiment_name}/{load_weights_path}_model.pt",
                       weights_only=True))
        print(f"\n----- Loaded weights from file ({save_weights_path}_model.pt) -----")
