import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from thesis_binn.model.Encoder import DenseBICoder
from thesis_binn.model.build_model import build_model


def get_module_weights(module: DenseBICoder):
    module_weights = []
    for layer in module.net_layers:
        if isinstance(layer, nn.Linear):
            module_weights.append(layer.weight.data)
    return module_weights


def get_soft_links_by_index(module: DenseBICoder):
    """Very inefficient way to filter soft links and their indices. (Has been replaced with sparse tensors)"""
    soft_link_values_per_layer = []
    soft_link_indices_per_layer = []
    layer_index = 0
    for layer in module.net_layers:
        if isinstance(layer, nn.Linear):
            soft_link_values = []
            soft_link_indices = []
            for i, row in enumerate(layer.weight.data):
                for j, col in enumerate(row):
                    if not module.edge_masks[layer_index][i][j]:
                        soft_link_values.append(col.item())
                        soft_link_indices.append((i, j))

            soft_link_values_per_layer.append(soft_link_values)
            soft_link_indices_per_layer.append(soft_link_indices)
            layer_index += 1
    return soft_link_values_per_layer, soft_link_indices_per_layer


def get_go_terms_by_index(module: DenseBICoder):
    """Returns a list of dictionaries (one per weight matrix), where the key [row, col] gives the two GO-terms that belong to that edge in the weight matrix."""
    go = module.go_layers
    index_dicts = []
    for i in range(len(go) - 1):
        index_to_go = {}
        for col, source_node in enumerate(go[i]):
            for row, sink_node in enumerate(go[i + 1]):
                index_to_go[row, col] = (source_node, sink_node)
        index_dicts.append(index_to_go)
    return index_dicts


def histogram_soft_links_per_layer(soft_links_per_layer, bin_width=0.01, i=None, a=1.0, model=""):
    values = soft_links_per_layer.values()
    bin_range = max(values) - min(values)
    n_bins = int(bin_range.item() / bin_width)
    plt.hist(soft_links_per_layer.values(), log=True, bins=n_bins, edgecolor="k", alpha=a)
    plt.xlabel("Value")
    plt.ylabel("# Soft Links")
    if i:
        plt.title(f"Distribution of Soft Link values between layers {str(i)} and {str(i + 1)}\nmodel: {model}")
    else:
        plt.title(f"Distribution of Soft Link values between Encoder and Decoder")
    # plt.show()


def split_weights(module: DenseBICoder):
    all_weights = get_module_weights(module)
    masks = module.edge_masks
    soft_links_masked = []
    soft_links_sparse = []
    hard_links_sparse = []
    for i in range(len(all_weights)):
        soft_links_masked.append(torch.masked_fill(all_weights[i], masks[i], -1))
        soft_links_sparse.append(torch.masked_fill(all_weights[i], masks[i], 0).to_sparse())
        hard_links_sparse.append(torch.masked_fill(all_weights[i], ~masks[i], 0).to_sparse())
    return soft_links_sparse, hard_links_sparse


if __name__ == "__main__":
    project_folder = "../../.."
    dataset_name = "TCGA_complete_bp_top1k"
    experiment_name = "AE_2.1"
    experiment_version = ".0"
    model_name = "decoder"
    seed = 42
    n_nan_cols = 5

    # Model construction
    model_type = "dense"
    biologically_informed = model_name  # change this for locally trained models
    soft_links = True
    random_version = None
    go_preprocessing = True
    merge_conditions = (1, 30, 50)
    n_go_layers_used = 5
    activation_fn = torch.nn.ReLU
    dtype = torch.float64

    print("----- START: Loading data -----")
    dataset = pd.read_csv(f"{project_folder}/data/{dataset_name}.csv.gz", compression="gzip")
    print("----- COMPLETED: Loading data -----")

    print("----- START: Building model -----")
    # Genes are only needed if there are no masks available from file for the model of interest
    if go_preprocessing:
        genes = list(dataset.columns[n_nan_cols:])
    else:
        genes = None
    model = build_model(model_type, biologically_informed, soft_links, dataset_name, go_preprocessing, merge_conditions,
                        n_go_layers_used, activation_fn, dtype, genes, random_version=random_version, package_call=True)
    model.load_state_dict(
        torch.load(
            f"{project_folder}/out/trained_models/{experiment_name}/{experiment_name + experiment_version}_{model_name}_model.pt",
            weights_only=True))

    module, soft_links_sparse, hard_links_sparse = None, None, None
    soft_links_sparse_enc, soft_links_sparse_dec, hard_links_enc, hard_links_dec = None, None, None, None
    if biologically_informed == "encoder":
        module = model.encoder
        soft_links_sparse, hard_links_sparse = split_weights(module)
    elif biologically_informed == "decoder":
        module = model.decoder
        soft_links_sparse, hard_links_sparse = split_weights(module)
    elif biologically_informed == "both":
        soft_links_sparse_enc, hard_links_sparse_enc = split_weights(model.encoder)
        soft_links_sparse_dec, hard_links_sparse_dec = split_weights(model.decoder)

    else:
        raise ValueError(f"Unknown biologically informed module: {biologically_informed}")

    if biologically_informed == "both":
        edge_dicts_enc = get_go_terms_by_index(model.encoder)
        edge_dicts_dec = get_go_terms_by_index(model.decoder)
        for i in range(len(soft_links_sparse_enc)):
            histogram_soft_links_per_layer(soft_links_sparse_enc[i], bin_width=0.01, a=0.5, model=model.name)
            # Debug: overlay histogram with hard links for comparison of distributions
            histogram_soft_links_per_layer(soft_links_sparse_dec[(n_go_layers_used-2)-i], bin_width=0.01, a=0.5, model=model.name)
            plt.legend(["Soft links Encoder", "Soft links Decoder"])
            plt.show()
    else:
        edge_dicts = get_go_terms_by_index(module)
        for i, layer_weights in enumerate(soft_links_sparse):
            histogram_soft_links_per_layer(layer_weights, bin_width=0.01, i=i, model=model.name)
            # Debug: overlay histogram with hard links for comparison of distributions
            histogram_soft_links_per_layer(hard_links_sparse[i], bin_width=0.01, i=i, a=0.5, model=model.name)
            plt.legend(["Soft links", "Hard links"])
            plt.show()

    pass
