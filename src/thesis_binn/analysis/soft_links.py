import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from goatools.obo_parser import GOTerm

from thesis_binn.data_processing.ProxyTerm import ProxyTerm
from thesis_binn.data_processing.go_preprocessing import construct_go_bp
from thesis_binn.model.Autoencoder import Autoencoder
from thesis_binn.model.Coder import DenseCoder
from thesis_binn.model.Encoder import DenseBICoder
from thesis_binn.model.build_model import build_model


def get_module_weights(module: DenseCoder):
    """Returns a list of the weight matrices of the given module."""
    module_weights = []
    for layer in module.net_layers:
        if isinstance(layer, nn.Linear):
            module_weights.append(layer.weight.data)
    return module_weights


def get_soft_links_by_index(module: DenseBICoder):
    """DEPRECATED: Very inefficient way to filter soft links and their indices. (Has been replaced with sparse tensor implementation.)"""
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
    """Returns a list of dictionaries (one per weight matrix), where the key [row, col] gives the two GO-terms that belong to that edge in the weight matrix.
    (Input module must be initialized with GO processing in order to obtain GO-term objects.)"""
    go = module.go_layers
    index_dicts = []
    for i in range(len(go) - 1):
        index_to_go = {}
        for col, source_node in enumerate(go[i]):
            for row, sink_node in enumerate(go[i + 1]):
                index_to_go[row, col] = (source_node, sink_node)
        index_dicts.append(index_to_go)
    return index_dicts


def histogram_weights_per_layer(weights_per_layer, bin_width=0.01, i=None, a=1.0):
    """Assumes a sparse tensor containing the weights from a specific layer and plots the distribution of their values."""
    values = weights_per_layer.values()
    # Adjust number of bins so that bin width is always constant
    bin_range = max(values) - min(values)
    n_bins = max(1, int(bin_range.item() / bin_width))
    # Plot histogram
    _, bins, _ = plt.hist(weights_per_layer.values(), log=True, bins=n_bins, alpha=a)
    plt.xlabel("Value")
    plt.ylabel("# Weights")
    plt.title(f"Distribution of weight values between layers {str(i)} and {str(i + 1)}")


def split_weights(module: DenseCoder):
    """Split the weight matrices of the given module into GO and non-GO weights. If the module does not have any masks, it returns the full weight matrix as sparse tensor."""
    all_weights = get_module_weights(module)
    if hasattr(module, "edge_masks"):
        masks = module.edge_masks
    else:
        # Model is fully connected
        return None, [weight_mat.to_sparse() for weight_mat in all_weights]

    # Split weight matrices into two sparse tensors, one for soft and one for fixed links
    soft_links_sparse = []
    fixed_links_sparse = []
    for i in range(len(all_weights)):
        soft_links_sparse.append(torch.masked_fill(all_weights[i], masks[i], 0).to_sparse())
        fixed_links_sparse.append(torch.masked_fill(all_weights[i], ~masks[i], 0).to_sparse())
    return fixed_links_sparse, soft_links_sparse


def split_weights_per_module(model: Autoencoder):
    weights = []
    weights.append(split_weights(model.encoder))
    weights.append(split_weights(model.decoder))
    return weights


def get_top_k_soft_links(soft_links_per_layer: [torch.Tensor], k: int, layer_index: int, go_dicts: [dict]):
    soft_links = soft_links_per_layer[layer_index]
    values = soft_links.values()
    indices = soft_links.indices()
    # Select the top k highest magnitudes
    topk = torch.topk(values.abs(), k)
    topk_values = values[topk.indices]
    topk_indices = indices[:, topk.indices]
    # Convert indices to GO-terms
    go_dict = go_dicts[layer_index]
    top_links = []
    for i in range(len(topk_values)):
        source_term, sink_term = go_dict[topk_indices[0][i].item(), topk_indices[1][i].item()]
        soft_link_terms = (source_term, sink_term, topk_values[i])
        top_links.append(soft_link_terms)
    return top_links


def print_soft_links(soft_link_list: list, go: dict[GOTerm]):
    for soft_link in soft_link_list:
        term1, term2, value = soft_link
        for term in (term1, term2):
            if isinstance(term, ProxyTerm):
                original_term_index = term.item_id.index("_") + 1
                term_id = term.item_id[original_term_index:]
            else:
                term_id = term.item_id
            name = go[term_id].name
            print(f"{term_id} -> {name}")
        print(f"Soft Link Value = {value:.3e}\n")


if __name__ == "__main__":
    project_folder = "../../.."
    dataset_name = "TCGA_complete_bp_top1k"
    experiment_name = "AE_3.2"
    experiment_version = ".3"
    model_name = "encoder"
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

    go_dict = construct_go_bp(genes, merge_conditions, print_go=True, package_call=True)

    # Build model
    model = build_model(model_type, biologically_informed, soft_links, dataset_name, go_preprocessing, merge_conditions, n_go_layers_used, activation_fn, dtype, genes, random_version=random_version, package_call=True, preprocessed_go_dict=go_dict)
    model.load_state_dict(torch.load(f"{project_folder}/out/trained_models/{experiment_name}/{experiment_name + experiment_version}_{model_name}_model.pt", weights_only=True))

    # Histogram comparing Fixed Links, Soft Links, FC
    model_GO = build_model(model_type, biologically_informed, False, dataset_name, go_preprocessing, merge_conditions, n_go_layers_used, activation_fn, dtype, genes, random_version=random_version, package_call=True, preprocessed_go_dict=go_dict)
    model_GO.load_state_dict(torch.load(f"{project_folder}/out/trained_models/AE_3.-1/AE_3.-1.2_{model_name}_model.pt", weights_only=True))

    model_FC = build_model(model_type, "none", False, dataset_name, go_preprocessing, merge_conditions, n_go_layers_used, activation_fn, dtype, genes, random_version=random_version, package_call=True, preprocessed_go_dict=go_dict)
    model_FC.load_state_dict(torch.load(f"{project_folder}/out/trained_models/AE_3.-1/AE_3.-1.3_none_model.pt", weights_only=True))

    # Get weight values for each model
    GO_weights = split_weights_per_module(model_GO)
    SL_weights = split_weights_per_module(model)
    FC_weights = split_weights_per_module(model_FC)

    module_index = 0 if model_name == "encoder" else 1
    fixed_GO = GO_weights[module_index][0]
    soft_GO = SL_weights[module_index][0]
    soft_FC = SL_weights[module_index][1]
    fixed_FC = FC_weights[module_index][1]
    for i in range(n_go_layers_used - 1):
    # Debug: Plot weights of a single layer
    # if i == 1:
        histogram_weights_per_layer(fixed_FC[i], bin_width=0.01, i=i, a=0.2)
        histogram_weights_per_layer(soft_FC[i], bin_width=0.01, i=i, a=0.3)
        histogram_weights_per_layer(soft_GO[i], bin_width=0.01, i=i, a=0.3)
        histogram_weights_per_layer(fixed_GO[i], bin_width=0.01, i=i, a=0.3)
        plt.legend(["FC Links", "Soft non-GO Links", "Soft GO Links", "Fixed GO Links"])
        plt.show()

    # Soft Link Identification
    index_to_go = get_go_terms_by_index(model_GO.encoder if model_name == "encoder" else model_GO.decoder)
    layer = 0
    soft_link_terms = get_top_k_soft_links(soft_FC, 10, 0, index_to_go)
    print_soft_links(soft_link_terms, go_dict)
    pass
