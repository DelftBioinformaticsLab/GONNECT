import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from goatools.obo_parser import GOTerm

from thesis_binn.data_processing.GeneTerm import GeneTerm
from thesis_binn.data_processing.ProxyTerm import ProxyTerm
from thesis_binn.data_processing.generate_masks import make_layers
from thesis_binn.model.Autoencoder import Autoencoder
from thesis_binn.model.build_model import build_model


def activations_per_term(model: Autoencoder, go_layers: [GOTerm], data: pd.DataFrame, bi_module: str):
    """Returns a DataFrame with GO-terms as columns and raw sample activations as rows."""
    # Generate activations by performing a forward pass over the provided data
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data.values)
        y = model(x)

    # Select which module we want to retrieve activations from and strip off gene layer
    if bi_module == "encoder":
        module = model.encoder
        go_layers = list(reversed(go_layers))[1:]
    else:
        module = model.decoder
        go_layers = go_layers[1:]

    # Match module activations to GO terms
    activation_dict = dict()
    for i, layer in enumerate(go_layers):
        for j, term in enumerate(layer):
            if not isinstance(term, GeneTerm) and not isinstance(term, ProxyTerm):
                # From the activations dict of module, get the activation of the linear layers (2*i) and select the column corresponding to the key GO term
                activation_dict[term.item_id] = list(module.activations.values())[2 * i].data[:, j].numpy()

    return pd.DataFrame(activation_dict)


def k_most_variable_terms(k: int, terms: pd.DataFrame):
    """Dirty filtering for large activation differences in activation DataFrame."""
    term_variances = terms.var()
    top_k_terms = term_variances.nlargest(k).index
    return top_k_terms


def setup_figure(data: pd.DataFrame, label: str, n_nan_cols: int, go: dict[str, GOTerm]):
    """Setup grid with GO-term columns and sample rows. Requires GO dict for naming GO-terms. Does not filter or sort data."""
    data = data.sort_values("cancer_type")
    data_values = data[data.columns[n_nan_cols:]]
    fig, ax = plt.subplots(figsize=(12, 15))
    ax.imshow(data_values, interpolation="none", aspect='auto')

    # Set column labels
    term_objects = [go[term_id] for term_id in data_values.columns]
    term_names = [term.name for term in term_objects]
    ax.set_xticks(np.arange(data_values.shape[1]))
    ax.set_xticklabels(term_names, rotation=270)

    # Set row labels
    labels = data[label].values
    label_positions = []
    label_names = []

    current_label = labels[0]
    label_positions.append(0)
    label_names.append(current_label)
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            label_positions.append(i)
            label_names.append(labels[i])
            current_label = labels[i]

    # Set y-ticks once per group
    ax.set_yticks(label_positions)
    ax.set_yticklabels(label_names)

    plt.tight_layout()


if __name__ == "__main__":
    # Experiment params
    experiment_name = "AE_1.0"
    experiment_version = ".0"
    model_name = "encoder"
    # Model params
    model_type = "dense"
    biologically_informed = "encoder"
    soft_links = False
    activation_fn = torch.nn.ReLU
    # GO params
    go_preprocessing = False
    merge_conditions = (1, 30, 50)  # min parents, min children, min terms per layer
    n_go_layers_used = 5
    # Data params
    dataset_name = "TCGA_complete_bp_top1k"
    n_nan_cols = 5

    # Model
    model = build_model(model_type, biologically_informed, soft_links, dataset_name, go_preprocessing, merge_conditions,
                        n_go_layers_used, activation_fn, dtype=torch.float64, package_call=True)
    model.load_state_dict(torch.load(
        f"../../../out/trained_models/{experiment_name}/{experiment_name + experiment_version}_{model_name}_model.pt",
        weights_only=True))

    # GO graph
    go_layers = make_layers(merge_conditions, dataset_name, n_nan_cols)[-n_go_layers_used:]
    latent_terms = [term.item_id for term in go_layers[1]]
    terms_list = []
    for layer in go_layers:
        terms_list += layer
    terms_dict = {term.item_id: term for term in terms_list}

    # Dataset
    dataset = pd.read_csv(f"../../../data/{dataset_name}.csv.gz", compression="gzip")
    data_values = dataset[dataset.columns[n_nan_cols:]]

    # All activations
    activation_values = activations_per_term(model, go_layers, data_values, "encoder")
    activation_data = pd.concat(
        [dataset[dataset.columns[:n_nan_cols]].reset_index(drop=True), activation_values.reset_index(drop=True)],
        axis=1)

    # Top k most variable terms
    top_k_most_variable_terms = k_most_variable_terms(30, activation_values)

    # Visualization
    cols = list(activation_data.columns[:n_nan_cols]) + list(top_k_most_variable_terms)
    plot_data = activation_data[cols]  # .sort_values("cancer_type")
    # plot_values = plot_data[plot_data.columns[n_nan_cols:]]
    # plt.imshow(plot_values[:200], interpolation="none", aspect="auto")  # cmap="RdYlGn"
    setup_figure(plot_data, "cancer_type", n_nan_cols, terms_dict)
    plt.show()
