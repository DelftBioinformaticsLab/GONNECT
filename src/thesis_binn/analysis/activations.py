import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
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


def k_most_abundant_labels(data: pd.DataFrame, label: str, k: int):
    labels = data[label]
    unique_labels = list(labels.unique())
    counts = []
    ordered_labels = []
    for label in unique_labels:
        counts.append(labels.count(label))
    for i in range(k):
        ordered_labels.append(unique_labels[counts.index(max(counts))])
        counts[counts.index(max(counts))] = -1
    return ordered_labels


def setup_figure(data: pd.DataFrame, label: str, n_nan_cols: int, go: dict[str, GOTerm]):
    """Setup grid with GO-term columns and sample rows. Requires GO dict for naming GO-terms. Does not filter or sort data."""
    data = data.sort_values("cancer_type")
    data_values = data[data.columns[n_nan_cols:]]

    # Normalize columns -> Debug: How to properly normalize?
    data_values = (data_values - data_values.mean()) / (data_values.std() / data_values.mean())

    fig, ax = plt.subplots(figsize=(12, 18))
    ax.imshow(data_values, interpolation="none", aspect="auto", vmin=data_values.min().min(),
              vmax=data_values.max().max(), cmap="plasma")

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


def activation_heatmap(activations: pd.DataFrame, label: str, n_nan_cols: int, go: dict[str, GOTerm], k=20):
    # Remove any sample with NaN as label
    activations = activations.dropna(subset=[label])

    # Order and group samples by label
    activations = activations.sort_values(label)
    activation_values = activations[activations.columns[n_nan_cols:]]
    labels = activations[label].values
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

    # Normalize before performing ANOVA (drop terms without variance)
    var_terms = [col for col in activation_values if activation_values[col].std() != 0]
    activation_values = activation_values[var_terms]
    activation_values = (activation_values - activation_values.mean()) / (activation_values.std())

    # Select k terms with best class separation
    f_values, p_values = f_classif(activation_values, labels)
    f_scores = pd.Series(f_values, index=activation_values.columns, name="f_values")
    top_k_terms = f_scores.nlargest(k).index  # Default k=20

    # Debug:
    # print(top_k_terms) # Print so that the same GO terms can be displayed for different models
    # top GO terms
    # top_k_terms = ['GO:0015810', 'GO:0043487', 'GO:0030194', 'GO:0048690', 'GO:0045763',
    #        'GO:0010903', 'GO:0048686', 'GO:0031641', 'GO:1900135', 'GO:0048731',
    #        'GO:0007596', 'GO:0048167', 'GO:0010902', 'GO:0060770', 'GO:0048688',
    #        'GO:0034377', 'GO:0045834', 'GO:0042445', 'GO:0009069', 'GO:2000504',
    #        'GO:0002087', 'GO:0048858', 'GO:0070778', 'GO:0008206', 'GO:0060291',
    #        'GO:0043102', 'GO:0040019', 'GO:0072530', 'GO:1902047', 'GO:0046907',
    #        'GO:0002034', 'GO:0006886', 'GO:0030195', 'GO:0019227', 'GO:0019229',
    #        'GO:0031643', 'GO:0090660', 'GO:1905869', 'GO:0010466', 'GO:1990961',
    #        'GO:0032532', 'GO:1903712', 'GO:0097186', 'GO:0019400', 'GO:0006591',
    #        'GO:0055086', 'GO:0033626', 'GO:0030516', 'GO:0120036', 'GO:0040029']
    # top Random terms
    # top_k_terms = ['GO:0043487', 'GO:0060264', 'GO:1901679', 'GO:2000182', 'GO:0098828',
    #    'GO:0015810', 'GO:1904015', 'GO:1903449', 'GO:0002018', 'GO:0007632',
    #    'GO:0008630', 'GO:0046878', 'GO:0060513', 'GO:0001990', 'GO:1905941',
    #    'GO:1904179', 'GO:0140367', 'GO:0033600', 'GO:0034444', 'GO:0032345',
    #    'GO:0071393', 'GO:1902459', 'GO:0051798', 'GO:0060253', 'GO:0010718',
    #    'GO:0032331', 'GO:0050867', 'GO:0045859', 'GO:0033555', 'GO:0071214',
    #    'GO:0051241', 'GO:1905906', 'GO:0033673', 'GO:0010544', 'GO:1905869',
    #    'GO:0090190', 'GO:0120163', 'GO:0071869', 'GO:0048167', 'GO:0050891',
    #    'GO:1902532', 'GO:0071466', 'GO:0045820', 'GO:0150094', 'GO:0071318',
    #    'GO:0016056', 'GO:0090191', 'GO:0010232', 'GO:2000261', 'GO:0006670']

    activation_values = activation_values[top_k_terms]

    # Cluster columns using clustermap
    print("\n----- START: CLustering columns -----")
    clustermap = sns.clustermap(activation_values,
                                metric='correlation',
                                method='average',
                                col_cluster=True,
                                row_cluster=False,
                                cmap='viridis',
                                cbar_pos=None,
                                xticklabels=True,
                                yticklabels=True)
    print("----- COMPLETED: CLustering columns -----")
    # Extract ordered column indices
    ordered_col_indices = clustermap.dendrogram_col.reordered_ind

    # Debug:
    # print(ordered_col_indices) # Print so that the same GO terms can be displayed for different models
    # GO cols Encoder
    # ordered_col_indices = [26, 49, 7, 17, 36, 22, 33, 1, 24, 48, 29, 4, 6, 14, 45, 44, 18, 10, 16, 23, 39, 27, 28, 40, 46, 42, 43, 38, 32, 2, 19, 12, 5, 15, 25, 30, 8, 13, 35, 20, 34, 41, 37, 31, 0, 3, 47, 21, 9, 11]
    # Random cols Encoder
    # ordered_col_indices = [30, 40, 29, 22, 28, 47, 45, 46, 34, 35, 16, 17, 21, 31, 37, 43, 27, 7, 1, 8, 10, 0, 3, 26, 12, 13, 11, 20, 5, 49, 44, 48, 33, 38, 32, 6, 18, 24, 23, 25, 42, 14, 19, 39, 15, 41, 36, 9, 2, 4]

    ordered_columns = activation_values.columns[ordered_col_indices]
    ordered_values = activation_values[ordered_columns]

    # Ensure symmetric colorbar
    colorbar_extreme_value = min(abs(ordered_values.min().min()), abs(ordered_values.max().max()))
    print(f"colorbar_extreme_values = {ordered_values.min().min()}, {ordered_values.max().max()}")

    plt.figure(figsize=(max(20, int(4 * (k / 10))), 28))
    sns.heatmap(ordered_values, cmap="coolwarm", xticklabels=False, yticklabels=False, cbar_kws={'label': 'Activation'},
                vmin=-colorbar_extreme_value, vmax=colorbar_extreme_value)

    # Set column labels
    term_objects = [go[term_id] for term_id in ordered_columns]
    term_names = [term.name for term in term_objects]
    plt.xticks(np.arange(ordered_values.shape[1]) + 0.5, labels=term_names, rotation=270)

    # Add one y-tick per cancer type group
    plt.yticks(label_positions, label_names)
    plt.xlabel('GO terms (Clustered by Similarity)')
    plt.ylabel(f'Samples grouped by {label}')
    plt.title('GO Term Activation Heatmap')
    plt.tight_layout()
    plt.show()


def histogram_for_class_activation(activations: pd.DataFrame, label_col: str, labels: [str], term: GOTerm, a=1.0):
    n_bins = 30
    bin_width = -1
    for label in labels:
        filtered_activations = activations[activations[label_col] == label]
        filtered_term_activation = filtered_activations[term.item_id]

        # Dynamically set bin width to ensure +- 30 bins per class
        bin_range = max(filtered_term_activation) - min(filtered_term_activation)
        if bin_width == -1:
            bin_width = bin_range / n_bins
        n_bins = max(1, int(bin_range / bin_width))
        # Plot histogram
        plt.hist(filtered_term_activation, bins=n_bins, alpha=a)

    plt.xlabel(f"Activation of {term.item_id}: {term.name}")
    plt.ylabel("# samples")
    plt.legend(labels)
    plt.title(f"Distribution of activations for {term.name}")
    plt.show()


def anova_distribution_per_module(activations: pd.DataFrame, label: str, module: str, n_nan_cols=5):
    # Remove any sample with NaN as label
    activations = activations.dropna(subset=[label])

    activation_values = activations[activations.columns[n_nan_cols:]]
    labels = activations[label].values

    # Normalize before performing ANOVA (drop terms without variance)
    var_terms = [col for col in activation_values if activation_values[col].std() != 0]
    activation_values = activation_values[var_terms]
    activation_values = (activation_values - activation_values.mean()) / (activation_values.std())

    # Get ANOVA scores
    f_values, p_values = f_classif(activation_values, labels)
    f_scores = pd.Series(f_values, index=activation_values.columns, name="f_values")

    # Set bin number
    bin_range = f_values.max().max()
    bin_width = 400
    n_bins = max(1, int(bin_range / bin_width))

    plt.hist(f_scores, bins=n_bins, alpha=0.5)
    plt.title(f"Distribution of ANOVA F-values per node for Biologically-Informed {module.capitalize()}")
    plt.xlabel("F-values")
    plt.ylabel("# nodes")
    # plt.show()


def anova_distribution(model: Autoencoder, go_layers: [GOTerm], dataset: pd.DataFrame, label: str, n_nan_cols=5):
    data_values = dataset[dataset.columns[n_nan_cols:]]
    activation_values_enc = activations_per_term(model, go_layers, data_values, "encoder")
    activation_values_dec = activations_per_term(model, go_layers, data_values, "decoder")
    activations_enc = pd.concat([dataset[dataset.columns[:n_nan_cols]].reset_index(drop=True), activation_values_enc.reset_index(drop=True)], axis=1)
    activations_dec = pd.concat([dataset[dataset.columns[:n_nan_cols]].reset_index(drop=True), activation_values_dec.reset_index(drop=True)], axis=1)
    anova_distribution_per_module(activations_enc, label, "encoder")
    anova_distribution_per_module(activations_dec, label, "decoder")
    plt.title(f"Distribution of ANOVA F-values for {model.name}")
    plt.legend([model.encoder.name, model.decoder.name])
    plt.xlim((-1000, 22000))
    plt.show()


if __name__ == "__main__":
    # Experiment params
    experiment_name = "AE_2.0"
    experiment_version = ".0"
    model_name = "encoder"
    # Model params
    model_type = "dense"
    biologically_informed = model_name
    soft_links = False
    random_version = None  # "1"
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
    terms_list = []
    for layer in go_layers:
        terms_list += layer
    terms_dict = {term.item_id: term for term in terms_list}

    # Dataset
    dataset = pd.read_csv(f"../../../data/{dataset_name}.csv.gz", compression="gzip")
    data_values = dataset[dataset.columns[n_nan_cols:]]

    # All activations
    activation_values = activations_per_term(model, go_layers, data_values, biologically_informed)
    activation_data = pd.concat(
        [dataset[dataset.columns[:n_nan_cols]].reset_index(drop=True), activation_values.reset_index(drop=True)],
        axis=1)

    # Top k most variable terms
    top_k_most_variable_terms = k_most_variable_terms(30, activation_values)

    # Visualization
    cols = list(dataset.columns[:n_nan_cols]) + list(top_k_most_variable_terms)
    plot_data = activation_data[cols]

    selected_labels = ["Bladder", "Kidney"]

    # Replace activation_data by activation_data[activation_data["tumor_tissue_site"].isin(selected_labels)] to plot only selected labels
    # activation_heatmap(activation_data, "tumor_tissue_site", n_nan_cols, terms_dict, k=50)

    # histogram_for_class_activation(activation_data, "tumor_tissue_site", ["Kidney", "Bladder"], terms_dict["GO:1902047"], a=0.5)

    # anova_distribution_per_module(activation_data, "tumor_tissue_site", biologically_informed)
    anova_distribution(model, go_layers, dataset, "cancer_type")#"tumor_tissue_site")

    # # Deprecated
    # setup_figure(plot_data, "cancer_type", n_nan_cols, terms_dict)
    # plt.show()
