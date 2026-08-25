import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
from goatools.obo_parser import GOTerm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from thesis_binn.data_processing.GeneTerm import GeneTerm
from thesis_binn.data_processing.ProxyTerm import ProxyTerm
from thesis_binn.data_processing.generate_masks import make_layers
from thesis_binn.model.Autoencoder import Autoencoder
from thesis_binn.model.build_model import build_model
from thesis_binn.train.train import split_data


def activations_per_term(model: Autoencoder, go_layers: [GOTerm], data: pd.DataFrame, bi_module: str):
    """Returns a DataFrame with GO-terms as columns and raw sample activations as rows. GO layers are automatically flipped if needed."""
    # Generate activations by performing a forward pass over the provided data
    model.eval()
    model.set_store_activations(True)
    with torch.no_grad():
        x = torch.tensor(data.values)
        y = model(x)
    model.set_store_activations(False)

    # Select which module we want to retrieve activations from and strip off gene layer
    if bi_module == "encoder":
        module = model.encoder
        go_layers = list(reversed(go_layers))[1:]
    else:
        module = model.decoder
        go_layers = go_layers[:-1]

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

    # Normalize columns
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


def activation_heatmap(activations: pd.DataFrame, label: str, n_nan_cols: int, go: dict[str, GOTerm], k=20,
                       term_selection=None):
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

    if term_selection:
        top_k_terms = term_selection
    else:
        # Select k terms with best class separation
        f_values, p_values = f_classif(activation_values, labels)
        f_scores = pd.Series(f_values, index=activation_values.columns, name="f_values")
        top_k_terms = f_scores.nlargest(k).index  # Default k=20

    activation_values = activation_values[top_k_terms]

    # Cluster columns using clustermap
    print("\n----- START: Clustering columns -----")
    clustermap = sns.clustermap(activation_values,
                                metric='correlation',
                                method='average',
                                col_cluster=True,
                                row_cluster=False,
                                cmap='viridis',
                                cbar_pos=None,
                                xticklabels=True,
                                yticklabels=True)
    print("----- COMPLETED: Clustering columns -----")
    # Extract ordered column indices
    ordered_col_indices = clustermap.dendrogram_col.reordered_ind
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
    plt.show()


def anova_distribution(model: Autoencoder, go_layers: [GOTerm], dataset: pd.DataFrame, label: str, n_nan_cols=5):
    data_values = dataset[dataset.columns[n_nan_cols:]]
    activation_values_enc = activations_per_term(model, go_layers, data_values, "encoder")
    activation_values_dec = activations_per_term(model, go_layers, data_values, "decoder")
    activations_enc = pd.concat(
        [dataset[dataset.columns[:n_nan_cols]].reset_index(drop=True), activation_values_enc.reset_index(drop=True)],
        axis=1)
    activations_dec = pd.concat(
        [dataset[dataset.columns[:n_nan_cols]].reset_index(drop=True), activation_values_dec.reset_index(drop=True)],
        axis=1)
    anova_distribution_per_module(activations_enc, label, "encoder")
    anova_distribution_per_module(activations_dec, label, "decoder")
    plt.title(f"Distribution of ANOVA F-values for {model.name}")
    plt.legend([model.encoder.name, model.decoder.name])
    plt.xlim((-1000, 22000))
    plt.show()


def node_all_class_auc(activation_data, label, scoring='roc_auc', n_nan_cols=5):
    activation_data = activation_data.dropna(subset=[label])
    labels = activation_data[label]
    activation_values = activation_data[activation_data.columns[n_nan_cols:]]
    lb = LabelBinarizer()
    Y_bin = lb.fit_transform(labels)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []

    for node in activation_values.columns:
        X_node = activation_values[[node]].values  # (n_samples, 1)

        # Compute macro average ROC-AUC manually
        aucs = []
        for i in range(Y_bin.shape[1]):
            y_bin_class = Y_bin[:, i]
            model = LogisticRegression(solver='liblinear')
            scores = cross_val_score(model, X_node, y_bin_class, cv=cv, scoring=scoring)
            aucs.append(scores.mean())
        avg_auc = np.mean(aucs)
        results.append((node, avg_auc))

    result_df = pd.DataFrame(results, columns=['node', 'roc_auc'])
    return result_df.sort_values(by='roc_auc', ascending=False).reset_index(drop=True)


def node_per_class_auc(activation_data, label, terms_dict, scoring='roc_auc', cv=5, n_nan_cols=5, term_selection=None):
    activation_data = activation_data.dropna(subset=[label])
    labels = activation_data[label]
    activation_values = activation_data[activation_data.columns[n_nan_cols:]]
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(labels)
    class_labels = lb.classes_
    x_train, x_test = split_data(activation_data, n_nan_cols=5, seed=6)
    y_train, y_test = split_data(pd.DataFrame(y_bin), n_nan_cols=0, seed=6)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    results = []
    if term_selection is None:
        term_selection = activation_values.columns
    for node in term_selection:
        # x_node = activation_values[[node]].values  # (n_samples, 1)
        x_train_class = x_train[[node]].values
        x_test_class = x_test[[node]].values
        row = {"term": node, "name": terms_dict[node].name}

        for i, class_label in enumerate(class_labels):
            # y_bin_class = y_bin[:, i]
            # model = LogisticRegression(solver='liblinear')
            # scores = cross_val_score(model, x_node, y_bin_class, cv=cv, scoring=scoring)
            y_train_class = y_train[:, i]
            y_test_class = y_test[:, i]
            model = LogisticRegression(solver='liblinear')
            model.fit(x_train_class, y_train_class)
            y_prob = model.predict_proba(x_test_class)[:, 1]
            scores = roc_auc_score(y_test_class, y_prob)
            # row[f'roc_auc_{class_label}'] = np.mean(scores)
            row[f'roc_auc_{class_label}'] = scores
        results.append(row)

    results_data = pd.DataFrame(results)
    return results_data


def auc_heatmap(auc_data, terms_list):
    # Select and order terms
    auc_data = auc_data[auc_data["term"].isin(terms_list)]
    auc_data = auc_data.sort_values(by="term", key=lambda x: x.map(terms_list.index))

    # Extract only the AUC columns
    auc_cols = list(auc_data.columns[2:])

    # Prepare heatmap data: node_name as index, classes as columns
    heatmap_data = auc_data.set_index('name')[auc_cols]

    # Rename columns to just class names (remove 'roc_auc_' prefix)
    heatmap_data.columns = [col.replace("roc_auc_", "") for col in heatmap_data.columns]

    perform_clustering = False
    if perform_clustering:
        # Cluster columns using clustermap
        print("\n----- START: Clustering columns -----")
        clustermap = sns.clustermap(heatmap_data,
                                    metric='correlation',
                                    method='average',
                                    col_cluster=True,
                                    row_cluster=False,
                                    cbar_pos=None,
                                    xticklabels=True,
                                    yticklabels=False)
        print("----- COMPLETED: Clustering columns -----")
        # Extract ordered column indices
        ordered_col_indices = clustermap.dendrogram_col.reordered_ind
        ordered_columns = heatmap_data.columns[ordered_col_indices]
    else:
        ordered_columns = ["BRCA", "LUAD", "LUSC", "KIRC", "KIRP", "KICH", "UCEC", "LGG", "HNSC", "THCA", "PRAD",
                           "SKCM",
                           "COAD", "OV", "STAD", "BLCA", "LIHC", "CESC", "PCPG", "ACC", "SARC", "ESCA", "PAAD", "READ",
                           "TGCT",
                           "LAML", "THYM", "MESO", "UVM", "UCS", "DLBC", "CHOL"]
    heatmap_data = heatmap_data[ordered_columns]

    # Plot heatmap "selected terms, thesis style"
    plt.figure(figsize=(7, 15))
    sns.heatmap(heatmap_data.transpose(), xticklabels=False, cbar_kws={'label': 'ROC-AUC'}, vmin=0.5, vmax=1.0,
                cmap="viridis")
    plt.xticks(np.arange(heatmap_data.shape[0]) + 0.5, labels=heatmap_data.index, rotation=270)

    plt.title("Per-Class ROC-AUC of GO-terms in GONNECT Encoder")
    plt.ylabel("GO-Term")
    plt.xlabel("Cancer Type")
    plt.tight_layout()
    plt.savefig("../../../../publication/auc_encoder.pdf", format="pdf")
    plt.show()


def activation_heatmap_average_per_label(activations: pd.DataFrame, label: str, n_nan_cols: int, go: dict[str, GOTerm],
                                         k=20, term_selection=None):
    # Remove any sample with NaN as label
    activations = activations.dropna(subset=[label])

    # Order and group samples by label
    activations = activations.sort_values(label)
    labels = activations[label].values

    group_values = [label] + list(activations.columns[n_nan_cols:])
    grouped_activations = activations[group_values].groupby(label).mean().reset_index()
    activation_values = grouped_activations.set_index(label)[grouped_activations.columns[1:]]

    # Normalize before performing ANOVA (skip terms without variance)
    var_terms = [col for col in activation_values if activation_values[col].std() != 0]
    activation_values[var_terms] = (activation_values[var_terms] - activation_values[var_terms].mean()) / (
        activation_values[var_terms].std())

    if term_selection:
        top_k_terms = [term for term in term_selection if term in activation_values.columns]
    else:
        # Select k terms with best class separation
        f_values, p_values = f_classif(activation_values, labels)
        f_scores = pd.Series(f_values, index=activation_values.columns, name="f_values")
        top_k_terms = f_scores.nlargest(k).index  # Default k=20

    activation_values = activation_values[top_k_terms]

    perform_clustering = False
    if perform_clustering:
        # Cluster columns using clustermap
        print("\n----- START: Clustering columns -----")
        clustermap = sns.clustermap(activation_values,
                                    metric='correlation',
                                    method='average',
                                    col_cluster=True,
                                    row_cluster=False,
                                    cmap='viridis',
                                    cbar_pos=None,
                                    xticklabels=True,
                                    yticklabels=True)
        print("----- COMPLETED: Clustering columns -----")
        # Extract ordered column indices
        ordered_col_indices = clustermap.dendrogram_col.reordered_ind
        ordered_columns = activation_values.columns[ordered_col_indices]
    else:
        ordered_columns = activation_values.columns
    ordered_values = activation_values[ordered_columns]

    # Ensure symmetric colorbar
    colorbar_extreme_value = min(abs(ordered_values.min().min()), abs(ordered_values.max().max()))
    print(f"colorbar_extreme_values = {ordered_values.min().min()}, {ordered_values.max().max()}")

    # Custom label ordering
    label_order = ["BRCA", "LUAD", "LUSC", "KIRC", "KIRP", "KICH", "UCEC", "LGG", "HNSC", "THCA", "PRAD", "SKCM",
                   "COAD", "OV", "STAD", "BLCA", "LIHC", "CESC", "PCPG", "ACC", "SARC", "ESCA", "PAAD", "READ", "TGCT",
                   "LAML", "THYM", "MESO", "UVM", "UCS", "DLBC", "CHOL"]
    ordered_values = ordered_values.sort_values(by=label, key=lambda x: x.map(label_order.index))

    plt.figure(figsize=(7, 10))
    sns.heatmap(np.abs(ordered_values), cmap="coolwarm", xticklabels=False, cbar_kws={'label': 'Activation'},
                vmin=-colorbar_extreme_value, vmax=colorbar_extreme_value)

    # Set column labels
    term_objects = [go[term_id] for term_id in ordered_columns]
    term_names = [term.name for term in term_objects]
    term_ids = [term.item_id for term in term_objects]
    plt.xticks(np.arange(ordered_values.shape[1]) + 0.5, labels=term_ids, rotation=270)

    # Set title and axis labels
    plt.xlabel('GO terms')
    plt.ylabel(f'Mean sample activation per cancer type')
    plt.title('GO Term Activation Heatmap')
    plt.tight_layout()
    plt.show()


def temporary_act_extraction(experiment_names):
    """Store activations of given experiments to csv."""
    # Experiment params
    experiment_versions = [2, 3, 4, 5, 6]
    model_names = ["encoder", "decoder"]
    # Model params
    model_type = "dense"
    random_version = None
    activation_fn = torch.nn.ReLU
    # GO params
    go_preprocessing = False
    merge_conditions = (1, 30, 50)  # min parents, min children, min terms per layer
    n_go_layers_used = 5
    # Data params
    dataset_name = "TCGA_complete_bp_top1k"
    n_nan_cols = 5

    # Dataset
    dataset = pd.read_csv(f"../../../data/{dataset_name}.csv.gz", compression="gzip")
    data_values = dataset[dataset.columns[n_nan_cols:]]

    # GO graph
    go_layers = make_layers(merge_conditions, dataset_name, n_nan_cols)[-n_go_layers_used:]

    for experiment_name in experiment_names:
        soft_links = (experiment_name[-1] == "1")

        for model_name in model_names:
            biologically_informed = model_name

            for experiment_version in experiment_versions:
                # Model
                model = build_model(model_type, biologically_informed, soft_links, dataset_name, go_preprocessing,
                                    merge_conditions, n_go_layers_used, activation_fn, dtype=torch.float64,
                                    package_call=True)
                model.load_state_dict(torch.load(
                    f"../../../out/trained_models/{experiment_name}/{experiment_name}.{experiment_version}_{model_name}_model.pt",
                    weights_only=True))

                # All activations
                activation_values = activations_per_term(model, go_layers, data_values, biologically_informed)
                activation_data = pd.concat([dataset[dataset.columns[:n_nan_cols]].reset_index(drop=True),
                                             activation_values.reset_index(drop=True)], axis=1)

                # Save activation data as csv
                activation_data.to_csv(
                    f"../../../out/activations/{experiment_name}.{experiment_version}_{model_name}_activations.csv.gz",
                    compression="gzip", index=False)
                print(f"Saved activations to file: {experiment_name}.{experiment_version}_{model_name}")

    return 0


if __name__ == "__main__":
    temporary_act_extraction(["AE_2.0", "AE_2.1"])
