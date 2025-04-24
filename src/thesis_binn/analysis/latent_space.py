import matplotlib.pyplot as plt
import pandas as pd
import torch
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as pio

from thesis_binn.model.Autoencoder import Autoencoder
from thesis_binn.model.build_model import build_model


def convert_labels(labels):
    label_set = labels.unique()
    label_to_int = {label: idx for idx, label in enumerate(label_set)}
    int_labels = labels.map(label_to_int)
    return int_labels, label_to_int


def size_by_label(label_name):
    fig_size = (8, 6)
    # Hardcode figure size
    if label_name == "tumor_tissue_site": fig_size = (8, 6)
    if label_name == "cancer_type": fig_size = (9, 8)
    if label_name == "stage_pathologic_stage": fig_size = (8, 6)
    return fig_size


def setup_figure(embedding, labels, fig_size=(8, 6), cmap="tab20", s=5, colored=True):
    # Convert labels to integers for plotting
    int_labels, label_map = convert_labels(labels)
    # Custom legend
    handles = []
    for label_str, label_int in label_map.items():
        handles.append(
            plt.Line2D([], [], marker='o', linestyle='', color=plt.colormaps["tab20"](label_int % 20), label=label_str))
    # Remove sample colors
    if not colored: int_labels = [0 for _ in range(len(int_labels))]
    # Create figure
    plt.figure(figsize=fig_size)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=int_labels, cmap=cmap, s=s)
    plt.legend(handles=handles, title=labels.name, bbox_to_anchor=(1.0, .5), loc='center left')
    return


def plot_umap(model: Autoencoder, data: pd.DataFrame, labels, seed=42, colored=True):
    """Calculate and plot the UMAP of the given model on the given data. Samples are colored by the given labels."""
    # Prepare model for evaluation
    model.eval()
    with torch.no_grad():
        # Compute latent representation of data
        x = torch.tensor(data.values)
        latent_x = model.encoder(x)
    # Calculate UMAP representation of latent space
    reducer = UMAP(random_state=seed)
    embedding = reducer.fit_transform(latent_x)
    # Plot UMAP coordinates, colored by label
    setup_figure(embedding, labels, fig_size=size_by_label(labels.name), colored=colored)
    plt.title(f'UMAP of latent space colored by {labels.name}, model: {model.name}')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.tight_layout()
    plt.show()


def plot_tsne(model: Autoencoder, data: pd.DataFrame, labels, seed=42, colored=True):
    # Prepare model for evaluation
    model.eval()
    with torch.no_grad():
        # Compute latent representation of data
        x = torch.tensor(data.values)
        latent_x = model.encoder(x)
    # Calculate t-SNE representation of latent space
    tsne = TSNE(n_components=2, random_state=seed)
    embedding = tsne.fit_transform(latent_x)
    # Plot t-SNE coordinates, colored by label
    setup_figure(embedding, labels, fig_size=size_by_label(labels.name), colored=colored)
    plt.title(f't-SNE of latent space colored by {labels.name}, model: {model.name}')
    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')
    plt.tight_layout()
    plt.show()

    # # Interactive plot in browser
    # pio.renderers.default = 'browser'
    # fig = px.scatter(x=embedding[:, 0], y=embedding[:, 1], color=labels)
    # fig.update_layout(legend_title_text=labels.name)
    # fig.show()


def plot_pca(model: Autoencoder, data: pd.DataFrame, labels, seed=42, colored=True):
    # Prepare model for evaluation
    model.eval()
    with torch.no_grad():
        # Compute latent representation of data
        x = torch.tensor(data.values)
        latent_x = model.encoder(x)
    # Calculate first two PCs of latent space
    pca = PCA(n_components=2, random_state=seed)
    embedding = pca.fit_transform(latent_x)
    # Plot PCA coordinates, colored by label
    setup_figure(embedding, labels, fig_size=size_by_label(labels.name), colored=colored)
    plt.title(f'PCA of latent space colored by {labels.name}, model: {model.name}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Best practice would be to make a separate script for the actual processing, but I didn't...
    project_folder = "../../.."
    dataset_name = "TCGA_complete_bp_top1k"
    experiment_name = "AE_1.0"
    experiment_version = ".1"
    model_name = "none"
    label = "cancer_type"  # nan_cols: patient_id, sample_type, cancer_type, tumor_tissue_site, stage_pathologic_stage
    seed = 42
    n_nan_cols = 5
    colored = True

    # Model construction
    model_type = "dense"
    biologically_informed = model_name
    soft_links = False
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
                        n_go_layers_used, activation_fn, dtype, genes, package_call=True)
    model.load_state_dict(
        torch.load(
            f"{project_folder}/out/trained_models/{experiment_name}/{experiment_name + experiment_version}_{model_name}_model.pt",
            weights_only=True))
    print("----- COMPLETED: Building model -----")

    print("----- START: Transforming latent space -----")
    plot_pca(model, data=dataset[dataset.columns[n_nan_cols:]], labels=dataset[label], seed=seed, colored=colored)
    plot_tsne(model, data=dataset[dataset.columns[n_nan_cols:]], labels=dataset[label], seed=seed, colored=colored)
    plot_umap(model, data=dataset[dataset.columns[n_nan_cols:]], labels=dataset[label], seed=seed, colored=colored)
