import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_samples

from thesis_binn.model.Autoencoder import Autoencoder
from thesis_binn.model.build_model import build_model
from thesis_binn.analysis.training import calculate_mse

colors32 = [
    "#e6194b", "#3cb44b", "#4363d8", "#ffe119",
    "#911eb4", "#46f0f0", "#f58231", "#008080",
    "#f032e6", "#bcf60c", "#000075", "#fabebe",
    "#808000", "#e6beff", "#9a6324", "#aaffc3",
    "#c71585", "#ffd8b1", "#1e90ff", "#800000",
    "#00ced1", "#f5a9bc", "#b0de5c", "#808080",
    "#ff7f50", "#dda0dd", "#b22222", "#7fffd4",
    "#a9a9f5", "#ff1493", "#daa520", "#fffac8"
]
map_32 = {i: color for i, color in enumerate(colors32)}


def convert_labels(labels):
    label_set = ["BRCA", "LUAD", "LUSC", "KIRC", "KIRP", "KICH", "UCEC", "LGG", "HNSC", "THCA", "PRAD", "SKCM",
                 "COAD", "OV", "STAD", "BLCA", "LIHC", "CESC", "PCPG", "ACC", "SARC", "ESCA", "PAAD", "READ", "TGCT",
                 "LAML", "THYM", "MESO", "UVM", "UCS", "DLBC", "CHOL"]
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


def setup_figure(embedding, labels, fig_size=(16, 12), cmap="tab20", s=8, colored=True):
    # Convert labels to integers for plotting
    int_labels, label_map = convert_labels(labels)
    # Custom legend
    handles = []
    for label_str, label_int in label_map.items():
        handles.append(
            plt.Line2D([], [], marker='o', linestyle='', color=map_32[label_int], label=label_str)
        )
    # Remove sample colors
    if not colored: int_labels = [0 for _ in range(len(int_labels))]
    # Create figure
    plt.figure(figsize=fig_size)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=[map_32[i] for i in int_labels], s=15, alpha=.8, edgecolor='none')
    plt.legend(handles=handles, title="Cancer Type", bbox_to_anchor=(1.0, .5), loc='center left')
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
    # plt.savefig("../../../../publication/input_umap.pdf", format="pdf")
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
    # plt.savefig("../../../../publication/input_tsne.pdf", format="pdf")
    plt.show()


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
    # plt.savefig("../../../../publication/input_pca.pdf", format="pdf")
    plt.show()


def calculate_silhouette_score(model: Autoencoder, data: pd.DataFrame, labels):
    model.eval()
    with torch.no_grad():
        # Compute latent representation of data
        x = torch.tensor(data.values)
        latent_x = model.encoder(x)

    ss = silhouette_score(latent_x, labels)
    return ss


def calculate_ari(model, data: pd.DataFrame, labels, seed=42):
    model.eval()
    with torch.no_grad():
        # Compute latent representation of data
        x = torch.tensor(data.values)
        latent_x = model.encoder(x)

    # Make k-means clustering of latent representations
    n_clusters = labels.nunique()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_assignments = kmeans.fit_predict(latent_x)

    # Compute Adjusted Rand Index (ARI)
    ari_score = adjusted_rand_score(labels, cluster_assignments)
    return ari_score


def calculate_nmi(model: Autoencoder, data: pd.DataFrame, labels, seed=42):
    model.eval()
    with torch.no_grad():
        # Compute latent representation of data
        x = torch.tensor(data.values)
        latent_x = model.encoder(x)

    # Make k-means clustering of latent representations
    n_clusters = labels.nunique()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_assignments = kmeans.fit_predict(latent_x)

    # Compute Normalized Mutual Information (NMI)
    nmi_score = normalized_mutual_info_score(labels, cluster_assignments)
    return nmi_score


def print_average_metric_scores():
    experiment_name = "AE_8.1"
    model_name = "both"
    biologically_informed = model_name
    soft_links = (experiment_name[-1] == "1")
    label = "Subclass"  # "cancer_type"  # nan_cols: patient_id, sample_type, cancer_type, tumor_tissue_site, stage_pathologic_stage
    # Don't change these parameters
    project_folder = "../../.."
    dataset_name = "SEAAD_A9_RNAseq_final-nuclei.2024-02-13_bp_top1k"  # "TCGA_complete_bp_top1k"
    cluster_seed = 42
    n_nan_cols = 133  # 5
    model_type = "dense"
    go_preprocessing = False
    merge_conditions = (1, 30, 50)
    n_go_layers_used = 5
    activation_fn = torch.nn.ReLU
    dtype = torch.float64
    genes = None

    dataset = pd.read_csv(f"{project_folder}/data/{dataset_name}.csv.gz", compression="gzip")
    dataset = dataset.dropna(subset=[label])

    mse = []
    ss = []
    ari = []
    nmi = []
    versions = [8, 9, 10, 11, 12]
    seeds = [2, 3, 4, 5, 6]
    missed_seeds = []
    mse_per_version = []

    for seed, version in zip(seeds, versions):
        random_version = version if experiment_name[-1] == "2" else None
        if experiment_name == "AE_2.2":
            random_version = version - 14  # odd random mask numbering for this experiment (8 instead of 22)
        print(f"----- START: calculating average metrics for seed {seed} -----")
        model = build_model(model_type, biologically_informed, soft_links, dataset_name, go_preprocessing,
                            merge_conditions, n_go_layers_used, activation_fn, dtype, genes,
                            random_version=random_version, package_call=True)
        try:
            model.load_state_dict(torch.load(f"{project_folder}/out/trained_models/{experiment_name}/{experiment_name}."
                                             f"{str(seed)}_{model_name}_model.pt", weights_only=True))

            mse.append(calculate_mse(experiment_name, seed, model_name))
            ss.append(calculate_silhouette_score(model, dataset[dataset.columns[n_nan_cols:]], dataset[label]))
            ari.append(calculate_ari(model, dataset[dataset.columns[n_nan_cols:]], dataset[label], seed=cluster_seed))
            nmi.append(calculate_nmi(model, dataset[dataset.columns[n_nan_cols:]], dataset[label], seed=cluster_seed))
        except:
            print(f"----- Missing data for seed {seed} -----")
            missed_seeds.append(seed)

        # ----- MSE PER CANCER TYPE -----
        from thesis_binn.train.loss import MSE, MSE_Masked
        if ((experiment_name == "AE_2.0") or (experiment_name == "AE_2.2")) and (model_name != "none"):
            loss = MSE_Masked(torch.load(f"{project_folder}/out/masks/genes/{merge_conditions}/{dataset_name}_gene_mask.pt", weights_only=True))
        else:
            loss = MSE()
        mse_per_version.append(mse_per_cancer_type(dataset, label, seed, model, loss))
        mse_mean_per_label = np.mean(mse_per_version, axis=0)
        for label_mean in mse_mean_per_label:
            print(f"{label_mean:.3f}")
        # ----- END -----

    print(f"Model: {model.name}")
    print(f"Mean Squared Error (MSE): {np.mean(mse):.3f} pm {np.std(mse):.3f}")  # 1 = Good, 0 = Random, -1 = Bad
    for i in mse:
        print(i)
    print(f"Silhouette Score (SS): {np.mean(ss):.3f} pm {np.std(ss):.3f}")  # 1 = Good, 0 = Random, -1 = Bad
    for i in ss:
        print(i)
    print(f"Adjusted Rand Index (ARI): {np.mean(ari):.3f} pm {np.std(ari):.3f}")  # 1 = Good, 0 = Random
    for i in ari:
        print(i)
    print(f"Normalized Mutual Information (NMI): {np.mean(nmi):.3f} pm {np.std(nmi):.3f}")  # 1 = Good, 0 = Bad
    for i in nmi:
        print(i)
    print(f"Missed seeds: {missed_seeds}")


def mse_per_cancer_type(dataset, label, seed, model, loss):
    """Helper function for calculating MSE per cluster for a given model and seed (built for TCGA)."""
    from thesis_binn.train.train import split_data, test
    from torch.utils.data import TensorDataset, DataLoader
    labels = ["BRCA", "LUAD", "LUSC", "KIRC", "KIRP", "KICH", "UCEC", "LGG", "HNSC", "THCA", "PRAD", "SKCM",
              "COAD", "OV", "STAD", "BLCA", "LIHC", "CESC", "PCPG", "ACC", "SARC", "ESCA", "PAAD", "READ", "TGCT",
              "LAML", "THYM", "MESO", "UVM", "UCS", "DLBC", "CHOL"]
    trainset, valset, testset = split_data(dataset, 0, seed=seed)
    testset = testset.dropna(subset=[label])
    mse_per_label = []
    for l in labels:
        testset_label = testset[testset[label].isin([l])]
        test_torch = TensorDataset(torch.from_numpy(testset_label[testset_label.columns[5:]].to_numpy()))
        testloader = DataLoader(test_torch, batch_size=len(testset_label), shuffle=False)
        mse_per_label.append(test(testloader, model, loss))
    return mse_per_label


def mse_per_cell_type(dataset, label, seed, model, loss):
    """Helper function for calculating MSE per cluster for a given model and seed (built for SEA-AD)."""
    from thesis_binn.train.train import split_data, test
    from torch.utils.data import TensorDataset, DataLoader
    labels = ['Oligodendrocyte', 'L6b', 'Pvalb', 'Vip', 'OPC', 'Lamp5 Lhx6',
              'Sst', 'L6 CT', 'L6 IT', 'Astrocyte', 'L6 IT Car3', 'L5 IT',
              'L2/3 IT', 'Microglia-PVM', 'L4 IT', 'Lamp5', 'L5/6 NP', 'Pax6',
              'Sncg', 'Chandelier', 'L5 ET', 'Endothelial', 'Sst Chodl', 'VLMC']
    trainset, valset, testset = split_data(dataset, 0, seed=seed)
    testset = testset.dropna(subset=[label])
    mse_per_label = []
    for label in labels:
        testset_label = testset[testset[label].isin([label])]
        test_torch = TensorDataset(torch.from_numpy(testset_label[testset_label.columns[133:]].to_numpy()))
        testloader = DataLoader(test_torch, batch_size=len(testset_label), shuffle=False)
        mse_per_label.append(test(testloader, model, loss))
    return mse_per_label


def plot_mse_per_cancer_type():
    import seaborn as sns
    data = pd.read_excel("../../../../mse_per_cancer_type.xlsx", index_col=0)
    plt.figure(figsize=(12, 5))
    sns.heatmap(np.transpose(data), cbar_kws={'label': 'MSE'})
    plt.xticks(rotation=90)
    plt.title("MSE per Cancer Type")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()


def silhouette_score_per_cluster():
    experiment_name = "AE_2.2"
    dataset_name = "TCGA_complete_bp_top1k"
    model_names = ["none", "encoder", "decoder", "both"] if experiment_name == "AE_2.0" else ["encoder", "decoder",
                                                                                              "both"]
    seeds = [2, 3, 4, 5, 6]
    versions = [22, 23, 24, 25, 26] if experiment_name == "AE_2.2" else seeds
    soft_links = (experiment_name[-1] == "1")
    label = "cancer_type"  # "Subclass"
    labels = ["BRCA", "LUAD", "LUSC", "KIRC", "KIRP", "KICH", "UCEC", "LGG", "HNSC", "THCA", "PRAD", "SKCM",
              "COAD", "OV", "STAD", "BLCA", "LIHC", "CESC", "PCPG", "ACC", "SARC", "ESCA", "PAAD", "READ", "TGCT",
              "LAML", "THYM", "MESO", "UVM", "UCS", "DLBC", "CHOL"]

    data = pd.read_csv(f"../../../data/{dataset_name}.csv.gz", compression="gzip")
    results = pd.DataFrame(index=labels)

    for model_name in model_names:
        sum_cluster_scores = [0] * len(data[label])
        for seed, version in zip(seeds, versions):
            if experiment_name == "AE_2.2":
                random_version = version - 14
            else:
                random_version = None
            print("----- Start building model -----")
            model = build_model("dense", model_name, soft_links, dataset_name, False, (1, 30, 50), 5, torch.nn.ReLU,
                                torch.float64, None, random_version=random_version, package_call=True)
            model.load_state_dict(torch.load(
                f"../../../out/trained_models/{experiment_name}/{experiment_name}.{str(version)}_{model_name}_model.pt",
                weights_only=True))

            # Calculate SS per cluster for current model
            model.eval()
            with torch.no_grad():
                # Compute latent representation of data
                x = torch.tensor(data[data.columns[5:]].values)
                latent_x = model.encoder(x)

            print("----- Start calculation SS -----")
            ss_samples = silhouette_samples(latent_x, data[label])

            cluster_scores = []
            for cluster in labels:
                cluster_score = ss_samples[data[label] == cluster].mean()
                cluster_scores.append(cluster_score)

            sum_cluster_scores = [old + new for old, new in zip(sum_cluster_scores, cluster_scores)]

        results[f"{experiment_name}_{model_name}"] = [score / len(versions) for score in sum_cluster_scores]

    # Print results for easy access
    for x in results.columns:
        print(x)
        for y in results[x]:
            print(y)
        print("")
    pass


if __name__ == '__main__':
    print_average_metric_scores()
    silhouette_score_per_cluster()
    breakpoint = 1 / 0
    project_folder = "../../.."
    dataset_name = "SEAAD_A9_RNAseq_final-nuclei.2024-02-13_bp_top1k"  # "TCGA_complete_bp_top1k"
    experiment_name = "AE_8.0"
    experiment_version = ".2"
    model_name = "none"
    label = "cancer_type"
    seed = 42
    n_nan_cols = 133
    colored = True

    # Model construction
    model_type = "dense"
    biologically_informed = model_name  # change this for locally trained models
    soft_links = False
    random_version = None
    go_preprocessing = False
    merge_conditions = (1, 30, 50)
    n_go_layers_used = 5
    activation_fn = torch.nn.ReLU
    dtype = torch.float64

    print("----- START: Loading data -----")
    dataset = pd.read_csv(f"{project_folder}/data/{dataset_name}.csv.gz", nrows=10000)
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
    print("----- COMPLETED: Building model -----")

    print("----- START: Transforming latent space -----")
    full_label_set = dataset[label].unique()
    label_subset = full_label_set
    filtered_data = dataset[dataset[label].isin(label_subset)]

    # Visualization
    plot_pca(model, data=filtered_data[filtered_data.columns[n_nan_cols:]], labels=filtered_data[label], seed=seed,
             colored=colored)
    plot_tsne(model, data=filtered_data[filtered_data.columns[n_nan_cols:]], labels=filtered_data[label], seed=seed,
              colored=colored)
    plot_umap(model, data=filtered_data[filtered_data.columns[n_nan_cols:]], labels=filtered_data[label], seed=seed,
              colored=colored)

    # Quantification
    ss = calculate_silhouette_score(model, dataset[dataset.columns[n_nan_cols:]], dataset[label])
    ari = calculate_ari(model, dataset[dataset.columns[n_nan_cols:]], dataset[label], seed=seed)
    nmi = calculate_nmi(model, dataset[dataset.columns[n_nan_cols:]], dataset[label])

    print(f"Model: {model.name}")
    print(f"Silhouette Score (SS): {ss:.4f}")  # 1 = Good, 0 = Random, -1 = Bad
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")  # 1 = Good, 0 = Random
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")  # 1 = Good, 0 = Bad
