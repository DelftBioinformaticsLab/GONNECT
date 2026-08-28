import pandas as pd
import torch
import os

from goatools.obo_parser import GOTerm
from gonnect.data_processing.go_preprocessing import construct_go_bp_layers
from gonnect.model.Decoder import DenseBIDecoder, SparseBIDecoder
from gonnect.model.Encoder import DenseBIEncoder, SparseBIEncoder
from gonnect.paths import resolve_masks_dir


def make_layers(merge_conditions, dataset_name, n_nan_cols, print_go=True, n_go_layers_used=5):
    """With the genes found in the provided dataset and the given merge conditions, apply all GO preprocessing steps to obtain the layerized GO graph."""
    print("\n----- START: GO preprocessing -----")
    data = pd.read_csv(f"../../../data/{dataset_name}.csv.gz", compression="gzip")
    genes = list(data.columns[n_nan_cols:])
    layers = construct_go_bp_layers(genes, merge_conditions, print_go=print_go, package_call=True,
                                    n_go_layers_used=n_go_layers_used)
    print("----- COMPLETED: GO preprocessing -----")
    return layers


def save_layers(layers, merge_conditions, dataset_name):
    """Save a Tensor with the same dimensions as the provided GO layers, organized by dataset and merge conditions."""
    layer_copy = [torch.zeros(len(layer)) for layer in layers]
    os.makedirs(f"../../../out/masks/layers/{str(merge_conditions)}", exist_ok=True)
    torch.save(layer_copy, f"../../../out/masks/layers/{str(merge_conditions)}/{dataset_name}_layers.pt")
    print("----- Saved layers to file -----")


def save_masks(layers: [GOTerm], merge_conditions, dataset_name, dtype, model_type="dense"):
    """Generate biologically-informed edge and proxy masks by initializing BICoder objects with layers containing GOTerms. These masks are saved as Tensors and organized by merge conditions, dataset and model type (sparse/dense)."""
    print(f"\n----- START: Generate {model_type} masks -----")
    if model_type == "sparse":
        encoder = SparseBIEncoder(layers, torch.nn.ReLU, dtype)
        decoder = SparseBIDecoder(layers, torch.nn.ReLU, dtype)
    else:
        encoder = DenseBIEncoder(layers, torch.nn.ReLU, dtype)
        decoder = DenseBIDecoder(layers, torch.nn.ReLU, dtype)
    print(f"----- COMPLETED: Generate {model_type} masks -----")
    os.makedirs(f"../../../out/masks/encoder/{merge_conditions}", exist_ok=True)
    torch.save(encoder.edge_masks,
               f"../../../out/masks/encoder/{merge_conditions}/{dataset_name}_{model_type}_edge_masks.pt")
    torch.save(encoder.proxy_masks,
               f"../../../out/masks/encoder/{merge_conditions}/{dataset_name}_{model_type}_proxy_masks.pt")
    os.makedirs(f"../../../out/masks/decoder/{merge_conditions}", exist_ok=True)
    torch.save(decoder.edge_masks,
               f"../../../out/masks/decoder/{merge_conditions}/{dataset_name}_{model_type}_edge_masks.pt")
    torch.save(decoder.proxy_masks,
               f"../../../out/masks/decoder/{merge_conditions}/{dataset_name}_{model_type}_proxy_masks.pt")
    os.makedirs(f"../../../out/masks/genes/{merge_conditions}", exist_ok=True)
    orphan_gene_mask = []
    print("----- Genes without terms: -----")
    for gene in layers[-1]:
        if len(gene.parents) == 0:
            print(gene.item_id)
            orphan_gene_mask.append(True)
        else:
            orphan_gene_mask.append(False)
    torch.save(torch.tensor(orphan_gene_mask),
               f"../../../out/masks/genes/{merge_conditions}/{dataset_name}_gene_mask.pt")
    print(f"----- Saved {model_type} masks to file -----")


def load_masks(module, merge_conditions, dataset_name, model_type, random_version=None, root_dir=None, masks_dir=None):
    """Load edge and proxy masks from file. Arguments are used to find the correct file path for the AE module. Masks are returned as a list of Tensors. The 'random_version' argument should be the integer of the randomized edge mask you want to use.

    Pass 'masks_dir' to point straight at a directory laid out like out/masks/.
    'root_dir' is the pre-1.0 spelling (a repository root, whose out/masks is
    used) and is still honoured; when neither is given, gonnect.paths falls back
    to $GONNECT_MASKS_DIR and then ./out/masks.
    """
    masks_dir = resolve_masks_dir(masks_dir, root_dir=root_dir)
    suffix = ""
    if random_version is not None:
        suffix = f"_random{str(random_version)}"

    masks = []
    if (module == "encoder") or (module == "both"):
        masks.append(
            torch.load(
                f"{masks_dir}/encoder/{str(merge_conditions)}/{dataset_name}_{model_type}_edge_masks{suffix}.pt",
                weights_only=True))
        masks.append(
            torch.load(
                f"{masks_dir}/encoder/{str(merge_conditions)}/{dataset_name}_{model_type}_proxy_masks.pt",
                weights_only=True))
    if (module == "decoder") or (module == "both"):
        masks.append(
            torch.load(
                f"{masks_dir}/decoder/{str(merge_conditions)}/{dataset_name}_{model_type}_edge_masks{suffix}.pt",
                weights_only=True))
        masks.append(
            torch.load(
                f"{masks_dir}/decoder/{str(merge_conditions)}/{dataset_name}_{model_type}_proxy_masks.pt",
                weights_only=True))
    if len(masks) == 0:
        return None
    return masks


def list_links_per_layer(masks):
    """Returns a list of the total amount of edges between consecutive layers. Used for edge randomization."""
    link_counts = []
    for mask in masks:
        link_counts.append(int(mask.sum()))
    return link_counts


def save_random_masks(module, merge_conditions, dataset_name, model_type, version):
    """Take an existing edge mask and shuffle the edges in a way that preserves the in- and out-degree of each node and save the new random mask."""
    original_masks = load_masks(module, merge_conditions, dataset_name, model_type, root_dir="../../..")
    randomized_masks = []
    # Skip performing swaps for the final mask, since this mask is for the GO root, which is fully connected, so there are no valid swaps possible
    if module == "encoder":
        masks = original_masks[0][:-1]
    else:
        masks = original_masks[0][1:]
    links_per_layer = list_links_per_layer(masks)

    print("----- Start shuffling... -----")
    for i, edge_mask in enumerate(masks):
        # Copy the original edge mask, and make sparse masks temporarily dense during swapping phase
        randomized_edge_mask = edge_mask.clone()
        if model_type == "sparse":
            randomized_edge_mask = randomized_edge_mask.to_dense()

        swaps = 0
        while swaps < 100 * links_per_layer[i]:
            # Randomly pick two edges
            edge_indices = torch.nonzero(randomized_edge_mask)
            edge_a = edge_indices[int(torch.rand(1).item() * len(edge_indices) - 1)]
            edge_b = edge_indices[int(torch.rand(1).item() * len(edge_indices) - 1)]
            # Check if the swapped edges already exist
            if randomized_edge_mask[edge_b[0]][edge_a[1]] or randomized_edge_mask[edge_a[0]][edge_b[1]]:
                continue
            # If not, swap the rows of the two selected edges to change the mask yet preserve node connectivity
            randomized_edge_mask[edge_a[0]][edge_a[1]] = False
            randomized_edge_mask[edge_b[0]][edge_a[1]] = True
            randomized_edge_mask[edge_b[0]][edge_b[1]] = False
            randomized_edge_mask[edge_a[0]][edge_b[1]] = True
            swaps += 1

        print(f"----- Layer shuffle {i} complete -----")
        if model_type == "sparse":
            randomized_masks.append(randomized_edge_mask.to_sparse())
        else:
            randomized_masks.append(randomized_edge_mask)

    torch.save(randomized_masks,
               f"../../../out/masks/{module}/{merge_conditions}/{dataset_name}_{model_type}_edge_masks_random{version}.pt")
    print(f"----- Saved {model_type} {module} random{version} masks to file -----")


def save_ablation_masks(module, merge_conditions, dataset_name, model_type, percentage):
    """Take an existing edge mask and remove X% of the edges for ablation study. Save the new masks."""
    # GO dictionary needed for matching index to GO term
    from gonnect.data_processing.go_preprocessing import construct_go_bp, construct_go_bp_layers
    genes = pd.read_csv(f"../../../data/{dataset_name}.csv.gz", compression="gzip", nrows=1)
    genes = genes.columns[5:]
    go_layers = construct_go_bp_layers(genes, merge_conditions, print_go=False, package_call=True)
    # Start loading masks, determining number of edge removals and executing removals
    original_masks = load_masks(module, merge_conditions, dataset_name, model_type, root_dir="../../..")
    ablated_masks = []
    # Skip removing edges for the final mask, since this mask is for the GO root, which will be removed during model construction
    if module == "encoder":
        masks = original_masks[0][:-1]
        layer_index = 0
        go_layers = list(reversed(go_layers))
    else:
        masks = original_masks[0][1:]
        layer_index = 5
        go_layers = go_layers[1:]
    links_per_layer = list_links_per_layer(masks)
    removals_per_layer = [int(n * percentage / 100) for n in links_per_layer]

    print("----- Start removing... -----")
    ablated_links = pd.DataFrame()
    layer_indices = []
    source_indices = []
    source_ids = []
    sink_indices = []
    sink_ids = []
    for i, edge_mask in enumerate(masks):
        # Copy the original edge mask, and make sparse masks temporarily dense during swapping phase
        ablated_edge_mask = edge_mask.clone()
        if model_type == "sparse":
            ablated_edge_mask = ablated_edge_mask.to_dense()

        for j in range(removals_per_layer[i]):
            # Randomly pick a GO link
            edge_indices = torch.nonzero(ablated_edge_mask)
            go_link = edge_indices[int(torch.rand(1).item() * len(edge_indices) - 1)]
            source_index = go_link[1]
            sink_index = go_link[0]

            # remove the randomly chosen GO link
            ablated_edge_mask[go_link[0]][go_link[1]] = False  # rows are sink, cols are source
            layer_indices.append(layer_index + i)
            source_indices.append(source_index.item())
            source_ids.append(go_layers[i][source_index].item_id)
            sink_indices.append(sink_index.item())
            sink_ids.append(go_layers[i + 1][sink_index].item_id)

        print(f"----- Layer ablation {i} complete -----")
        if model_type == "sparse":
            ablated_masks.append(ablated_edge_mask.to_sparse())
        else:
            ablated_masks.append(ablated_edge_mask)

    # Save data on removed GO links
    ablated_links["layer_index"] = layer_indices
    ablated_links["source_index"] = source_indices
    ablated_links["source_id"] = source_ids
    ablated_links["sink_index"] = sink_indices
    ablated_links["sink_id"] = sink_ids
    ablated_links.to_csv(
        f"../../../out/masks/{module}/{merge_conditions}/{dataset_name}_{model_type}_edge_masks_random{percentage}%_removed.csv.gz")
    print("Removed GO links saved successfully to file.")

    torch.save(ablated_masks,
               f"../../../out/masks/{module}/{merge_conditions}/{dataset_name}_{model_type}_edge_masks_random{percentage}%.pt")
    print(f"----- Saved {model_type} {module} random{percentage} masks to file -----")


def save_random_masks_non_degree_preserving(module, merge_conditions, dataset_name, model_type, version):
    """Take an existing edge mask and shuffle the edges in a non-degree preserving way and save the new random mask."""
    original_masks = load_masks(module, merge_conditions, dataset_name, model_type, root_dir="../../..")
    randomized_masks = []
    # Skip performing swaps for the final mask, since this mask is for the GO root, which is fully connected, so there are no valid swaps possible
    if module == "encoder":
        masks = original_masks[0][:-1]
    else:
        masks = original_masks[0][1:]
    links_per_layer = list_links_per_layer(masks)

    print("----- Start shuffling... -----")
    for i, edge_mask in enumerate(masks):
        # Copy the original edge mask, and make sparse masks temporarily dense during swapping phase
        randomized_edge_mask = edge_mask.clone()
        if model_type == "sparse":
            randomized_edge_mask = randomized_edge_mask.to_dense()

        flat = randomized_edge_mask.flatten()
        shuffled = flat[torch.randperm(flat.numel())]
        randomized_edge_mask = shuffled.view_as(randomized_edge_mask)

        print(f"----- Layer shuffle {i} complete -----")
        if model_type == "sparse":
            randomized_masks.append(randomized_edge_mask.to_sparse())
        else:
            randomized_masks.append(randomized_edge_mask)

    torch.save(randomized_masks,
               f"../../../out/masks/{module}/{merge_conditions}/{dataset_name}_{model_type}_edge_masks_random{version}.pt")
    print(f"----- Saved {model_type} {module} random{version} masks to file -----")


if __name__ == "__main__":
    merge_conditions = (1, 30, 50)
    dataset_name = "SEAAD_A9_RNAseq_final-nuclei.2024-02-13_bp_top1k"  # "TCGA_complete_bp_top1k"
    n_go_layers_used = 5
    n_nan_cols = 133  # 5
    dtype = torch.float64

    layers = make_layers(merge_conditions, dataset_name, n_nan_cols, n_go_layers_used=n_go_layers_used)
    save_layers(layers, merge_conditions, dataset_name)
    save_masks(layers, merge_conditions, dataset_name, dtype, model_type="sparse")
    save_masks(layers, merge_conditions, dataset_name, dtype, model_type="dense")
    save_random_masks("encoder", merge_conditions, dataset_name, "dense", version=2)
    save_random_masks("decoder", merge_conditions, dataset_name, "dense", version=2)
    save_random_masks("encoder", merge_conditions, dataset_name, "dense", version=3)
    save_random_masks("decoder", merge_conditions, dataset_name, "dense", version=3)
    save_random_masks("encoder", merge_conditions, dataset_name, "dense", version=4)
    save_random_masks("decoder", merge_conditions, dataset_name, "dense", version=4)
    save_random_masks("encoder", merge_conditions, dataset_name, "dense", version=5)
    save_random_masks("decoder", merge_conditions, dataset_name, "dense", version=5)
    save_random_masks("encoder", merge_conditions, dataset_name, "dense", version=6)
    save_random_masks("decoder", merge_conditions, dataset_name, "dense", version=6)
    save_ablation_masks("encoder", merge_conditions, dataset_name, model_type="dense", percentage=10)
    save_ablation_masks("decoder", merge_conditions, dataset_name, model_type="dense", percentage=10)

    save_random_masks_non_degree_preserving("encoder", merge_conditions, dataset_name, "dense", version=22)
    save_random_masks_non_degree_preserving("decoder", merge_conditions, dataset_name, "dense", version=22)
    save_random_masks_non_degree_preserving("encoder", merge_conditions, dataset_name, "dense", version=23)
    save_random_masks_non_degree_preserving("decoder", merge_conditions, dataset_name, "dense", version=23)
    save_random_masks_non_degree_preserving("encoder", merge_conditions, dataset_name, "dense", version=24)
    save_random_masks_non_degree_preserving("decoder", merge_conditions, dataset_name, "dense", version=24)
    save_random_masks_non_degree_preserving("encoder", merge_conditions, dataset_name, "dense", version=25)
    save_random_masks_non_degree_preserving("decoder", merge_conditions, dataset_name, "dense", version=25)
    save_random_masks_non_degree_preserving("encoder", merge_conditions, dataset_name, "dense", version=26)
    save_random_masks_non_degree_preserving("decoder", merge_conditions, dataset_name, "dense", version=26)
