import pandas as pd
import torch
import os

from goatools.obo_parser import GOTerm
from thesis_binn.data_processing.go_preprocessing import construct_go_bp_layers
from thesis_binn.model.Decoder import DenseBIDecoder, SparseBIDecoder
from thesis_binn.model.Encoder import DenseBIEncoder, SparseBIEncoder


def make_layers(merge_conditions, dataset_name, n_nan_cols, print_go=True):
    """With the genes found in the provided dataset and the given merge conditions, apply all GO preprocessing steps to obtain the layerized GO graph."""
    print("\n----- START: GO preprocessing -----")
    # data = pd.read_csv(f"../../data/{dataset_name}.csv.gz", usecols=["gene id"],
    #                    compression="gzip").sort_values("gene id")
    # genes = list(data["gene id"])
    data = pd.read_csv(f"../../../data/{dataset_name}.csv.gz", compression="gzip")
    genes = list(data.columns[n_nan_cols:])
    layers = construct_go_bp_layers(genes, merge_conditions, print_go=print_go, package_call=True)
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
    for gene in layers[-1]:
        if len(gene.parents) == 0:
            orphan_gene_mask.append(True)
        else:
            orphan_gene_mask.append(False)
    torch.save(torch.tensor(orphan_gene_mask),
               f"../../../out/masks/genes/{merge_conditions}/{dataset_name}_gene_mask.pt")
    print(f"----- Saved {model_type} masks to file -----")


def load_masks(module, merge_conditions, dataset_name, model_type, cluster=False):
    """Load edge and proxy masks from file. Arguments are used to find the correct file path for the AE module. Masks are returned as a list of Tensors."""
    root_dir = ".."
    if cluster:
        root_dir = "/opt/app"
    masks = []
    if (module == "encoder") or (module == "both"):
        masks.append(
            torch.load(f"{root_dir}/out/masks/encoder/{str(merge_conditions)}/{dataset_name}_{model_type}_edge_masks.pt",
                       weights_only=True))
        masks.append(
            torch.load(f"{root_dir}/out/masks/encoder/{str(merge_conditions)}/{dataset_name}_{model_type}_proxy_masks.pt",
                       weights_only=True))
    if (module == "decoder") or (module == "both"):
        masks.append(
            torch.load(f"{root_dir}/out/masks/decoder/{str(merge_conditions)}/{dataset_name}_{model_type}_edge_masks.pt",
                       weights_only=True))
        masks.append(
            torch.load(f"{root_dir}/out/masks/decoder/{str(merge_conditions)}/{dataset_name}_{model_type}_proxy_masks.pt",
                       weights_only=True))
    if len(masks) == 0:
        return None
    return masks


if __name__ == "__main__":
    merge_conditions = (1, 30, 50)
    dataset_name = "TCGA_complete_bp_top1k"
    n_nan_cols = 5
    dtype = torch.float64

    layers = make_layers(merge_conditions, dataset_name, n_nan_cols)
    save_layers(layers, merge_conditions, dataset_name)
    save_masks(layers, merge_conditions, dataset_name, dtype, model_type="sparse")
    save_masks(layers, merge_conditions, dataset_name, dtype, model_type="dense")
