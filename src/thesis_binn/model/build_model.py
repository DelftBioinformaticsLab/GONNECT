import torch

from thesis_binn.data_processing.dag_analysis import print_layers
from thesis_binn.data_processing.generate_masks import load_masks
from thesis_binn.data_processing.go_preprocessing import construct_go_bp_layers
from thesis_binn.model.Autoencoder import Autoencoder
from thesis_binn.model.Decoder import SparseBIDecoder, DenseBIDecoder, Decoder
from thesis_binn.model.Encoder import SparseBIEncoder, DenseBIEncoder, Encoder


def build_model(model_type: str, biologically_informed: str, soft_links: bool, dataset_name: str,
                go_preprocessing: bool, merge_conditions, n_go_layers_used: int, activation_fn, dtype, genes=None,
                package_call=False, cluster=False, random_version=None):
    # GO processing
    if go_preprocessing:
        if not genes:
            raise Exception("Cannot perform GO preprocessing without gene list")

        print("\n----- START: GO preprocessing -----")
        go_layers = construct_go_bp_layers(genes, merge_conditions, print_go=True, package_call=package_call,
                                           cluster=cluster)
        masks = None
        print("----- COMPLETED: GO preprocessing -----")

    else:
        if cluster:
            root_dir = "/opt/app"
        else:
            root_dir = f"{package_call * "../../"}.."
        go_layers = torch.load(f"{root_dir}/out/masks/layers/{str(merge_conditions)}/{dataset_name}_layers.pt",
                               weights_only=True)
        masks = load_masks(biologically_informed, merge_conditions, dataset_name, model_type,
                           random_version=random_version, root_dir=root_dir)
        print("\n----- COMPLETED: Loading GO from file -----")

    # Model construction
    used_go_layers = go_layers[-min(n_go_layers_used, len(go_layers)):]
    print("Layers used in model:")
    print_layers(used_go_layers)
    if (biologically_informed == "encoder") or (biologically_informed == "both"):
        if masks:
            # Discard masks of unused GO layers
            module_masks = (masks.pop(0)[:min(n_go_layers_used - 1, len(go_layers))],
                            masks.pop(0)[:min(n_go_layers_used - 1, len(go_layers))])
        else:
            module_masks = None
        if model_type == "sparse":
            encoder = SparseBIEncoder(used_go_layers, activation_fn, dtype, module_masks)
        else:
            encoder = DenseBIEncoder(used_go_layers, activation_fn, dtype, masks=module_masks, soft_links=soft_links)
    else:
        encoder = Encoder(used_go_layers, activation_fn, dtype)

    if (biologically_informed == "decoder") or (biologically_informed == "both"):
        if masks:
            # Discard masks of unused GO layers
            module_masks = (masks.pop(0)[-min(n_go_layers_used - 1, len(go_layers)):],
                            masks.pop(0)[-min(n_go_layers_used - 1, len(go_layers)):])
        else:
            module_masks = None
        if model_type == "sparse":
            decoder = SparseBIDecoder(used_go_layers, activation_fn, dtype, masks=module_masks)
        else:
            decoder = DenseBIDecoder(used_go_layers, activation_fn, dtype, masks=module_masks, soft_links=soft_links)
    else:
        decoder = Decoder(used_go_layers, activation_fn, dtype)

    print("\n----- MODEL SUMMARY -----")
    print(f"Autoencoder tensor type: {model_type}")
    print(f"Biologically informed module: {biologically_informed}")
    print(f"Uses soft links: {soft_links}")
    print(f"Randomized edges: {random_version}")
    print(f"Real-time GO-processing: {go_preprocessing}")
    print(f"Merge conditions: {merge_conditions} (min parents, min children, min terms per layer)")
    print(f"Number of GO layers used: {len(used_go_layers)}\n")

    model = Autoencoder(encoder, decoder)
    # Rename model for special conditions
    if soft_links:
        model.name += " (SL)"
    if random_version:
        model.name += " (Randomized)"
    return model
