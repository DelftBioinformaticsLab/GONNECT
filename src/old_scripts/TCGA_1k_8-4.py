import torch
import pandas as pd
from matplotlib import pyplot as plt
from thesis_binn.data_processing.dag_analysis import print_layers
from thesis_binn.data_processing.generate_masks import load_masks
from thesis_binn.data_processing.go_preprocessing import construct_go_bp_layers
from thesis_binn.model.Autoencoder import Autoencoder
from thesis_binn.model.Decoder import SparseBIDecoder, DenseBIDecoder, Decoder
from thesis_binn.model.Encoder import SparseBIEncoder, Encoder, DenseBIEncoder
from thesis_binn.train.loss import MSE, MSE_Soft_Link_Sum
from thesis_binn.train.train import make_data_splits, train_with_validation, save_training_losses

if __name__ == "__main__":
    experiment_name = ""
    # Model params
    model_type = "dense"
    biologically_informed = "encoder"
    soft_links = False
    activation_fn = torch.nn.ReLU
    loss_fn = MSE_Soft_Link_Sum(alpha=1.0) if soft_links else MSE()
    # GO params
    go_preprocessing = True
    merge_conditions = (1, 30)
    n_go_layers_used = 4
    # Training params
    dataset_name = "TCGA_complete_bp_top1k"
    n_samples = 10000
    batch_size = 100
    n_epochs = 3
    learning_rate = 0.01
    momentum = 0.9
    patience = 5
    device = "cpu"
    save_losses = False
    loss_path = experiment_name
    # Model storage params
    save_weights = False
    load_weights = False
    weights_path = experiment_name
    # Additional params
    data_split = 0.7
    seed = 1
    dtype = torch.float64
    n_nan_cols = 5

    # Data processing
    data = pd.read_csv(f"../data/{dataset_name}.csv.gz", nrows=min(n_samples, 9797),
                       usecols=range(n_nan_cols + 1000), compression="gzip")
    dataloader, trainloader, validationloader, testloader = make_data_splits(data, n_nan_cols, n_samples, batch_size,
                                                                             data_split, seed)

    # GO processing
    if go_preprocessing:
        print("\n----- START: GO preprocessing -----")
        genes = list(data.columns[n_nan_cols:])
        go_layers = construct_go_bp_layers(genes, merge_conditions, print_go=True)
        masks = None
        print("----- COMPLETED: GO preprocessing -----")

    else:
        go_layers = torch.load(f"../out/masks/layers/{str(merge_conditions)}/{dataset_name}_layers.pt",
                               weights_only=True)
        masks = load_masks(biologically_informed, merge_conditions, dataset_name, model_type)
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

    model = Autoencoder(encoder, decoder)
    if load_weights:
        model.load_state_dict(
            torch.load(f"../out/trained_models/{dataset_name}/{weights_path}_model.pt", weights_only=True))

    # Training
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    epoch_losses = train_with_validation(n_epochs, trainloader, testloader, validationloader, model, optimizer, loss_fn,
                                         patience, device)
    if save_losses:
        save_training_losses(epoch_losses, f"../out/trained_models/{dataset_name}/{loss_path}_results.txt")

    # Debug: show network weights as colored grid
    # plt.imshow(encoder.net_layers._modules["0"].weight.data, cmap="RdYlGn")
    if save_weights:
        torch.save(model.state_dict(), f"../out/trained_models/{dataset_name}/{weights_path}_model.pt")
        print(f"\n----- Saved weights to file ({weights_path}_model.pt) -----")

    # Plot train and test loss
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses)
    plt.title(
        f"Loss for {model_type} BI-module: {biologically_informed} (n = {int(min(n_samples, 9797) * data_split)})")
    plt.xlabel("Epoch")
    plt.ylabel(loss_fn.name)
    plt.legend(["train", "validation", "test"])
    plt.show()
