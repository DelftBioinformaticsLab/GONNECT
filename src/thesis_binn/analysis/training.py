import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt

from thesis_binn.model.build_model import build_model
from thesis_binn.train.loss import MSE
from thesis_binn.train.train import make_data_splits, test


def plot_training_loss(path, loss_type):
    with open(path, "r") as f:
        loss_file_content = f.readlines()[1:]

    epoch_losses = []
    train_losses = []
    test_losses = []
    val_losses = []
    mse_losses = []
    for line in loss_file_content:
        losses = line.split("\t")
        epoch_losses.append([float(loss) for loss in losses])
        train_losses.append(float(losses[0]))
        test_losses.append(float(losses[1]))
        val_losses.append(float(losses[2]))
        if loss_type == "mse":
            mse_losses.append(float(losses[3]))

    if loss_type == "train":
        plot_loss = train_losses
    elif loss_type == "test":
        plot_loss = test_losses
    elif loss_type == "validation":
        plot_loss = val_losses
    elif loss_type == "mse":
        plot_loss = mse_losses
    else:
        raise Exception(f"Invalid loss type: {loss_type}")
    plt.plot(range(1, len(epoch_losses) + 1), plot_loss)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")


def plot_loss_curves(experiment, version, loss_type):
    bi_modules = ["encoder", "decoder", "both"]
    for bi_module in bi_modules:
        plot_training_loss(f"../../../out/trained_models/{experiment}/{experiment}.{version}_{bi_module}_results.txt",
                           loss_type=loss_type)

    # Baseline FC-FC
    plot_training_loss(f"../../../out/trained_models/AE_2.0/AE_2.0.0_none_results.txt", loss_type=loss_type)

    plt.title(f"{loss_type} loss (???)")
    plt.legend(["BI-FC", "FC-BI", "BI-BI", "FC-FC"])
    plt.show()


def bar_plot_final_loss():
    """Plot a bar chart of the average test loss over multiple training rounds, comparing different model configurations."""
    experiments = ["AE_2.1", "AE_2.0", "AE_2.2"]  # SL, HL, R
    bi_modules = ["encoder", "decoder", "both"]
    versions = ["2", "3", "4", "5", "6"]

    bars = ["Fully Connected\nAutoencoder", "", "Soft Links", "", "", "GONNECT\nAutoencoder", "", "", "Randomized Fixed Links", ""]
    bar_labels = ["dual MLP",
                  "GONNECT encoder", "GONNECT decoder", "dual GONNECT",
                  "_GONNECT encoder", "_GONNECT decoder", "_dual GONNECT",
                  "_GONNECT encoder", "_GONNECT decoder", "_dual GONNECT"]
    bar_colors = ["#8516D1",
                  "#1171BE", "#DD5400", "#EDB120",
                  "#1171BE", "#DD5400", "#EDB120",
                  "#1171BE", "#DD5400", "#EDB120"]
    x_positions = [0, 1.5, 2.5, 3.5, 5, 6, 7, 8.5, 9.5, 10.5]

    losses_per_model = []

    # Collect final loss values for FC-FC
    fc_losses = []
    for version in versions:
        loss_path = f"../../../out/trained_models/AE_2.0/AE_2.0.{version}_none_results.txt"
        with open(loss_path, "r") as f:
            loss_file_content = f.readlines()
        final_losses = [float(loss) for loss in loss_file_content[-1].split("\t")]
        final_test_loss = final_losses[3]
        fc_losses.append(final_test_loss)
    losses_per_model.append(fc_losses)

    # Collect final loss value for each BI-model in the experiment
    for experiment in experiments:
        if experiment == "AE_2.2":  # Randomized links has been retrained with new randomness heuristic
            versions = ["8", "9", "10", "11", "12"]
        for bi_module in bi_modules:
            bi_losses = []
            for version in versions:
                loss_path = f"../../../out/trained_models/{experiment}/{experiment}.{version}_{bi_module}_results.txt"
                with open(loss_path, "r") as f:
                    loss_file_content = f.readlines()
                final_losses = [float(loss) for loss in loss_file_content[-1].split("\t")]
                # Use regular MSE for SL models (disregarding the regularization term for fair performance comparison)
                final_test_loss = final_losses[2] if (experiment != "AE_2.1") else final_losses[3]
                bi_losses.append(final_test_loss)
            losses_per_model.append(bi_losses)

    # Compute means and standard deviations
    mean_losses = [np.mean(losses) for losses in losses_per_model]
    std_losses = [np.std(losses) for losses in losses_per_model]

    # # x_positions = [x_positions[0], x_positions[4], x_positions[5], x_positions[6]]
    # x_positions = x_positions[:7]
    # mean_losses = [mean_losses[0], mean_losses[4], mean_losses[5], mean_losses[6], mean_losses[7], mean_losses[8], mean_losses[9]]
    # std_losses = [std_losses[0], std_losses[4], std_losses[5], std_losses[6], std_losses[7], std_losses[8], std_losses[9]]
    # # bar_labels = [bar_labels[0], bar_labels[4], bar_labels[5], bar_labels[6]]
    # bar_labels = bar_labels[:7]
    # bar_colors = [bar_colors[0], bar_colors[4], bar_colors[5], bar_colors[6], bar_colors[7], bar_colors[8], bar_colors[9]]
    # bars = ["", "", "", "", "", "", "", "", "", ""] # Presentation style

    # plt.figure(figsize=(10, 8)) # Presentation style
    plt.figure(figsize=(8, 8))
    plt.title(f"Average MSE on Gene Expression Reconstruction")
    plt.bar(x_positions, mean_losses, yerr=std_losses, label=bar_labels, color=bar_colors, capsize=5)
    # Add MLP reference as dashed line
    plt.axhline(float(mean_losses[0]), linestyle='--', color="gray")
    plt.ylabel("MSE")
    plt.legend()
    plt.xticks(x_positions, bars, rotation=0)
    plt.tight_layout()
    plt.show()


def calculate_final_loss():
    """Manually change method to quickly find the post-training test loss with a different loss function than the model was trained with."""
    experiment = "AE_2.2"
    version = "2"
    bi_module = "decoder"
    soft_links = True
    random_version = None
    loss_fn = MSE()
    seed = 1

    # Generate MSE losses without regularization term
    data = pd.read_csv(f"../../../data/TCGA_complete_bp_top1k.csv.gz", nrows=9797,
                       usecols=range(5 + 1000), compression="gzip")
    model = build_model(model_type="dense", biologically_informed=bi_module, soft_links=soft_links,
                           dataset_name="TCGA_complete_bp_top1k",
                           go_preprocessing=False, merge_conditions=(1, 30, 50), n_go_layers_used=5,
                           activation_fn=torch.nn.ReLU, dtype=torch.float64, genes=None,
                           package_call=True, cluster=False, random_version=random_version)
    model.load_state_dict(torch.load(
        f"../../../out/trained_models/{experiment}/{experiment}.{version}_{bi_module}_model.pt", weights_only=True))
    dataloader, trainloader, validationloader, testloader = make_data_splits(data,
                                                                             n_nan_cols=5,
                                                                             n_samples=9797,
                                                                             batch_size=100,
                                                                             data_split=0.7,
                                                                             seed=seed)
    loss = test(testloader, model, loss_fn)
    return loss


if __name__ == "__main__":
    experiment = "AE_2.2"
    version = "3"
    bi_module = "decoder"

    # plot_loss_curves(experiment, version, "train")
    bar_plot_final_loss()

    # Evaluate Soft Link alpha parameter
    # experiments = ["AE_3.0", "AE_3.1", "AE_3.2", "AE_3.3"]
    # plot_training_loss(f"../../../out/trained_models/AE_3.-1/AE_3.-1.2_none_results.txt", loss_type="test")
    # for experiment in experiments:
    #     plot_training_loss(f"../../../out/trained_models/{experiment}/{experiment}.{version}_{bi_module}_results.txt", loss_type="mse")
    # plot_training_loss(f"../../../out/trained_models/AE_3.-1/AE_3.-1.2_{bi_module}_results.txt", loss_type="test")
    # plt.legend([r"MLP", r"$\alpha$ = 1e2", r"$\alpha$ = 1e3", r"$\alpha$ = 1e4", r"$\alpha$ = 1e5", r"Fixed links"])
    # plt.title(f"MSE during training of GONNECT-SL decoder model")
    # plt.show()
