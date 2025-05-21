from matplotlib import pyplot as plt


def plot_training_loss(path, loss_type):
    with open(path, "r") as f:
        loss_file_content = f.readlines()[1:]

    epoch_losses = []
    train_losses = []
    test_losses = []
    val_losses = []
    for line in loss_file_content:
        losses = line.split("\t")
        epoch_losses.append([float(loss) for loss in losses])
        train_losses.append(float(losses[0]))
        test_losses.append(float(losses[1]))
        val_losses.append(float(losses[2]))

    if loss_type == "train":
        plot_loss = train_losses
    elif loss_type == "test":
        plot_loss = test_losses
    elif loss_type == "validation":
        plot_loss = val_losses
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
    experiments = ["AE_2.1", "AE_2.0", "AE_2.2"] # SL, HL, R
    bi_modules = ["encoder", "decoder", "both"]
    version = "0"

    # When changing version, not only these lines swap, but the two for loops also need to be swapped, and the rotation has to be set to 45
    # bars = ["FC-FC", "BI-FC (SL)", "BI-FC", "BI-FC (R)", "FC-BI (SL)", "FC-BI", "FC-BI (R)", "BI-BI (SL)", "BI-BI", "BI-BI (R)"]
    # bar_labels = ["Fully connected", "Soft links", "Hard links", "Randomized hard links", "_Soft links", "_Hard links", "_Randomized hard links", "_Soft links", "_Hard links", "_Randomized hard links"]
    bars = ["Fully Connected", "    ", "Soft Links", "     ", "", "Hard Links", " ", "  ", "Randomized Hard Links", "   "]
    bar_labels = ["No Biology", "Biological Encoder", "Biological Decoder", "Biological Encoder and Decoder", "_Biological Encoder", "_Biological Decoder", "_Biological Encoder and Decoder", "_Biological Encoder", "_Biological Decoder", "_Biological Encoder and Decoder"]
    bar_colors = ["#8516D1", "#1171BE", "#DD5400", "#EDB120", "#1171BE", "#DD5400", "#EDB120", "#1171BE", "#DD5400", "#EDB120"]

    values = []
    # Collect final loss value for FC-FC
    loss_path = f"../../../out/trained_models/AE_2.0/AE_2.0.0_none_results.txt"
    with open(loss_path, "r") as f:
        loss_file_content = f.readlines()
    final_losses = [float(loss) for loss in loss_file_content[-1].split("\t")]
    final_test_loss = final_losses[2]
    values.append(final_test_loss)
    # Collect final loss value for each BI-model in the experiment
    for experiment in experiments:
        for bi_module in bi_modules:
            loss_path = f"../../../out/trained_models/{experiment}/{experiment}.{version}_{bi_module}_results.txt"
            with open(loss_path, "r") as f:
                loss_file_content = f.readlines()
            final_losses = [float(loss) for loss in loss_file_content[-1].split("\t")]
            final_test_loss = final_losses[2]
            values.append(final_test_loss)

    plt.figure(figsize=(8, 8))
    plt.title("MSE on test set after training")
    plt.bar(bars, values, label=bar_labels, color=bar_colors)
    plt.ylabel("MSE")
    plt.legend()
    plt.xticks(bars, rotation=0)
    plt.show()

if __name__ == "__main__":
    experiment = "AE_2.2"
    version = "0"

    # plot_loss_curves(experiment, version, "train")
    bar_plot_final_loss()
