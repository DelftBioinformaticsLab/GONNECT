from matplotlib import pyplot as plt


def plot_training_loss(path):
    with open(path, "r") as f:
        loss_file_content = f.readlines()[1:]

    epoch_losses = []
    for line in loss_file_content:
        losses = line.split("\t")
        epoch_losses.append([float(loss) for loss in losses])

    # Plot train and test loss
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses)
    plt.title("")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(["train", "validation", "test"])
    plt.show()

if __name__ == "__main__":
    plot_training_loss("../../../out/trained_models/AE_1.0/AE_1.0.0_both_results.txt")