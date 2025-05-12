import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from thesis_binn.data_processing.go_preprocessing import *
from thesis_binn.data_processing.DAGGenerator import DAGGenerator
from thesis_binn.model.deprecated.OldEncoder import Encoder
from thesis_binn.model.deprecated.OldDecoder import Decoder
from thesis_binn.model.Autoencoder import Autoencoder
from thesis_binn.train.loss import MSE


def loss_kl_divergence(inputs, outputs, net):
    """To be used for VAE training..."""
    return ((inputs - outputs) ** 2).sum() + net.encoder.kl


def split_data(data, n_nan_cols, split=0.7, seed=1):
    """Split the given dataframe in train, validation and test sets. The split argument sets the training fraction, the remainder is split 50/50 between validation and test."""
    validation_test_split = 0.5
    # Strip any non-sample column before making the splits
    gene_expression = data.copy()
    gene_expression = gene_expression[gene_expression.columns[n_nan_cols:]]
    train_set, remaining_set = train_test_split(gene_expression, train_size=split, random_state=seed)
    validation_set, test_set = train_test_split(remaining_set, train_size=validation_test_split, random_state=seed)
    return train_set, validation_set, test_set


def make_data_splits(data, n_nan_cols, n_samples, batch_size, data_split, seed):
    """Split the data provided fot training into the full, train, validation and test sets, and return them as DataLoader objects."""
    train_set, validation_set, test_set = split_data(data, n_nan_cols, data_split, seed)
    data_np = data.iloc[:, n_nan_cols:].to_numpy()
    data_torch = TensorDataset(torch.from_numpy(data_np))
    dataloader = DataLoader(data_torch, batch_size=min(n_samples, batch_size), shuffle=False)
    train_torch = TensorDataset(torch.from_numpy(train_set.to_numpy()))
    validation_torch = TensorDataset(torch.from_numpy(validation_set.to_numpy()))
    test_torch = TensorDataset(torch.from_numpy(test_set.to_numpy()))
    trainloader = DataLoader(train_torch, batch_size=min(n_samples, batch_size), shuffle=False)
    validationloader = DataLoader(validation_torch, batch_size=min(n_samples, batch_size), shuffle=False)
    testloader = DataLoader(test_torch, batch_size=min(n_samples, batch_size), shuffle=False)
    return dataloader, trainloader, validationloader, testloader


def train(train_loader, net: Autoencoder, optimizer, loss_fn, device="cpu"):
    """Trains network for one epoch in batches.
    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        loss_fn: Loss function.
        device: Whether the network runs on CPU or GPU."""

    # Additional setup for special models
    net.to(device)
    net.masks_to(device)
    if hasattr(loss_fn, "mask"):
        loss_fn.mask.to(device)

    avg_loss = 0
    # Iterate over batches
    for i, data in enumerate(train_loader):
        inputs = data[0]
        inputs = inputs.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Backward pass
        loss = loss_fn(outputs, inputs)
        loss.backward()

        optimizer.step()

        # Force biologically-informed weights
        net.mask_weights()

        # keep track of loss and accuracy
        avg_loss += loss

    return avg_loss / len(train_loader)


def test(test_loader, net, loss_fn, device="cpu"):
    """Test current model performance on a validation or test set. Used to prevent overfitting during training."""
    net.to(device)
    avg_loss = 0
    # No gradient computation needed for forward pass only
    with torch.no_grad():
        # Iterate over batches
        for i, data in enumerate(test_loader):
            inputs = data[0]
            inputs = inputs.to(device)

            # Forward pass
            outputs = net(inputs)
            loss = loss_fn(outputs, inputs)

            # keep track of loss and accuracy
            avg_loss += loss

    return avg_loss / len(test_loader)


def train_with_validation(max_epochs, trainloader, testloader, validationloader, net, optimizer, loss_function,
                          patience, device="cpu"):
    """Function to execute the full training process. The provided model is trained on the train set, and evaluated on both validation and test sets. The patience argument is used for early stopping based on performance on the validation set."""
    epoch_losses = []
    t_start = time.time()
    for epoch in range(max_epochs):  # loop over the dataset multiple times
        train_loss = train(trainloader, net, optimizer, loss_fn=loss_function, device=device)
        validation_loss = test(validationloader, net, loss_fn=loss_function, device=device)
        test_loss = test(testloader, net, loss_fn=loss_function, device=device)
        print(f"Train loss after epoch {epoch + 1}:\t{train_loss}\t\t"
              f"Validation loss after epoch {epoch + 1}:\t{validation_loss}\t\t"
              f"Test loss after epoch {epoch + 1}:\t{test_loss}")
        epoch_losses.append([train_loss.item(), validation_loss.item(), test_loss.item()])

        # Initialize early stopping variables
        if epoch == 0:
            best_validation_loss = validation_loss.item()
            patience_count = 0

        # Early stopping evaluation
        if validation_loss.item() < best_validation_loss:
            best_validation_loss = validation_loss.item()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    t_end = time.time() - t_start
    print(f"Total training time: {t_end // 3600:.0f}h {(t_end % 3600) // 60:.0f}m {t_end % 60:.0f}s")

    return epoch_losses


def save_training_losses(epoch_losses, file_path):
    """Save losses on train, validation and test sets after each epoch of the training process to the provided file path."""
    with open(file_path, "w") as f:
        f.write("Train loss\tValidation loss\tTest loss\n")
        for epoch_loss in epoch_losses:
            f.write(f"{epoch_loss[0]}\t{epoch_loss[1]}\t{epoch_loss[2]}\n")


if __name__ == "__main__":
    n_samples = 10
    batch_size = 10
    n_epochs = 1000
    lr = 1e-1
    dtype = torch.float32

    # DAG to layers
    dag = DAGGenerator.dag4()
    go = copy_dag(dag)
    balance_until_convergence(go, root_id="A")
    pull_leaves_down(go, len(dag))
    print_dag_info(go)
    layers = create_layers(go)

    # Load data (samples, genes)
    data = TensorDataset(torch.randn(n_samples, len(layers[-1]), dtype=dtype))

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    model = Autoencoder(Encoder(layers, dtype=dtype), Decoder(layers, dtype=dtype))

    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Set the number of epochs for training
    epochs = n_epochs
    epoch_losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = train(dataloader, model, optimizer, MSE())
        epoch_losses.append(train_loss.item())
        print(f"Training loss after epoch {epoch + 1}: {train_loss}")

    plt.plot(epoch_losses)
    plt.show()
    pass
