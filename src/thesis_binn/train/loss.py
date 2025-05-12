import torch
import torch.nn as nn

from thesis_binn.model.Autoencoder import Autoencoder
from thesis_binn.model.Coder import SparseCoder
from thesis_binn.model.Decoder import Decoder, DenseBIDecoder
from thesis_binn.model.Encoder import DenseBIEncoder


class MSE(nn.Module):
    """Wrapper class for torch.nn.MSELoss to allow additional arguments for easy swapping of loss functions."""

    def __init__(self):
        super(MSE, self).__init__()
        self.name = "MSE Loss"
        self.mse = nn.MSELoss()

    def forward(self, x, y, **kwargs):
        return self.mse(x, y)


class MSE_Masked(nn.Module):
    """Mean Squared Error loss for reconstructed gene expression where genes without GO-terms are masked."""

    def __init__(self, mask, device="cpu"):
        super(MSE_Masked, self).__init__()
        self.name = "MSE Loss Guess Corrected"
        self.mask = mask.to(device)
        self.device = device
        self.mse = nn.MSELoss()

    def forward(self, x, y, **kwargs):
        mask = ~self.mask
        if self.mask.dim() == 1:
            mask = mask.unsqueeze(0)
        x_masked = x * mask
        y_masked = y * mask
        return self.mse(x_masked, y_masked)


class MSE_Soft_Link_Sum(nn.Module):
    """Standard MSE with an additional term, weighted by alpha, for the sum of soft link weights."""

    def __init__(self, alpha=1.0):
        super(MSE_Soft_Link_Sum, self).__init__()
        self.name = "MSE + soft link sum"
        self.alpha = alpha

    def forward(self, x, y, model: Autoencoder, **kwargs):
        if isinstance(model.encoder, SparseCoder) or isinstance(model.decoder, SparseCoder):
            raise Exception("Soft links are not supported for models containing SparseTensors")

        mse = nn.functional.mse_loss(x, y)
        soft_weight_sum = 0
        n_soft_weights = 0
        if hasattr(model.encoder, "edge_masks"):
            layer_sum, layer_n = soft_link_sum(model.encoder)
            soft_weight_sum += layer_sum
            n_soft_weights += layer_n
        if hasattr(model.decoder, "edge_masks"):
            layer_sum, layer_n = soft_link_sum(model.decoder)
            soft_weight_sum += layer_sum
            n_soft_weights += layer_n
        # Debug
        # print("soft_weight_sum: ", soft_weight_sum)
        return mse + self.alpha * (soft_weight_sum / n_soft_weights)


def soft_link_sum(module):
    """For a given network (encoder or decoder), return the sum of absolute values of the weights that are considered soft links because they are masked by the edge mask of the network."""
    soft_weight_sum = 0
    mask_index = 0
    n = 0
    for layer in module.net_layers:
        if isinstance(layer, nn.Linear):
            # For each linear layer of the network, sum the absolute values of the masked weights
            soft_weight_sum += torch.sum(layer.weight.abs() * ~module.edge_masks[mask_index])
            n += torch.sum(~module.edge_masks[mask_index])
            mask_index += 1
    return soft_weight_sum, n


if __name__ == '__main__':
    loss_fn = MSE_Soft_Link_Sum()
    layers = torch.randn((2, 3))
    mask_e = [torch.Tensor([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]]).bool()]
    encoder_weights = torch.Tensor([
        [1, 2, 3],
        [4, -5, 6],
        [7, 8, 9]])
    masks = [mask_e, [torch.randn(3, 1) > 0.5]]
    encoder = DenseBIEncoder(layers, nn.ReLU(), torch.float64, masks)
    encoder.net_layers[0].weight.data = encoder_weights
    print(f"Test result should be 15. Result: {soft_link_sum(encoder)}")

    decoder = Decoder(layers, nn.ReLU(), torch.float64)
    ae = Autoencoder(encoder, decoder)
    x = torch.Tensor([1, 2, 3]).requires_grad_()
    print(
        f"Test result should be 15/3. Result: {loss_fn(x, torch.Tensor([1, 2, 3]), ae)}")
    print(
        f"Test result should be 1+15/3. Result: {loss_fn(x, torch.Tensor([0, 1, 2]), ae)}")
    decoder = DenseBIDecoder(layers, nn.ReLU(), torch.float64, masks)
    decoder.net_layers[0].weight.data = encoder_weights
    ae = Autoencoder(encoder, decoder)
    print(
        f"Test result should be 30/6. Result: {loss_fn(x, torch.Tensor([1, 2, 3]), ae)}")
    print(
        f"Test result should be 1+30/6. Result: {loss_fn(x, torch.Tensor([0, 1, 2]), ae)}")
