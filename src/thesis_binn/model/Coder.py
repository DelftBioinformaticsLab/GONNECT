import torch
import torch.nn as nn

from thesis_binn.data_processing.ProxyTerm import ProxyTerm
from thesis_binn.model.SparseLinear import SparseLinear


class DenseCoder(nn.Module):
    """Base class for shared functionality between dense encoder and decoder."""

    def __init__(self, go_layers, activation_fn, dtype):
        super(DenseCoder, self).__init__()

        # Initialize architecture (conversion to ModuleList is passed down to implementation)
        self.go_layers = go_layers
        self.activation_fn = activation_fn
        network_layers = []
        for i in range(len(self.go_layers) - 1):
            network_layers.append(nn.Linear(len(self.go_layers[i]), len(self.go_layers[i + 1]), dtype=dtype))
            if i < len(self.go_layers) - 2:
                network_layers.append(self.activation_fn())
        # ModuleList conversion should appear here, but by passing that down it allows for additional activations to be added
        self.net_layers = network_layers

        # Hooks to store activations during forward pass
        self.activations = {}
        self._register_hooks()

    def forward(self, x):
        for layer in self.net_layers:
            x = layer(x)
        return x

    def mask_weights(self):
        return

    def mask_gradients(self):
        return

    def masks_to(self, device):
        return

    def _register_hooks(self):
        for idx, layer in enumerate(self.net_layers):
            if isinstance(layer, nn.Module):
                layer.register_forward_hook(self._get_activation_hook(f"layer_{idx}"))

    def _get_activation_hook(self, name):
        def hook(module, x, output):
            self.activations[name] = output.detach().clone()

        return hook


class DenseBICoder(DenseCoder):
    """Base class for shared functionality between masked dense encoder and decoder."""

    def __init__(self, go_layers, activation_fn, dtype, masks, soft_links):
        super(DenseBICoder, self).__init__(go_layers, activation_fn, dtype)

        # Initialize biologically-informed masks
        if masks:
            self.edge_masks = masks[0]
            self.proxy_masks = masks[1]
        else:
            self.proxy_masks = self._create_proxy_masks()
            self.edge_masks = self._create_edge_masks()

        self.soft_links = soft_links

    def mask_weights(self):
        """Using the internal dense mask matrices, mask the dense weights and biases after each training step."""
        # Mask weights using proxy and edge masks
        mask_index = 0
        for layer in self.net_layers:
            if isinstance(layer, nn.Linear):
                # Ensure that devices match
                if self.proxy_masks[mask_index].device != layer.weight.device:
                    self.proxy_masks[mask_index] = self.proxy_masks[mask_index].to(layer.weight.device)
                if self.edge_masks[mask_index].device != layer.weight.device:
                    self.edge_masks[mask_index] = self.edge_masks[mask_index].to(layer.weight.device)

                # Combine edge and proxy masks to set BI-weights to proxies
                proxy_weight_mask = self.edge_masks[mask_index] * self.proxy_masks[mask_index]
                # Set weights towards proxies to 1
                layer.weight.data = torch.masked_fill(layer.weight.data, proxy_weight_mask, value=1)
                # Set bias of proxy terms to 0
                for i in range(len(layer.bias.data)):
                    if self.proxy_masks[mask_index][i]:
                        layer.bias.data[i] = 0

                # Set weights without edges to 0, unless soft links is enabled
                if not self.soft_links:
                    layer.weight.data = torch.masked_fill(layer.weight.data, self.edge_masks[mask_index] == 0, value=0)
                mask_index += 1

    def mask_gradients(self):
        """Mask the dense gradients of weights and biases after the backwards pass to prevent masked values getting updates."""
        # Mask gradients using proxy and edge masks
        mask_index = 0
        for layer in self.net_layers:
            if isinstance(layer, nn.Linear):
                # Set gradients of proxy weights and biases to 0
                layer.weight.grad = torch.masked_fill(layer.weight.grad, self.proxy_masks[mask_index], value=0)
                layer.bias.grad = torch.masked_fill(layer.bias.grad, self.proxy_masks[mask_index].T, value=0)
                # Set gradients of weights without edges to 0
                layer.weight.grad = torch.masked_fill(layer.weight.grad, self.edge_masks[mask_index] == 0, value=0)
                mask_index += 1

    def _create_proxy_masks(self):
        """Returns a list of dense 1D boolean tensors that represent each network layer. Each non-zero entry means that the corresponding term in that layer is a ProxyTerm."""
        proxy_masks = []
        for n in range(len(self.go_layers) - 1):
            next_layer = self.go_layers[n + 1]
            proxy_mask = torch.zeros(len(next_layer), 1, dtype=torch.bool)
            for i in range(len(next_layer)):
                if isinstance(next_layer[i], ProxyTerm):
                    proxy_mask[i] = 1
            proxy_masks.append(proxy_mask)
        return proxy_masks

    def _create_edge_masks(self):
        """Encoder/Decoder dependent. Implemented in respective subclasses."""
        return [torch.empty() for _ in range(len(self.go_layers))]

    def masks_to(self, device):
        """When model is transferred to another device, this method must be called to move the masks as well."""
        self.edge_masks = [mask.to(device) for mask in self.edge_masks]
        self.proxy_masks = [mask.to(device) for mask in self.proxy_masks]


class SparseCoder(nn.Module):
    """Base class for shared functionality between sparse encoder and decoder."""

    def __init__(self, go_layers, activation_fn, dtype, masks):
        super(SparseCoder, self).__init__()
        self.go_layers = go_layers

        # Initialize masks
        if masks:
            self.edge_masks = masks[0]
            self.proxy_masks = masks[1]
        else:
            self.edge_masks = self._create_edge_masks()
            self.proxy_masks = self._create_proxy_masks()

        # Initialize architecture (conversion to ModuleList is passed down to implementation)
        network_layers = []
        self.activation_fn = activation_fn
        for i in range(len(self.go_layers) - 1):
            network_layers.append(
                SparseLinear(len(self.go_layers[i]), len(self.go_layers[i + 1]), self.edge_masks[i], dtype=dtype))
            if i < len(self.go_layers) - 2:
                network_layers.append(self.activation_fn())
        # ModuleList conversion should appear here, but by passing that down it allows for additional activations to be added
        self.net_layers = network_layers

        # Hooks to store activations during forward pass
        self.activations = {}
        self._register_hooks()

    def forward(self, x):
        for layer in self.net_layers:
            x = layer(x)
        return x

    def mask_weights(self):
        """Sparse weight matrices ensure that edgeless weights remain zero. Dense proxy masks are used to set the non-zero weights corresponding to a ProxyTerm to 1, and their bias to 0."""
        mask_index = 0
        for layer in self.net_layers:
            if isinstance(layer, SparseLinear):
                # Ensure that devices match
                if self.proxy_masks[mask_index].device != layer.weight.device:
                    self.proxy_masks[mask_index] = self.proxy_masks[mask_index].to(layer.weight.device)

                nnz_rows = layer.weight.data.coalesce().indices()[0]
                proxy_mask = self.proxy_masks[mask_index]
                # If a row of the sparse weight matrix corresponds to a ProxyTerm, all non-zero values in that row are set to 1
                for j in range(len(nnz_rows)):
                    if proxy_mask[nnz_rows[j]]:
                        layer.weight.data = layer.weight.data.coalesce()
                        layer.weight.data.values()[j] = 1
                # Mask ProxyTerm bias
                for i in range(len(layer.bias.data)):
                    if self.proxy_masks[mask_index][i]:
                        layer.bias.data[i] = 0
                mask_index += 1

    def _create_proxy_masks(self):
        """Returns a list of dense 1D boolean tensors that represent each network layer. Each non-zero entry means that the corresponding term in that layer is a ProxyTerm."""
        proxy_masks = []
        for n in range(len(self.go_layers) - 1):
            next_layer = self.go_layers[n + 1]
            proxy_mask = torch.zeros(len(next_layer), 1, dtype=torch.bool)
            for i in range(len(next_layer)):
                if isinstance(next_layer[i], ProxyTerm):
                    proxy_mask[i] = 1
            proxy_masks.append(proxy_mask)
        return proxy_masks

    def _create_edge_masks(self):
        """Implemented in Encoder/Decoder (sub)classes"""
        return [torch.empty() for _ in range(len(self.go_layers))]

    def _register_hooks(self):
        for idx, layer in enumerate(self.net_layers):
            if isinstance(layer, nn.Module):
                layer.register_forward_hook(self._get_activation_hook(f"layer_{idx}"))

    def _get_activation_hook(self, name):
        def hook(module, x, output):
            self.activations[name] = output.detach().clone()

        return hook
