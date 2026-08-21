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

        # Hooks to store activations during forward pass. Storing is opt-in, because cloning every
        # intermediate output on every batch is overhead during training; only the analysis code reads it.
        self.activations = {}
        self.store_activations = False
        self._register_hooks()

    def forward(self, x):
        for layer in self.net_layers:
            x = layer(x)
        return x

    def mask_weights(self):
        return

    def masks_to(self, device):
        return

    def _register_hooks(self):
        for idx, layer in enumerate(self.net_layers):
            if isinstance(layer, nn.Module):
                layer.register_forward_hook(self._get_activation_hook(f"layer_{idx}"))

    def _get_activation_hook(self, name):
        def hook(module, x, output):
            if self.store_activations:
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
        self._cache_masks()
        if soft_links:
            self._initialize_soft_links()

    def _cache_masks(self):
        """Derive the masks used every optimization step once, instead of rebuilding them per step.

        All of these are functions of the edge and proxy masks alone, so they are constant for the
        lifetime of the module. See mask_weights and thesis_binn.train.loss.soft_link_sum."""
        # Weights towards proxy terms, fixed at 1
        self.proxy_weight_masks = [e & p for e, p in zip(self.edge_masks, self.proxy_masks)]
        # Weights without a GO edge: fixed at 0 for GONNECT, regularized as soft links for GONNECT-SL
        self.non_edge_masks = [~e for e in self.edge_masks]
        # Biases of proxy terms, fixed at 0
        self.proxy_bias_masks = [p.squeeze(-1) for p in self.proxy_masks]
        # Soft links towards proxy terms, used by MSE_Soft_Link_Proxyless
        self.non_edge_proxy_masks = [n & p for n, p in zip(self.non_edge_masks, self.proxy_masks)]
        # The element counts are constants as well, so they must not be recomputed inside the loss
        self.n_non_edge = int(sum(int(m.sum()) for m in self.non_edge_masks))
        self.n_non_edge_proxy = int(sum(int(m.sum()) for m in self.non_edge_proxy_masks))

    def _initialize_soft_links(self):
        # Re-initialize soft links with small values
        mask_index = 0
        for layer in self.net_layers:
            if isinstance(layer, nn.Linear):
                mask = ~self.edge_masks[mask_index]
                with torch.no_grad():
                    # Draw soft link values
                    soft_link_weights = torch.empty_like(layer.weight).normal_(mean=0.0, std=1e-3)
                    soft_link_weights = soft_link_weights.to(layer.weight)
                    # Apply only where mask is True
                    layer.weight[mask] = soft_link_weights[mask]
                mask_index += 1

    def mask_weights(self):
        """Using the internal dense mask matrices, mask the dense weights and biases after each training step.

        This runs after every optimization step, so it uses the masks cached by _cache_masks and fills
        in place. Masking the biases element by element in Python forced a host-device synchronization
        per node, which dominated the training time."""
        # Ensure that devices match
        if self.proxy_weight_masks[0].device != self.net_layers[0].weight.device:
            self.masks_to(self.net_layers[0].weight.device)

        # Mask weights using proxy and edge masks
        mask_index = 0
        for layer in self.net_layers:
            if isinstance(layer, nn.Linear):
                # Set weights towards proxies to 1
                layer.weight.data.masked_fill_(self.proxy_weight_masks[mask_index], value=1)
                # Set bias of proxy terms to 0
                layer.bias.data.masked_fill_(self.proxy_bias_masks[mask_index], value=0)
                # Set weights without edges to 0, unless soft links is enabled
                if not self.soft_links:
                    layer.weight.data.masked_fill_(self.non_edge_masks[mask_index], value=0)
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
        device = torch.device(device)
        if self.edge_masks[0].device == device:
            return
        self.edge_masks = [mask.to(device) for mask in self.edge_masks]
        self.proxy_masks = [mask.to(device) for mask in self.proxy_masks]
        self._cache_masks()


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

        # Hooks to store activations during forward pass (opt-in, see DenseCoder)
        self.activations = {}
        self.store_activations = False
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
            if self.store_activations:
                self.activations[name] = output.detach().clone()

        return hook
