import torch
import torch.nn as nn

from thesis_binn.model.Coder import DenseCoder, DenseBICoder, SparseCoder
from thesis_binn.model.SparseLinear import SparseLinear


class Encoder(DenseCoder):
    def __init__(self, go_layers, activation_fn, dtype):
        # GO layers are originally ordered by ascending depth, so they need to be reversed for the encoder architecture
        go_layers = list(reversed(go_layers))
        super(Encoder, self).__init__(go_layers, activation_fn, dtype)
        self.name = "Encoder"

        # Optionally add activation after final linear layer
        pass

        # Convert list of layers to ModuleList for easy propagation
        self.net_layers = nn.ModuleList(self.net_layers)

        # (Re)initialize all weights
        for layer in self.net_layers:
            if isinstance(layer, nn.Linear):
                # Option for initialization other than standard (Kaiming uniform)
                pass
                # nn.init.kaiming_normal_(layer.weight)


class DenseBIEncoder(DenseBICoder):
    def __init__(self, go_layers, activation_fn, dtype, masks=None, soft_links=False):
        # GO layers are originally ordered by ascending depth, so they need to be reversed for the encoder architecture
        go_layers = list(reversed(go_layers))
        super(DenseBIEncoder, self).__init__(go_layers, activation_fn, dtype, masks, soft_links)
        self.name = "BI-Encoder"

        # Optionally add activation after final linear layer
        pass

        # Convert list of layers to ModuleList for easy propagation
        self.net_layers = nn.ModuleList(self.net_layers)

        # (Re)initialize all weights
        for layer in self.net_layers:
            if isinstance(layer, nn.Linear):
                # Option for initialization other than standard (Kaiming uniform)
                pass
                # nn.init.kaiming_normal_(layer.weight)

        # Mask weights using proxy and edge masks
        self.mask_weights()

    def _create_edge_masks(self):
        """Returns a list of dense 2D boolean tensors. Each tensor functions as an adjacency matrix between network layers, derived from GO."""
        edge_masks = []
        for n in range(len(self.go_layers) - 1):
            current_layer = self.go_layers[n]
            next_layer = self.go_layers[n + 1]
            edge_mask = torch.zeros(len(next_layer), len(current_layer), dtype=torch.bool)

            for i in range(len(current_layer)):
                child = current_layer[i]
                parent_ids = [parent.item_id for parent in child.parents]
                next_layer_ids = [term.item_id for term in next_layer]
                for j in range(len(next_layer)):
                    if next_layer_ids[j] in parent_ids:
                        edge_mask[j][i] = 1
            edge_masks.append(edge_mask)
        return edge_masks


class SparseBIEncoder(SparseCoder):
    def __init__(self, go_layers, activation_fn, dtype, masks=None):
        # GO layers are originally ordered by ascending depth, so they need to be reversed for the encoder architecture
        go_layers = list(reversed(go_layers))
        super(SparseBIEncoder, self).__init__(go_layers, activation_fn, dtype, masks)
        self.name = "sparse BI-Encoder"

        # Optionally add activation after final linear layer
        pass

        # Convert list of layers to ModuleList for easy propagation
        self.net_layers = nn.ModuleList(self.net_layers)

        # (Re)initialize all weights
        for layer in self.net_layers:
            if isinstance(layer, SparseLinear):
                # Kaiming uniform initialization for SparseTensors is implemented in the SparseLinear class, no need to re-initialize
                pass

        # Mask weights using proxy masks
        self.mask_weights()

    def _create_edge_masks(self):
        """Returns a list of sparse 2D boolean tensors. Each tensor functions as an adjacency matrix between network layers, derived from GO."""
        edge_masks = []
        for n in range(len(self.go_layers) - 1):
            current_layer = self.go_layers[n]
            next_layer = self.go_layers[n + 1]
            non_zero_indices = []
            # For each layer, find the coordinates of the non-zero values of the adjacency matrix
            for i in range(len(current_layer)):
                child = current_layer[i]
                parent_ids = [parent.item_id for parent in child.parents]
                next_layer_ids = [term.item_id for term in next_layer]
                for j in range(len(next_layer)):
                    if next_layer_ids[j] in parent_ids:
                        non_zero_indices.append([j, i])

            # Construct a sparse boolean mask using the non-zero coordinates
            edge_mask = torch.sparse_coo_tensor(torch.tensor(non_zero_indices).t(), torch.ones(len(non_zero_indices)),
                                                size=(len(next_layer), len(current_layer)), dtype=torch.bool)
            edge_masks.append(edge_mask.coalesce())
        return edge_masks
