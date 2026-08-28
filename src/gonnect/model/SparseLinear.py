import math

import torch
import torch.nn as nn


class SparseLinear(nn.Module):
    """Linear layer holding only the connections the ontology allows.

    Where DenseBICoder stores a full in x out weight matrix and resets the disallowed
    entries after every optimization step, this layer never materializes them. The
    parameters are a flat vector of the learnable edge values, so the optimizer's
    parameter set -- and its momentum state -- contains exactly the weights the ontology
    permits.

    Two groups of values are fixed by the architecture rather than learned, matching
    DenseBICoder.mask_weights():
      - edges pointing at a ProxyTerm are held at 1, so a proxy passes its inputs on
      - biases of ProxyTerms are held at 0
    Both are kept as buffers outside the parameter set, so they are structurally
    constant instead of being updated and then overwritten.
    """

    def __init__(self, in_features, out_features, mask: torch.Tensor, proxy_mask: torch.Tensor = None,
                 bias=True, device=None, dtype=None, protocol="coo"):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.edge_mask = mask
        if protocol not in ("coo", "csr"):
            raise Exception(f"Unknown sparsity protocol: {protocol}")
        self.protocol = protocol

        # Edge coordinates, as (2, nnz) row/column indices
        if mask.is_sparse:
            indices = mask.coalesce().indices()
        else:
            indices = mask.nonzero().t()
        rows = indices[0]
        nnz = indices.shape[1]

        # Split the edges into those the architecture fixes at 1 (pointing at a proxy
        # term) and those that are learnable
        if proxy_mask is None:
            proxy_nodes = torch.zeros(out_features, dtype=torch.bool)
        else:
            proxy_nodes = proxy_mask.reshape(-1).bool()
        proxy_edge = proxy_nodes.to(rows.device)[rows]

        self.register_buffer("indices", indices)
        self.register_buffer("proxy_nodes", proxy_nodes)
        self.register_buffer("live_edges", torch.nonzero(~proxy_edge).reshape(-1))
        # 1 at every fixed proxy edge, 0 elsewhere; learnable values are added on top
        self.register_buffer("fixed_values", proxy_edge.to(**factory_kwargs))
        self.weight = nn.Parameter(torch.empty(int((~proxy_edge).sum()), **factory_kwargs))

        if bias:
            self.register_buffer("live_biases", torch.nonzero(~proxy_nodes).reshape(-1))
            self.bias = nn.Parameter(torch.empty(int((~proxy_nodes).sum()), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
            self.register_buffer("live_biases", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match nn.Linear: Kaiming uniform with a = sqrt(5) over the full fan-in, so a
        sparse layer is initialized from the same distribution as its dense counterpart."""
        bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def edge_values(self) -> torch.Tensor:
        """Full nnz-length value vector: learnable weights placed among the fixed ones."""
        return self.fixed_values.index_add(0, self.live_edges, self.weight)

    def bias_vector(self) -> torch.Tensor:
        """Full out_features bias, zero at every ProxyTerm."""
        full = torch.zeros(self.out_features, dtype=self.weight.dtype, device=self.weight.device)
        return full.index_add(0, self.live_biases, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.sparse_coo_tensor(self.indices, self.edge_values(),
                                    (self.out_features, self.in_features))
        if self.protocol == "csr":
            w = w.to_sparse_csr()
        y = torch.sparse.mm(w, x.t()).t()
        if self.bias is not None:
            y = y + self.bias_vector()
        return y

    def dense_weight(self) -> torch.Tensor:
        """Materialize the equivalent dense weight matrix, for comparison against
        DenseBICoder layers. Not used during training."""
        return torch.sparse_coo_tensor(self.indices, self.edge_values().detach(),
                                       (self.out_features, self.in_features)).to_dense()

    def load_from_dense(self, weight: torch.Tensor, bias: torch.Tensor = None) -> None:
        """Copy the learnable entries out of an equivalent masked-dense layer, so both
        implementations can be started from identical weights."""
        with torch.no_grad():
            rows, cols = self.indices[0], self.indices[1]
            self.weight.copy_(weight[rows, cols][self.live_edges])
            if self.bias is not None and bias is not None:
                self.bias.copy_(bias[self.live_biases])

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"edges={self.indices.shape[1]}, learnable={self.weight.numel()}, "
                f"bias={self.bias is not None}")
