"""Non-equivariant GNN baseline (GCN stack)."""

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from .common import GeometricModel


class GnnModel(GeometricModel):
    """GCN layers with the same depth/width/pooling as Cosmo.

    Topology only: node labels and the provided graph, no coordinates.
    That solves scramble (and rotation, whose test graphs are rotated copies
    of the same combinatorial ship) and cannot see chirality or mast tilt.

    ``gnn_style: residual`` projects to ``hidden_dim`` then uses LayerNorm
    residual GCNs (BioLiP). ``plain`` keeps the original ReLU stack.
    """

    def __init__(self, in_channels, out_channels, hidden_dim=128, num_layers=4, **kwargs):
        kwargs.setdefault("head", "identity")
        super().__init__(in_channels, out_channels, hidden_dim, num_layers, **kwargs)
        self.residual = str(kwargs.get("gnn_style", "plain")).lower() in {
            "residual",
            "norm",
        }
        if self.residual:
            self.input_proj = nn.Linear(in_channels, hidden_dim)
            self.layers = nn.ModuleList(
                [gnn.GCNConv(hidden_dim, hidden_dim, add_self_loops=True) for _ in range(num_layers)]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        else:
            last = out_channels if self.head_type == "identity" else hidden_dim
            channels = [in_channels] + [hidden_dim] * (num_layers - 1) + [last]
            self.input_proj = None
            self.norms = None
            self.layers = nn.ModuleList(
                [
                    gnn.GCNConv(channels[i], channels[i + 1], add_self_loops=True)
                    for i in range(num_layers)
                ]
            )

    def node_features(self, features, pos, edge_index):
        if self.residual:
            x = self.input_proj(features)
            for conv, norm in zip(self.layers, self.norms):
                x = norm(x + F.relu(conv(x, edge_index)))
            return x
        x = features
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x
