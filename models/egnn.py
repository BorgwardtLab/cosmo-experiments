"""E(n)-equivariant GNN baseline (Satorras et al. 2021).

``egnn_style: toy`` keeps the ship-cloud layer (RBF distances, LayerNorm,
active coordinate updates). ``egnn_style: standard`` follows the official
EGCL (squared distance, sum messages, mean coordinate updates, residual,
no LayerNorm / tanh).
"""

import torch
import torch.nn as nn
from torch_scatter import scatter

from .common import GeometricModel


def _small_last_linear(module: nn.Sequential, gain=0.001) -> None:
    last = module[-1]
    nn.init.xavier_uniform_(last.weight, gain=gain)
    if last.bias is not None:
        nn.init.zeros_(last.bias)


class ToyEGNNConv(nn.Module):
    """EGCL with coordinate updates and additive RBF distances.

    Nodes move along neighbor directions so later layers can re-measure
    distances after the cloud deforms. The coordinate MLP uses the official
    small Xavier init (no tanh) so updates start tiny but are active.
    """

    def __init__(self, hidden_dim=128, n_rbf=16, rbf_max=4.0):
        super().__init__()
        mu = torch.linspace(0.0, rbf_max, n_rbf)
        self.register_buffer("rbf_mu", mu)
        self.rbf_sigma = rbf_max / n_rbf
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.phi_r = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        _small_last_linear(self.phi_x)
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, pos, features, edge_index):
        src, dst = edge_index
        rel = pos[dst] - pos[src]
        dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        rel_hat = rel / (dist + 1.0)
        rbf = torch.exp(-((dist - self.rbf_mu) / self.rbf_sigma).pow(2))
        messages = self.phi_e(torch.cat([features[dst], features[src]], dim=-1)) + self.phi_r(
            rbf
        )
        coord_step = rel_hat * self.phi_x(messages)
        pos = pos + scatter(coord_step, dst, dim=0, dim_size=pos.size(0), reduce="mean")
        aggregated = scatter(
            messages, dst, dim=0, dim_size=features.size(0), reduce="mean"
        )
        features = self.norm(
            features + self.phi_h(torch.cat([features, aggregated], dim=-1))
        )
        return features, pos


class EGNNConv(nn.Module):
    """Official-style EGCL (Satorras et al. 2021, ``egnn_clean``).

    Squared Euclidean distance on the edge MLP, sum-aggregated messages,
    mean coordinate updates, residual features. The coordinate MLP uses
    Xavier gain 0.001 and no tanh, matching the published QM9 / n-body
    defaults rather than the toy-cloud damping.
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        _small_last_linear(self.coord_mlp)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, pos, features, edge_index):
        src, dst = edge_index
        rel = pos[dst] - pos[src]
        radial = rel.pow(2).sum(dim=-1, keepdim=True)
        messages = self.edge_mlp(
            torch.cat([features[dst], features[src], radial], dim=-1)
        )
        pos = pos + scatter(
            rel * self.coord_mlp(messages),
            dst,
            dim=0,
            dim_size=pos.size(0),
            reduce="mean",
        )
        aggregated = scatter(
            messages, dst, dim=0, dim_size=features.size(0), reduce="sum"
        )
        features = features + self.node_mlp(torch.cat([features, aggregated], dim=-1))
        return features, pos


class EgnnModel(GeometricModel):
    """EGNN layers with the same depth/width/pooling as Cosmo."""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=128,
        num_layers=4,
        **kwargs,
    ):
        kwargs["head"] = "linear"
        style = str(kwargs.get("egnn_style", "toy")).lower()
        super().__init__(in_channels, out_channels, hidden_dim, num_layers, **kwargs)
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        if style in ("standard", "official", "biolip"):
            self.layers = nn.ModuleList(
                [EGNNConv(hidden_dim) for _ in range(num_layers)]
            )
            self.readout_norm = nn.Identity()
        else:
            rbf_max = max(4.0, 2.0 * float(self.radius))
            self.layers = nn.ModuleList(
                [ToyEGNNConv(hidden_dim, rbf_max=rbf_max) for _ in range(num_layers)]
            )
            self.readout_norm = nn.LayerNorm(hidden_dim)

    def node_features(self, features, pos, edge_index):
        h = self.input_proj(features)
        for layer in self.layers:
            h, pos = layer(pos, h, edge_index)
        return self.readout_norm(h)
