"""SE(3)-Transformer baseline (Fuchs et al. 2020) via e3nn tensor-product convolution."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import Activation, FullyConnectedNet
from torch_scatter import scatter

from .common import GeometricModel


def _pad_pos3(pos):
    if pos.shape[-1] == 3:
        return pos
    zeros = pos.new_zeros(pos.shape[0], 3 - pos.shape[-1])
    return torch.cat([pos, zeros], dim=-1)


def _zero_last_radial_layer(radial: nn.Module) -> None:
    last = list(radial.children())[-1]
    last.weight.data.zero_()


class EquivariantNorm(nn.Module):
    """LayerNorm on even scalars; RMS-normalize higher-order irreps per node."""

    def __init__(self, irreps, eps=1e-5):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        n_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        self.norm = nn.LayerNorm(n_scalar) if n_scalar else None

    def forward(self, x):
        pieces = []
        ix = 0
        for mul, ir in self.irreps:
            field = x[:, ix : ix + mul * ir.dim]
            ix += mul * ir.dim
            if ir.l == 0 and ir.p == 1:
                field = self.norm(field) if self.norm is not None else field
            else:
                field = field.reshape(-1, mul, ir.dim)
                rms = field.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
                field = (field / rms).reshape(x.shape[0], -1)
            pieces.append(field)
        return torch.cat(pieces, dim=-1)


class SE3TransformerLayer(nn.Module):
    """SE(3)-equivariant TFN convolution with residual, SiLU, and RBF radial.

    Mean-aggregated TFN messages keep the Fuchs tensor-field core. Attention
    and per-edge tensor-product weights both destabilized enzyme training, so
    this layer uses a shared kernel gated by a distance RBF. Radial weights
    are tanh-bounded around a unit bias so geometry is visible from step 0.
    """

    def __init__(
        self, irreps_hidden, lmax=1, radial_dim=64, rbf_max=4.0, zero_radial=True
    ):
        super().__init__()
        self.irreps = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.n_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps,
            self.irreps_sh,
            self.irreps,
            irrep_normalization="component",
            path_normalization="element",
        )
        self.n_rbf = 16
        self.register_buffer("rbf_mu", torch.linspace(0.0, rbf_max, self.n_rbf))
        self.rbf_sigma = rbf_max / self.n_rbf
        self.radial = FullyConnectedNet([self.n_rbf, radial_dim, 1], F.silu)
        if zero_radial:
            _zero_last_radial_layer(self.radial)
        self.radial_bias = nn.Parameter(torch.tensor(1.0))
        self.rbf_lin = nn.Linear(self.n_rbf, self.n_scalar)
        self.norm = EquivariantNorm(self.irreps)
        self.act = Activation(
            self.irreps,
            [F.silu if ir.l == 0 else None for _, ir in self.irreps],
        )

    def forward(self, x, pos, edge_index):
        src, dst = edge_index
        rel = pos[src] - pos[dst]
        dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sh = o3.spherical_harmonics(
            self.irreps_sh, rel, normalize=True, normalization="component"
        )
        rbf = torch.exp(-((dist - self.rbf_mu) / self.rbf_sigma).pow(2))
        radial = torch.tanh(self.radial(rbf)) + self.radial_bias
        msg = self.tp(x[src], sh) * radial
        out = scatter(msg, dst, dim=0, dim_size=x.shape[0], reduce="mean")
        rbf_n = scatter(rbf, dst, dim=0, dim_size=x.shape[0], reduce="mean")
        out = torch.cat(
            [out[:, : self.n_scalar] + self.rbf_lin(rbf_n), out[:, self.n_scalar :]],
            dim=-1,
        )
        return self.act(self.norm(x + out))


class Se3TransformerModel(GeometricModel):
    """SE(3)-Transformer with the same depth/width/pooling as Cosmo.

    Planar (2D) inputs are lifted to R^3 with a zero z-coordinate so the same
    SE(3) layers apply to both toy ships and enzyme structures.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=128,
        num_layers=4,
        **kwargs,
    ):
        kwargs["head"] = "linear"
        super().__init__(in_channels, out_channels, hidden_dim, num_layers, **kwargs)
        n_vec = max(hidden_dim // 4, 8)
        self.irreps_hidden = o3.Irreps(f"{hidden_dim}x0e + {n_vec}x1o")
        irreps_in = o3.Irreps(f"{in_channels}x0e")
        self.input_proj = o3.Linear(irreps_in, self.irreps_hidden)
        rbf_max = max(4.0, 2.0 * float(self.radius))
        zero_radial = bool(kwargs.get("se3_zero_radial", True))
        self.layers = nn.ModuleList(
            [
                SE3TransformerLayer(
                    self.irreps_hidden, rbf_max=rbf_max, zero_radial=zero_radial
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = o3.Linear(self.irreps_hidden, o3.Irreps(f"{hidden_dim}x0e"))
        self.readout_norm = nn.LayerNorm(hidden_dim)

    def node_features(self, features, pos, edge_index):
        pos = _pad_pos3(pos)
        x = self.input_proj(features)
        for layer in self.layers:
            x = layer(x, pos, edge_index)
        return self.readout_norm(self.readout(x))
