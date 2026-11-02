"""Rigid Kernel Point Convolution baseline (Thomas et al. 2019)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import add_self_loops
from torch_scatter import scatter

from .common import GeometricModel, build_kernel_points


def default_kernel_points(dim, radius=1.0, radii=None, diagonals=False):
    return build_kernel_points(dim, radius=radius, radii=radii, diagonals=diagonals)


def _min_kernel_spacing(kernel_points):
    if kernel_points.shape[0] <= 1:
        return 1.0
    dist = torch.cdist(kernel_points.float(), kernel_points.float())
    inf = dist.new_full((), dist.max().clamp(min=1.0) * 10.0)
    dist = dist + torch.eye(kernel_points.shape[0], device=dist.device, dtype=dist.dtype) * inf
    return float(dist.min().clamp(min=1e-4))


def _is_gaussian_style(style):
    return str(style).lower() in {"gaussian", "linear", "2019", "kpconv"}


class KPConv(nn.Module):
    """Kernel-point convolution in a rigid global frame.

    Same nearest-kernel filter as Cosmo ``KernelPointCosmo`` (KPConvX). The
    intended Cosmo/KPConv gap is the coordinate frame: Cosmo evaluates the
    kernels in each lifted local frame, this layer uses global ``pos[src]-pos[dst]``.
    """

    def __init__(self, in_channels, out_channels, kernel_points, style="nearest"):
        super().__init__()
        self.style = "gaussian" if _is_gaussian_style(style) else "nearest"
        points = kernel_points.float()
        self.register_buffer("kernel_points", points)
        self.weight = nn.Parameter(torch.empty(points.shape[0], in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        if self.style == "gaussian":
            self.register_buffer("sigma", torch.tensor(_min_kernel_spacing(points)))

    def forward(self, x, pos, edge_index):
        if edge_index.numel() == 0 or x.shape[0] == 0:
            return x.new_zeros(x.shape[0], self.weight.shape[-1]) + self.bias
        src = edge_index[0].long()
        dst = edge_index[1].long()
        rel = pos[src] - pos[dst]
        xs = x[src]
        if self.style == "gaussian":
            influence = (1.0 - torch.cdist(rel, self.kernel_points) / self.sigma.clamp(min=1e-6)).clamp(
                min=0.0
            )
            msg = torch.einsum("ek,kio,ei->eo", influence, self.weight, xs)
        else:
            with torch.no_grad():
                nn_idx = torch.cdist(rel, self.kernel_points).argmin(dim=1)
            msg = xs.new_zeros(xs.shape[0], self.weight.shape[-1])
            for k in range(self.weight.shape[0]):
                sel = nn_idx == k
                if bool(sel.any()):
                    msg[sel] = xs[sel] @ self.weight[k]
        out = scatter(msg, dst, dim=0, dim_size=x.shape[0], reduce="mean")
        return out + self.bias


class KpconvModel(GeometricModel):
    """KPConv stack with the same depth/width/pooling as Cosmo."""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=128,
        num_layers=4,
        radius=1.0,
        dim=3,
        **kwargs,
    ):
        kwargs.setdefault("head", "identity")
        super().__init__(
            in_channels,
            out_channels,
            hidden_dim,
            num_layers,
            radius=radius,
            dim=dim,
            **kwargs,
        )
        radii = kwargs.get("kernel_radii")
        kernel_points = default_kernel_points(
            dim,
            radius=radius,
            radii=radii,
            diagonals=bool(kwargs.get("kernel_diagonals", False)),
        )
        self.residual = bool(kwargs.get("kpconv_residual", False))
        self.style = str(kwargs.get("kpconv_style", "nearest")).lower()
        # Self-loops always hit the origin kernel in the global frame, which
        # turns nearest KPConv into a rotation-invariant GCN. Cosmo has no
        # extra self-loops either; BioLiP residual stacks keep them.
        self.use_self_loops = self.residual
        layer_kwargs = dict(style=self.style)
        if self.residual:
            self.input_proj = (
                nn.Identity()
                if in_channels == hidden_dim
                else nn.Linear(in_channels, hidden_dim)
            )
            self.layers = nn.ModuleList(
                [
                    KPConv(hidden_dim, hidden_dim, kernel_points, **layer_kwargs)
                    for _ in range(num_layers)
                ]
            )
        else:
            self.input_proj = None
            last = out_channels if self.head_type == "identity" else hidden_dim
            channels = [in_channels] + [hidden_dim] * (num_layers - 1) + [last]
            self.layers = nn.ModuleList(
                [
                    KPConv(channels[i], channels[i + 1], kernel_points, **layer_kwargs)
                    for i in range(num_layers)
                ]
            )

    def node_features(self, features, pos, edge_index):
        if self.use_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=features.shape[0])
        if self.residual:
            x = self.input_proj(features)
            for layer in self.layers:
                x = x + F.relu(layer(x, pos, edge_index))
            return x
        x = features
        for i, layer in enumerate(self.layers):
            x = layer(x, pos, edge_index)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x
