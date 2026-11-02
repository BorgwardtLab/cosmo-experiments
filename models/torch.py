"""Bioverse-compatible model wrapper dispatching to architecture-specific trunks."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import apply_node_dropout, pool_nodes
from .cosmo import CosmoModel
from .egnn import EgnnModel
from .gnn import GnnModel
from .kpconv import KpconvModel
from .se3_transformer import Se3TransformerModel

_ARCHITECTURES = {
    "gnn": GnnModel,
    "kpconv": KpconvModel,
    "cosmo": CosmoModel,
    "egnn": EgnnModel,
    "se3_transformer": Se3TransformerModel,
}


def _make_trunk(arch, **kwargs):
    try:
        cls = _ARCHITECTURES[arch]
    except KeyError as exc:
        raise ValueError(f"Unknown arch: {arch}") from exc
    return cls(**kwargs)


class Model(nn.Module):
    """Uniform interface expected by the Bioverse trainer."""

    def __init__(
        self,
        arch="cosmo",
        node_level=True,
        in_channels=1,
        out_channels=1,
        loss="cross_entropy",
        learning_rate=0.0003,
        resolution="atom",
        hidden_dim=128,
        num_layers=4,
        radius=1.0,
        dim=3,
        filter="neural_field",
        pool="mean",
        cosmo_pool="max",
        head="identity",
        activation=None,
        field_channels=32,
        field_layers=3,
        field_dropout=0.0,
        attribute=False,
        minimum_angle=0.0,
        ig_steps=32,
        node_dropout=0.0,
        weight_decay=5e-5,
        kernel_radii=None,
        kernel_diagonals=False,
        split_readout=None,
        ligand_feature_offset=0,
        egnn_style="toy",
        protein_pool="mean",
        softmax_temperature=1.0,
        mask_ligand_features=False,
        cosmo_pool_to="node",
        gnn_style="plain",
        kpconv_residual=False,
        kpconv_style="nearest",
        se3_zero_radial=True,
        pl_radius=0.0,
    ):
        super().__init__()
        self.node_level = node_level
        self.resolution = resolution
        self.arch = arch
        self.score_attribution = bool(attribute)
        self.node_dropout = float(node_dropout)
        self.weight_decay = float(weight_decay)
        trunk_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            radius=radius,
            dim=dim,
            resolution=resolution,
            node_level=node_level,
            pool=pool,
            cosmo_pool=cosmo_pool,
            head=head,
            ig_steps=ig_steps,
            kernel_radii=kernel_radii,
            kernel_diagonals=kernel_diagonals,
            split_readout=split_readout,
            ligand_feature_offset=ligand_feature_offset,
            egnn_style=egnn_style,
            protein_pool=protein_pool,
            softmax_temperature=softmax_temperature,
            mask_ligand_features=mask_ligand_features,
            cosmo_pool_to=cosmo_pool_to,
            gnn_style=gnn_style,
            kpconv_residual=kpconv_residual,
            kpconv_style=kpconv_style,
            se3_zero_radial=se3_zero_radial,
            pl_radius=pl_radius,
        )
        if arch == "cosmo":
            trunk_kwargs.update(
                filter=filter,
                activation=activation,
                field_channels=field_channels,
                field_layers=field_layers,
                field_dropout=field_dropout,
                minimum_angle=minimum_angle,
            )
        self.trunk = _make_trunk(arch, **trunk_kwargs)
        self.loss = getattr(F, loss)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=self.weight_decay
        )

    def _predict(self, data):
        data = apply_node_dropout(
            data,
            self.node_dropout,
            training=self.training,
            resolution=self.resolution,
        )
        output = self.trunk(data)
        if not self.node_level and not getattr(self.trunk, "handles_pooling", False):
            output = pool_nodes(output, data.vertex2molecule, "mean")
        if output.ndim == 2 and output.shape[1] == 1:
            output = output.squeeze(1)
        return output, data

    def train_step(self, Xy, data):
        self._dump_params_once()
        output, data = self._predict(data)
        y = data.y.long() if self.loss == F.cross_entropy else data.y.float()
        if self.loss == F.cross_entropy and y.ndim > 1:
            y = y.reshape(-1)
        loss = self.loss(output, y)
        return loss, output

    def _dump_params_once(self):
        if getattr(self, "_dumped_params", False):
            return
        self._dumped_params = True
        name = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("name=")), None)
        if not name:
            return
        path = Path("results") / name / "model_params.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"trainable_parameters: {self.count_trainable_parameters()}\n")

    def eval_step(self, Xy, data):
        if self.score_attribution:
            scores = self.attribution(data, normalize=False)
            if not torch.is_tensor(scores):
                device = data.y.device if torch.is_tensor(getattr(data, "y", None)) else next(
                    self.parameters()
                ).device
                scores = torch.as_tensor(scores, dtype=torch.float32, device=device)
            return scores.new_zeros(()), scores
        return self.train_step(Xy, data)

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def attribution(self, data, method="auto", **kwargs):
        """Return per-node attribution scores for visualization or ranking.

        ``method="ig"`` uses feature-only integrated gradients when the trunk
        exposes ``ig_attribution`` (Cosmo completeness is the ``auto`` path).
        """
        if method == "ig":
            ig = getattr(self.trunk, "ig_attribution", None)
            if ig is not None:
                return ig(data, **kwargs)
        return self.trunk.attribution(data, **kwargs)
