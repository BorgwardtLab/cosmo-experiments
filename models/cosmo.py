"""Cosmo trunk wrapping ``cosmic.CosmoModel`` for Bioverse experiments."""

import numpy as np
import torch
import torch.nn as nn
from cosmic.model import CosmoModel as CosmicCosmoModel

from .common import (
    ProteinLigandReadout,
    build_kernel_points,
    extract_inputs,
    feature_baseline,
    integrated_gradients,
    is_protein_readout,
    normalize_scores,
    pool_nodes,
    reduce_scores,
)


def _stabilize_neural_fields(net):
    """Scale neural-field output maps so a linear Cosmo stack starts near zero.

    Identity activations keep completeness attribution exact, but an unscaled
    4-layer linear stack explodes. Xavier on the last field linear, then ×0.1,
    keeps early logits small without changing the architecture.
    """
    for layer in net.cosmo_layers:
        field = getattr(layer, "neural_field", None)
        if field is None:
            continue
        last_linear = None
        for module in reversed(list(field)):
            if isinstance(module, nn.Linear):
                last_linear = module
                break
        if last_linear is None:
            continue
        nn.init.xavier_uniform_(last_linear.weight)
        last_linear.weight.data.mul_(0.1)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)


def _stabilize_kernel_points(net):
    """Scale the last kernel-point map so a linear KP stack starts small.

    KernelPointCosmo has no per-layer bias. Xavier weights plus scatter-sum
    over a 4-layer product produce huge logits and the train-loss spikes seen
    on BioLiP. Same ×0.1 as the neural-field maps; completeness is unchanged.
    """
    for layer in reversed(list(net.cosmo_layers)):
        weight = getattr(layer, "w", None)
        if weight is None:
            continue
        weight.data.mul_(0.1)
        return


class CosmoModel(nn.Module):
    """Bioverse adapter around the cosmic-torch Cosmo network and attribution."""

    handles_pooling = True

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=128,
        num_layers=4,
        radius=1.0,
        dim=3,
        resolution="atom",
        node_level=True,
        filter="neural_field",
        pool="mean",
        cosmo_pool="max",
        head="identity",
        activation=None,
        field_channels=32,
        field_layers=3,
        field_dropout=0.0,
        minimum_angle=0.0,
        ig_steps=32,
        kernel_radii=None,
        kernel_diagonals=False,
        split_readout=None,
        ligand_feature_offset=0,
        protein_pool="mean",
        softmax_temperature=1.0,
        mask_ligand_features=False,
        cosmo_pool_to="node",
        pl_radius=0.0,
        **_unused,
    ):
        super().__init__()
        self.resolution = resolution
        self.dim = dim
        self.node_level = node_level
        self.graph_pool = pool
        self.split_readout = None if split_readout in (None, "none", "") else split_readout
        self.ligand_feature_offset = int(ligand_feature_offset or 0)
        self.mask_ligand_features = bool(mask_ligand_features)
        self.pl_radius = float(pl_radius or 0.0)
        self.ig_steps = int(ig_steps)
        self.cosmo_pool_to = "graph" if str(cosmo_pool_to).lower() == "graph" else "node"
        self.graph_cosmo = (not node_level) and self.cosmo_pool_to == "graph"
        split_graph = (not self.graph_cosmo) and is_protein_readout(
            self.split_readout, node_level
        )
        include_ligand = self.split_readout == "protein_ligand"
        cosmic_out = hidden_dim if split_graph else out_channels
        cosmic_head = "linear" if self.graph_cosmo else head
        kernel_points = build_kernel_points(
            dim,
            radius=radius,
            radii=kernel_radii,
            diagonals=bool(kernel_diagonals),
        )
        # Graph Cosmo pools lifted frames onto the instance so completeness
        # scores sum to the logit. Otherwise pool frames onto nodes, then
        # protein mean / softmax / max (ligand masked when protein_only).
        self.net = CosmicCosmoModel(
            in_channels=in_channels,
            out_channels=cosmic_out,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dim=dim,
            filter=filter,
            radius=radius,
            activation=activation,
            pool=cosmo_pool,
            pool_to="graph" if self.graph_cosmo else "node",
            head=cosmic_head,
            kernel_points=kernel_points,
            field_channels=field_channels,
            field_layers=field_layers,
            field_dropout=field_dropout,
            minimum_angle=minimum_angle,
        )
        _stabilize_neural_fields(self.net)
        _stabilize_kernel_points(self.net)
        if split_graph:
            self.graph_head = nn.Linear(
                (2 * hidden_dim) if include_ligand else hidden_dim, out_channels
            )
            self.graph_readout = ProteinLigandReadout(
                hidden_dim,
                ligand_feature_offset=self.ligand_feature_offset,
                protein_pool=protein_pool,
                softmax_temperature=softmax_temperature,
                include_ligand=include_ligand,
            )
        else:
            self.graph_head = nn.Identity()
            self.graph_readout = None
        # Graph-level offset so a linear Cosmo does not have to manufacture
        # the target mean through the deep weight product. Split readout and
        # graph Cosmo already have a biased linear head.
        self.output_bias = (
            None
            if node_level or split_graph or self.graph_cosmo
            else nn.Parameter(torch.zeros(out_channels))
        )

    def _args(self, data):
        return extract_inputs(
            data,
            self.resolution,
            self.dim,
            ligand_feature_offset=self.ligand_feature_offset,
            mask_ligand_features=self.mask_ligand_features,
            pl_radius=self.pl_radius,
        )

    def _readout(self, node_h, batch, features):
        if self.node_level:
            return node_h
        if self.graph_readout is not None:
            return self.graph_head(self.graph_readout(node_h, batch, features))
        return pool_nodes(node_h, batch, self.graph_pool)

    def _add_output_bias(self, logits):
        if self.output_bias is None:
            return logits
        if logits.ndim == 1:
            return logits + self.output_bias.reshape(-1)[0]
        return logits + self.output_bias

    def forward(self, data):
        features, pos, edge_index, batch = self._args(data)
        hidden = self.net(features, pos, edge_index, batch)
        if self.graph_cosmo:
            return hidden
        return self._add_output_bias(self._readout(hidden, batch, features))

    def attribute_tensors(self, data):
        return self.net.attribute(*self._args(data))

    def attribution(self, data, normalize=False, **_unused):
        """Per-node scores from Cosmo completeness attribution.

        Graph Cosmo scores complete to the predicted logit. With a node-level
        Cosmo plus protein readout, ranking is the protein-head projection
        times the pooling weight (mean, softmax, sparsemax, or max-channel
        winners).
        Pass ``normalize=True`` for visualization in ``[0, 1]``.
        """
        features, pos, edge_index, batch = self._args(data)
        logits, scores = self.net.attribute(features, pos, edge_index, batch)
        if self.graph_readout is not None and not self.node_level and scores.ndim == 2:
            hidden = scores.shape[-1]
            weight = self.graph_head.weight
            if self.split_readout == "protein_ligand":
                weight = weight[:, :hidden]
            if weight.shape[0] == 1:
                projected = scores @ weight.reshape(-1)
            else:
                pred = logits.argmax(dim=-1)
                projected = (scores * weight[pred[batch]]).sum(dim=-1)
            alpha = self.graph_readout.protein_weights(logits, batch, features)
            scores_1d = projected * alpha
        else:
            scores_1d = reduce_scores(scores, logits, batch)
        if normalize:
            return normalize_scores(scores_1d, batch)
        return scores_1d.detach().cpu().numpy().astype(np.float32)

    def ig_attribution(
        self,
        data,
        normalize=False,
        ig_steps=None,
        ig_baseline="zero",
        **_unused,
    ):
        """Per-node IG of the predicted logit w.r.t. node features only."""
        features, pos, edge_index, batch = self._args(data)
        steps = self.ig_steps if ig_steps is None else int(ig_steps)
        was_training = self.training
        self.eval()
        try:
            with torch.enable_grad():
                baseline = feature_baseline(ig_baseline, features, batch)

                def _forward(x):
                    hidden = self.net(x, pos, edge_index, batch)
                    if self.graph_cosmo:
                        return hidden
                    return self._add_output_bias(
                        self._readout(hidden, batch, features)
                    )

                igs = integrated_gradients(
                    _forward,
                    features,
                    n_steps=steps,
                    baselines=[baseline],
                )
        finally:
            self.train(was_training)
        scores = igs[0].sum(dim=-1)
        if normalize:
            return normalize_scores(scores, batch)
        return scores.detach().cpu().numpy().astype(np.float32)
