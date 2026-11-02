"""Shared geometric-graph utilities, pooling, and integrated-gradients attribution."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_scatter import scatter, scatter_max, scatter_min


def as_tensor(value, *, dtype=None, device=None):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if dtype is not None:
        value = value.to(dtype=dtype)
    if device is not None:
        value = value.to(device)
    return value


def _unit_kernel_directions(dim, diagonals=False):
    """Axis unit vectors, plus coordinate-plane diagonals when requested."""
    directions = []
    eye = torch.eye(dim)
    for axis in eye:
        directions.extend([axis, -axis])
    if diagonals and dim >= 2:
        for i in range(dim):
            for j in range(i + 1, dim):
                for sign_i in (1.0, -1.0):
                    for sign_j in (1.0, -1.0):
                        vector = torch.zeros(dim)
                        vector[i] = sign_i
                        vector[j] = sign_j
                        directions.append(vector / torch.linalg.norm(vector))
    return directions


def build_kernel_points(dim, radius=1.0, radii=None, diagonals=False):
    """Kernel points at one or more radii, with optional face diagonals.

    The default (single radius, axes only) matches Thomas et al. 2019: origin
    plus ±e_i. Two radii plus diagonals cover ligand-bond and residue-contact
    lengths and 45° neighborhoods that otherwise collapse to the origin under
    Cosmo's nearest-kernel assignment.
    """
    if radii is None:
        shells = [float(radius)]
    else:
        shells = [float(item) for item in radii]
    points = [torch.zeros(dim)]
    for scale in shells:
        for direction in _unit_kernel_directions(dim, diagonals=diagonals):
            points.append(direction * scale)
    return torch.stack(points, dim=0)


def node2inst(data, pos):
    if hasattr(data, "vertex2molecule") and data.vertex2molecule is not None:
        return as_tensor(data.vertex2molecule, dtype=torch.long, device=pos.device)
    return torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)


def protein_ligand_radius_index(pos, batch, protein, ligand, radius):
    """Undirected protein–ligand pairs with distance ``<= radius``. No self-loops."""
    p_idx = protein.nonzero(as_tuple=False).view(-1)
    l_idx = ligand.nonzero(as_tuple=False).view(-1)
    empty = pos.new_zeros((2, 0), dtype=torch.long)
    if p_idx.numel() == 0 or l_idx.numel() == 0:
        return empty
    same = batch[p_idx][:, None] == batch[l_idx][None, :]
    dist = torch.cdist(pos[p_idx], pos[l_idx])
    keep = same & (dist <= radius) & (p_idx[:, None] != l_idx[None, :])
    if not bool(keep.any()):
        return empty
    pi, li = keep.nonzero(as_tuple=True)
    return torch.stack([p_idx[pi], l_idx[li]], dim=0)


def edge_index(data, pos, batch, k=8, protein=None, ligand=None, pl_radius=0.0):
    if hasattr(data, "molecule_edges") and data.molecule_edges is not None:
        edges = as_tensor(data.molecule_edges, dtype=torch.long, device=pos.device)
        if edges.numel() == 0:
            index = edges.reshape(2, 0)
        elif edges.ndim == 2 and edges.shape[0] == 2:
            index = edges
        else:
            index = edges.reshape(-1, 2).T.contiguous()
    else:
        n = pos.shape[0]
        k = min(k, max(n - 1, 1))
        index = gnn.knn_graph(pos, k=k, batch=batch)
    radius = float(pl_radius or 0.0)
    if radius > 0 and protein is not None and ligand is not None:
        extra = protein_ligand_radius_index(pos, batch, protein, ligand, radius)
        if extra.numel():
            index = extra if index.numel() == 0 else torch.cat([index, extra], dim=1)
    if index.numel() == 0:
        return index.reshape(2, 0)
    index = torch.cat([index, index.flip(0)], dim=1)
    return torch.unique(index, dim=1)


def extract_inputs(
    data,
    resolution="atom",
    dim=3,
    ligand_feature_offset=0,
    mask_ligand_features=False,
    pl_radius=0.0,
):
    """Pull node features, coordinates, edges, and batch from a Bioverse batch."""
    pos = as_tensor(getattr(data, f"{resolution}_pos"), dtype=torch.float32)
    if pos.shape[-1] > dim:
        pos = pos[:, :dim]
    key = f"{resolution}_features"
    if hasattr(data, key) and getattr(data, key) is not None:
        features = as_tensor(getattr(data, key), dtype=torch.float32, device=pos.device)
    else:
        features = torch.ones(pos.shape[0], 1, device=pos.device, dtype=pos.dtype)
    if mask_ligand_features:
        features = collapse_ligand_features(features, ligand_feature_offset)
    batch = node2inst(data, pos)
    protein, ligand = protein_ligand_masks(features, ligand_feature_offset)
    index = edge_index(
        data,
        pos,
        batch,
        protein=protein,
        ligand=ligand,
        pl_radius=pl_radius,
    )
    return features, pos, index, batch


def apply_node_dropout(data, p, *, training, resolution="atom"):
    """Drop nodes independently with probability ``p`` (training only).

    Each graph keeps at least one node. Edges with a dropped endpoint are
    removed and surviving endpoints are reindexed.
    """
    if (not training) or p is None or float(p) <= 0.0:
        return data
    p = float(p)
    pos_key = f"{resolution}_pos"
    if not hasattr(data, pos_key) or getattr(data, pos_key) is None:
        return data
    pos = as_tensor(getattr(data, pos_key))
    n = pos.shape[0]
    if n == 0:
        return data
    batch = node2inst(data, pos)
    keep = torch.rand(n, device=pos.device) >= p
    for graph_id in torch.unique(batch):
        mask = batch == graph_id
        if keep[mask].any():
            continue
        idx = mask.nonzero(as_tuple=False).view(-1)
        keep[idx[torch.randint(idx.numel(), (1,), device=pos.device)]] = True
    if bool(keep.all()):
        return data

    new_index = torch.full((n,), -1, dtype=torch.long, device=pos.device)
    new_index[keep] = torch.arange(int(keep.sum()), device=pos.device)

    out = copy.copy(data)
    setattr(out, pos_key, pos[keep])
    feat_key = f"{resolution}_features"
    if hasattr(data, feat_key) and getattr(data, feat_key) is not None:
        setattr(out, feat_key, as_tensor(getattr(data, feat_key))[keep])
    if hasattr(data, "vertex2molecule") and data.vertex2molecule is not None:
        out.vertex2molecule = as_tensor(data.vertex2molecule, dtype=torch.long)[keep]
    if hasattr(data, "molecule_edges") and data.molecule_edges is not None:
        edges = as_tensor(data.molecule_edges, dtype=torch.long, device=pos.device)
        if edges.numel() == 0:
            out.molecule_edges = edges.reshape(edges.shape[0], 0) if edges.ndim == 2 else edges
        else:
            if edges.ndim == 2 and edges.shape[0] == 2:
                src, dst = edges[0], edges[1]
                both = keep[src] & keep[dst]
                remapped = torch.stack([new_index[src[both]], new_index[dst[both]]], dim=0)
            else:
                pairs = edges.reshape(-1, 2)
                both = keep[pairs[:, 0]] & keep[pairs[:, 1]]
                remapped = torch.stack(
                    [new_index[pairs[both, 0]], new_index[pairs[both, 1]]], dim=1
                )
            out.molecule_edges = remapped
    return out


def pool_nodes(x, batch, pool="mean"):
    if pool == "max":
        return gnn.global_max_pool(x, batch)
    if pool == "mean":
        return gnn.global_mean_pool(x, batch)
    if pool == "sum":
        return gnn.global_add_pool(x, batch)
    raise ValueError(f"Unknown pool {pool!r}. Expected 'max', 'mean', or 'sum'.")


PROTEIN_READOUTS = ("protein_ligand", "protein_only")


def is_protein_readout(split_readout, node_level=False):
    return (not node_level) and split_readout in PROTEIN_READOUTS


def protein_ligand_masks(features, ligand_feature_offset):
    """Boolean protein/ligand masks from TokenizeByChainType one-hots.

    Protein tokens occupy ``features[:, :offset]``; ligand tokens the rest.
    """
    offset = int(ligand_feature_offset or 0)
    n = features.shape[0]
    if offset <= 0 or features.shape[-1] <= offset:
        protein = torch.ones(n, dtype=torch.bool, device=features.device)
        return protein, ~protein
    token = features.argmax(dim=-1)
    protein = token < offset
    return protein, ~protein


def collapse_ligand_features(features, ligand_feature_offset):
    """Replace ligand chemotype with a single token; keep protein/ligand split.

    Ligand rows become a one-hot at ``ligand_feature_offset`` (first ligand
    token). Protein rows and all coordinates are unchanged.
    """
    offset = int(ligand_feature_offset or 0)
    if offset <= 0 or features.shape[-1] <= offset:
        return features
    _protein, ligand = protein_ligand_masks(features, offset)
    if not bool(ligand.any()):
        return features
    masked = features.clone()
    masked[ligand] = 0
    masked[ligand, offset] = 1
    return masked


def masked_mean(x, batch, mask):
    """Per-graph mean of ``x`` over ``mask``. Empty graphs contribute zeros."""
    if batch.numel() == 0:
        return x.new_zeros(0, x.shape[-1])
    n_graph = int(batch.max().item()) + 1
    weights = mask.to(dtype=x.dtype).unsqueeze(-1)
    summed = scatter(x * weights, batch, dim=0, dim_size=n_graph, reduce="sum")
    counts = scatter(weights, batch, dim=0, dim_size=n_graph, reduce="sum").clamp(min=1.0)
    return summed / counts


def masked_max(x, batch, mask):
    """Per-graph, per-channel max of ``x`` over ``mask``. Empty graphs are 0."""
    if batch.numel() == 0:
        return x.new_zeros(0, x.shape[-1])
    n_graph = int(batch.max().item()) + 1
    fill = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
    values = scatter_max(fill, batch, dim=0, dim_size=n_graph)[0]
    return values.masked_fill(values <= torch.finfo(x.dtype).min / 2, 0.0)


def protein_ligand_readout(h, batch, features, ligand_feature_offset):
    """``[mean(protein); mean(ligand)]`` per graph, zeros if a part is missing."""
    protein, ligand = protein_ligand_masks(features, ligand_feature_offset)
    return torch.cat(
        [masked_mean(h, batch, protein), masked_mean(h, batch, ligand)],
        dim=-1,
    )


def masked_softmax_weights(logits, batch, mask):
    """Per-graph softmax over ``mask``; other nodes get weight 0."""
    if batch.numel() == 0:
        return logits.new_zeros(logits.shape[0])
    n_graph = int(batch.max().item()) + 1
    fill = torch.where(mask, logits, torch.full_like(logits, -1e9))
    shift = scatter(fill, batch, dim=0, dim_size=n_graph, reduce="max")[batch]
    exp = torch.where(mask, torch.exp(logits - shift), torch.zeros_like(logits))
    denom = scatter(exp, batch, dim=0, dim_size=n_graph, reduce="sum").clamp(min=1e-8)
    return exp / denom[batch]


def sparsemax_1d(logits):
    """Sparsemax on a 1-D vector (Martins & Astudillo 2016)."""
    n = logits.numel()
    if n == 0:
        return logits
    sorted_logits, _ = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(sorted_logits, dim=0) - 1
    support = torch.arange(1, n + 1, device=logits.device, dtype=logits.dtype)
    keep = sorted_logits > cumulative / support
    rho = int(keep.long().sum().clamp(min=1).item())
    tau = cumulative[rho - 1] / support[rho - 1]
    return torch.clamp(logits - tau, min=0.0)


def masked_sparsemax_weights(logits, batch, mask):
    """Per-graph sparsemax over ``mask``; other nodes get weight 0."""
    if batch.numel() == 0:
        return logits.new_zeros(logits.shape[0])
    weights = logits.new_zeros(logits.shape[0])
    n_graph = int(batch.max().item()) + 1
    for graph_id in range(n_graph):
        idx = (batch == graph_id) & mask
        if not bool(idx.any()):
            continue
        weights[idx] = sparsemax_1d(logits[idx])
    return weights


class ProteinLigandReadout(nn.Module):
    """Protein pool, optionally concatenated with mean(ligand)."""

    def __init__(
        self,
        hidden_dim,
        ligand_feature_offset=0,
        protein_pool="mean",
        softmax_temperature=1.0,
        include_ligand=True,
    ):
        super().__init__()
        self.ligand_feature_offset = int(ligand_feature_offset or 0)
        pool = str(protein_pool).lower()
        self.protein_pool = (
            pool if pool in ("mean", "softmax", "max", "sparsemax") else "mean"
        )
        self.temperature = max(float(softmax_temperature or 1.0), 1e-6)
        self.include_ligand = bool(include_ligand)
        self.gate = (
            nn.Linear(hidden_dim, 1)
            if self.protein_pool in ("softmax", "sparsemax")
            else None
        )
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def protein_weights(self, h, batch, features):
        protein, _ligand = protein_ligand_masks(features, self.ligand_feature_offset)
        if batch.numel() == 0:
            return h.new_zeros(h.shape[0])
        n_graph = int(batch.max().item()) + 1
        if self.protein_pool == "softmax":
            logits = self.gate(h).squeeze(-1) / self.temperature
            return masked_softmax_weights(logits, batch, protein)
        if self.protein_pool == "sparsemax":
            logits = self.gate(h).squeeze(-1) / self.temperature
            return masked_sparsemax_weights(logits, batch, protein)
        if self.protein_pool == "max":
            fill = h.masked_fill(~protein.unsqueeze(-1), torch.finfo(h.dtype).min)
            values, idx = scatter_max(fill, batch, dim=0, dim_size=n_graph)
            valid = values > torch.finfo(h.dtype).min / 2
            wins = h.new_zeros(h.shape[0])
            flat_idx = idx.reshape(-1)
            flat_ok = valid.reshape(-1)
            if bool(flat_ok.any()):
                wins.scatter_add_(
                    0,
                    flat_idx[flat_ok],
                    h.new_ones(int(flat_ok.sum())),
                )
            wins = wins * protein.to(dtype=h.dtype)
            counts = scatter(wins, batch, dim=0, dim_size=n_graph, reduce="sum").clamp(
                min=1.0
            )
            return wins / counts[batch]
        counts = scatter(
            protein.to(dtype=h.dtype), batch, dim=0, dim_size=n_graph, reduce="sum"
        ).clamp(min=1.0)
        return protein.to(dtype=h.dtype) / counts[batch]

    def forward(self, h, batch, features):
        protein, ligand = protein_ligand_masks(features, self.ligand_feature_offset)
        if self.protein_pool == "max":
            protein_h = masked_max(h, batch, protein)
        elif self.protein_pool in ("softmax", "sparsemax"):
            alpha = self.protein_weights(h, batch, features)
            n_graph = int(batch.max().item()) + 1 if batch.numel() else 0
            protein_h = scatter(
                h * alpha.unsqueeze(-1), batch, dim=0, dim_size=n_graph, reduce="sum"
            )
        else:
            protein_h = masked_mean(h, batch, protein)
        if not self.include_ligand:
            return protein_h
        return torch.cat([protein_h, masked_mean(h, batch, ligand)], dim=-1)


def reduce_scores(scores, logits, batch):
    """Turn completeness scores ``(n, C)`` into a per-node ranking vector."""
    if scores.ndim == 1:
        return scores
    if scores.shape[1] == 1:
        return scores.squeeze(-1)
    pred = logits.argmax(dim=-1)
    return scores[torch.arange(scores.shape[0], device=scores.device), pred[batch]]


def normalize_scores(scores, batch=None):
    """Map scores to ``[0, 1]`` for visualization (per graph when batched)."""
    scores = scores.detach().float()
    if scores.numel() == 0:
        return scores.cpu().numpy()
    if batch is None:
        lo = scores.min()
        hi = scores.max()
        return ((scores - lo) / (hi - lo).clamp(min=1e-8)).cpu().numpy()
    lo = scatter_min(scores, batch, dim=0)[0][batch]
    hi = scatter_max(scores, batch, dim=0)[0][batch]
    return ((scores - lo) / (hi - lo).clamp(min=1e-8)).cpu().numpy()


def _target_logit(logits, pred):
    if logits.ndim == 1:
        return logits.sum()
    if logits.shape[-1] == 1:
        return logits.sum()
    return logits[torch.arange(logits.shape[0], device=logits.device), pred].sum()


IG_INPUTS = "features"


def _ig_baselines(mode, features, pos, batch):
    if mode in (None, "zero"):
        return [torch.zeros_like(features), torch.zeros_like(pos)]
    if mode == "graph_mean":
        pos_base = torch.zeros_like(pos)
        feat_base = torch.zeros_like(features)
        for graph_id in torch.unique(batch):
            mask = batch == graph_id
            pos_base[mask] = pos[mask].mean(dim=0, keepdim=True)
            feat_base[mask] = features[mask].mean(dim=0, keepdim=True)
        return [feat_base, pos_base]
    raise ValueError(f"Unknown IG baseline mode {mode!r}")


def feature_baseline(mode, features, batch):
    """Feature-channel IG baseline; coordinates are not interpolated."""
    return _ig_baselines(mode, features, features.new_zeros(features.shape[0], 1), batch)[0]


def integrated_gradients(forward_fn, *tensors, n_steps=32, baselines=None):
    """IG of a scalar graph objective w.r.t. one or more inputs.

    ``forward_fn`` is called as ``forward_fn(*interpolated_tensors)``. BioLiP
    / enzyme ranking interpolates node features only and holds coordinates
    fixed so site maps are not a proximity detector.
    """
    tensors = [t.detach() for t in tensors]
    if baselines is None:
        baselines = [torch.zeros_like(t) for t in tensors]
    else:
        if len(baselines) != len(tensors):
            raise ValueError("baselines must match number of tensors")
        baselines = [b.detach().to(t.device).to(t.dtype) for b, t in zip(baselines, tensors)]
    deltas = [t - b for t, b in zip(tensors, baselines)]
    with torch.no_grad():
        logits0 = forward_fn(*tensors)
        if logits0.ndim == 2 and logits0.shape[-1] > 1:
            pred = logits0.argmax(dim=-1)
        else:
            pred = None

    grads = [torch.zeros_like(t) for t in tensors]
    alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=tensors[0].device)
    for alpha in alphas:
        xs = [(b + alpha * d).detach().requires_grad_(True) for b, d in zip(baselines, deltas)]
        logits = forward_fn(*xs)
        scalar = _target_logit(logits, pred)
        computed = torch.autograd.grad(
            scalar, xs, retain_graph=False, create_graph=False, allow_unused=True
        )
        grads = [
            g + (c if c is not None else 0) for g, c in zip(grads, computed)
        ]
    return [d * g / n_steps for d, g in zip(deltas, grads)]


class GeometricModel(nn.Module):
    """Shared encode / pool / IG attribution interface for graph baselines."""

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
        pool="mean",
        head="linear",
        ig_steps=32,
        split_readout=None,
        ligand_feature_offset=0,
        protein_pool="mean",
        softmax_temperature=1.0,
        mask_ligand_features=False,
        pl_radius=0.0,
        **_unused,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.radius = radius
        self.dim = dim
        self.resolution = resolution
        self.node_level = node_level
        self.pool = pool
        self.split_readout = None if split_readout in (None, "none", "") else split_readout
        self.ligand_feature_offset = int(ligand_feature_offset or 0)
        self.mask_ligand_features = bool(mask_ligand_features)
        self.pl_radius = float(pl_radius or 0.0)
        self.ig_steps = int(ig_steps)
        split_graph = is_protein_readout(self.split_readout, node_level)
        if split_graph:
            include_ligand = self.split_readout == "protein_ligand"
            self.head_type = "linear"
            self.head = nn.Linear(
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
            self.head_type = "identity" if head in (None, "identity") else "linear"
            self.graph_readout = None
            if self.head_type == "linear":
                self.head = nn.Linear(hidden_dim, out_channels)
            else:
                self.head = nn.Identity()

    def node_features(self, features, pos, edge_index):
        """Return per-node embeddings of shape ``(n, hidden_dim)`` or ``(n, out)``."""
        raise NotImplementedError

    def encode(self, features, pos, edge_index, batch, part_features=None):
        h = self.node_features(features, pos, edge_index)
        if self.node_level:
            return self.head(h)
        parts = features if part_features is None else part_features
        if is_protein_readout(self.split_readout):
            return self.head(self.graph_readout(h, batch, parts))
        if self.head_type == "identity":
            return pool_nodes(h, batch, self.pool)
        return self.head(pool_nodes(h, batch, self.pool))

    def _args(self, data):
        return extract_inputs(
            data,
            self.resolution,
            self.dim,
            ligand_feature_offset=self.ligand_feature_offset,
            mask_ligand_features=self.mask_ligand_features,
            pl_radius=self.pl_radius,
        )

    def forward(self, data):
        features, pos, edge_index, batch = self._args(data)
        return self.encode(features, pos, edge_index, batch)

    def attribution(
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
                igs = integrated_gradients(
                    lambda x: self.encode(
                        x, pos, edge_index, batch, part_features=features
                    ),
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
