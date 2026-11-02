"""Load checkpoints and run attribution on Bioverse batches or plot examples."""

from __future__ import annotations

import glob
import os
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from .torch import Model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(ROOT, "configs")
RESULTS = os.path.join(ROOT, "results")

ARCH_YAML = {
    "gnn": "gnn.yaml",
    "kpconv": "kpconv.yaml",
    "cosmo": "cosmo.yaml",
    "egnn": "egnn.yaml",
    "se3_transformer": "se3_transformer.yaml",
}


TOY_RUN_PREFIX = "toy"
TOY_MODEL_DIM = 2


def configure_toy_line(*, prefix: str = "toy", dim: int = 2) -> None:
    """Select the 2D ``toy_`` line or the planar-in-3D ``toy3d_`` sanity line."""
    global TOY_RUN_PREFIX, TOY_MODEL_DIM
    TOY_RUN_PREFIX = prefix
    TOY_MODEL_DIM = int(dim)
    _TOY_MODEL_CACHE.clear()


def merge_configs(*names: str, **overrides):
    files = [os.path.join(CONFIGS, name) if not os.path.isabs(name) else name for name in names]
    cfg = OmegaConf.merge(*[OmegaConf.load(path) for path in files])
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def build_model(config: dict) -> Model:
    spec = config["model"]
    _, kwargs = next(iter(spec.items()))
    return Model(**kwargs)


def find_run_dir(pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    for run_dir in matches:
        ckpt = os.path.join(run_dir, "checkpoint.pt")
        if os.path.isfile(ckpt):
            return run_dir
    return matches[0] if matches else None


def find_completed_run_dir(pattern: str) -> Optional[str]:
    """Run directory with both a checkpoint and a finished test.

    A checkpoint alone is not enough: failed mid-train jobs still write
    ``checkpoint.pt``, and using those numbers looks like a real result.
    """
    matches = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    for run_dir in matches:
        ckpt = os.path.join(run_dir, "checkpoint.pt")
        test = os.path.join(run_dir, "test_results.yaml")
        if os.path.isfile(ckpt) and os.path.isfile(test):
            return run_dir
    return None


def load_checkpoint(model: torch.nn.Module, path: str, map_location="cpu") -> torch.nn.Module:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(state, dict):
        state = state.state_dict()
    cleaned = {}
    for key, value in state.items():
        for prefix in ("_forward_module.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def checkpoint_sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run_from_dir(run_dir: str, *config_names: str, **overrides) -> Optional[Model]:
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    if not os.path.isfile(ckpt):
        return None
    model = build_model(merge_configs(*config_names, **overrides))
    return load_checkpoint(model, ckpt)


def load_run(
    arch: str,
    pattern: str,
    *config_names: str,
    require_completed: bool = False,
    **overrides,
) -> Optional[Model]:
    run_dir = find_completed_run_dir(pattern) if require_completed else find_run_dir(pattern)
    if run_dir is None:
        return None
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    if not os.path.isfile(ckpt):
        return None
    if require_completed and not os.path.isfile(os.path.join(run_dir, "test_results.yaml")):
        return None
    model = build_model(merge_configs(*config_names, **overrides))
    return load_checkpoint(model, ckpt)


def load_cosmo_run(pattern: str, *config_names: str, **overrides) -> Optional[Model]:
    return load_run("cosmo", pattern, *config_names, **overrides)


def ship_example_to_data(example, dim=2, device=None):
    """Build a minimal geometric batch from a plotted ship example."""
    pos = np.asarray(example.pos, dtype=np.float32)
    if pos.shape[1] < dim:
        pos = np.pad(pos, ((0, 0), (0, dim - pos.shape[1])))
    motif = np.asarray(example.atom_is_motif, dtype=bool)
    features = np.stack([motif.astype(np.float32), (~motif).astype(np.float32)], axis=1)
    edges = np.asarray(example.edges, dtype=np.int64)
    if device is None:
        device = toy_device()
    return SimpleNamespace(
        atom_pos=torch.from_numpy(pos[:, :dim]).to(device),
        atom_features=torch.from_numpy(features).to(device),
        molecule_edges=torch.from_numpy(edges).to(device),
        vertex2molecule=torch.zeros(pos.shape[0], dtype=torch.long, device=device),
    )


_TOY_MODEL_CACHE: dict = {}


def toy_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def toy_run_configs(arch: str, ship: str) -> tuple[str, ...]:
    """Same yaml merge order as ``submit_toy.sh`` training."""
    yaml_name = ARCH_YAML[arch]
    return ("base.yaml", yaml_name, "ship.yaml", f"{ship}.yaml")


def get_toy_model(
    arch: str,
    ship: str,
    *,
    dim: Optional[int] = None,
    run_prefix: Optional[str] = None,
) -> Optional[Model]:
    yaml_name = ARCH_YAML.get(arch)
    if yaml_name is None:
        return None
    dim = TOY_MODEL_DIM if dim is None else int(dim)
    run_prefix = TOY_RUN_PREFIX if run_prefix is None else run_prefix
    cache_key = (run_prefix, dim, arch, ship)
    model = _TOY_MODEL_CACHE.get(cache_key)
    if model is None:
        overrides = {"dim": dim}
        if arch in ("cosmo", "kpconv"):
            # ship.yaml width is 128; Cosmo/KPConv stay at 64 (submit_toy.sh).
            overrides["hidden_dim"] = 64
        model = load_run(
            arch,
            f"{run_prefix}_{arch}_{ship}_*",
            *toy_run_configs(arch, ship),
            **overrides,
        )
        if model is None:
            return None
        model = model.to(toy_device())
        _TOY_MODEL_CACHE[cache_key] = model
    return model


def toy_attribution(
    example,
    ship: str,
    arch: str,
    method: str = "auto",
    ig_baseline: str = "graph_mean",
    dim: Optional[int] = None,
    run_prefix: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Raw signed per-node scores. Colormap min-max is a plot-only step."""
    dim = TOY_MODEL_DIM if dim is None else int(dim)
    model = get_toy_model(arch, ship, dim=dim, run_prefix=run_prefix)
    if model is None:
        return None
    data = ship_example_to_data(example, dim=dim)
    use_ig = method == "ig" and hasattr(model.trunk, "ig_attribution")
    raw = (
        model.trunk.ig_attribution(data, normalize=False, ig_baseline=ig_baseline)
        if use_ig
        else model.attribution(data, normalize=False, ig_baseline=ig_baseline)
    )
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().numpy()
    return np.asarray(raw, dtype=np.float32).reshape(-1)


def toy_cosmo_attribution(example, ship: str) -> Optional[np.ndarray]:
    return toy_attribution(example, ship, "cosmo")


def _molecule_id(X) -> str:
    import awkward as ak

    return str(ak.to_numpy(ak.ravel(X.molecules.molecule_id))[0])


_ENZYME_MODEL_CACHE: dict[str, Model] = {}
_ENZYME_BATCH_CACHE: dict[str, object] = {}


def enzyme_checkpoint_path(arch: str) -> Optional[str]:
    run_dir = find_completed_run_dir(f"enzyme_{arch}_*")
    if run_dir is None:
        return None
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    return ckpt if os.path.isfile(ckpt) else None


def _get_enzyme_model(arch: str) -> Optional[Model]:
    if arch not in _ENZYME_MODEL_CACHE:
        yaml_name = ARCH_YAML.get(arch)
        if yaml_name is None:
            return None
        model = load_run(
            arch,
            f"enzyme_{arch}_*",
            "base.yaml",
            "enzyme.yaml",
            yaml_name,
            require_completed=True,
        )
        if model is not None:
            _ENZYME_MODEL_CACHE[arch] = model
    return _ENZYME_MODEL_CACHE.get(arch)


def preload_enzyme_batches(molecule_ids: Iterable[str], archs: Iterable[str]) -> None:
    """Load attribution batches with one partition scan per model architecture."""
    from bioverse.collaters import LongCollater
    from bioverse.factory import BenchmarkFactory, TransformFactory

    wanted = set(molecule_ids)
    for arch in archs:
        yaml_name = ARCH_YAML.get(arch)
        if yaml_name is None:
            continue
        missing = {
            molecule_id
            for molecule_id in wanted
            if f"{arch}:{molecule_id}" not in _ENZYME_BATCH_CACHE
        }
        if not missing:
            continue
        attr_cfg = merge_configs(
            "base.yaml", "enzyme.yaml", "enzyme_attr.yaml", yaml_name
        )
        bench = BenchmarkFactory(attr_cfg["benchmark"])
        bench.apply(TransformFactory(list(attr_cfg["transforms"])))
        loader = bench.loader(
            "test",
            collater=LongCollater(),
            batch_size=1,
            batch_on="molecules",
            shuffle=False,
            progress=False,
        )
        for (X, _y), data in loader:
            molecule_id = _molecule_id(X)
            if molecule_id in missing:
                _ENZYME_BATCH_CACHE[f"{arch}:{molecule_id}"] = data
                missing.remove(molecule_id)
            if not missing:
                break


def _get_enzyme_batch(molecule_id: str, arch: str):
    key = f"{arch}:{molecule_id}"
    if key in _ENZYME_BATCH_CACHE:
        return _ENZYME_BATCH_CACHE[key]
    yaml_name = ARCH_YAML.get(arch)
    if yaml_name is None:
        return None
    attr_cfg = merge_configs(
        "base.yaml", "enzyme.yaml", "enzyme_attr.yaml", yaml_name
    )
    data = _load_named_batch(
        attr_cfg["benchmark"],
        attr_cfg["transforms"],
        molecule_id,
        partition="test",
    )
    if data is not None:
        _ENZYME_BATCH_CACHE[key] = data
    return data


def enzyme_attribution(
    molecule_id: str,
    arch: str,
    *,
    data=None,
    ig_steps=None,
    normalize=True,
) -> Optional[np.ndarray]:
    yaml_name = ARCH_YAML.get(arch)
    if yaml_name is None:
        return None
    model = _get_enzyme_model(arch)
    if model is None:
        return None
    if data is None:
        data = _get_enzyme_batch(molecule_id, arch)
    if data is None:
        return None
    kwargs = {"normalize": normalize}
    if ig_steps is not None:
        kwargs["ig_steps"] = ig_steps
    return model.attribution(data, **kwargs)


def enzyme_cosmo_attribution(molecule_id: str) -> Optional[np.ndarray]:
    return enzyme_attribution(molecule_id, "cosmo")


_BIOLIP_MODEL_CACHE: dict[str, Model] = {}
_BIOLIP_BATCH_CACHE: dict[str, object] = {}


def biolip_checkpoint_path(arch: str) -> Optional[str]:
    run_dir = find_completed_run_dir(f"biolip_{arch}_*")
    if run_dir is None:
        return None
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    return ckpt if os.path.isfile(ckpt) else None


def _get_biolip_model(arch: str) -> Optional[Model]:
    if arch not in _BIOLIP_MODEL_CACHE:
        yaml_name = ARCH_YAML.get(arch)
        if yaml_name is None:
            return None
        model = load_run(
            arch,
            f"biolip_{arch}_*",
            "base.yaml",
            "biolip.yaml",
            yaml_name,
            require_completed=True,
        )
        if model is not None:
            _BIOLIP_MODEL_CACHE[arch] = model
    return _BIOLIP_MODEL_CACHE.get(arch)


def preload_biolip_batches(molecule_ids: Iterable[str], archs: Iterable[str]) -> None:
    """Load BIOLIP batches with one partition scan per model architecture."""
    from bioverse.collaters import LongCollater
    from bioverse.factory import BenchmarkFactory, TransformFactory
    from bioverse.utilities import config as bioverse_config

    bioverse_config.workers = 1
    wanted = set(molecule_ids)
    for arch in archs:
        yaml_name = ARCH_YAML.get(arch)
        if yaml_name is None:
            continue
        missing = {
            molecule_id
            for molecule_id in wanted
            if f"{arch}:{molecule_id}" not in _BIOLIP_BATCH_CACHE
        }
        if not missing:
            continue
        cfg = merge_configs("base.yaml", "biolip.yaml", yaml_name)
        bench = BenchmarkFactory(cfg["benchmark"])
        bench.apply(TransformFactory(list(cfg["transforms"])))
        loader = bench.loader(
            "test",
            collater=LongCollater(),
            batch_size=1,
            batch_on="molecules",
            shuffle=False,
            progress=False,
        )
        for (X, _y), data in loader:
            if X is None:
                continue
            molecule_id = _molecule_id(X)
            if molecule_id in missing:
                _BIOLIP_BATCH_CACHE[f"{arch}:{molecule_id}"] = data
                missing.remove(molecule_id)
            if not missing:
                break


def _get_biolip_batch(molecule_id: str, arch: str):
    key = f"{arch}:{molecule_id}"
    if key in _BIOLIP_BATCH_CACHE:
        return _BIOLIP_BATCH_CACHE[key]
    yaml_name = ARCH_YAML.get(arch)
    if yaml_name is None:
        return None
    cfg = merge_configs("base.yaml", "biolip.yaml", yaml_name)
    data = _load_named_batch(
        cfg["benchmark"],
        cfg["transforms"],
        molecule_id,
        partition="test",
    )
    if data is not None:
        _BIOLIP_BATCH_CACHE[key] = data
    return data


def biolip_attribution(
    molecule_id: str,
    arch: str,
    *,
    data=None,
    ig_steps=None,
    normalize=True,
    method="auto",
) -> Optional[np.ndarray]:
    yaml_name = ARCH_YAML.get(arch)
    if yaml_name is None:
        return None
    model = _get_biolip_model(arch)
    if model is None:
        return None
    if data is None:
        data = _get_biolip_batch(molecule_id, arch)
    if data is None:
        return None
    kwargs = {"normalize": normalize, "method": method}
    if ig_steps is not None:
        kwargs["ig_steps"] = ig_steps
    return model.attribution(data, **kwargs)


def _load_named_batch(benchmark, transforms: Iterable, molecule_id: str, partition="test"):
    from bioverse.collaters import LongCollater
    from bioverse.factory import BenchmarkFactory, TransformFactory
    from bioverse.utilities import config as bioverse_config

    bioverse_config.workers = 1
    bench = BenchmarkFactory(benchmark)
    bench.apply(TransformFactory(list(transforms)))
    loader = bench.loader(
        partition,
        collater=LongCollater(),
        batch_size=1,
        batch_on="molecules",
        shuffle=False,
        progress=False,
    )
    for (X, _y), data in loader:
        if X is None:
            continue
        if _molecule_id(X) == molecule_id:
            return data
    return None
