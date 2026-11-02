"""Shared site-ranking metrics and GPU batch helpers for attribution eval."""

from __future__ import annotations

import os

import numpy as np
import torch
import yaml

from models.common import IG_INPUTS
from models.io import checkpoint_sha256


def as_scores(raw) -> np.ndarray:
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    return np.asarray(raw, dtype=float).reshape(-1)


def as_ranking_scores(raw) -> np.ndarray:
    """``|score|`` for site ranking (magnitude, not sign of the logit)."""
    return np.abs(as_scores(raw))


def protein_chain_scores(scores: np.ndarray, start: int, n_protein: int):
    """Slice protein nodes using the chain offset, not ``[:n_protein]``."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    start = int(start)
    n_protein = int(n_protein)
    end = start + n_protein
    if start < 0 or n_protein <= 0 or scores.shape[0] < end:
        return None
    return scores[start:end]


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    greater = float((pos[:, None] > neg[None, :]).mean())
    equal = float((pos[:, None] == neg[None, :]).mean())
    return greater + 0.5 * equal


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")[::-1]
    ranked = labels[order]
    hits = np.cumsum(ranked)
    precision = hits / np.arange(1, ranked.size + 1)
    return float(precision[ranked].mean())


def recall_at_site(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_site = int(labels.sum())
    if n_site == 0 or labels.size == 0:
        return float("nan")
    top = np.argsort(scores, kind="mergesort")[::-1][:n_site]
    return float(labels[top].sum()) / n_site


def to_device(data, device):
    for name in dir(data):
        if name.startswith("_"):
            continue
        try:
            value = getattr(data, name)
        except Exception:
            continue
        if torch.is_tensor(value):
            setattr(data, name, value.to(device))
    for name in (
        "residue_pos",
        "residue_features",
        "atom_pos",
        "atom_features",
        "molecule_edges",
        "vertex2molecule",
        "y",
    ):
        value = getattr(data, name, None)
        if value is None or torch.is_tensor(value):
            continue
        try:
            setattr(data, name, torch.as_tensor(value, device=device))
        except Exception:
            continue
    return data


def write_attr_results(run_dir: str, values: dict) -> str:
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    payload = {
        "Site AUROC": float(values["auroc"]),
        "Site AUPRC": float(values["auprc"]),
        "Recall@site": float(values["recall"]),
        "checkpoint_sha256": checkpoint_sha256(ckpt),
    }
    payload.update({key: values[key] for key in values if key not in ("auroc", "auprc", "recall")})
    path = os.path.join(run_dir, "attr_results.yaml")
    with open(path, "w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def attr_results_for_checkpoint(run_dir: str) -> dict | None:
    """Return attr yaml only if it was written for the current checkpoint.pt."""
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    path = os.path.join(run_dir, "attr_results.yaml")
    if not (os.path.isfile(ckpt) and os.path.isfile(path)):
        return None
    with open(path) as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        return None
    recorded = data.get("checkpoint_sha256")
    if not recorded or recorded != checkpoint_sha256(ckpt):
        return None
    return data


def site_metrics_from_attr(
    attr: dict | None, *, method: str = "auto"
) -> tuple[float | None, float | None, float | None]:
    """Site AUROC / AUPRC / Recall@site for completeness (``auto``) or feature IG."""
    if not attr:
        return None, None, None
    if method == "ig":
        if attr.get("ig_inputs") != IG_INPUTS:
            return None, None, None
        if "IG Site AUROC" not in attr or "IG Recall@site" not in attr:
            return None, None, None
        auprc = attr.get("IG Site AUPRC")
        return (
            float(attr["IG Site AUROC"]),
            None if auprc is None else float(auprc),
            float(attr["IG Recall@site"]),
        )
    if attr.get("attribution") == "completeness" or attr.get("ig_inputs") == IG_INPUTS:
        auprc = attr.get("Site AUPRC")
        return (
            float(attr["Site AUROC"]),
            None if auprc is None else float(auprc),
            float(attr["Recall@site"]),
        )
    return None, None, None
