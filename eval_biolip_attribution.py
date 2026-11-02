#!/usr/bin/env python3
"""Rank BioLiP attribution against residue–ligand contact labels.

Scores the current ``checkpoint.pt`` in a run directory (no result cache).
Writes ``attr_results.yaml`` with ``checkpoint_sha256`` so the comparison
table can refuse scores from a different checkpoint.

Intended to run in the same SLURM test job as ``bioverse test``.
"""

from __future__ import annotations

import argparse
import os

import awkward as ak
import numpy as np
from bioverse.collaters import LongCollater
from bioverse.factory import BenchmarkFactory, TransformFactory
from bioverse.utilities import config as bioverse_config

from eval_common import (
    as_ranking_scores,
    average_precision,
    binary_auroc,
    protein_chain_scores,
    recall_at_site,
    to_device,
    write_attr_results,
)
from models.common import IG_INPUTS
from models.io import (
    ARCH_YAML,
    RESULTS,
    _molecule_id,
    load_run_from_dir,
    merge_configs,
)
from plot_style import MODELS

DEFAULT_MAX_COMPLEXES = 200
DEFAULT_IG_STEPS = 8


def _protein_contact_split(X):
    types = [str(t) for t in ak.to_list(ak.ravel(X.chains.chain_type))]
    contacts = ak.to_numpy(ak.ravel(X.molecules.residue_contact)).astype(bool)
    n_res_per_chain = [int(n) for n in ak.to_numpy(ak.ravel(X.toc["residue"]))]
    if len(types) != 2 or set(types) != {"protein", "ligand"}:
        return None
    protein_idx = types.index("protein")
    start = int(sum(n_res_per_chain[:protein_idx]))
    n_protein = n_res_per_chain[protein_idx]
    mask = contacts[start : start + n_protein]
    if n_protein < 8 or int(mask.sum()) < 2 or int((~mask).sum()) < 2:
        return None
    return start, n_protein, mask


def evaluate_run(
    name: str,
    model: str,
    *,
    max_complexes: int = DEFAULT_MAX_COMPLEXES,
    ig_steps: int = DEFAULT_IG_STEPS,
) -> dict:
    import torch

    yaml_name = ARCH_YAML.get(model)
    if yaml_name is None:
        raise ValueError(f"Unknown model {model!r}")
    run_dir = name if os.path.isabs(name) else os.path.join(RESULTS, name)
    net = load_run_from_dir(run_dir, "base.yaml", "biolip.yaml", yaml_name)
    if net is None:
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    print(f"Evaluating {name} ({model}) on {device}")

    bioverse_config.workers = 1
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

    auroc, auprc, recall = [], [], []
    ig_auroc, ig_auprc, ig_recall = [], [], []
    used = 0
    skipped = 0
    also_ig = model == "cosmo"
    for (X, _y), data in loader:
        if X is None:
            continue
        split = _protein_contact_split(X)
        if split is None:
            continue
        start, n_protein, mask = split
        data = to_device(data, device)
        try:
            raw = net.attribution(data, normalize=False, ig_steps=ig_steps)
            raw_ig = (
                net.attribution(data, normalize=False, ig_steps=ig_steps, method="ig")
                if also_ig
                else None
            )
        except Exception as exc:
            skipped += 1
            if skipped <= 3:
                print(f"  skip {_molecule_id(X)}: {exc}")
            continue
        scores = protein_chain_scores(as_ranking_scores(raw), start, n_protein)
        if scores is None:
            continue
        auroc.append(binary_auroc(scores, mask))
        auprc.append(average_precision(scores, mask))
        recall.append(recall_at_site(scores, mask))
        if raw_ig is not None:
            ig_scores = protein_chain_scores(as_ranking_scores(raw_ig), start, n_protein)
            if ig_scores is not None:
                ig_auroc.append(binary_auroc(ig_scores, mask))
                ig_auprc.append(average_precision(ig_scores, mask))
                ig_recall.append(recall_at_site(ig_scores, mask))
        used += 1
        if used % 25 == 0:
            print(f"  scored {used}/{max_complexes}")
        if used >= max_complexes:
            break

    if used == 0:
        raise RuntimeError(f"No usable BioLiP attribution for {name}")
    summary = {
        "auroc": float(np.nanmean(auroc)),
        "auprc": float(np.nanmean(auprc)),
        "recall": float(np.nanmean(recall)),
        "n_complexes": used,
        "ig_steps": ig_steps,
        "ig_inputs": IG_INPUTS,
        "attribution": "completeness" if also_ig else "ig",
        "ranking_scores": "abs",
    }
    if ig_auroc:
        summary["IG Site AUROC"] = float(np.nanmean(ig_auroc))
        summary["IG Site AUPRC"] = float(np.nanmean(ig_auprc))
        summary["IG Recall@site"] = float(np.nanmean(ig_recall))
    path = write_attr_results(run_dir, summary)
    extra = ""
    if "IG Site AUROC" in summary:
        extra = (
            f"  IG AUROC={summary['IG Site AUROC']:.3f}  "
            f"IG R@site={summary['IG Recall@site']:.3f}"
        )
    print(
        f"{model:16s}  AUROC={summary['auroc']:.3f}  "
        f"AUPRC={summary['auprc']:.3f}  "
        f"R@site={summary['recall']:.3f}  n={used}{extra}"
    )
    print(f"Wrote {path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Run directory name, e.g. biolip_gnn_1")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--max-complexes", type=int, default=DEFAULT_MAX_COMPLEXES)
    parser.add_argument("--ig-steps", type=int, default=DEFAULT_IG_STEPS)
    args = parser.parse_args()
    evaluate_run(
        args.name,
        args.model,
        max_complexes=args.max_complexes,
        ig_steps=args.ig_steps,
    )


if __name__ == "__main__":
    main()
