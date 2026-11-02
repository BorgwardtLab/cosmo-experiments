"""Mol* rendering for BioLiP attribution: protein surface + ligand wireframe."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import awkward as ak
import numpy as np
from bioverse.factory import BenchmarkFactory
from bioverse.utilities import config
from molviewspec import create_builder
from molviewspec.nodes import MVSJ

from molstar_common import (
    PLAYWRIGHT_BROWSERS,
    RENDER_SIZE,
    SURFACE_GREY,
    _meta_is_current,
    _read_cache_meta,
    _render_panels_batch,
    _score_to_hex,
    _scores_digest,
    _write_cache_meta,
    build_viewer_html,
    download_pdb,
)
from eval_common import recall_at_site
from models.common import IG_INPUTS
from plot_style import BIOLIP_MODEL_COLUMNS, THEME_ACCENT

# Binding site is prepended by the figure; remaining columns match the table.
BIOLIP_ATTRIBUTION_COLUMNS = BIOLIP_MODEL_COLUMNS


def _column_id(arch: str, method: str) -> str:
    return f"{arch}_ig" if method == "ig" else arch


def _attribution_archs(columns=None) -> list[str]:
    columns = BIOLIP_ATTRIBUTION_COLUMNS if columns is None else columns
    return list(dict.fromkeys(arch for arch, _method, _label in columns))


BIOLIP_ATTRIBUTION_MODELS = _attribution_archs()

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "figures" / "cache" / "biolip_attribution"
PANEL_DIR = CACHE_DIR / "panels"
SCORES_DIR = CACHE_DIR / "scores"
HTML_DIR = CACHE_DIR / "html"
EXAMPLES_CACHE = CACHE_DIR / "selected_examples.json"

N_EXAMPLES = 3
MIN_PROTEIN_RESIDUES = 80
MAX_PROTEIN_RESIDUES = 280
MIN_CONTACT_RESIDUES = 3
# Relative SASA (Tien 2013) above which a contact is treated as surface-exposed.
MIN_CONTACT_RSA = 0.20
MAX_LIGAND_ATOMS = 60
MIN_LIGAND_ATOMS = 8
MAX_CANDIDATES = 16
SELECTION_IG_STEPS = 8
LIGAND_COLOR = "#000000"
EXCLUDED_PDB_IDS = {"2q2a"}
MIN_LIGAND_OFFSET = 8.0
# Bump when the camera / framing / ligand drawing changes so cached panels are rebuilt.
VIEW_VERSION = 11

# Empirical residue SASA maxima from Tien et al., PLOS ONE 2013.
_MAX_RESIDUE_SASA = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}


@dataclass(frozen=True)
class BiolipExample:
    pdb_id: str
    molecule_id: str
    receptor_chain: str
    ligand_id: str
    ligand_chain: str
    ligand_residue_number: int
    residue_numbers: np.ndarray
    residue_insertions: np.ndarray
    contact_mask: np.ndarray
    n_protein: int
    n_ligand: int
    protein_start: int
    pkd: float

    @property
    def contact_residues(self) -> np.ndarray:
        return self.residue_numbers[self.contact_mask]


def _ravel_str(value) -> str:
    return str(ak.to_numpy(ak.ravel(value))[0])


def _ravel_float(value) -> float:
    return float(ak.to_numpy(ak.ravel(value))[0])


def _protein_ligand_split(X):
    types = [str(t) for t in ak.to_list(ak.ravel(X.chains.chain_type))]
    residue_numbers = ak.to_numpy(ak.ravel(X.molecules.residue_number)).astype(int)
    insertions = [
        "" if v is None else str(v)
        for v in ak.to_list(ak.ravel(X.molecules.residue_insertion))
    ]
    contacts = ak.to_numpy(ak.ravel(X.molecules.residue_contact)).astype(bool)
    n_res_per_chain = [int(n) for n in ak.to_numpy(ak.ravel(X.toc["residue"]))]
    n_atom_per_chain = [
        int(n) for n in ak.to_numpy(ak.sum(X.toc["atom"], axis=-1).ravel())
    ]
    if len(types) != 2 or set(types) != {"protein", "ligand"}:
        return None
    protein_idx = types.index("protein")
    ligand_idx = types.index("ligand")
    starts = np.cumsum([0, *n_res_per_chain[:-1]])
    p0, p1 = int(starts[protein_idx]), int(starts[protein_idx] + n_res_per_chain[protein_idx])
    l0 = int(starts[ligand_idx])
    return {
        "residue_numbers": residue_numbers[p0:p1],
        "residue_insertions": np.asarray(insertions[p0:p1], dtype=object),
        "contact_mask": contacts[p0:p1],
        "n_protein": p1 - p0,
        "n_ligand": int(n_atom_per_chain[ligand_idx]),
        "protein_start": p0,
        "ligand_residue_number": int(residue_numbers[l0]),
    }


def _residue_key(number: int, insertion: str) -> tuple[int, str]:
    return int(number), str(insertion or "").strip()


def _n_surface_contacts(example: BiolipExample) -> int:
    """Count BioLiP contacts with relative SASA >= MIN_CONTACT_RSA.

    SASA is computed on the plotted receptor chain only (no ligand, no other
    chains) so buried cleft residues do not count as visible surface sites.
    """
    from Bio.PDB import MMCIFParser, ShrakeRupley

    wanted = {
        _residue_key(number, insertion)
        for number, insertion, is_contact in zip(
            example.residue_numbers,
            example.residue_insertions,
            example.contact_mask,
        )
        if is_contact
    }
    if not wanted:
        return 0
    try:
        cif_path = download_pdb(example.pdb_id)
        structure = MMCIFParser(QUIET=True).get_structure(example.pdb_id, str(cif_path))
        chain = structure[0][example.receptor_chain]
        for residue_id in [residue.id for residue in list(chain) if residue.id[0].strip()]:
            chain.detach_child(residue_id)
        ShrakeRupley().compute(chain, level="R")
    except Exception:
        return 0
    n_surface = 0
    for residue in chain:
        _het, number, insertion = residue.id
        if _residue_key(number, insertion) not in wanted:
            continue
        sasa = float(getattr(residue, "sasa", 0.0) or 0.0)
        max_sasa = _MAX_RESIDUE_SASA.get(residue.get_resname().strip().upper(), 200.0)
        if max_sasa > 0.0 and (sasa / max_sasa) >= MIN_CONTACT_RSA:
            n_surface += 1
    return n_surface


def _has_surface_site(example: BiolipExample) -> bool:
    return _n_surface_contacts(example) >= MIN_CONTACT_RESIDUES


def _spread_candidates(
    candidates: list[BiolipExample], limit: int = MAX_CANDIDATES
) -> list[BiolipExample]:
    if len(candidates) <= limit:
        return candidates
    ordered = sorted(candidates, key=lambda item: (item.n_protein, item.molecule_id))
    idxs = np.unique(np.linspace(0, len(ordered) - 1, limit).round().astype(int))
    return [ordered[int(i)] for i in idxs]


def _collect_biolip_candidates() -> list[BiolipExample]:
    config.workers = 1
    loader = BenchmarkFactory("B_BIOLIP").loader(
        partition="test",
        batch_size=1,
        batch_on="molecules",
        shuffle=False,
        progress=False,
    )
    candidates = []
    for (X, _y), _ in loader:
        split = _protein_ligand_split(X)
        if split is None:
            continue
        n_protein = split["n_protein"]
        n_ligand = split["n_ligand"]
        n_contact = int(split["contact_mask"].sum())
        if n_protein < MIN_PROTEIN_RESIDUES or n_protein > MAX_PROTEIN_RESIDUES:
            continue
        if n_ligand < MIN_LIGAND_ATOMS or n_ligand > MAX_LIGAND_ATOMS:
            continue
        if n_contact < MIN_CONTACT_RESIDUES:
            continue
        candidates.append(
            BiolipExample(
                pdb_id=_ravel_str(X.molecules.molecule_pdb_id),
                molecule_id=_ravel_str(X.molecules.molecule_id),
                receptor_chain=_ravel_str(X.molecules.molecule_receptor_chain),
                ligand_id=_ravel_str(X.molecules.molecule_ligand_id),
                ligand_chain=_ravel_str(X.molecules.molecule_ligand_chain),
                ligand_residue_number=split["ligand_residue_number"],
                residue_numbers=split["residue_numbers"],
                residue_insertions=split["residue_insertions"],
                contact_mask=split["contact_mask"],
                n_protein=n_protein,
                n_ligand=n_ligand,
                protein_start=int(split["protein_start"]),
                pkd=_ravel_float(X.molecules.molecule_pkd),
            )
        )
    return _spread_candidates(candidates)


def _selection_cache_key(models: Iterable[str]) -> dict:
    return {
        "models": list(models),
        "n_examples": N_EXAMPLES,
        "min_protein": MIN_PROTEIN_RESIDUES,
        "max_protein": MAX_PROTEIN_RESIDUES,
        "min_contact": MIN_CONTACT_RESIDUES,
        "min_contact_rsa": MIN_CONTACT_RSA,
        "max_candidates": MAX_CANDIDATES,
        "ig_steps": SELECTION_IG_STEPS,
        "excluded_pdbs": sorted(EXCLUDED_PDB_IDS),
        "min_ligand_offset": MIN_LIGAND_OFFSET,
        "ranking": "abs_recall",
        "rsa_after_spread": True,
        "selection_version": 4,
        "checkpoints": {model: _biolip_checkpoint_meta(model) for model in models},
    }


def _example_to_payload(example: BiolipExample) -> dict:
    return {
        "pdb_id": example.pdb_id,
        "molecule_id": example.molecule_id,
        "receptor_chain": example.receptor_chain,
        "ligand_id": example.ligand_id,
        "ligand_chain": example.ligand_chain,
        "ligand_residue_number": example.ligand_residue_number,
        "residue_numbers": example.residue_numbers.tolist(),
        "residue_insertions": [str(v) for v in example.residue_insertions],
        "contact_mask": example.contact_mask.astype(bool).tolist(),
        "n_protein": example.n_protein,
        "n_ligand": example.n_ligand,
        "protein_start": example.protein_start,
        "pkd": example.pkd,
    }


def _example_from_payload(payload: dict) -> BiolipExample:
    return BiolipExample(
        pdb_id=payload["pdb_id"],
        molecule_id=payload["molecule_id"],
        receptor_chain=payload["receptor_chain"],
        ligand_id=payload["ligand_id"],
        ligand_chain=payload["ligand_chain"],
        ligand_residue_number=int(payload["ligand_residue_number"]),
        residue_numbers=np.asarray(payload["residue_numbers"], dtype=int),
        residue_insertions=np.asarray(payload["residue_insertions"], dtype=object),
        contact_mask=np.asarray(payload["contact_mask"], dtype=bool),
        n_protein=int(payload["n_protein"]),
        n_ligand=int(payload["n_ligand"]),
        protein_start=int(payload.get("protein_start", 0)),
        pkd=float(payload["pkd"]),
    )


def _load_selected_examples(models: Iterable[str]) -> Optional[list[BiolipExample]]:
    cached = _read_cache_meta(EXAMPLES_CACHE)
    if cached is None or cached.get("key") != _selection_cache_key(models):
        return None
    examples = [_example_from_payload(item) for item in cached.get("examples", [])]
    return examples if len(examples) == N_EXAMPLES else None


def _save_selected_examples(
    examples: list[BiolipExample],
    models: Iterable[str],
    diagnostics: list[dict],
) -> None:
    _write_cache_meta(
        EXAMPLES_CACHE,
        {
            "key": _selection_cache_key(models),
            "examples": [_example_to_payload(example) for example in examples],
            "diagnostics": diagnostics,
        },
    )


def _score_example_alignment(
    example: BiolipExample,
    models: Iterable[str],
    *,
    ig_steps: int = SELECTION_IG_STEPS,
) -> dict[str, float]:
    from models.io import biolip_attribution

    per_model = {}
    for model in models:
        try:
            scores = biolip_attribution(
                example.molecule_id,
                model,
                ig_steps=ig_steps,
                normalize=False,
                method="auto",
            )
        except Exception:
            per_model[model] = 0.0
            continue
        if scores is None:
            per_model[model] = 0.0
            continue
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy()
        scores = np.abs(np.asarray(scores, dtype=float).reshape(-1))
        start = example.protein_start
        end = start + example.n_protein
        if scores.shape[0] < end:
            per_model[model] = 0.0
            continue
        per_model[model] = recall_at_site(scores[start:end], example.contact_mask)
    return per_model


def load_biolip_examples(models: Iterable[str] | None = None) -> list[BiolipExample]:
    models = _attribution_archs() if models is None else list(models)
    require_biolip_checkpoints(models)
    cached = _load_selected_examples(models)
    if cached is not None:
        if all(_has_surface_site(example) for example in cached):
            return cached

    candidates = _collect_biolip_candidates()
    if len(candidates) < N_EXAMPLES:
        raise RuntimeError(
            f"Need at least {N_EXAMPLES} BioLiP figure candidates, found {len(candidates)}."
        )

    replacements = [
        example
        for example in candidates
        if example.pdb_id not in EXCLUDED_PDB_IDS
        and _ligand_protein_offset(example) >= MIN_LIGAND_OFFSET
        and _has_surface_site(example)
    ]
    from models.io import preload_biolip_batches

    preload_biolip_batches([example.molecule_id for example in replacements], models)
    ranked = []
    for example in replacements:
        per_model = _score_example_alignment(example, models)
        mean_recall = float(np.nanmean(list(per_model.values()))) if per_model else 0.0
        ranked.append((mean_recall, example, per_model))
    ranked.sort(key=lambda item: -item[0])
    selected: list[BiolipExample] = []
    diagnostics: list[dict] = []
    used_pdbs: set[str] = set()
    for mean_recall, example, per_model in ranked:
        if example.pdb_id in used_pdbs:
            continue
        n_surface = _n_surface_contacts(example)
        if n_surface < MIN_CONTACT_RESIDUES:
            continue
        selected.append(example)
        used_pdbs.add(example.pdb_id)
        diagnostics.append(
            {
                "molecule_id": example.molecule_id,
                "pdb_id": example.pdb_id,
                "ligand_id": example.ligand_id,
                "n_protein": example.n_protein,
                "n_contact": int(example.contact_mask.sum()),
                "n_surface_contact": n_surface,
                "abs_recall": mean_recall,
                "models": per_model,
            }
        )
        print(
            f"  {example.pdb_id} {example.ligand_id} ({example.molecule_id}, "
            f"{example.n_protein} residues, {example.n_ligand} ligand atoms, "
            f"{n_surface} surface contacts, Recall@site={mean_recall:.3f})"
        )
        if len(selected) == N_EXAMPLES:
            break
    if len(selected) < N_EXAMPLES:
        raise RuntimeError("Not enough distinct-PDB BioLiP examples after ranking.")
    _save_selected_examples(selected, models, diagnostics)
    return selected


def require_biolip_checkpoints(models: Iterable[str] | None = None) -> None:
    from models.io import biolip_checkpoint_path

    models = _attribution_archs() if models is None else list(models)
    missing = [model for model in models if biolip_checkpoint_path(model) is None]
    if missing:
        raise RuntimeError(
            "BioLiP attribution checkpoints missing for: "
            + ", ".join(missing)
            + ". Expected results/biolip_{model}_*/checkpoint.pt."
        )


def _biolip_checkpoint_meta(model: str) -> Optional[dict]:
    from models.io import biolip_checkpoint_path

    ckpt = biolip_checkpoint_path(model)
    if ckpt is None:
        return None
    return {"checkpoint": ckpt, "mtime": os.path.getmtime(ckpt)}


def _scores_cache_path(model: str, molecule_id: str, method: str = "auto") -> Path:
    return SCORES_DIR / f"{model}_{method}_{molecule_id.replace('/', '_')}.npz"


def _normalize_protein_scores(
    scores: np.ndarray, n_protein: int, start: int = 0
) -> np.ndarray:
    """Min-max ``|score|`` on protein residues to ``[0, 1]``, ignoring ligand."""
    start = int(start)
    protein = np.abs(np.asarray(scores, dtype=float).reshape(-1)[start : start + n_protein])
    lo = float(protein.min())
    hi = float(protein.max())
    return (protein - lo) / max(hi - lo, 1e-8)


def _load_cached_scores(
    model: str,
    molecule_id: str,
    n_protein: int,
    n_nodes: int,
    method: str = "auto",
) -> Optional[np.ndarray]:
    npz_path = _scores_cache_path(model, molecule_id, method)
    meta = _read_cache_meta(npz_path.with_suffix(".json"))
    if not _meta_is_current(meta, _biolip_checkpoint_meta(model)) or not npz_path.exists():
        return None
    if method == "ig" or model != "cosmo":
        if meta.get("ig_inputs") != IG_INPUTS:
            return None
    scores = np.asarray(np.load(npz_path)["scores"], dtype=float)
    if scores.shape[0] < n_protein:
        return None
    if scores.shape[0] not in {n_protein, n_nodes}:
        return None
    return scores


def _save_cached_scores(
    model: str, molecule_id: str, scores: np.ndarray, method: str = "auto"
) -> None:
    checkpoint_meta = _biolip_checkpoint_meta(model)
    if checkpoint_meta is None:
        return
    npz_path = _scores_cache_path(model, molecule_id, method)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, scores=np.asarray(scores, dtype=np.float32))
    extra = {"scores_kind": "raw"}
    if method == "ig" or model != "cosmo":
        extra["ig_inputs"] = IG_INPUTS
    _write_cache_meta(npz_path.with_suffix(".json"), {**checkpoint_meta, **extra})


def _model_scores(
    example: BiolipExample,
    model: str,
    *,
    method: str = "auto",
    force: bool = False,
) -> np.ndarray:
    from models.io import biolip_attribution

    n_nodes = example.n_protein + example.n_ligand
    scores = None if force else _load_cached_scores(
        model, example.molecule_id, example.n_protein, n_nodes, method
    )
    if scores is None:
        scores = biolip_attribution(
            example.molecule_id, model, normalize=False, method=method
        )
        if scores is None:
            raise RuntimeError(
                f"No BioLiP attribution for model {model!r} on {example.molecule_id}. "
                f"Checkpoint or test-set graph missing "
                f"(expected results/biolip_{model}_*/checkpoint.pt)."
            )
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if scores.shape[0] < example.protein_start + example.n_protein:
            raise RuntimeError(
                f"{model} attribution has {scores.shape[0]} scores but "
                f"{example.molecule_id} has {example.n_protein} protein residues "
                f"starting at {example.protein_start}."
            )
        _save_cached_scores(model, example.molecule_id, scores, method)
    return _normalize_protein_scores(
        scores, example.n_protein, start=example.protein_start
    )


def _residue_selector(example: BiolipExample, residue_number: int, insertion: str) -> dict:
    selector = {
        "auth_asym_id": example.receptor_chain,
        "auth_seq_id": int(residue_number),
    }
    if insertion:
        selector["pdbx_PDB_ins_code"] = insertion
    return selector


def _ligand_selector(example: BiolipExample, cif_path: Path | None = None) -> list[dict]:
    """Select the BioLiP ligand, including free amino acids stored as HETATM.

    RCSB mmCIF often puts those on a distinct ``label_asym_id`` while
    ``auth_asym_id`` still matches the receptor chain.
    """
    resi = int(example.ligand_residue_number)
    ligand_id = example.ligand_id
    selectors = [
        {
            "auth_asym_id": example.ligand_chain,
            "auth_comp_id": ligand_id,
            "auth_seq_id": resi,
        },
        {
            "auth_asym_id": example.ligand_chain,
            "label_comp_id": ligand_id,
            "auth_seq_id": resi,
        },
    ]
    if cif_path is not None:
        import gemmi

        structure = gemmi.read_structure(str(cif_path))
        for model in structure:
            for chain in model:
                for residue in chain:
                    try:
                        number = int(residue.seqid.num)
                    except (TypeError, ValueError):
                        continue
                    if residue.name.strip().upper() != ligand_id.upper() or number != resi:
                        continue
                    if chain.name not in {example.ligand_chain, example.receptor_chain}:
                        continue
                    label_chain = residue.subchain or chain.name
                    selectors.insert(
                        0,
                        {
                            "label_asym_id": label_chain,
                            "label_comp_id": residue.name.strip(),
                            "auth_seq_id": resi,
                        },
                    )
                    return selectors
    return selectors


def _atom_xyz(atom) -> np.ndarray:
    pos = atom.pos
    return np.array([pos.x, pos.y, pos.z], dtype=float)


def _ligand_centers(cif_path: Path, example: BiolipExample) -> tuple[np.ndarray, np.ndarray]:
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    protein_pts: list[np.ndarray] = []
    ligand_pts: list[np.ndarray] = []
    ligand_id = example.ligand_id.upper()
    receptor_chain = example.receptor_chain
    ligand_chain = example.ligand_chain
    ligand_resi = int(example.ligand_residue_number)

    for model in structure:
        for chain in model:
            chain_id = chain.name
            for residue in chain:
                name = residue.name.strip().upper()
                try:
                    resi = int(residue.seqid.num)
                except (TypeError, ValueError):
                    continue
                is_ligand = (
                    name == ligand_id
                    and resi == ligand_resi
                    and chain_id in {ligand_chain, receptor_chain}
                )
                if is_ligand:
                    ligand_pts.extend(_atom_xyz(atom) for atom in residue)
                    continue
                kind = gemmi.find_tabulated_residue(name)
                if chain_id == receptor_chain and kind.is_amino_acid:
                    protein_pts.extend(_atom_xyz(atom) for atom in residue)

    protein_center = (
        np.mean(np.stack(protein_pts), axis=0) if protein_pts else np.zeros(3)
    )
    ligand_center = (
        np.mean(np.stack(ligand_pts), axis=0) if ligand_pts else protein_center
    )
    return protein_center, ligand_center


def _ligand_protein_offset(example: BiolipExample) -> float:
    cif_path = download_pdb(example.pdb_id)
    protein_center, ligand_center = _ligand_centers(cif_path, example)
    return float(np.linalg.norm(protein_center - ligand_center))


def _camera_from_cif(cif_path: Path, example: BiolipExample) -> tuple[list[float], list[float]]:
    """Look from the ligand side toward the protein so the pocket faces the camera."""
    protein_center, ligand_center = _ligand_centers(cif_path, example)
    direction = protein_center - ligand_center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]
    direction = direction / norm
    helper = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(direction, helper))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    right = np.cross(helper, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    up /= np.linalg.norm(up)
    return direction.tolist(), up.tolist()


def build_complex_state(
    cif_path: Path,
    example: BiolipExample,
    *,
    highlight_contacts: bool = False,
    scores: Optional[np.ndarray] = None,
):
    structure = (
        create_builder()
        .canvas(background_color="#ffffff")
        .download(url=cif_path.resolve().as_uri())
        .parse(format="mmcif")
        .model_structure()
    )
    direction, up = _camera_from_cif(cif_path, example)
    ligand_selectors = _ligand_selector(example, cif_path)
    framed = structure.component(
        selector=[
            *[
                _residue_selector(example, number, str(insertion))
                for number, insertion in zip(example.residue_numbers, example.residue_insertions)
            ],
            *ligand_selectors,
        ]
    )
    framed.focus(direction=direction, up=up, radius_factor=1.08, radius_extent=2)

    # Surface only receptor residues so same-chain ligands are not meshed
    # into the grey protein.
    polymer = structure.component(
        selector=[
            _residue_selector(example, number, str(insertion))
            for number, insertion in zip(example.residue_numbers, example.residue_insertions)
        ]
    )
    representation = polymer.representation(type="surface").color(color=SURFACE_GREY)
    if scores is not None:
        protein_scores = np.asarray(scores, dtype=float).reshape(-1)[: example.n_protein]
        for residue_number, insertion, score in zip(
            example.residue_numbers,
            example.residue_insertions,
            protein_scores,
        ):
            representation = representation.color(
                color=_score_to_hex(score),
                selector=_residue_selector(example, residue_number, str(insertion)),
            )
    elif highlight_contacts:
        for residue_number, insertion in zip(
            example.residue_numbers[example.contact_mask],
            example.residue_insertions[example.contact_mask],
        ):
            representation = representation.color(
                color=THEME_ACCENT,
                selector=_residue_selector(example, residue_number, str(insertion)),
            )
    size_factor = float(
        np.clip(0.012 * example.n_protein * (18.0 / max(example.n_ligand, 8)), 1.8, 4.0)
    )
    structure.component(selector=ligand_selectors).representation(
        type="ball_and_stick",
        ignore_hydrogens=True,
        size_factor=size_factor,
    ).color(color=LIGAND_COLOR)
    return structure._get_root().get_state()


def _ground_truth_meta(example: BiolipExample) -> dict:
    return {
        "kind": "ground_truth",
        "view_version": VIEW_VERSION,
        "molecule_id": example.molecule_id,
        "residue_digest": _scores_digest(example.residue_numbers.astype(np.float64)),
        "contact_digest": _scores_digest(example.contact_mask.astype(np.float64)),
    }


def _panel_has_molecule(png_path: Path) -> bool:
    from PIL import Image

    rgb = np.asarray(Image.open(png_path).convert("RGB"))
    return bool((rgb < 248).any(axis=2).mean() > 0.02)


def _panel_cache_valid(
    png_path: Path,
    *,
    model: Optional[str],
    example: BiolipExample,
    scores: Optional[np.ndarray],
) -> bool:
    if not png_path.exists() or not _panel_has_molecule(png_path):
        return False
    meta = _read_cache_meta(png_path.with_suffix(".json"))
    if meta is None:
        return False
    if model is None:
        expected = _ground_truth_meta(example)
        return meta == expected
    if scores is None or not _meta_is_current(meta, _biolip_checkpoint_meta(model)):
        return False
    return (
        meta.get("view_version") == VIEW_VERSION
        and meta.get("scores_digest") == _scores_digest(scores)
    )


def _write_panel_meta(
    png_path: Path,
    *,
    model: Optional[str],
    example: BiolipExample,
    scores: Optional[np.ndarray],
) -> None:
    if model is None:
        meta = _ground_truth_meta(example)
    else:
        checkpoint_meta = _biolip_checkpoint_meta(model)
        if checkpoint_meta is None or scores is None:
            return
        meta = {
            **checkpoint_meta,
            "view_version": VIEW_VERSION,
            "scores_digest": _scores_digest(scores),
        }
    _write_cache_meta(png_path.with_suffix(".json"), meta)


def render_all_panels(
    force: bool = False,
    models: list[str] | None = None,
    columns: list[tuple[str, str, str]] | None = None,
) -> Dict[str, Dict[str, Path]]:
    from models.io import preload_biolip_batches

    columns = BIOLIP_ATTRIBUTION_COLUMNS if columns is None else columns
    models = _attribution_archs(columns) if models is None else models
    require_biolip_checkpoints(models)
    examples = load_biolip_examples(models)
    preload_biolip_batches([example.molecule_id for example in examples], models)
    rendered: Dict[str, Dict[str, Path]] = {}
    jobs: list[tuple[Path, Path, int, dict]] = []
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    for example in examples:
        cif_path = download_pdb(example.pdb_id)
        rendered[example.molecule_id] = {}
        specs: list[tuple[str, Optional[str], Optional[np.ndarray], bool]] = [
            ("ground_truth", None, None, True)
        ]
        for arch, method, _label in columns:
            specs.append(
                (
                    _column_id(arch, method),
                    arch,
                    _model_scores(example, arch, method=method, force=force),
                    False,
                )
            )

        for column, model_name, scores, highlight_contacts in specs:
            panel_id = f"{example.pdb_id.lower()}_{example.ligand_id.lower()}_{column}"
            png_path = PANEL_DIR / f"{panel_id}.png"
            rendered[example.molecule_id][column] = png_path
            if not force and _panel_cache_valid(
                png_path, model=model_name, example=example, scores=scores
            ):
                continue
            state = build_complex_state(
                cif_path,
                example,
                highlight_contacts=highlight_contacts,
                scores=scores,
            )
            html_path = HTML_DIR / f"{panel_id}.html"
            html_path.write_text(build_viewer_html(MVSJ(data=state).dumps(indent=None), size=RENDER_SIZE))
            jobs.append(
                (
                    html_path,
                    png_path,
                    RENDER_SIZE,
                    {"model": model_name, "example": example, "scores": scores},
                )
            )

    if jobs:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(PLAYWRIGHT_BROWSERS)
        batch = [(html_path, png_path, size) for html_path, png_path, size, _ in jobs]
        asyncio.run(_render_panels_batch(batch))
        empties = [
            (html_path, png_path, size)
            for html_path, png_path, size, _ in jobs
            if not _panel_has_molecule(png_path)
        ]
        if empties:
            asyncio.run(_render_panels_batch(empties, concurrency=1))
        for _, png_path, _, meta in jobs:
            if not _panel_has_molecule(png_path):
                continue
            _write_panel_meta(
                png_path,
                model=meta["model"],
                example=meta["example"],
                scores=meta["scores"],
            )
    return rendered


if __name__ == "__main__":
    paths = render_all_panels(force="--force" in sys.argv)
    for molecule_id, panels in paths.items():
        for name, path in panels.items():
            print(f"{molecule_id} {name}: {path}")
