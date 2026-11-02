# Cosmo Validation Experiments (RECOMB 2026)

Experiment repository for validating the equivariant **Cosmo** architecture ([cosmic-torch](https://github.com/...)) on Bioverse benchmarks. Structure mirrors [bioverse_experiments](../bioverse_experiments).

## Experiments

### 1. Toy example — Picasso ship motif (`toy_*`)

Train five models on five ship perturbations from Bioverse:

| Config | Dataset | Perturbation |
|--------|---------|--------------|
| `ship.yaml` + `ship00.yaml` | D_SHIP00 | Scramble (baseline) |
| `ship.yaml` + `shipr0.yaml` | D_SHIPR0 | Scramble; train upright, test rotated |
| `ship.yaml` + `shipch.yaml` | D_SHIPCH | Chirality flip |
| `ship.yaml` + `shippc.yaml` | D_SHIPPC | Picasso compositional |
| `ship.yaml` + `shiptl.yaml` | D_SHIPTL | Mast tilt |

**Models:** GNN (GCN), KPConv, Cosmo, EGNN, SE(3)-Transformer. All share the same depth, width, mean-pooling graph readout, and training setup; only the convolution / attention layer differs. Cosmo max-pools lifted frames onto each node before that readout.

**Figures:**
- `plot_toy_performance.py` — example scatter plots (row 1) + accuracy bar plots (row 2)
- `plot_toy_attribution.py` — ground-truth motif (row 1) + per-model attribution maps

### 2. Enzyme classification — D_ECACSA (`enzyme_*`)

Train on `B_ECCSAP` (EC1 classification), then evaluate attribution on `B_ECCSAA` (catalytic-site AUROC / AUPRC / Recall@site). Cosmo uses completeness attribution; GNN, KPConv, EGNN, and SE(3)-Transformer use integrated gradients.

**Figures:**
- `plot_enzyme_table.py` — LaTeX table (`figures/enzyme_comparison.tex`)
- `eval_enzyme_attribution.py` — site AUROC / AUPRC / Recall@site vs catalytic residues
- `plot_enzyme_attribution.py` — 3D structures colored by attribution, one column per EC class

### 3. Protein–ligand binding — D_BIOLIP (`biolip_*`)

Train on `B_BIOLIP` (pKd = −log₁₀ Kd regression). Each complex is a single connected graph of protein residues and exploded ligand atoms (`ExplodeLigandAtoms` → `ResiduePositions` → `TokenizeByChainType` → residue k-NN, k=6, plus all protein–ligand pairs within 8 Å). Batch size is 32 complexes, 50 epochs. Ligand atom identities are collapsed to one token (coordinates kept). All models max-pool protein residues (ligand masked at readout). Cosmo and KPConv use kernel shells [1.5, 4.5, 8.0] and hidden 64; EGNN uses hidden 256 and SE3T hidden 512. Cosmo max-pools lifted frames onto nodes first. Site metrics and figures use $|$attribution$|$ on the protein chain. KPConv uses nearest-kernel assignment (KPConvX). Baselines use integrated gradients.

**Figures:**
- `plot_biolip_table.py` — LaTeX table (`figures/biolip_comparison.tex`)
- `eval_biolip_attribution.py` — site AUROC / Recall@site vs BioLiP contact residues
- `plot_biolip_attribution.py` — protein surface colored by attribution, ligand as ball-and-stick

### 4. QM9 quantum-chemical properties — D_QNTMA9 (`qm9_*`)

Train Cosmo on all twelve QM9 targets from Brandstetter et al. (ICLR 2022, Table 2): $\alpha$, gap, HOMO, LUMO, $\mu$, $C_v$, $G$, $H$, $R^2$, $U$, $U_0$, ZPVE. Atom graphs are $k$-NN with $k=8$, 100 epochs, L1 loss. Paper baselines (NMP, SchNet, Cormorant, L1Net, LieConv, TFN, SE(3)-Transformer, DimeNet++, SphereNet, PaiNN, EGNN, SEGNN) are copied into the table; only Cosmo is trained here.

**Figures:**
- `plot_qm9_table.py` — LaTeX table (`figures/qm9_comparison.tex`)

## Usage

```bash
# Single run (toy Cosmo)
bioverse train configs/base.yaml configs/ship.yaml configs/ship00.yaml configs/cosmo.yaml seed=1 name=toy_cosmo_ship00_1

# Single run (BioLiP Cosmo)
bioverse train configs/base.yaml configs/biolip.yaml configs/cosmo.yaml seed=1 name=biolip_cosmo_1
bioverse test  configs/base.yaml configs/biolip.yaml configs/cosmo.yaml seed=1 name=biolip_cosmo_1

# Single run (QM9 Cosmo, isotropic polarizability)
bioverse train configs/base.yaml configs/qm9.yaml configs/cosmo.yaml property=alpha seed=1 name=qm9_cosmo_alpha_1
bioverse test  configs/base.yaml configs/qm9.yaml configs/cosmo.yaml property=alpha seed=1 name=qm9_cosmo_alpha_1

# Batch submission (SLURM). train = train → test/eval → figures.
chmod +x submit_toy.sh submit_toy3d.sh submit_enzyme.sh submit_biolip.sh submit_qm9.sh
./submit_toy.sh train
./submit_enzyme.sh train
./submit_biolip.sh train
./submit_qm9.sh train
# Figures only (CPU Mol* / Plotly jobs; afterok on tests when using train/test)
./submit_biolip.sh plot
./submit_qm9.sh plot
```

## Layout

```
configs/          Composable YAML (base, task, model)
models/           Bioverse-compatible model implementations
results/          Training outputs (gitignored)
figures/          Generated plots and LaTeX tables
submit_toy.sh     SLURM grid for ship experiments
submit_enzyme.sh  SLURM grid for enzyme experiments
submit_biolip.sh  SLURM grid for BioLiP binding experiments
submit_qm9.sh     SLURM grid for QM9 property experiments
plot_*.py         Figure generation scripts
legacy/           Earlier prototype code
```

## Dependencies

- [bioverse](https://github.com/...) — dataset loading and trainer
- [cosmic-torch](https://github.com/...) — Cosmo network (`CosmoModel`) with completeness attribution
- plotly, kaleido, pyyaml, numpy
