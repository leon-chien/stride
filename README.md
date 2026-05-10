# STRIDE

**Structured Trajectory Representation and Intelligent Dynamic Exploration**

STRIDE is a goal-conditioned deep learning system for rare-event molecular
simulation. It is designed to sit above molecular dynamics engines and WESTPA:
the simulation engine advances the physics, WESTPA manages weighted-ensemble
resampling, and STRIDE predicts which active trajectory lineages are most likely
to become scientifically useful.

In one sentence:

> STRIDE learns the future value of simulation walkers, then converts that value
> into adaptive bins and priorities that WESTPA can use for split/merge
> resampling.

The project is currently a research prototype focused on atomistic,
goal-conditioned value learning. It includes a structured goal interface,
atomistic coordinate-window datasets, WESTPA HDF5 lineage extraction, a
WESTPA-style value bin mapper, and a goal-conditioned eGNN + Temporal
Transformer value model.

## Why STRIDE Exists

Weighted ensemble methods are powerful, but their performance often depends on
manual progress coordinates such as distances, angles, RMSD, contacts, or
hand-designed state definitions. Those coordinates can miss the important
question:

```text
Which current walker is most likely to produce useful future target flux?
```

STRIDE reframes adaptive sampling as a trajectory-value learning problem:

```text
recent lineage history + user goal -> future event probability and flux value
```

Instead of only asking whether a state is new, STRIDE asks whether a path is
going somewhere useful for the scientific target the user chose.

## Architecture

The target architecture is:

```text
molecular frames
  -> eGNN frame encoder
  -> Temporal Transformer over trajectory history
  -> structured goal conditioning
  -> value heads
  -> WESTPA-compatible bins and priorities
```

### 1. Frame Representation

Each molecular frame is encoded with an equivariant graph neural network
(`EGNNFrameEncoder`). The eGNN consumes atom coordinates and atom features, then
produces a frame embedding that is stable under global translation and rotation.

This gives STRIDE a physics-aware representation without requiring the user to
hand-pick every distance, angle, or contact feature.

### 2. Temporal Modeling

Time is learned by a Temporal Transformer. The eGNN answers:

```text
What molecular state is this frame in?
```

The Temporal Transformer answers:

```text
Where is this trajectory going?
```

It reads a window of frame embeddings and learns direction, commitment,
pre-event patterns, and other history-dependent signals that a single-frame
classifier would miss.

### 3. Goal Conditioning

STRIDE uses structured goal specifications. A goal can describe a target such as
ligand binding, unbinding, RMSD-to-state, conformational change, or contact
formation.

Example:

```yaml
goal:
  name: ligand_contact_asp42
  type: contact
  selections:
    - ligand
    - ASP42
  operator: less_than
  threshold: 0.45
  horizon_iterations: 50
  value_target: event_and_flux
```

The structured goal is converted into deterministic numeric features, then
passed through a learned goal encoder. This keeps the interface auditable while
allowing one model to eventually support many targets.

Dihedral-window goals are supported for small conformational benchmarks such as
alanine dipeptide:

```yaml
goal:
  name: alanine_phi_window
  type: dihedral_window
  selections:
    - resid:1&atom:C
    - resid:2&atom:N
    - resid:2&atom:CA
    - resid:2&atom:C
  operator: inside
  threshold: -80.0
  lower_bound: -120.0
  upper_bound: -40.0
  horizon_iterations: 25
  value_target: event_and_flux
```

### 4. Value Heads

The model produces WESTPA-facing value predictions:

```python
{
    "p_event": probability_of_reaching_target,
    "flux_value": expected_descendant_probability_flux,
    "uncertainty": model_uncertainty_or_exploration_bonus,
    "stride_score": combined_control_score,
}
```

Those continuous scores can be converted into quantile bins for WESTPA-style
resampling.

## Training Objective

The core training idea is delayed descendant supervision.

For a walker at iteration `t`, STRIDE looks ahead over its descendants:

```text
input:  trajectory window ending at t
goal:   user-chosen target
label:  did this walker or any descendant reach the target by t + H?
label:  how much probability weight reached the target by t + H?
```

This trains the model to recognize pre-event trajectory patterns before a rare
event is obvious from a simple progress coordinate.

The current value loss combines:

- binary event loss for `p_event`
- flux regression loss for `flux_value`
- uncertainty regularization
- optional score regression

## Current Status

Implemented:

- Goal specification interface.
- Atomistic dataset schema for coordinate windows, atom/residue features, masks,
  goal features, and labels.
- Pure PyTorch eGNN frame encoder.
- Temporal Transformer value model.
- Multi-head value loss.
- STRIDE value-score binning helpers.
- WESTPA HDF5 lineage reconstruction for segment IDs, parents, weights, and
  pcoords.
- Delayed descendant event/flux label extraction from WESTPA lineages.
- Multi-model PDB to atomistic STRIDE dataset conversion.
- MDAnalysis trajectory conversion for topology plus XTC/DCD/NC/TRR-style
  trajectory files.
- Dihedral-window goal labels for conformational transition training.
- mdshare alanine dipeptide download helper.
- A generated protein-ligand contact smoke-test dataset.
- Atomistic model training, checkpointing, and offline scoring scripts.
- WESTPA-style value bin mapper implementing `assign(coords, mask=None,
  output=None)`.
- Tests for goal encoding, eGNN invariance, model heads, value loss, and
  binning, PDB conversion, dihedral labels, and atomistic training.

Not complete yet:

- Frame-to-segment mapping between real WESTPA walkers and coordinate files.
- Runtime scorer that computes STRIDE scores for active WESTPA walkers inside a
  live WESTPA run.
- Live production WESTPA plugin packaging.
- Large-scale multi-goal training.

## Repository Layout

```text
src/stride/goals.py                  Structured goal specifications
src/stride/data/atomistic.py         Atomistic dataset schema and featurization
src/stride/data/mdanalysis_converter.py
                                     Topology + trajectory converter
src/stride/data/pdb_converter.py     Multi-model PDB trajectory converter
src/stride/data/sample.py            Tiny generated atomistic smoke-test dataset
src/stride/models/egnn.py            eGNN molecular frame encoder
src/stride/models/stride_value_model.py
                                     Goal-conditioned eGNN + Transformer model
src/stride/training/atomistic.py     Train/checkpoint/score atomistic models
src/stride/training/stride_value.py  Multi-head delayed-descendant loss
src/stride/binning/                  Score and quantile binning utilities
src/stride/westpa_plugin/            WESTPA adapter and HDF5 bridge scaffolding
configs/goals/                       Example structured goal specs
scripts/                             Dataset, training, scoring, and WESTPA CLIs
tests/                               Focused unit tests
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate stride
```

Or install into an existing Python environment:

```bash
pip install -e .
```

The project currently targets Python 3.11 and uses PyTorch, NumPy,
scikit-learn, h5py, PyYAML, Matplotlib, MDAnalysis, and mdshare.

## Testing

Run the full suite:

```bash
pytest tests
```

If working from another active shell environment, run through the `stride` conda
environment:

```bash
conda run -n stride pytest tests
```

Expected current result:

```text
16 passed
```

PyTorch may emit a Transformer nested-tensor performance warning. That warning
does not indicate a correctness failure.

## Small Dataset

Create the local smoke-test dataset:

```bash
python scripts/create_sample_dataset.py
```

This writes ignored local artifacts:

```text
outputs/sample_ligand_contact.npz
outputs/sample_ligand_contact.pdb
```

The dataset is intentionally tiny and synthetic. It proves that coordinates,
atom/residue identity features, structured goals, delayed proxy labels, the eGNN
encoder, the Temporal Transformer, and checkpoint scoring all connect. It is not
the final biological training corpus.

Convert a multi-model PDB into the same dataset format:

```bash
python scripts/build_atomistic_dataset.py outputs/sample_ligand_contact.pdb configs/goals/ligand_contact_asp42.yaml outputs/sample_from_pdb.npz --window-size 4 --horizon 2
```

Train a small checkpoint:

```bash
python scripts/train_atomistic.py outputs/sample_ligand_contact.npz outputs/sample_ligand_contact.pt --epochs 1 --hidden-dim 16 --egnn-layers 1 --transformer-layers 1 --transformer-heads 4 --dropout 0.0
```

Score the dataset with that checkpoint:

```bash
python scripts/score_atomistic.py outputs/sample_ligand_contact.npz outputs/sample_ligand_contact.pt outputs/sample_ligand_contact_scores.npz
```

## First Real Dataset

The recommended first real dataset is mdshare alanine dipeptide. It is small
enough for laptop debugging and useful for validating conformational transition
learning before moving to protein-ligand systems.

Update the environment after pulling dependency changes:

```bash
conda env update -f environment.yml
conda activate stride
```

Download alanine dipeptide:

```bash
python scripts/download_mdshare_dataset.py alanine_dipeptide
```

Build a STRIDE dataset from topology plus trajectory:

```bash
python scripts/build_mdanalysis_dataset.py \
  outputs/mdshare/alanine_dipeptide/alanine-dipeptide-nowater.pdb \
  outputs/mdshare/alanine_dipeptide/alanine-dipeptide-0-250ns-nowater.xtc \
  configs/goals/alanine_phi_window.yaml \
  outputs/alanine_phi_stride.npz \
  --window-size 8 \
  --horizon 25 \
  --stride 4
```

Run a short laptop debug training job:

```bash
python scripts/train_atomistic.py \
  outputs/alanine_phi_stride.npz \
  outputs/alanine_phi_stride.pt \
  --epochs 1 \
  --batch-size 16 \
  --hidden-dim 64 \
  --egnn-layers 2 \
  --transformer-layers 1 \
  --transformer-heads 4 \
  --device auto \
  --event-positive-weight auto
```

Use a remote GPU for longer runs after this one-epoch workflow succeeds.
For rare-event labels, keep `--event-positive-weight auto` unless you have a
reason to set the class weight manually.

## Example Workflows

Inspect a WESTPA HDF5 file:

```bash
python scripts/inspect_westpa_h5.py path/to/west.h5
```

Extract pcoord lineage windows and delayed labels from a WESTPA file:

```bash
python scripts/extract_westpa_dataset.py path/to/west.h5 configs/goals/ligand_contact_asp42.yaml outputs/stride_dataset.npz
```

Use the atomistic dataset utilities from Python:

```python
from stride.data import build_atomistic_dataset_from_pdb
from stride.goals import GoalSpec

goal = GoalSpec.from_yaml("configs/goals/ligand_contact_asp42.yaml")
dataset = build_atomistic_dataset_from_pdb(
    "outputs/sample_ligand_contact.pdb",
    goal=goal,
    window_size=4,
    horizon=2,
)
```

Use a real MD trajectory through MDAnalysis:

```python
from stride.data import build_atomistic_dataset_from_mdanalysis
from stride.goals import GoalSpec

goal = GoalSpec.from_yaml("configs/goals/alanine_phi_window.yaml")
dataset = build_atomistic_dataset_from_mdanalysis(
    "outputs/mdshare/alanine_dipeptide/alanine-dipeptide-nowater.pdb",
    "outputs/mdshare/alanine_dipeptide/alanine-dipeptide-0-250ns-nowater.xtc",
    goal=goal,
    window_size=8,
    horizon=25,
    stride=4,
)
```

## Roadmap

### Phase 1: WESTPA Data Bridge

- Reconstruct walker lineages from `west.h5`.
- Extract delayed descendant event and flux labels.
- Build a canonical STRIDE dataset format from WESTPA runs.
- Map WESTPA segments to coordinate trajectories for eGNN inputs.

### Phase 2: Live STRIDE Binning

- Wrap STRIDE scoring behind a WESTPA-compatible BinMapper interface.
- Return bin IDs, value scores, uncertainty, and diagnostics.
- Preserve probability-weight semantics and provide robust fallback behavior.

### Phase 3: Geometry-Aware Learning

- Connect real coordinate trajectories and topologies to the eGNN path.
- Add atom feature construction and frame-to-segment mapping.
- Benchmark against hand-designed progress coordinates.

### Phase 4: Goal-Conditioned Generalization

- Train on multiple targets and systems.
- Reuse the same model backbone across binding, unbinding, and transition goals.
- Add richer goal encoders only after structured goal specs are reliable.

## Design Philosophy

STRIDE should not replace molecular physics. It should learn where molecular
physics should spend its compute.

The long-term goal is not just to find novel states. The goal is to guide
simulation toward user-conditioned rare events with better sample efficiency,
better lineage prioritization, and fewer hand-designed progress coordinates.
