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

The project is currently a research prototype. It includes toy weighted-ensemble
benchmarks, a GRU baseline, a synthetic NaCl association benchmark, WESTPA HDF5
inspection scaffolding, learned bin mappers, and the first goal-conditioned
eGNN + Temporal Transformer value model.

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
NaCl association, ligand binding, unbinding, RMSD-to-state, or contact
formation.

Example:

```yaml
goal:
  name: nacl_association
  type: distance_threshold
  selection_a: Na+
  selection_b: Cl-
  operator: less_than
  threshold: 0.35
  horizon_iterations: 50
  value_target: event_and_flux
```

The structured goal is converted into deterministic numeric features, then
passed through a learned goal encoder. This keeps the interface auditable while
allowing one model to eventually support many targets.

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

- 2D rare-event toy simulator.
- Weighted-ensemble-like split/merge resampling.
- GRU trajectory-value baseline.
- Delayed-label training for toy and NaCl-style data.
- Offline replay evaluation.
- Synthetic NaCl association benchmark.
- WESTPA HDF5 inspection and pcoord reading scaffolding.
- NaCl WESTPA-style adapter and learned score/quantile bin mappers.
- Goal specification interface.
- Pure PyTorch eGNN frame encoder.
- Temporal Transformer value model.
- Multi-head value loss.
- STRIDE value-score binning helpers.
- Tests for goal encoding, eGNN invariance, model heads, value loss, and
  binning.

Not complete yet:

- Full production WESTPA `west.h5` lineage reconstruction.
- Coordinate trajectory extraction and frame-to-segment mapping for real MD
  systems.
- Live production WESTPA plugin packaging.
- Large-scale multi-goal training.

## Repository Layout

```text
src/stride/goals.py                  Structured goal specifications
src/stride/models/egnn.py            eGNN molecular frame encoder
src/stride/models/stride_value_model.py
                                     Goal-conditioned eGNN + Transformer model
src/stride/models/gru_ranker.py      GRU baseline model
src/stride/training/stride_value.py  Multi-head delayed-descendant loss
src/stride/training/train_toy.py     Toy model training
src/stride/training/train_nacl.py    NaCl benchmark training
src/stride/binning/                  Score and quantile binning utilities
src/stride/sampling/                 Weighted-ensemble-like toy sampling
src/stride/replay/                   Offline replay evaluation
src/stride/westpa_plugin/            WESTPA adapter and HDF5 bridge scaffolding
configs/                             Toy and NaCl configs
scripts/                             Benchmark, training, replay, and demo scripts
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
scikit-learn, h5py, PyYAML, and Matplotlib.

## Testing

Run the deep model tests:

```bash
pytest tests/test_deep_value_model.py
```

Run the toy tests:

```bash
pytest tests/test_toy2d.py
```

If working from another active shell environment, run through the `stride` conda
environment:

```bash
conda run -n stride pytest tests/test_deep_value_model.py
```

Expected current result for the deep model tests:

```text
4 passed
```

PyTorch may emit a Transformer nested-tensor performance warning. That warning
does not indicate a correctness failure.

## Example Workflows

Generate the synthetic NaCl dataset:

```bash
python scripts/run_nacl_dataset.py
```

Train the NaCl GRU baseline:

```bash
python scripts/train_nacl.py
```

Inspect a WESTPA HDF5 file:

```bash
python scripts/inspect_westpa_h5.py path/to/west.h5
```

Run the learned-bin mapper demo:

```bash
python scripts/demo_stride_binmapper.py
```

## Roadmap

### Phase 1: WESTPA Data Bridge

- Reconstruct walker lineages from `west.h5`.
- Extract delayed descendant event and flux labels.
- Build a canonical STRIDE dataset format from WESTPA runs.

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
