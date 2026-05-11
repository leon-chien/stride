# STRIDE Agent Notes

This file gives future coding agents the current project context and the next
goals. Keep it updated when major architecture or workflow decisions change.

## Project Vision

STRIDE is a goal-conditioned deep learning layer for WESTPA-style weighted
ensemble molecular simulation. MD engines handle local physics. STRIDE predicts
which active walker lineages are most likely to reach a user-chosen biological
event, then converts those predictions into bins and priorities WESTPA can use
for split/merge resampling.

The target architecture is:

```text
molecular frames
  -> eGNN frame encoder
  -> Temporal Transformer over trajectory history
  -> structured goal conditioning
  -> value heads: p_event, flux_value, uncertainty, stride_score
  -> WESTPA-compatible bin IDs
```

Key design principle: STRIDE should learn future value for many possible
biological events, not just novelty or next-frame dynamics.

## What Has Been Implemented

Current repository foundation:

- Structured goal specs for user-conditioned molecular events.
- Atomistic coordinate-window dataset utilities with atom/residue identity
  features, masks, proxy labels, and `.npz` save/load.
- HDF5 inspection/reading scaffolding for WESTPA `west.h5` files.
- Generalized WESTPA segment record loading, lineage reconstruction,
  descendant traversal, delayed pcoord event/flux labels, and pcoord lineage
  window extraction.
- A `scripts/extract_westpa_dataset.py` CLI that turns a `west.h5` file and a
  structured goal YAML into a STRIDE `.npz` training artifact.
- A WESTPA-style `StrideValueBinMapper` that implements `assign(coords,
  mask=None, output=None)` for scalar STRIDE value scores.
- Canonical atomistic STRIDE dataset utilities in `src/stride/data/atomistic.py`
  for coordinate windows, atom/residue features, atom masks, frame masks, goal
  features, proxy event labels, and `.npz` save/load.
- `src/stride/data/pdb_converter.py`
  - Loads single-model or multi-model PDB files into STRIDE coordinates and
    `AtomRecord` metadata.
  - Converts those trajectories into the canonical `AtomisticDataset` schema.
- `src/stride/data/mdanalysis_converter.py`
  - Loads topology plus real MD trajectory files through MDAnalysis.
  - Stores STRIDE real-data coordinates in nanometers by default.
  - Converts PDB+XTC/DCD/NC/TRR-style inputs into `AtomisticDataset`.
- `src/stride/data/sample.py`
  - Generates a tiny ASP42/ligand contact trajectory for local smoke tests.
  - Writes both `.npz` datasets and multi-model PDB files.
- `src/stride/training/atomistic.py`
  - Trains the eGNN + Temporal Transformer on `AtomisticDataset`.
  - Saves/loads PyTorch checkpoints.
  - Scores datasets offline with `p_event`, `flux_value`, `uncertainty`, and
    `stride_score`.
- CLI scripts:
  - `scripts/create_sample_dataset.py`
  - `scripts/build_atomistic_dataset.py`
  - `scripts/build_mdanalysis_dataset.py`
  - `scripts/download_mdshare_dataset.py`
  - `scripts/train_atomistic.py`
  - `scripts/score_atomistic.py`
- Atomistic tests that pass a small protein-ligand contact dataset through the
  existing eGNN + Temporal Transformer value model.
- PDB conversion, dihedral labeling, and atomistic checkpoint tests.

Recent deep learning architecture additions:

- `src/stride/goals.py`
  - Adds `GoalSpec` for structured YAML/dict user targets.
  - Converts auditable goal specs into deterministic numeric feature vectors.

- `src/stride/models/egnn.py`
  - Adds a pure PyTorch `EGNNFrameEncoder`.
  - Encodes one molecular coordinate frame into an invariant frame embedding.

- `src/stride/models/stride_value_model.py`
  - Adds `StrideValueModel`.
  - Uses eGNN per frame, a Temporal Transformer over frame embeddings, and a
    learned numeric goal encoder.
  - Produces WESTPA-facing heads:
    `p_event`, `flux_value`, `uncertainty`, `stride_score`.

- `src/stride/training/stride_value.py`
  - Adds delayed-descendant value loss.
  - Combines event BCE, flux MSE, uncertainty regularization, and optional score
    regression.

- `src/stride/binning/value_binner.py`
  - Adds helper functions to combine value heads and map continuous scores to
    quantile bins.

- `tests/test_deep_value_model.py`
  - Covers goal vector determinism, eGNN rotation/translation invariance, full
    model output shape/range, value loss, and quantile binning.

## Architecture Decisions

- The user goal should start as structured YAML/dict data, not natural language.
  This keeps label generation auditable and training reproducible.
- `dihedral_window` goals use four exact atom selections, `operator: inside`,
  and degree-valued `lower_bound`/`upper_bound` fields.
- STRIDE should be simulation-agnostic. The goal is broad biological rare-event
  prediction: binding, unbinding, conformational transitions, contact
  formation, and target-state membership.
- eGNN learns spatial molecular state for each frame.
- Temporal Transformer learns time: trajectory direction, commitment, momentum,
  and pre-event history signals across frame embeddings.
- The main STRIDE model is eGNN + Temporal Transformer + goal conditioning.
- Delayed descendant labels are the core training signal:

```text
recent lineage window at iteration t
  -> did this walker or any descendant reach the target by t + H?
  -> how much probability flux did descendants carry into the target?
```

- WESTPA integration should preserve WESTPA's physics and probability-weight
  semantics. STRIDE should provide bin IDs, scores, priorities, and diagnostics.

## Environment Notes

- The active base Python in this shell may not have project dependencies.
- The `stride` conda environment has the current test dependencies available:

```bash
conda run -n stride python ...
```

Compilation was checked with:

```bash
conda run -n stride python -m compileall src scripts tests
```

Current full test command:

```bash
conda run -n stride pytest tests
```

Current expected result:

```text
16 passed
```

Local smoke-test artifacts can be regenerated with:

```bash
conda run -n stride python scripts/create_sample_dataset.py
conda run -n stride python scripts/train_atomistic.py outputs/sample_ligand_contact.npz outputs/sample_ligand_contact.pt --epochs 1 --hidden-dim 16 --egnn-layers 1 --transformer-layers 1 --transformer-heads 4 --dropout 0.0
conda run -n stride python scripts/score_atomistic.py outputs/sample_ligand_contact.npz outputs/sample_ligand_contact.pt outputs/sample_ligand_contact_scores.npz
```

Generated files under `outputs/` are ignored and should not be committed.

Training CLI notes:

- `scripts/train_atomistic.py` prints split positive rates before training and
  per-epoch progress after each validation pass.
- It saves the final checkpoint to the requested checkpoint path and a best
  checkpoint to `<checkpoint>.best.pt` unless `--no-save-best` is used.
- Best checkpoint selection defaults to `--save-best-metric val_auroc
  --save-best-mode max`.
- Resume training with `--resume-from CHECKPOINT`; `--epochs` is interpreted as
  the desired final epoch number, not additional epochs.

First real dataset workflow:

```bash
conda env update -f environment.yml
conda run -n stride python scripts/download_mdshare_dataset.py alanine_dipeptide
conda run -n stride python scripts/build_mdanalysis_dataset.py outputs/mdshare/alanine_dipeptide/alanine-dipeptide-nowater.pdb outputs/mdshare/alanine_dipeptide/alanine-dipeptide-0-250ns-nowater.xtc configs/goals/alanine_phi_window.yaml outputs/alanine_phi_stride.npz --window-size 8 --horizon 25 --stride 4
conda run -n stride python scripts/train_atomistic.py outputs/alanine_phi_stride.npz outputs/alanine_phi_stride.pt --epochs 1 --batch-size 16 --hidden-dim 64 --egnn-layers 2 --transformer-layers 1 --transformer-heads 4 --device auto --event-positive-weight auto
```

The alanine phi goal currently targets a rare positive basin:
`lower_bound: 60.0`, `upper_bound: 100.0`. The full local build produced about
`0.0327` event positives with `--window-size 8 --horizon 25 --stride 4`.

## Next Goals

Prioritize the one-person build path. Do not jump straight to a large
foundation model, and do not do more serious training until generalized
WESTPA/data infrastructure exists.

1. Expand the WESTPA dataset bridge from pcoord-only to coordinate-aware data.
   - Keep the existing pcoord lineage extractor as the smoke-test path.
   - Add optional coordinate/topology references and frame-to-segment mapping.
   - Preserve `window_mask` support for variable-length trajectory histories.

2. Validate real offline training on alanine dipeptide.
   - Run the mdshare download, MDAnalysis dataset build, one-epoch training, and
     scoring path.
   - Tune window/horizon/stride only after confirming event positive rate is
     reasonable.
   - Public MD proxy labels can train event prediction, but true flux labels
     still require WESTPA weights and descendants.

3. Move longer training to a GPU server.
   - Use laptop only for conversion and short debug epochs.
   - Use CUDA for real multi-epoch runs once the alanine `.npz` is built.
   - Track top-k enrichment, AUPRC, calibration, and offline scoring utility.

4. Connect model scoring to live WESTPA binning.
   - Use `StrideValueBinMapper` as the WESTPA-facing assignment surface.
   - Add a runtime scorer that computes STRIDE scores from active walker
     histories and exposes them to the mapper.
   - Include fallback to simple value/distance bins if model loading fails.

5. Add full coordinate data support for the eGNN path.
   - Define atom feature construction.
   - Use alanine dipeptide as the first serious geometry benchmark.
   - Then add a biological benchmark such as ligand binding/unbinding or a
     protein conformational transition.

6. Benchmark distance bins vs STRIDE bins.
   - Compare time to first event, target flux estimate, effective sample size,
     lineage diversity, probability conservation, and bin occupancy stability.

## Coding Guidance

- Keep interfaces stable and small. Add adapters rather than rewriting working
  baseline code.
- Preserve existing uncommitted work unless the user explicitly asks to revert
  it.
- Keep generated datasets, checkpoints, WESTPA files, and benchmark outputs out
  of git. Source code and small configs should be tracked; artifacts should be
  regenerated or stored externally.
- Prefer pure PyTorch for the first eGNN implementation to avoid heavy graph
  library setup.
- Add focused tests for lineage labels, invariance, shape contracts, and
  probability-weight conservation.
- Avoid making STRIDE depend on live WESTPA execution for basic unit tests.
