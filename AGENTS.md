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
- Optional coordinate-aware WESTPA extraction:
  - `scripts/build_westpa_segment_coordinates.py` builds the required
    per-segment coordinate store from a `west.h5`, topology, trajectory root,
    and a format pattern such as `{n_iter:06d}/{seg_id:06d}/seg.xtc`.
  - `scripts/extract_westpa_dataset.py --segment-coordinates-npz ...` maps
    segment keyed coordinate frames into canonical atomistic lineage windows.
  - WESTPA provenance arrays are saved alongside atomistic datasets:
    `westpa_n_iter`, `westpa_seg_id`, lineage keys, and segment weights.
- WESTPA lineage evaluation:
  - `scripts/evaluate_westpa_lineage.py` evaluates pcoord lineage artifacts
    against pcoord baselines under held-out iteration splits.
- A WESTPA-style `StrideValueBinMapper` that implements `assign(coords,
  mask=None, output=None)` for scalar STRIDE value scores.
- A `StrideRuntimeScorer` adapter that scores active atomistic walker histories
  with a checkpoint and falls back to configured scalar scores if model loading
  or scoring fails.
- A `PcoordLineageRuntimeScorer` adapter that scores active pcoord lineage
  histories with a pcoord-lineage checkpoint and uses the same fallback score
  contract.
- WESTPA steering replay tooling:
  - `scripts/replay_westpa_steering.py` compares STRIDE walker priorities and
    bins against simple pcoord control baselines on held-out WESTPA artifacts.
  - Outputs WESTPA-facing arrays: `stride_score`, `stride_bin`, and
    `stride_priority_rank`.
  - Writes steering diagnostics for top-k enrichment, bin occupancy,
    positive-event gradients, score/label correlation, and per-goal/per-cell
    groups.
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
  - `scripts/evaluate_atomistic.py`
  - `scripts/evaluate_westpa_lineage.py`
  - `scripts/build_westpa_segment_coordinates.py`
- Atomistic tests that pass a small protein-ligand contact dataset through the
  existing eGNN + Temporal Transformer value model.
- PDB conversion, dihedral labeling, and atomistic checkpoint tests.
- Evaluation utilities for checkpoint/score reports:
  - Score distribution summaries.
  - AUROC, AUPRC, top-k enrichment at 1%, 5%, 10%, and 25%.
  - Precision/recall at score quantiles.
  - Random and dihedral phi-window baselines.
  - Markdown, CSV, and plot outputs under `outputs/reports/`.
- Training controls for more rigorous validation:
  - Random, contiguous, blocked, and blocked-tail trajectory splits.
  - Early stopping on the selected best-checkpoint metric.
  - Optional cosine and plateau learning-rate schedulers.

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
- Local development uses the `stride` conda environment. The remote training
  server uses micromamba, so replace `conda run -n stride` with
  `micromamba run -n stride` in remote commands.

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
52 passed
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
- Use `--split-strategy blocked` for a purged random held-out trajectory block
  and `--split-strategy blocked_tail` for a purged tail chunk. For small WESTPA
  tutorial datasets, use `--split-strategy iteration_balanced` to hold out whole
  iterations while keeping positives in both train and validation.
- Early stopping is available with `--early-stopping-patience`; it monitors the
  same metric configured by `--save-best-metric`.
- Optional scheduler choices are `--lr-scheduler cosine` and
  `--lr-scheduler plateau`.

Evaluation report example:

```bash
conda run -n stride python scripts/evaluate_atomistic.py outputs/alanine_phi_stride_rare.npz --checkpoint outputs/alanine_phi_gpu_w2_lr1e4.best.pt --goal-yaml configs/goals/alanine_phi_window.yaml --topology-pdb outputs/mdshare/alanine_dipeptide/alanine-dipeptide-nowater.pdb --output-dir outputs/reports/alanine_phi_gpu_w2_lr1e4
```

WESTPA lineage report example:

```bash
conda run -n stride python scripts/evaluate_westpa_lineage.py outputs/stride_dataset.npz --eval-split validation --iteration-split-strategy tail --output-dir outputs/reports/westpa_lineage_validation
```

WESTPA pcoord-lineage training example:

```bash
conda run -n stride python scripts/train_westpa_lineage.py outputs/tutorial35_cell0_dim1_thr0.5.npz outputs/tutorial35_cell0_dim1_thr0.5.pt --epochs 50 --batch-size 128 --learning-rate 1e-4 --hidden-dim 64 --transformer-layers 1 --transformer-heads 4 --device cuda --event-positive-weight auto --split-strategy tail --save-best-metric val_auprc --save-best-mode max --early-stopping-patience 8
conda run -n stride python scripts/score_westpa_lineage.py outputs/tutorial35_cell0_dim1_thr0.5.npz outputs/tutorial35_cell0_dim1_thr0.5.best.pt outputs/tutorial35_cell0_dim1_thr0.5_scores.npz --device cuda
conda run -n stride python scripts/evaluate_westpa_lineage.py outputs/tutorial35_cell0_dim1_thr0.5.npz --stride-scores-npz outputs/tutorial35_cell0_dim1_thr0.5_scores.npz --rank-key p_event --eval-split validation --iteration-split-strategy tail --validation-fraction 0.2 --pcoord-target 0.5 --pcoord-dim 1 --output-dir outputs/reports/tutorial35_cell0_dim1_thr0.5_stride_validation
```

Tutorial 3.5 `cell_0`, `pcoord_dim=1`, `threshold=0.5` is the current pcoord
lineage benchmark. The baseline to beat is `last_pcoord_low` with about
`AUROC=0.6793` and `AUPRC=0.5056` on the tail validation split.

WESTPA multi-goal benchmark example:

```bash
conda run -n stride python scripts/build_westpa_multigoal_lineage.py configs/benchmarks/tutorial35_multigoal.yaml outputs/tutorial35_multigoal.npz
conda run -n stride python scripts/train_westpa_lineage.py outputs/tutorial35_multigoal.npz outputs/tutorial35_multigoal.pt --epochs 50 --batch-size 128 --learning-rate 1e-4 --hidden-dim 64 --transformer-layers 1 --transformer-heads 4 --device cuda --event-positive-weight auto --split-strategy tail --save-best-metric val_auprc --save-best-mode max --early-stopping-patience 8
conda run -n stride python scripts/score_westpa_lineage.py outputs/tutorial35_multigoal.npz outputs/tutorial35_multigoal.best.pt outputs/tutorial35_multigoal_scores.npz --device cuda
conda run -n stride python scripts/evaluate_westpa_lineage.py outputs/tutorial35_multigoal.npz --stride-scores-npz outputs/tutorial35_multigoal_scores.npz --rank-key p_event --eval-split validation --iteration-split-strategy tail --validation-fraction 0.2 --output-dir outputs/reports/tutorial35_multigoal_validation
conda run -n stride python scripts/summarize_westpa_reports.py outputs/reports/tutorial35_multigoal_seed*_validation --output-csv outputs/reports/tutorial35_multigoal_seed_summary.csv --output-md outputs/reports/tutorial35_multigoal_seed_summary.md
```

Multi-goal lineage artifacts include `cell_id`, `goal_id`, `pcoord_dim`,
`threshold`, and `horizon_iterations` metadata. `scripts/train_westpa_lineage.py`
and `scripts/evaluate_westpa_lineage.py` support `heldout_goal` and
`heldout_cell` split strategies for generalization checks.

WESTPA steering replay example:

```bash
micromamba run -n stride python scripts/replay_westpa_steering.py outputs/tutorial35_multigoal.npz --stride-scores-npz outputs/tutorial35_multigoal_scores.npz --rank-key p_event --baseline-key last_pcoord_low --eval-split validation --iteration-split-strategy tail --validation-fraction 0.2 --num-bins 8 --binning quantile --bin-reference train --per-iteration --output-dir outputs/reports/tutorial35_multigoal_steering_replay_train_bins
```

Replay reports answer the steering-readiness question: whether STRIDE would
prioritize better held-out walkers than simple pcoord ranking. The assignment
artifact and `stride_control_config.json` are the WESTPA-facing handoff for live
integration experiments. Use `--bin-reference train` for deployment-realistic
frozen bins; `--bin-reference eval` is only a diagnostic mode.

End-to-end steering benchmark wrapper:

```bash
micromamba run -n stride python scripts/run_westpa_steering_benchmark.py outputs/tutorial35_multigoal.npz outputs/tutorial35_multigoal_heldout_cell --mode heldout_cell --device cuda
micromamba run -n stride python scripts/run_westpa_steering_benchmark.py outputs/tutorial35_multigoal.npz outputs/tutorial35_multigoal_heldout_goal --mode heldout_goal --device cuda
```

WESTPA segment coordinate store example:

```bash
conda run -n stride python scripts/build_westpa_segment_coordinates.py path/to/west.h5 path/to/topology.pdb path/to/traj_segs outputs/segment_coordinates.npz --trajectory-pattern "{n_iter:06d}/{seg_id:06d}/seg.xtc" --frame-index -1 --coordinate-units nm
conda run -n stride python scripts/extract_westpa_dataset.py path/to/west.h5 configs/goals/westpa_distance_threshold.yaml outputs/westpa_atomistic_stride.npz --segment-coordinates-npz outputs/segment_coordinates.npz --window-iterations 8
```

NaCl WESTPA tutorial benchmark:

```bash
conda run -n stride python scripts/extract_westpa_dataset.py /Users/leonchien/Projects/westpa_tutorials/tutorials7.1-7.4/tutorial7.1-basic-nacl/west.h5 configs/goals/nacl_association.yaml outputs/nacl_westpa_pcoord_stride.npz --window-iterations 4 --horizon-iterations 4
conda run -n stride python scripts/build_westpa_segment_coordinates.py /Users/leonchien/Projects/westpa_tutorials/tutorials7.1-7.4/tutorial7.1-basic-nacl/west.h5 /Users/leonchien/Projects/westpa_tutorials/tutorials7.1-7.4/tutorial7.1-basic-nacl/common_files/bstate.pdb /Users/leonchien/Projects/westpa_tutorials/tutorials7.1-7.4/tutorial7.1-basic-nacl/traj_segs outputs/nacl_ions_segment_coordinates.npz --trajectory-pattern "{n_iter:06d}/{seg_id:06d}/seg.dcd" --frame-index -1 --coordinate-units angstrom --mda-selection "not resname HOH" --allow-missing
conda run -n stride python scripts/extract_westpa_dataset.py /Users/leonchien/Projects/westpa_tutorials/tutorials7.1-7.4/tutorial7.1-basic-nacl/west.h5 configs/goals/nacl_association.yaml outputs/nacl_ions_westpa_atomistic_stride.npz --segment-coordinates-npz outputs/nacl_ions_segment_coordinates.npz --window-iterations 4 --horizon-iterations 4
conda run -n stride python scripts/train_atomistic.py outputs/nacl_ions_westpa_atomistic_stride.npz outputs/nacl_ions_westpa_balanced.pt --epochs 10 --batch-size 32 --learning-rate 1e-4 --hidden-dim 32 --egnn-layers 1 --transformer-layers 1 --transformer-heads 4 --dropout 0.0 --device cpu --event-positive-weight auto --split-strategy iteration_balanced --save-best-metric val_auprc --save-best-mode max --early-stopping-patience 4
conda run -n stride python scripts/evaluate_atomistic.py outputs/nacl_ions_westpa_atomistic_stride.npz --checkpoint outputs/nacl_ions_westpa_balanced.best.pt --goal-yaml configs/goals/nacl_association.yaml --atom-pair-indices 0,1 --distance-direction low --eval-split validation --split-strategy iteration_balanced --output-dir outputs/reports/nacl_ions_westpa_balanced_validation_westpa_baselines --device cpu
```

Current NaCl smoke result with ion-only coordinates:
`val_auroc=0.6096`, `val_auprc=0.2829`, validation positive rate `0.2154`.
The validation report includes STRIDE, atom-pair distance, WESTPA pcoord, and
random baselines; STRIDE beats the simple non-random baselines on this tiny
held-out split. This is a pipeline benchmark, not a final model result.

Last-frame ablations can be trained by adding `--history-frames 1` to
`scripts/train_atomistic.py` and evaluated with the same flag in
`scripts/evaluate_atomistic.py`.

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

1. Use the WESTPA dataset bridge on a real or fixture WESTPA run.
   - Keep the existing pcoord lineage extractor as the smoke-test path.
   - Provide a segment coordinate `.npz` keyed by `n_iter` and `seg_id` when
     training the atomistic path from WESTPA lineages.
   - Preserve `window_mask`/`frame_mask` support for variable-length histories.

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
   - Start from `StrideRuntimeScorer`, which computes STRIDE scores from active
     walker histories and exposes them to the mapper.
   - Keep fallback to simple value/distance bins if model loading fails.

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
