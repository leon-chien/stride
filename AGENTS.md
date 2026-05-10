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
biological events, not just novelty, next-frame dynamics, or one NaCl benchmark.

## What Has Been Implemented

Current repository foundation:

- Toy 2D rare-event simulator and weighted-ensemble-like sampling code.
- GRU trajectory-value baseline.
- Delayed-label training and replay evaluation for toy systems.
- Synthetic NaCl reduced-distance benchmark used only as a smoke test.
- Prototype NaCl training, replay, and WESTPA-style adapter files. These are
  benchmark scaffolding, not the product target.
- HDF5 inspection/reading scaffolding for WESTPA `west.h5` files.
- Prototype learned score and quantile bin mappers.

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
- STRIDE should be simulation-agnostic. NaCl is useful for fast debugging, but
  the project goal is broad biological rare-event prediction: binding,
  unbinding, conformational transitions, contact formation, and target-state
  membership.
- eGNN learns spatial molecular state for each frame.
- Temporal Transformer learns time: trajectory direction, commitment, momentum,
  and pre-event history signals across frame embeddings.
- GRU remains a baseline, but the main STRIDE model should be
  eGNN + Temporal Transformer + goal conditioning.
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
- The `stride` conda environment has NumPy and Torch available:

```bash
conda run -n stride python ...
```

- `pytest` was not installed in the checked environment when the deep-learning
  modules were added. Direct function-level validation was run with:

```bash
conda run -n stride python -c "import sys; sys.path.insert(0, 'tests'); import test_deep_value_model as t; t.test_goal_spec_feature_vector_is_deterministic(); t.test_egnn_frame_embedding_is_translation_and_rotation_invariant(); t.test_stride_value_model_outputs_westpa_scoring_heads(); t.test_stride_value_loss_and_quantile_binning_interfaces(); print('deep value tests passed')"
```

Compilation was checked with:

```bash
conda run -n stride python -m compileall src tests
```

## Next Goals

Prioritize the one-person build path. Do not jump straight to a large
foundation model, and do not do more serious training until generalized
WESTPA/data infrastructure exists.

1. Finish generalized WESTPA lineage reconstruction.
   - Extend `src/stride/westpa_plugin/h5_reader.py`.
   - Read iteration groups, segment indices, parent IDs, pcoords, and weights.
   - Reconstruct walker ancestry and descendant trees.
   - Add tiny hand-built lineage tests where event and flux labels are known.

2. Build a simulation-agnostic STRIDE dataset extractor.
   - Input: `west.h5`, goal spec, optional coordinate/topology references.
   - Output: windows, labels, weights, lineage IDs, goal features, metadata.
   - Start with pcoord/reduced-feature windows before requiring coordinates, but
     design the artifact schema so eGNN coordinate windows can be added without
     changing the training API.

3. Wire delayed-descendant labels into training.
   - Use `StrideValueTargets` and `stride_value_loss`.
   - Use NaCl only as a smoke test for the full extraction/training path.
   - Train next on alanine dipeptide or another small geometry benchmark before
     moving to protein/ligand or large conformational systems.
   - Track top-k enrichment, AUPRC, calibration, and replay utility.

4. Add a WESTPA-facing scorer/binning adapter for the new value model.
   - Input: active walker histories and goal spec.
   - Output: bin IDs, value heads, combined score, diagnostics.
   - Include fallback to distance bins or current GRU mapper if model loading
     fails.

5. Add coordinate data support for the eGNN path.
   - Define frame-to-segment coordinate mapping.
   - Add atom feature construction.
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
- Keep NaCl source code as a benchmark until there are stronger biological
  examples, but do not let generated NaCl artifacts or naming dominate the
  project structure.
- Prefer pure PyTorch for the first eGNN implementation to avoid heavy graph
  library setup.
- Add focused tests for lineage labels, invariance, shape contracts, and
  probability-weight conservation.
- Avoid making STRIDE depend on live WESTPA execution for basic unit tests.
