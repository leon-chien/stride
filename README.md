# STRIDE

**Structured Trajectory Representation and Intelligent Dynamic Binning Engine**

STRIDE is a machine learning system for **goal-conditioned rare-event molecular simulation**. The long-term goal is to build a WESTPA-native learned BinMapper that uses trajectory histories to generate adaptive progress-coordinate bins for weighted ensemble simulation.

In plain terms:

> STRIDE learns which simulation walkers are likely to become scientifically valuable, then converts those predictions into adaptive bins that WESTPA can use for weighted resampling.

The current implementation contains a complete toy proof-of-concept: a 2D rare-event simulator, a GRU trajectory-value model, offline replay evaluation, and a first adaptive sampling controller.

---

## Motivation

Weighted ensemble methods such as WESTPA are powerful for rare-event molecular simulation, but their performance often depends heavily on the choice of progress coordinates and binning strategy.

STRIDE aims to learn these binning signals from data.

Instead of manually defining only a distance, RMSD, angle, or contact coordinate, STRIDE learns from trajectory histories:

```text
recent trajectory motion → future rare-event value → adaptive bin assignment
```

## Current Status

STRIDE has completed its first full toy-stage milestone: a WESTPA-like weighted ensemble benchmark with learned trajectory-value binning.

Implemented so far:

- 2D rare-event simulator
- GRU trajectory-value model
- Delayed-label training
- Offline replay evaluation
- Weighted walkers
- Bin-wise split/merge resampling
- Probability-weight conservation
- Static, model-score, and hybrid binning strategies
- Lineage-diversity tracking
- Multi-seed evaluation
- Gated rare-event benchmark where distance alone is insufficient

---

## Version 0–1 Summary: Toy Weighted Ensemble Prototype

The current STRIDE prototype tests whether a learned trajectory-value model can improve rare-event sampling in a simplified weighted ensemble setting.

The model is trained with delayed labels:

```text
recent trajectory window → future rare-event success