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

### Version 1: WE-style toy resampling

STRIDE now supports a simplified weighted ensemble toy loop with weighted walkers, score-based bin assignment, within-bin resampling, and probability-weight conservation.

On a 2D rare-event benchmark, model-score bins achieved higher final target-reaching probability weight than static distance bins:

| Method | First reached iteration | Final target weight | Target walkers | Unique lineages |
|---|---:|---:|---:|---:|
| Static distance bins | 7 | 0.1250 | 32/256 | 8 |
| STRIDE model-score bins | 10 | 0.2578 | 66/256 | 5 |

This shows that learned trajectory-value bins can concentrate more probability weight into target-reaching regions, while also revealing a diversity tradeoff that motivates novelty- and uncertainty-aware binning.
