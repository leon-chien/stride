AGENT.md

This file provides guidance to Agent Code (http://agent.ai/code) when working with code in this repository.

## Repository state

The repo is in Stage 1.2 pilot setup. It contains the architecture docs, uv/hatch project metadata, Typer CLI, Pydantic settings, mdCATH Tier 1/smoke domain lists, a Stage A mdCATH HDF5-to-Zarr pilot converter, protocol placeholders, and CPU-safe tests. The implementation below remains the design contract; update it as code lands. The repo dir is `stride`; the package is `stride` (`src/stride/`).

## Project: STRIDE

**Adaptive Trajectory Learning and Scheduling for Protein Molecular Dynamics.** Given the recent history of a protein MD trajectory, predict imminent conformational transitions and use those predictions plus a coverage-driven novelty signal to allocate scarce simulation compute toward trajectories most likely to discover new conformational states. Eventually drives a real SLURM/Ray cluster (today: offline trajectory replay).

**This is not** a structure predictor, a folding predictor, or a replacement for MD. The contribution is a *trajectory intelligence + scheduler* that beats FAST (Zimmerman & Bowman, *JCTC* 2015) and REAP (Shamsi, Cheng, Shukla, *J. Phys. Chem. B* 2018) on rare-state discovery per simulated nanosecond. Without beating FAST and REAP on a pre-registered benchmark, there is no paper.

The scheduler is the load-bearing scientific contribution. Prediction quality is in service of the scheduler — design tradeoffs always favor the prediction → scheduler → discovery-efficiency loop over standalone classifier polish.

## Tooling and commands

Python only (no C++/Rust/Java/CUDA kernels in our code; we depend on prebuilt CUDA wheels). Pinned stack: **CUDA 12.4, Python 3.11, PyTorch 2.4, `mamba-ssm` 1.2.x, `causal-conv1d` 1.2.x**. Once `pyproject.toml` exists:

| Task | Command |
|---|---|
| Install / sync deps | `uv sync` |
| Run CLI | `uv run stride <subcommand>` |
| Tests (all) | `uv run pytest` |
| Tests (single) | `uv run pytest tests/path/to/test_file.py::test_name` |
| Lint + format | `uv run ruff check .` / `uv run ruff format .` |
| Type check | `uv run pyright` |

**Per-step ritual:** each sub-stage (§1.1, §1.2, §2.1, …) ends with doc updates + ruff/pyright/pytest + commit + push to `main`. Full checklist in IMPLEMENTATION.md → "Per-step ritual" section.

Stack: `uv` + `hatchling` build backend, `typer` CLI, **Lightning Fabric** trainer (not Lightning Trainer, not Accelerate, not raw PyTorch — Fabric owns ~80 LoC of explicit train loop with FSDP/DDP/bf16 baked in), **Pydantic v2** configs (not Hydra — pyright-friendly, validators for free), **W&B free tier** for tracking (Aim local fallback). PyG ≥ 2.5 for `torch.compile` integration, `torch_cluster` for k-NN, `MDAnalysis` + `mdtraj` for MD I/O, `deeptime` for VAMPnets / TICA / PCCA+, **OpenMM 8.x** for live MD (not GROMACS — the Python streaming-frames story is what makes online scheduling tractable).

## CLI surface (`stride` via Typer)

- `stride preprocess` — Stage A: convert mdCATH HDF5 → Cα/feature Zarr shards; the pilot accepts `--domains`, `--input-root`, `--output-root`, `--dry-run`, and `--force`
- `stride pretrain-vampnets` — Stage A0: per-CATH-topology VAMPnet labels (offline, batch)
- `stride train --stage {1,2}` — Stage 1 (encoder + VICReg + GRL) or Stage 2 (frozen encoder + VAMP-2 head + transition head + Mamba)
- `stride evaluate` — held-out test evaluation against literature-named states
- `stride benchmark` — scheduler comparison vs Random / Round-Robin / RMSD-velocity / tICA-distance / MSM-counts / FAST / REAP / AdaptiveBandit
- `stride serve` — bring up Ray actors (Scheduler, NoveltyArchive, InferenceActors)
- `stride visualize` — figures, embedding visualizations, coverage diagnostics

## Datasets

| Dataset | Role | Size | License | Status |
|---|---|---|---|---|
| **mdCATH** (Mirarchi, Giorgino, De Fabritiis, *Sci Data* 11:1299, 2024) | Primary training | 3.3 TB HDF5 → ~620 GB Zarr after processing; 5,398 CATH domains × 5 T (320/348/379/413/450 K) × 5 replicates × ~464 ns | CC-BY-4.0 | ✓ verified, `huggingface.co/datasets/compsciencelab/mdCATH` |
| **STRIDE-MD** (Vander Meersche et al., *NAR* 2024) | Stage-1 pretraining + held-out OOD eval (different FF: CHARMM36m vs mdCATH's CHARMM22*) | 1500+ proteins × 3 × 100 ns | CC-BY (verify) | ⚠ **license + bulk-download URL must be re-verified before use** |
| **GPCRmd** (Rodríguez-Espigares et al., *Nat. Methods* 2020) | Held-out membrane-protein eval (named states: ionic lock, NPxxY, TM6 swing) | GPCR-specific | varies per deposition | ⚠ **per-deposition license audit required before training** |
| **DESRES Anton — BPTI 1 ms** (Shaw et al., *Science* 2010) | **Headline benchmark** — 5 PCCA+ macrostates are the gold-standard reference | small | DESRES request, terms-of-use restricts redistribution | ⚠ access by request only |
| **Public re-simulations of fast folders** (villin / Trp-cage / WW / chignolin / NTL9, e.g. on Zenodo) | Time-to-first-fold benchmark | small | per-trajectory | ⚠ **license per trajectory** |
| **Alanine dipeptide** (Chodera et al., *J. Chem. Phys.* 2007) | Sanity check — must recover (φ, ψ) basins above random | trivial | open | ✓ |
| **CryptoSite** (Cimermancic et al., *J. Mol. Biol.* 2016) | Qualitative case study only — *not* an MD benchmark, it scores cryptic-pocket pairs | small | open | v2 stretch |

**Out of scope:** MISATO (different problem — protein-ligand binding), BioEmu samples (synthetic from a generative model — not ground-truth dynamics), Folding@home aggregate (FF heterogeneity unmanageable for v1).

**Verification debt** (resolve before code touches these): STRIDE-MD bulk endpoint + license; GPCRmd per-deposition license; public fast-folder re-simulation licenses. mdCATH alone (CC-BY-4.0) is safe for unblocked work.

## Pipeline architecture (converged)

```
Stage A0: per-CATH-topology VAMPnet pretraining (offline, batch on A100s, ~10 GPU-days)
            ↓ produces label tensors labels/{protein_id}/vamp_{N}.npy
Stage A:  mdCATH HDF5 → Cα + Cβ + (φ,ψ,ω) + DSSP + SASA Zarr-v3 shards (~620 GB)
            ↓
Stage 1:  Encoder pretraining (GVP-GNN + VICReg novelty head + GRL FF-adversarial), ~2 days A100
            ↓ produces FF-invariant frozen encoder
Stage 2:  Main training (frozen/0.1×-LR encoder + Mamba temporal + VAMP-2 head + transition head), ~5 days A100
            ↓ produces teacher model
[optional]  Distill teacher GVP → EGNN student for low-latency online inference
            ↓
Inference:  Ray actors score live trajectory windows
            → Hybrid acquisition: score = μ_pred + κ·σ_pred + c·√(log t / N_visits(region))
            → Scheduler dispatches MD jobs via SimulationDriver protocol
```

## Input features (per residue, per frame)

Stored on disk (Stage A output, ~27 B/residue/frame, fp16 where safe):
- **Cα coordinates** (3 floats fp16)
- **Virtual Cβ pseudo-atom** (3 floats fp16; geometric construction for GLY) — recovers ~80% of the rotameric signal at minimal cost
- **Backbone torsions** (φ, ψ, ω) as (sin, cos) pairs (6 floats fp16)
- **DSSP 8-class secondary structure** (uint8)
- **SASA** (Shrake–Rupley, 1 float fp16)
- **Residue type embedding index** (uint8)
- **Force-field tag** (uint8) — every frame is FF-tagged at the source

Computed on-the-fly in the dataloader:
- **k-NN edges, k=16, radius cap 12 Å** — `torch_cluster.knn_graph` on (W, N_res, 3), ~200 µs at W=64
- **RBF distance** (16 Gaussians, 2–20 Å) — SchNet-style (Schütt et al., *J. Chem. Phys.* 2018)
- **Local IPA-style frame** (N–Cα–C → orthonormal triad, AlphaFold-IPA sense; Jumper et al., *Nature* 2021) — ~5 µs/residue
- **Relative-orientation quaternion** between local frames i and j on each edge
- **Sequence separation RBF** + sequence-distance virtual edges (i, i+k for k ∈ {1,2,3,5,10}) to carry long-range allosteric coupling that pure spatial k-NN misses
- **χ1 sidechain torsion** (sin, cos) — read from a *separate* heavy-atom-sidechain Zarr at 200 ps stride for the labeled subset (~200 GB)

**Deferred to v2:** χ2–χ4, full sidechain heavy-atoms, explicit solvent.

**Centering: Cα-centroid per frame, no Kabsch alignment to a reference.** Aligning to a reference *defines* what counts as motion and biases discovery. Equivariance is the model's job.

**Stride: 100 ps for primary corpus** (4640 frames per replicate; ~37.5M training frames total). Adaptive per-protein stride driven by the per-topology VAMPnet's slowest implied timescale (target 5–10 frames per relaxation). A 10 ps sub-dataset (~10 proteins, ~20 GB) exists for stride-sensitivity ablation only. **Do not claim** prediction of surface χ1 hops at 100 ps stride — they are aliased.

## Storage

Zarr v3, Blosc-Zstd-3 + BITSHUFFLE, **256-frame × 512-residue × fp16 chunks (~50 MB compressed)**. fp16 is safe for Cα coords (range ±100 Å, precision 0.01 Å well below thermal 1 Å); validated by roundtrip RMSD < 0.05 Å on a sample. Layout:

```
stride-data/
  coords.zarr/         (N_traj, T_frames, N_res_max, 3) fp16, NaN-padded
  features.zarr/       backbone torsions, DSSP, SASA, residue type
  sidechain.zarr/      (separate, 200 ps stride, labeled subset only)
  residue_mask.zarr/
  metadata.parquet     (traj_id, domain_id, cath_class, T_kelvin, replicate, n_frames, n_res, ff_tag, sha256_chunk0)
  labels/{protein_id}/vamp_{N}.npy   (Stage A0 output, frozen)
  splits/by_topology.json            (CATH-T-level 80/10/10, version-controlled)
```

Stage A is one-shot, idempotent, produces a `manifest.json` with `dataset_hash = sha256(sorted(per-trajectory hashes))`. Every training run records `dataset_hash`; mismatch → loud failure. The current pilot records actual frame cadence when present and otherwise uses a documented `frame_dt_default_ps=1000.0`; it does not upsample frames.

## Model architecture

### Per-frame encoder (teacher) — GVP-GNN, committed

**GVP-GNN** (Jing, Eismann, Suriana, Townshend, Dror, *ICLR* 2021) is the architectural commitment for the teacher. Native scalar + vector channels propagate orientation through message passing — the property EGNN can't deliver, and the one we need for sidechain reorientation, helix-axis rotation, and ionic-lock dynamics. This is the strongest version of the science argument and it determines the recruiter-readable signature of the project.

**Target config:** `s_dim=128 / v_dim=16`, **3 GVP message-passing layers**, k=16, attention pooling per frame. Smaller than the original 256-dim / 4-layer proposal — see the trainability section below for why this is the committed config, not a fallback.

**Per-residue node encoding (one node per residue, *not* per atom):**
- **Scalar features (s):** residue-type embedding, (φ, ψ, ω) sin/cos, DSSP 8-class one-hot, SASA, force-field tag.
- **Vector features (v):** Cα→Cβ unit vector (Cβ as a vector channel on the Cα node, *not* a second node — keeps N at residue count, ~300 max), Cα→Cα_prev, Cα→Cα_next. Standard GVP-protein input pattern from Jing et al.

**Edge encoding (k=16, 12 Å radius cap):**
- Scalar: RBF distance (16 Gaussians, 2–20 Å), sequence separation RBF, contact-type indicator.
- Vector: Cα_i → Cα_j unit vector. Relative orientation between local IPA frames.

Per-frame pooling: **attention pooling** to a single token per frame. Mean pooling destroys per-residue allosteric signal.

### Trainability — GVP on a 4080 at the committed config

**The 4080 is a debugging GPU, not a primary training GPU for the teacher.** Stage 1 + Stage 2 teacher training run on A100s. The 4080 must support: forward+backward at the committed config to validate gradients flow, equivariance unit tests, scheduler/Ray work end-to-end, single-protein ablations, and small-batch correctness checks. That's the constraint we size the 4080 envelope for.

**Memory budget at `s_dim=128 / v_dim=16`, 3 GVP layers, k=16, N=300, W=64, B=8, gradient accumulation = 4** (effective batch 32):

| Component | Estimated peak VRAM (bf16, with grad checkpointing on GVP) |
|---|---|
| GVP encoder (3 layers, edge messages dominate) | ~3.0 GB |
| Per-frame attention pooling | ~0.3 GB |
| Mamba temporal head (W=64, D=128, 2 layers) | ~0.8 GB |
| Heads (Koopman residual, VAMP-2 scratch in fp32, VICReg) | ~0.6 GB |
| Autograd graph + PyG batching scratch | ~2.0 GB |
| Optimizer state (AdamW master fp32) | ~0.05 GB |
| CUDA workspaces / fragmentation | ~1.5 GB |
| **Total** | **~8.3 GB** ← fits in 16 GB with 7 GB headroom |

**Mandatory levers** (these are *defaults*, not optimizations):
1. **Gradient checkpointing on every GVP layer.** ~30% wall-clock cost, ~70% activation-memory savings. Use `torch.utils.checkpoint.checkpoint` per layer's forward.
2. **bf16-mixed everywhere** with `autocast(enabled=False)` around (a) the equivariant vector-update step (numerically fragile near zero-norm vectors), (b) VAMP-2 covariance whitening.
3. **PyG batching with bucket-by-residue-count.** Pad each batch to the next bucket boundary in {128, 192, 256, 320}. Fixed buckets prevent worst-case-protein blowup; the dataloader rejects single proteins with N > 320 (a small fraction of mdCATH).
4. **Window length W=64 frames** (6.4 ns at 100 ps stride). W=32 is the fallback knob if memory is tight in practice.

**If the day-1 measurement on the 4080 disagrees with this estimate** — `torch.cuda.max_memory_allocated()` at B=8, W=64, N=300 — adjust knobs in this order *before* changing architecture:
1. W: 64 → 32 (halves temporal memory)
2. B: 8 → 4 with accumulation 8 (halves activation memory)
3. GVP layers: 3 → 2 (cuts encoder memory ~33%)
4. `s_dim`: 128 → 96 (cuts ~25% of edge-message memory)
5. **Only if all of the above fail:** fall back to EGNN with vector channels (`s_dim=128, v_dim=16`, 4 layers) and document the reason in the README. This is a documented fallback, not the plan.

**A100 budgets** — the GPUs that actually train the teacher:

| GPU | Stage | Config | Batch | Wall-clock estimate |
|---|---|---|---|---|
| A100 80 GB | Stage 1 (encoder pretrain) | GVP s_dim=128/v_dim=16, 3L, W=64 | B=64 | ~2 days |
| A100 80 GB | Stage 2 (main train) | + Mamba + heads, encoder frozen / 0.1× LR | B=64 | ~5 days |
| A100 40 GB | either stage | same | B=32 + accum 2 | ~1.4× the 80 GB time |

**No FSDP, no DeepSpeed.** ~4 M params is laughable for sharding; communication overhead dominates until ~1 B params. Plain DDP across multiple A100s gives near-linear scaling.

### GPU access plan — what you actually need

You have a 4080 and *say* you have A100 access. Confirm or budget for the following before starting Stage 1, because Stage A0 + Stage 1 + Stage 2 + benchmark cumulatively need real A100 time:

| Stage | A100 time needed | Notes |
|---|---|---|
| Stage A0 — VAMPnet labels (Tier 1 + Tier 2) | ~22 GPU-days on 1× A100, embarrassingly parallel | 4× A100 = ~5.5 days |
| Stage 1 — encoder pretrain | ~2 GPU-days on 1× A100 80 GB | One-shot |
| Stage 2 — main train | ~5 GPU-days on 1× A100 80 GB | Multiple seeds × ~5 |
| Headline benchmark (≥ 5 seeds × ~10 proteins × ~1 µs MD) | ~10–20 GPU-days A100 for the MD itself | OpenMM is the bottleneck, not the model |
| **Total minimum** | **~40–60 A100 GPU-days** | Cloud cost at $2/hr ≈ $2k–$3k; free if university/lab access |

If A100 access is uncertain, the order of operations to de-risk: (1) confirm access first, (2) Stage A on the 4080 (CPU + I/O bound, no A100 needed), (3) Tier 1 VAMPnets on a single A100 to validate the label pipeline before committing to Tier 2. If A100 access falls through, the whole project ceiling drops by an order of magnitude — accept it, scope Stage 1 to ~50 proteins instead of 2000, and ship a smaller paper.

**Equivariance unit test** is mandatory in CI: rotate every input uniformly at random, assert predictions unchanged within fp32 numerical tolerance. People publish "equivariant" models that aren't, due to alignment / centering bugs.

### Online student (low-latency scheduler inference)

**Distill teacher GVP → student EGNN, `dim=128`, 4 layers.** Teacher inference at 300 residues is ~25–30 ms on a 4080; the scheduler tick budget can't absorb this for the candidate sets we evaluate. Student EGNN runs at ~8 ms/candidate. The scheduler does not need the teacher's full expressivity — it needs a fast novelty/transition score. Student is what gets shipped in the Ray inference actor; teacher stays in the offline training pipeline.

### Temporal head

**Mamba-1** (Gu & Dao, 2023; `mamba-ssm==1.2.x`, `causal-conv1d==1.2.x`) primary — linear in sequence length, native to 10⁴–10⁵ frame trajectories where attention is quadratic-dead. Stable on Ada (4080, SM 8.9) and Ampere (A100, SM 8.0) since v1.2. **Plus chunked self-attention (window=128 frames, rotary positional encoding** per Su et al., 2021) on top of Mamba for local-pattern detection.

`torch._dynamo.disable` on the Mamba module (custom CUDA kernel doesn't trace cleanly). After the GVP encoder produces per-frame embeddings, pool to (B, W=64, D=256) — Mamba's bread and butter — before the SSM call.

**Fallback ladder if `mamba-ssm` flakes on a CUDA/PyTorch update:** S5 (Smith et al., *ICLR* 2023; pure PyTorch, ~2× slower), then GRU. GRU is acceptable at W=64 — Mamba's parallel scan only earns its keep at W ≥ 256.

### Heads

1. **Transition head:** **Koopman-residual regression**, *not* binary classification. The target is the VAMPnet-derived continuous Koopman residual ‖K·z(t) − z(t+τ)‖ — high residual ⇒ likely transition. Continuous, computable from a streaming window with no global archive, which gives offline ↔ online consistency. **Bayesian last-layer (Laplace approximation, recomputed once per epoch)** for epistemic uncertainty σ_pred — used by the acquisition function.
2. **VAMP-2 head:** single-resolution N=16, multi-task auxiliary loss `0.1 · VAMP-2`. Multi-resolution {N=4, 16, 64} is used at *label generation* time only (Stage A0). VAMP-2's generalized-eigenvalue objective requires fp32 around the covariance whitening — wrap with `autocast(False)`.
3. **Novelty embedding head (separate sub-network, Stage 1 only):** **VICReg** (Bardes, Ponce, LeCun, *ICLR* 2022), output dim 64. Variance + covariance regularizers prevent dimensional collapse — the *exact* failure mode that biases the scheduler toward already-seen modes. Reject SimCLR (negative pairs ill-defined for trajectories) and BYOL (collapse pathologies). Positive pairs: same-trajectory frames within Δt < 1 ns + same-frame SE(3) augmentation (random rotation; coords σ=0.5 Å Gaussian).
4. **GRL FF-adversarial head (Stage 1 only):** Ganin et al., 2016 — gradient-reversal on FF identity, λ schedule warming 0→1.0 over first 20% of Stage-1 steps. Lives in Stage 1 only; Stage 2 inherits an FF-invariant frozen encoder. Do not run GRL jointly with Stage 2's multi-task loss — known instability, would tip us into FSDP-debugging hell on a 2-week tail.

## Training stages

**Two-stage, not joint.** Joint training of (transition + VAMP-2 + VICReg + GRL) on a ~4 M-param equivariant encoder is unstable at bf16. The decomposition:

| Stage | Inputs | Heads active | Encoder | Duration |
|---|---|---|---|---|
| **A0** offline VAMPnet pretrain | mdCATH per CATH-topology | VAMPnets at N ∈ {4, 16, 64} (multi-resolution) | small MLPs in `deeptime` | ~10 GPU-days, batch |
| **1** encoder pretrain | mdCATH + STRIDE-MD | VICReg + GRL | GVP-GNN trainable | ~2 days A100 |
| **2** main train | mdCATH labeled corpus | Transition (Koopman-residual) + VAMP-2 (N=16) | encoder frozen or 0.1× LR | ~5 days A100 |

**Tiered VAMPnet labeling** (Stage A0 reality — fitting 5400 VAMPnets sequentially is 75 days on one A100):
- **Tier 1 (~50 headline + ablation proteins):** full multi-resolution VAMPnets, ~50 GPU-hours
- **Tier 2 (~2000 mdCATH domains):** single-resolution N=16 VAMPnets, ~500 GPU-hours
- **Tier 3 (~3400 remaining domains):** TICA + k-means surrogate (~30 s/protein), used for auxiliary pretraining only

A `vampnet_health` gate (VAMP-2 score on a held-out lag time vs random-projection baseline) auto-falls-back to TICA when fitting fails to converge per-protein.

**Mixed precision: bf16-mixed on both 4080 and A100** (Ada has full bf16 throughput). NOT fp16 — recurrent activations underflow to NaN. fp32 wrap (`autocast(enabled=False)`) around: (a) GVP/EGNN equivariant coordinate updates, (b) VAMP-2 covariance whitening.

**Checkpointing:** safetensors weights + optimizer.pt (http://optimizer.pt/) + rng_state.pt (http://rng_state.pt/) + dataset_cursor.json, every 30 min wall-clock, atomic via tmp+rename. Retention: last 3 + every 10th + best-by-val-AUROC. Resume verifies dataset_hash and replays dataloader cursor; CI test asserts uninterrupted-vs-resumed runs match losses to 1e-5.

## Cluster runtime (Ray)

**Ray, not Dask, not raw SLURM.** Ray's actor model is the right primitive for the shared-mutable novelty archive. SLURM `sbatch` submission latency (~hundreds of ms) is fine for launching MD jobs, too slow for the 5-second scheduler control loop.

**Topology:**
- 1 × `SchedulerActor` — priority queue, single-writer
- 1 × `NoveltyArchiveActor` — FAISS-IVFPQ index (4096 lists, 8-byte PQ codes), single-writer; readers refresh local mirrors every 30 s; snapshots every 5 min
- N × `InferenceActor` — GPU-pinned, holds the distilled student model, scores incoming frame-windows
- M × `FrameIngester` — `inotify` (local) or filesystem polling (SLURM shared FS), validates frame integrity (no NaNs, atom-count matches), pushes to inference queue

**Latency budget:** end-to-end frame-arrives → score-computed → priority-updated **< 200 ms p50, < 1 s p99**. Student forward at W=64 is ~5 ms on A100; FAISS k-NN over 1 M archive entries ~2 ms; Bayesian last-layer σ_pred ~1 ms. Headroom 30×.

**Scheduler tick: every 5 s.** 50 frames of 100 ps biological signal — faster decisions are noise.

**Failure modes:**
| Mode | Detection | Mitigation |
|---|---|---|
| Inference actor dies | Ray health-check 10 s | Auto-respawn, pending windows re-queued |
| Archive actor dies | same | **Hard halt** — pause scheduling until rehydrate from snapshot |
| Stuck MD job (no frames 60 s) | Driver `poll` empty for N polls | Halve trajectory priority, alert |
| One trajectory hogs queue | Per-trajectory time-on-CPU monitor | **Hard cap: ≤ 25% of compute budget per trajectory at any time** |
| Priority deadlock (low-prior never picked) | Coverage diagnostic | **ε-floor 0.05** — minimum random selection rate |
| Inference queue backpressure | Queue depth > 1000 | Pause new MD launches — better under-utilize than degrade decisions |
| FAISS writer queue overflow | Queue depth > 10 k | Reservoir + most-recent sub-sample to bound compute |

## SimulationDriver protocol — locked from day 1

The same `SimulationDriver` interface ships in three implementations so the scheduler logic is identical offline and online:

```
class SimulationDriver(Protocol):
    def list_active(self) -> list[TrajectoryHandle]
    def submit(self, init_state: State, budget_ns: float, priority: float) -> TrajectoryHandle
    def poll(self, handle) -> Optional[FrameWindow]   # non-blocking
    def extend(self, handle, additional_ns: float) -> None
    def terminate(self, handle) -> None
```

| Phase | Driver | Cluster | Goal |
|---|---|---|---|
| 0 (now) | `ReplaySimulationDriver` (reads pre-recorded mdCATH from Zarr, pretends to be live) | none | Train + validate scheduler beats random on replay |
| 1 | `OpenMMLocalDriver` (subprocess + shmem / Unix socket frame stream) | 4080 workstation | Validate end-to-end model→scheduler→MD→frames loop |
| 2 | `SlurmSimulationDriver` (sbatch → shared FS → watcher) | 4–8 GPU SLURM | Validate at concurrency ≥ 8 |
| 3 | same | full A100 | Headline experiment |

**Restarts: snapshot-only, 10 ns checkpoint cadence** via OpenMM `Simulation.saveState()`. Restart from arbitrary frame is 10× the disk and not in v1.

**Container: single `stride/runtime:{git-sha}`, < 4 GB.** CUDA 12.4 / Python 3.11 / OpenMM 8.x / PyTorch 2.4 / mamba-ssm 1.2.x. Apptainer/Singularity wrapper for academic SLURM clusters.

## Acquisition function (the load-bearing scheduler decision)

**Reject the strawman `α·p_transition + β·novelty` linear blend.** Two reasons:
1. `p_transition` peaks at *known* basin boundaries — pulls compute *back* to explored regions, the opposite of what we want.
2. Aleatoric/epistemic prediction uncertainty and coverage-driven novelty are different kinds of uncertainty; adding them with fixed weights is dimensionally incoherent.

**Hybrid acquisition, ~3 ms/candidate:**

```
score(τ_i) = μ_pred(τ_i) + κ · σ_pred(τ_i) + c · √(log t / N_visits(region(τ_i)))
```

- `μ_pred` — Koopman-residual transition prediction (mean of Bayesian last-layer)
- `σ_pred` — epistemic uncertainty from Bayesian last-layer (Laplace approximation, recomputed once per epoch). This is the Thompson-sampling-flavored exploration over *prediction* uncertainty.
- `√(log t / N_visits(region))` — UCB1-style count-based novelty bonus. `N_visits` from FAISS range query against the IVFPQ archive; `region` is the IVF cell. Provable regret bounds for stationary bandits.
- Two knobs (κ, c). Both interpretable in same units as the scaled prediction.

**Novelty embedding archive: FAISS `IndexIVFPQ`,** 4096 lists, 8-byte PQ codes, trained on first 100 k embeddings. At 1 M entries: ~10 MB index, ~1 ms query. At 10 M: ~100 MB, ~5 ms. **Single writer** → no index-corruption races. **MinHash/LSH dedup actor** — prevents one big basin from dominating the "known" set and starving novelty.

**Per-batch z-score normalization** of the novelty score within the active candidate pool — robust to embedding drift across training.

**Deferred to v2:** MAP-Elites / quality-diversity grid (Mouret & Clune, 2015) — 1500-protein bookkeeping in the archive actor is weeks of engineering for incremental science gain over count-based + LSH dedup. **Periodic VAMPnet refit on the archive** also v2 — 6-week build with on-call burden; gate v2 work on instrumented evidence (track frame fraction with novelty score > 95th pretraining percentile).

## Evaluation rules — anti-circularity is non-negotiable

**The biggest project-killer is using the same clustering for both training labels and "states discovered" evaluation.** A reviewer kills this immediately.

**Training labels** ← Stage-A0 VAMPnets (per-CATH-topology, frozen).
**Evaluation states** ← *independent* of the model. Two sources, in priority order:

1. **Literature-named states with pre-registered structural definitions** — for proteins where the field has consensus:
   - BPTI: 5 PCCA+ macrostates from Shaw et al. 2010 long equilibrium MSM
   - villin / Trp-cage / WW / chignolin / NTL9: folded / unfolded / intermediate
   - kinases (src, Abl): DFG-in / DFG-out / DFG-up; αC-in / αC-out (each defined by explicit torsion thresholds, e.g., DFG-out = ψ(D) ∈ [-180°, -90°] AND χ1(F) ∈ [60° ± 60°])
   - GPCRs (β2-AR if license confirms): ionic-lock R3.50–E6.30 distance, NPxxY, TM6 outward swing
2. **PCCA+ macrostates from a reference long-equilibrium MSM** — built once, offline, with literature-standard parameters, using a feature basis **disjoint** from what the model trained on (e.g., heavy-atom-sidechain TICA when the model trained on Cα+Cβ).

The reference MSM / named-state structural definitions are **locked before any adaptive runs** in `experiments/headline/protocol.md`. Pre-registration is load-bearing — once unlocked, the test set is touched twice (headline + camera-ready) and no more.

### Headline metric

**AUC of states-discovered curve vs simulated nanoseconds** on a fixed eval list of ~10–15 proteins (BPTI primary; villin/Trp-cage/WW/NTL9/chignolin/src-kinase/β2-AR-if-licensed). "State discovered" = at least one frame from the adaptive run lands within the structural definition of a named state, OR ≥ 1 ns of accumulated simulation time within tolerance.

**Statistical test:** paired bootstrap over (protein × seed), 1000 resamples, 95% CI on AUC delta vs each baseline. **Win iff CI excludes 0.** ≥ 5 seeds.

### Splits

**CATH-topology level (the third H — Topology / fold), 80/10/10.** Train and test never share a fold. Splits are version-controlled, regenerated only via a fixed-seed script when adding new data.

### Required baselines

Without beating items 5, 6, and 7 there is no paper.

1. Round-robin (uniform allocation) — trivial floor
2. Random with replacement — sanity check
3. RMSD-velocity heuristic — folklore baseline, surprisingly hard to beat
4. tICA-distance-to-nearest-discovered-state heuristic
5. **MSM counts-based adaptive sampling** (Bowman, Ensign, Pande, *JCTC* 2010) — the canonical approach
6. **FAST** (Zimmerman & Bowman, *JCTC* 2015) — modern standard
7. **REAP** (Shamsi, Cheng, Shukla, *J. Phys. Chem. B* 2018) — direct competitor in spirit
8. **AdaptiveBandit** (Pérez et al., *JCTC* 2020) — bandit-style adaptive sampler
9. Ablations: ours minus novelty head; ours minus transition head; novelty embedding from frozen ESM-2 vs VICReg-trained vs raw-PCA

### Mandatory sanity checks

- **Alanine dipeptide φ/ψ basin recovery** — the MSM "hello world." If we don't beat random here, we are broken.
- **Equivariance unit test** — rotate input uniformly at random, predictions invariant. People publish "equivariant" models that aren't.
- **Time-shuffle permutation test** — shuffle the time axis. Transition predictor performance must collapse, else the model is using non-temporal structural shortcuts.
- **Time-reversal symmetry test** — VAMP scores invariant under trajectory reversal in equilibrium regions.
- **FF-swap generalization test** — train on CHARMM, evaluate on AMBER held-out; report the gap honestly even if we don't fix it.
- **Calibration of `μ_pred`** — temperature scaling (Guo et al. 2017) post-hoc; report ECE not just AUROC.

### Coverage diagnostics (logged every run)

- TICA-coverage on a 50×50 grid in held-out reference TICA-2D space
- Hidden-state recall: time-to-first-visit per named state
- Entropy of state visitation — must beat random's natural-Boltzmann entropy

## Hardware budget summary

The detailed encoder budget lives in the **Trainability** section above. This is the cross-config table for sanity at a glance. Verify on day 1 with `torch.cuda.max_memory_allocated()`; if measurement disagrees with the spreadsheet, the spreadsheet is wrong, not the GPU.

| Configuration | Params | 4080 (16 GB) | A100 80 GB | A100 40 GB |
|---|---|---|---|---|
| **Teacher GVP** (s=128, v=16, **3L**, k=16) + Mamba + heads, W=64, N≤300 | ~4 M | B=8 + accum=4 *(debug only)* | B=64 | B=32 + accum=2 |
| Student EGNN (d=128, 4L) + Mamba + heads, W=64, N=200 | ~2.5 M | B=32 | trivial | trivial |
| dim → 256 or W → 256 | — | tight / A100-only | OK | tight |
| All-atom graphs (~2000 nodes) | — | infeasible | B=8 | infeasible |

**FSDP / DeepSpeed: not used.** ~4 M params is laughable for sharding; communication overhead dominates until ~1 B params. Plain DDP across multiple A100s gives near-linear scaling. **Heuristic:** the 4080 is for debugging, equivariance tests, single-protein ablations, and Ray/scheduler work. Anything that runs > 8 hours moves to A100.

**Online inference at full cluster scale:** ≤ 200 candidates per 5 s tick × ~10 ms student forward = ≤ 2 s per tick. Headroom 2.5×.

## Phasing (build order)

Earlier phases gate meaningful evaluation in later ones. Do not skip ahead.

1. Stage A pipeline (mdCATH HDF5 → Cα/feature Zarr) + integrity manifest
2. Stage A0 — VAMPnet labels (Tier 1 first, ~50 proteins, validate before running Tiers 2–3)
3. Baselines: contact-map MLP, contact-map GRU
4. Stage 1 + Stage 2 training + sanity checks (alanine, equivariance, time-shuffle)
5. `ReplaySimulationDriver` + UCB1+TS scheduler + offline benchmark vs MSM-counts/FAST/REAP
6. Distill teacher → student; deploy in Ray inference actor
7. `OpenMMLocalDriver` end-to-end on the 4080 workstation
8. `SlurmSimulationDriver` — first cluster run
9. Headline benchmark (pre-registered protocol, BPTI primary)

## Decisions explicitly rejected (and why)

- **PCA + MiniBatchKMeans labels** — clusters by structural variance, not kinetic separation. Use VAMPnets (Mardt, Pasquali, Wu, Noé, *Nat. Commun.* 2018).
- **Plain EGNN as teacher** — GVP-GNN is the committed teacher because vector channels propagate orientation through message passing; EGNN's coord-update term carries orientation only as scalar magnitudes. EGNN remains the *student* (distilled, low-latency online inference) and the documented fallback if GVP fails the day-1 memory measurement at every knob below `s_dim=96`.
- **GRU as primary temporal head** — won't capture μs-scale structure in 10⁵-frame sequences. Mamba.
- **Plain Transformer over frames** — quadratic memory at length 10⁵ is unaffordable.
- **Mean pooling per frame** — destroys per-residue allosteric signal.
- **Kabsch alignment to a reference frame** — defines what counts as motion; biases discovery.
- **Hard binary transition labels with a fixed 3–5 frame persistence threshold** — replaced by continuous Koopman-residual regression + per-protein lag time τ from implied-timescales plot.
- **`α·p_transition + β·novelty` linear blend** — wrong shape (peaks at *known* basin boundaries); replaced by hybrid `μ + κσ + c·√(log t/N)`.
- **Full GP Thompson sampling** — O(n³) cubic in archive size, dies before the Ray actor does. Bayesian last-layer Laplace approximation gets us σ_pred at ~1 ms.
- **MISATO, BioEmu samples, full DESRES Anton training** — different problem, generative samples vs ground truth, license-restricted respectively.
- **Joint multi-objective end-to-end training of (transition + VAMP-2 + VICReg + GRL)** — unstable at bf16 on a ~4 M-param equivariant encoder. Two-stage instead.
- **MAP-Elites grid + periodic VAMPnet refit in v1** — engineering cost dominates incremental science return at v1; gate v2 on instrumented mode-collapse evidence.
- **Hydra config system** — overkill, pyright-hostile. Pydantic v2 + Typer.
- **Frame-level train/val/test splits** — frames same trajectory correlate over > 100 ns; leakage is severe.
- **Same clustering for training labels and evaluation states** — circular; project-killer.

## Reference reading

- Markov State Models: Husic & Pande, *JACS* 2018; Prinz et al., *J. Chem. Phys.* 2011; Bowman, Pande, Noé eds., Springer 2014; Chodera & Noé, *Curr. Opin. Struct. Biol.* 2014.
- VAMPnets / VAMP-2: Mardt, Pasquali, Wu, Noé, *Nat. Commun.* 2018; Wu & Noé, *J. Nonlinear Sci.* 2020.
- tICA: Pérez-Hernández et al., *J. Chem. Phys.* 2013; Schwantes & Pande, *JCTC* 2013.
- PCCA+: Röblitz & Weber, *Adv. Data Anal. Classif.* 2013.
- GVP-GNN: Jing, Eismann, Suriana, Townshend, Dror, *ICLR* 2021.
- EGNN: Satorras, Hoogeboom, Welling, *ICML* 2021.
- Mamba: Gu & Dao, 2023. S5: Smith, Warrington, Linderman, *ICLR* 2023.
- VICReg: Bardes, Ponce, LeCun, *ICLR* 2022.
- DANN / GRL: Ganin et al., *JMLR* 2016.
- Adaptive sampling baselines: Bowman, Ensign, Pande, *JCTC* 2010 (counts-based); Zimmerman & Bowman, *JCTC* 2015 (FAST); Shamsi, Cheng, Shukla, *J. Phys. Chem. B* 2018 (REAP); Pérez et al., *JCTC* 2020 (AdaptiveBandit).
- Anton long trajectories: Shaw et al., *Science* 2010 (BPTI); Lindorff-Larsen et al., *Science* 2011 (fast folders).
- Datasets: Mirarchi, Giorgino, De Fabritiis, *Sci Data* 2024 (mdCATH); Vander Meersche et al., *NAR* 2024 (STRIDE-MD); Rodríguez-Espigares et al., *Nat. Methods* 2020 (GPCRmd).
- Calibration: Guo, Pleiss, Sun, Weinberger, *ICML* 2017.
