STRIDE Implementation Plan

This is the operational guide. AGENT.md is the *what* and *why*; this is the *how* and *in what order*. If you find yourself unsure which file something belongs in: architecture decisions and rejected alternatives → AGENT.md; concrete steps, commands, and gates → here.

## Workflow: laptop → git → server

You code on a MacBook with no usable GPU. Real work happens on remote machines. The discipline:

| Where | Runs |
|---|---|
| **Laptop (local)** | Editing, `ruff`, `pyright`, unit tests on synthetic data (CPU only), small-scale `pytest` integration tests with mocked Zarr/Ray, design and writing |
| **4080 server** | Stage A data conversion, day-1 memory measurements, equivariance tests, single-protein ablations, OpenMM live-MD development, Ray topology development, distilled-student training |
| **A100 (cluster or rented)** | Stage A0 VAMPnet fitting, Stage 1 + Stage 2 main training, headline benchmark MD |

Pattern for every change: write on laptop → `git push` → `ssh server` → `git pull` → run. Nothing is edited on the server. No "it works on my machine" excuses because the laptop never runs the heavy path.

**Repo conventions to lock in early:**
- `pyproject.toml` lists CPU + CUDA dependency groups separately. Laptop installs the CPU group via `uv sync --group cpu`. Server installs `uv sync --group cuda`. Same source tree, different lockfile resolution.
- `.env.local` (gitignored) holds machine-specific paths: `STRIDE_DATA_ROOT`, `STRIDE_MODELS_ROOT`, `STRIDE_W_AND_B_API_KEY`. Pydantic config reads `os.environ`.
- Every script that needs a GPU starts with `assert torch.cuda.is_available(), "this script does not run on the laptop"` so a stray laptop run fails fast.
- Heavy artifacts (Zarr, model weights, FAISS indices) live on the server filesystem and are *never* committed. Laptop has access to the *manifest* (a small JSON describing what's where) so eval scripts can sanity-check what's available without pulling terabytes.

**SSH / data ergonomics:**
- Set up `mosh` or `ssh` with `ServerAliveInterval` so long jobs don't drop.
- Use `tmux` on the server so a dropped connection doesn't kill a 5-day training run.
- W&B logs from the server back to your laptop browser — that's how you watch training without a persistent SSH session.
- For pulling small artifacts (eval reports, plots) back to the laptop: `rsync -avz server:~/stride/experiments/ ./experiments/` excluding the bulk dirs.

---

## One-time prerequisites

### On the laptop
```bash
# Once, ever:
brew install uv git pre-commit
git clone <stride-repo> ~/Projects/stride
cd ~/Projects/stride
uv sync --group cpu
pre-commit install
```

### On the 4080 server
```bash
# Once, after first SSH-in:
# Assumes Linux x86_64, NVIDIA driver for CUDA 12.4 already installed.
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <stride-repo> ~/stride
cd ~/stride

# Step 1: sync the lockfile-tracked stack (torch CUDA + PyG + Ray + OpenMM + ...).
uv sync --group dev --group cuda

# Step 2: install the GPU extras that can't go in the lockfile because they
# need wheels matched to torch + CUDA + Python (mamba-ssm, causal-conv1d,
# torch-cluster, torch-scatter, faiss-gpu).
./scripts/install_server_extras.sh

# Verify:
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
uv run python -c "from mamba_ssm import Mamba; print('mamba ok')"
uv run stride --help
```

### A100 access
Confirm one of: university SLURM allocation, lab cluster, or willingness to spend ~$2k–$3k on rented A100 (Lambda Labs, RunPod, vast.ai (http://vast.ai/)). **Do this first.** If A100 access is shaky, the project ceiling drops 10× — better to know now than at Stage 3.

---

## Datasets — download only, never simulate

You do not run any MD to build the training set. Every byte comes from a public download. The MD you eventually run is *driven by the trained scheduler*, not used to make labels.

| Dataset | How to get it | Where it goes | Approx size |
|---|---|---|---|
| **mdCATH** | `huggingface-cli download compsciencelab/mdCATH --repo-type dataset --local-dir $STRIDE_DATA_ROOT/mdcath_raw` (3.3 TB; budget download time) | `$STRIDE_DATA_ROOT/mdcath_raw/` | 3.3 TB |
| **STRIDE-MD** | `wget` from `https://www.dsimb.inserm.fr/STRIDE/api/STRIDE/info` to discover bulk endpoint, then mirror with `wget -r` or `rsync` if available. **Verify license terms in the download README before mirroring.** | `$STRIDE_DATA_ROOT/stride_md_raw/` | ~600 GB |
| **GPCRmd** (eval only) | Per-deposition download from `https://www.gpcrmd.org/` API. Many depositions are open; some are PI-restricted. Audit license per deposition before pulling. | `$STRIDE_DATA_ROOT/gpcrmd_raw/` | varies |
| **Public fast-folder re-simulations** (eval) | Specific Zenodo deposits — search "Trp-cage molecular dynamics open" / "villin folding trajectory open". License per deposit. | `$STRIDE_DATA_ROOT/fastfolders/` | ~50 GB |
| **Alanine dipeptide** (sanity) | Tiny — included in `deeptime` examples, or download `https://markovmodel.github.io/mdshare/ALA2/` (mdshare). | `$STRIDE_DATA_ROOT/ala2/` | ~50 MB |
| **DESRES BPTI / Anton fast folders** (headline) | Request from D.E. Shaw Research at `https://www.deshawresearch.com/downloads/download_trajectory_science2010.cgi`. Several-week turnaround. License restricts redistribution. | `$STRIDE_DATA_ROOT/desres_bpti/` | small |
| **CryptoSite** (v2 stretch) | `https://modbase.compbio.ucsf.edu/cryptosite/` static structure pairs. | `$STRIDE_DATA_ROOT/cryptosite/` | small |

**Practical:**
- Run all dataset downloads on the 4080 server, not the laptop. They're terabytes and benefit from server-grade I/O.
- Use `tmux` and `aria2c` (or `wget --continue`) so a dropped SSH session doesn't restart a 12-hour download.
- Verify SHA256 of every download against the source's published manifest. Record in `data_provenance.md`.
- DESRES request goes out **week 1** of Stage 1 — you'll wait weeks for it, so start the clock immediately even though you don't need it until Stage 4.

---

## Stage 1 — Foundation (3–4 weeks calendar)

### 1.1 Repo skeleton (laptop, day 1)
```
stride/
  pyproject.toml          # uv + hatchling, CPU + CUDA dep groups
  .env.example            # machine-local environment template
  src/stride/
    __init__.py
    cli.py                # typer entry: `stride <subcommand>`
    config/               # Pydantic v2 schemas
    data/                 # download verification, Zarr writer (Stage A)
    labeling/             # VAMPnet pipeline (Stage A0), TICA fallback
    models/               # GVP, Mamba, heads
    scheduler/            # Ray actors, acquisition, FAISS archive
    drivers/              # SimulationDriver protocol + 3 impls
    eval/                 # named-state definitions, MSM reference, AUC computation
    sanity/               # equivariance test, time-shuffle, time-reversal, FF-swap
  configs/                # yaml configs loaded into Pydantic
  experiments/
    headline/protocol.md  # locked at end of Stage 1
  tests/                  # pytest, runs on laptop CPU
  scripts/
    download_*.sh         # dataset download recipes
  IMPLEMENTATION.md
  AGENT.md
```

Push the empty skeleton, set up CI on day 1 (GitHub Actions: ruff + pyright + pytest). CI runs on the laptop's GitHub runner — no GPU needed for skeleton tests.

### 1.2 Stage A — Zarr conversion pipeline (4080 server, ~1 week)

Implement `stride preprocess` end-to-end:
- Reads mdCATH HDF5 per-domain
- Extracts Cα, virtual Cβ, (φ, ψ, ω), DSSP (via `MDAnalysis.analysis.dssp`), SASA (Shrake–Rupley via FreeSASA), residue type, FF tag
- Writes Zarr v3 with the chunk layout from AGENT.md (256-frame × 512-residue × fp16 chunks, Blosc-Zstd-3 + BITSHUFFLE)
- 100 ps stride
- Folded-state filter: drop frames where Cα-RMSD-to-experimental > 4 Å for the 450 K replicates
- Emits `manifest.json` with per-trajectory SHA256 and a global `dataset_hash`

Current pilot status:
- `configs/mdcath_tier1_domains.txt` seeds 50 real mdCATH domain IDs from the Hugging Face dataset tree; `configs/mdcath_smoke_domains.txt` contains the first 3 for fast server smoke runs.
- `scripts/download_mdcath.sh` uses `huggingface-cli download` with include patterns for selected domain HDF5 files plus `mdcath_source.h5`.
- `stride preprocess` writes `coords.zarr`, `features.zarr`, `residue_mask.zarr`, `metadata.parquet`, `manifest.json`, and `splits/by_topology.json`.
- The pilot reads mdCATH's actual HDF5 schema (`domain/temperature/replicate/{coords,dssp,rmsd,...}`) and embedded PDB atom metadata. Frame cadence is recorded as `frame_dt_ps`; no upsampling is performed.
- Sidechain/χ1 Zarr remains deferred, as planned.

**Test:**
- Round-trip a single mdCATH HDF5 → Zarr → load: RMSD < 0.05 Å between original and reloaded coords (fp16 quantization sanity).
- Re-run on identical input: byte-identical `dataset_hash`.
- Total compressed output ~620 GB primary + ~200 GB sidechain auxiliary on a 2 TB NVMe.
- Random-window load benchmark: `(traj, t_start, W=64)` returns in p50 < 50 ms, p99 < 200 ms.

### 1.3 Stage A0 — VAMPnet labels (A100, ~1 week wall, ~10 GPU-days)

Implement `stride pretrain-vampnets`:
- For each CATH topology in the train split, fit a small VAMPnet (4-layer MLP, dim 128) using `deeptime`
- Tier 1 (~50 headline + ablation proteins): multi-resolution at N ∈ {4, 16, 64}
- Tier 2 (~2000 mdCATH): single resolution N=16
- Tier 3 (~3400 remaining): TICA + k-means surrogate via `deeptime`
- `vampnet_health` gate: VAMP-2 score on held-out lag time vs random-projection baseline. Failure → automatic TICA fallback for that protein.
- Emits per-protein label tensors `labels/{protein_id}/vamp_{N}.npy` and a `vampnet_manifest.json`

**Test:**
- Implied-timescales plot converges (ITS within 10% across last 3 lag-time evaluations) for ≥ 80% of Tier 1 proteins.
- Chapman–Kolmogorov test passes for ≥ 70% of Tier 1 proteins at the chosen lag.
- Alanine dipeptide VAMPnet recovers (φ, ψ) basins (AMI vs ground-truth tICA labels ≥ 0.5 bits over random).
- Run on a 4080 first with 1 protein to validate pipeline → push → pull on A100 → run at scale.

### 1.4 Lock the headline protocol (laptop, ~3 days)

Write `experiments/headline/protocol.md` and `git tag headline-protocol-v1`:
- The 10–15-protein eval list (BPTI, villin, Trp-cage, WW, NTL9, chignolin, src-kinase, β2-AR if licensed, plus 2–3 mdCATH-held-out kinases or transporters)
- Per-state structural definitions in code (e.g., `def is_dfg_out(traj_frame): return psi_D in [-180, -90] and chi1_F in [60-60, 60+60]`)
- The exact metric formula: AUC of states-discovered curve vs simulated nanoseconds, paired bootstrap, 95% CI
- Pre-registration timestamp in commit message

**Stage 1 completion gate:**
- All five gates from the implementation plan in AGENT.md (data round-trip, VAMPnet convergence, CK-test, hash determinism, license sign-off)
- `headline-protocol-v1` git-tagged and pushed
- DESRES BPTI access request submitted

---

## Stage 2 — Encoder (3–4 weeks calendar)

### 2.1 Day-1 memory measurement (4080 server, day 1)

`scripts/measure_4080_memory.py`:
- Loads `s_dim=128, v_dim=16, 3-layer GVP-GNN`, `B=8, W=64, N=300, k=16`, gradient checkpointing on
- Forward + backward + AdamW step
- Records `torch.cuda.max_memory_allocated()` to `experiments/measurements/day1_4080.json`
- If peak > 13 GB, walks the knob-tuning order from AGENT.md (W → B → layers → s_dim → EGNN fallback)

**This is the single most important measurement of the project.** Don't proceed until this lands.

### 2.2 GVP-GNN implementation (laptop write, server test, ~1.5 weeks)

- Port `drorlab/gvp-pytorch` reference implementation into `src/stride/models/gvp.py`. Adapt to PyG `MessagePassing`.
- Two-stream layer (scalars s, vectors v) with the GVP forward.
- bf16-mixed everywhere with `autocast(enabled=False)` around the equivariant vector-update step.
- Gradient checkpointing on each layer's forward.

**Test on the laptop (CPU):**
- Equivariance unit test: random SO(3) rotation of input → output unchanged within 1e-4 fp32 (CPU fp32 makes the test more stringent).
- Permutation invariance: shuffle node order → same graph-level output.
- Shape contracts on tiny synthetic graphs.

**Test on the 4080:**
- 1000-step forward/backward/optimizer-step at the committed config without NaN or Inf.
- Equivariance test in bf16: tolerance 1e-2 acceptable.

### 2.3 VICReg novelty head + GRL FF head (~1 week)

- VICReg loss in `src/stride/models/heads/vicreg.py`: variance + invariance + covariance terms.
- Positive pairs: same-trajectory frames Δt < 1 ns, plus same-frame SE(3) augmentation (random rotation; coords σ=0.5 Å Gaussian).
- GRL head: gradient-reversal layer (autograd.Function) on FF identity, λ schedule warming 0→1.0 over first 20% of Stage-1 steps.

**Test:**
- VICReg dimensional-collapse check: rank-effective dimensionality of held-out embeddings ≥ 48 of 64. Run on a 1k-frame slice, then full Stage 1.
- GRL probe: a separate linear classifier trained on encoder embeddings to predict FF identity scores ≤ 60% (chance = 50%).

### 2.4 Stage 1 training run (A100, ~2 GPU-days, but 1 week wall including debug)

```bash
# On A100:
git pull
uv run stride train --stage 1 \
    --config configs/stage1_default.yaml \
    --data $STRIDE_DATA_ROOT/stride-data \
    --out $STRIDE_MODELS_ROOT/encoder_v1 \
    --wandb-project stride-stride --wandb-run-name stage1-{git-sha}
```

Single A100 80 GB, B=64, ~2 days. 5+ seeds is excessive for the encoder — 1 seed is fine if all gates pass, since the encoder is just a feature extractor. Add seeds only if VICReg or GRL look unstable.

**Stage 2 completion gate:**
- Equivariance test passes at fp32 tolerance 1e-4 (laptop CPU) and bf16 tolerance 1e-2 (4080).
- VICReg rank ≥ 48/64.
- GRL probe ≤ 60%.
- t-SNE colored by DSSP class shows non-trivial separation (silhouette score reported).
- Frozen encoder shipped: `models/encoder_stage1_{git_sha}.safetensors` and a small embedding dump for the eval set.

---

## Stage 3 — Predictor (4–5 weeks calendar)

### 3.1 Mamba temporal head (laptop write, server test, ~1 week)

- Wrap `mamba_ssm.Mamba` with a PyG-compatible adapter: per-frame embeddings pool to (B, W=64, D=128), feed Mamba.
- Two Mamba layers + chunked self-attention block (window=128 frames, rotary positional encoding) on top.
- `torch._dynamo.disable` on the Mamba module.
- GRU and S5 fallback implementations behind a single config switch.

**Test (4080):**
- Forward pass at W=64 returns finite tensor.
- Run with `mamba-ssm` disabled → falls back to GRU automatically without code change.

### 3.2 Heads: Koopman residual + VAMP-2 + Bayesian last-layer (~1 week)

- Koopman residual head: regression target is `||K · z(t) - z(t+τ)||` from the frozen Stage A0 VAMPnet's Koopman operator. MSE loss.
- VAMP-2 head: single resolution N=16, fp32 wrap around covariance whitening, multi-task auxiliary loss `0.1 · -VAMP-2_score`.
- Bayesian last-layer: Laplace approximation via `laplace-torch` library, recomputed once per epoch on a held-out slice. Produces σ_pred per prediction.

**Test:**
- Loss curves go down on a single-protein overfit run (sanity check that gradients flow).
- VAMP-2 score on val protein increases monotonically across epochs.
- Calibration: post-hoc temperature scaling on val brings ECE ≤ 5%.

### 3.3 Stage 2 main training run (A100, ~5 GPU-days × 3 seeds = ~15 GPU-days)

```bash
# On A100:
uv run stride train --stage 2 \
    --config configs/stage2_default.yaml \
    --encoder $STRIDE_MODELS_ROOT/encoder_v1/encoder_stage1_{sha}.safetensors \
    --vampnet-labels $STRIDE_DATA_ROOT/stride-data/labels \
    --out $STRIDE_MODELS_ROOT/stride_v1 \
    --seed 0  # repeat with --seed 1, --seed 2
```

### 3.4 Sanity tests (`stride evaluate` family)

Implement and run, in order:
- **Alanine dipeptide φ/ψ basin recovery** — must beat random projection by ≥ 0.5 bits MI.
- **Equivariance test on trained model** — random rotation, prediction unchanged.
- **Time-shuffle permutation** — shuffle time axis, AUROC drops ≥ 30%.
- **Time-reversal symmetry** — VAMP-2 invariant to within 5% in equilibrium regions.
- **FF-swap generalization** — train on mdCATH, evaluate on STRIDE-MD held-out, AUROC degradation ≤ 15%.
- **Beats RMSD-velocity** — paired Wilcoxon p < 0.05 on ablation protein set.

Output: `experiments/stage3/sanity_report.md` with all tests + pass/fail.

### 3.5 Distillation: teacher → student EGNN (~3 days)

`stride distill --teacher ... --student-config ...`:
- Student: EGNN, dim 128, 4 layers
- Distill on mdCATH train set, KL on Koopman residual prediction + MSE on novelty embedding
- Target: student performance within 10% of teacher on the eval set

**Stage 3 completion gate:**
- All 6 sanity tests pass (alanine, equivariance, time-shuffle, time-reversal, FF-swap, beats-RMSD-velocity).
- Calibration ECE ≤ 5%.
- Distillation gap ≤ 10%.
- Per-protein AUROC table on held-out CATH topologies, ≥ 3 seeds, bootstrap CIs reported.

---

## Stage 4 — Offline scheduler benchmark (3–4 weeks calendar)

### 4.1 Ray topology (laptop write, 4080 test, ~1.5 weeks)

Implement actors in `src/stride/scheduler/`:
- `SchedulerActor` — priority queue, single-writer
- `NoveltyArchiveActor` — FAISS-IVFPQ index, single-writer, 5-min snapshot
- `InferenceActor` — GPU-pinned, holds distilled student
- `FrameIngester` — `inotify` (local) or polling (SLURM) for incoming frames

Test on the 4080 with synthetic frame streams. Verify:
- 5-second tick cadence
- Latency p50 < 200 ms, p99 < 1 s
- ε-floor enforcement (random selection rate ≥ 0.05)
- 25%-per-trajectory cap enforcement
- Single-writer invariant under simulated archive-actor death

### 4.2 ReplaySimulationDriver (~3 days)

Implement `src/stride/drivers/replay.py` reading mdCATH Zarr and emulating live frame arrival. Same `SimulationDriver` protocol as the future OpenMM and SLURM drivers.

### 4.3 Acquisition function (~3 days)

```
score(τ) = μ_pred(τ) + κ · σ_pred(τ) + c · √(log t / N_visits(region(τ)))
```
- `μ_pred`, `σ_pred` from Bayesian last-layer Laplace head
- `N_visits` via FAISS range query against the IVFPQ archive

Default `κ=1.0, c=1.0`. Sweep on the ablation set in 4.5.

### 4.4 All 8 baselines (~1 week)

Implement each as its own `Scheduler` subclass:
1. Round-robin
2. Random with replacement
3. RMSD-velocity heuristic
4. tICA-distance to nearest discovered state
5. MSM counts-based (Bowman 2010)
6. FAST (Zimmerman 2015)
7. REAP (Shamsi 2018)
8. AdaptiveBandit (Pérez 2020)

Each must pass the same protocol harness — they all consume the same trajectory replay and produce the same metrics output. **Write the harness first, then plug each baseline into it.**

### 4.5 Headline benchmark run (~1 week)

```bash
# On 4080 server (replay does not need A100):
uv run stride benchmark \
    --protocol experiments/headline/protocol.md \
    --proteins experiments/headline/eval_proteins.json \
    --schedulers stride,round-robin,random,rmsd-velocity,tica-distance,msm-counts,fast,reap,adaptive-bandit \
    --seeds 0,1,2,3,4 \
    --budget-ns 1000 \
    --out experiments/stage4/results
```

10+ proteins × 5 seeds × 9 schedulers ≈ 450 runs. Replay is fast (no real MD), so total wall time is ~1 week of compute.

Generate `experiments/stage4/report.md`:
- AUC table per scheduler per protein
- Paired bootstrap (1000 resamples) CIs on AUC delta vs each baseline
- Coverage entropy per scheduler
- Ablation runs: STRIDE-no-novelty, STRIDE-no-σ_pred (pure UCB), VICReg vs ESM-2 vs raw-PCA novelty embedding

**Stage 4 completion gate:**
- STRIDE beats MSM-counts, FAST, REAP — paired bootstrap 95% CI on AUC delta excludes 0. *Existential gate.*
- STRIDE beats Random by ≥ 2× AUC.
- Coverage entropy beats Random's natural Boltzmann on every eval protein.
- All ablations run and reported.

**If STRIDE doesn't beat FAST/REAP here:** stop and diagnose. Run a κ/c grid sweep. If still failing, the contribution becomes "we tried and here's where it's hard" — write that paper instead of forcing the win.

---

## Stage 5 — Live MD on the workstation (2–3 weeks calendar)

### 5.1 OpenMMLocalDriver (4080, ~1 week)

Implement `src/stride/drivers/openmm_local.py`:
- Launches OpenMM in a subprocess
- Streams frames via Unix socket (preferred) or shared memory
- Same `SimulationDriver` protocol as `ReplaySimulationDriver`
- 10 ns checkpoint cadence via `Simulation.saveState()`

Test on a tiny system (alanine dipeptide) end-to-end before larger proteins.

### 5.2 24-hour soak (4080, 24h continuous)

Run with 3 proteins (alanine, Trp-cage, src-kinase fragment) for 24 hours:
- Zero crashes, zero OOM
- RSS growth < 500 MB across 24 h
- Decision-equivalence vs offline replay: rank-Spearman ρ ≥ 0.9 on identical seed trajectories
- Latency p50 < 200 ms p99 < 1 s end-to-end
- Backpressure works under artificially-slowed inference

**Stage 5 completion gate:** all soak metrics pass; decision-equivalence holds.

---

## Stage 6 — Cluster headline (4–6 weeks calendar)

### 6.1 SlurmSimulationDriver + Apptainer image (~2 weeks)

- Build `stride/runtime:{git-sha}` Docker image, < 4 GB
- Apptainer wrapper for the cluster
- `SlurmSimulationDriver` submits sbatch jobs, watches shared FS for frame outputs
- Ray-on-SLURM bring-up via `ray.init(address=...)` inside an sbatch allocation

Validate on a 4–8 GPU subset of the cluster before scaling up.

### 6.2 Headline benchmark (~2 weeks, ~10–20 GPU-days A100 for MD)

```bash
# On cluster login node:
sbatch scripts/headline_benchmark.sbatch
# launches Ray cluster + scheduler, drives ~10 proteins × 5 seeds × 1 µs MD
```

**Stage 6 completion gate (the headline claim):**
- Cluster results match Stage 4 offline AUC within 10%.
- STRIDE beats FAST and REAP at scale, paired bootstrap 95% CI excludes 0.
- Time-to-first-discovery for ≥ 7 of 10 named states significantly faster than best baseline (paired Wilcoxon p < 0.05).
- STRIDE overhead ≤ 10% of MD compute.
- Reproducibility manifest: dataset hash, model hash, protocol hash all locked.

---

## Continuous practices

These run alongside every stage, not as separate steps.

**Pre-commit (laptop):** `ruff format`, `ruff check`, `pyright`, `pytest -m "not slow"`. Anything tagged `slow` or `needs_gpu` skips on laptop and runs in CI on the server.

**Server CI:** on every push, server pulls and runs `pytest -m "needs_gpu"` on a small smoke-test config. Catches regressions before you discover them mid-training.

**W&B:** every training run logs config hash, dataset hash, code git SHA, GPU hours, and the headline metric. The dashboard becomes the project's lab notebook.

**Reproducibility CI:** a nightly job on the server runs the kill-and-resume test (Stage 2 / 3 training, kill at step 1000, resume, run to step 2000, assert losses match to 1e-5).

**Verification debt log:** any decision that depends on an unverified claim (license terms, dataset URL, library behavior) goes into `experiments/verification_debt.md` with an owner and a deadline. Resolve before that stage's completion gate.

---

## Per-step ritual — finish like this every time

Every sub-stage in this plan (each numbered subsection — §1.1, §1.2, §2.1, …) ends the same way. The discipline is what keeps the three docs aligned with code and the remote always recoverable. One commit per sub-stage, pushed direct to `main`.

1. **Update the docs.** For the sub-stage you just finished, ask:
   - Did anything change about the architecture, the rejected alternatives, the dataset list, the hardware budget, or a pinned version? → edit **AGENT.md**.
   - Did the operational steps for this or a future sub-stage change? → edit **IMPLEMENTATION.md**.
   - Did the user-facing quickstart change? → edit **README.md**.
   - If all three are no, that's a valid outcome — say so in the commit body (`docs: no changes`).
2. **Run all gates locally** — must all be clean before you commit:
   ```bash
   uv run pytest
   uv run ruff check
   uv run ruff format --check
   uv run pyright
   ```
3. **Run pre-commit on staged files** — `pre-commit run --files <staged paths>` to catch trailing whitespace, large files, end-of-file fixes before push.
4. **Commit with a structured message:**
   ```
   <stage>.<sub>: <verb> <what>

   - <bullet on what landed in code>
   - <bullet on doc updates, or "docs: no changes">
   ```
   Example: `1.2: implement Stage A Zarr writer` with body listing the writer module, the round-trip test, and any IMPLEMENTATION.md tweaks.
5. **Push** — `git push origin main`.
6. **Confirm CI green** on the GitHub Actions page (or `gh run list --limit 1`) before starting the next sub-stage. Red CI is a stop-the-world event — fix before moving on.

**Failure modes to call out:**
- **Doc drift caught later** — if you notice AGENT.md disagrees with code in a *later* sub-stage, fix it as part of that sub-stage's commit, not as a separate "docs cleanup" commit. The doc update belongs to the change that caused the drift.
- **CI flakes** — debug, do not retry blindly. A flake on a deterministic project like this almost always means a real cross-platform issue (Linux runner vs macOS dev).
- **Dataset / model artifacts** — never committed (`.gitignore` covers `*.zarr/`, `*.safetensors`, `*.pt`, `wandb/`). Verification debt and pre-registered protocols (`experiments/headline/protocol.md`) ARE committed.

---

## Exit ramps

You don't have to finish all six stages to have a publishable artifact:

| If you stop after | You have |
|---|---|
| Stage 1 | Curated, hashed mdCATH-derived dataset + VAMPnet labels — dataset paper. |
| Stage 2 | A frozen FF-invariant trajectory encoder — representation-learning paper. |
| Stage 3 | A trajectory-transition predictor that beats RMSD-velocity — workshop paper. |
| Stage 4 | **An offline adaptive sampler that beats FAST/REAP — main result.** |
| Stage 5 | + a live-MD demonstration on a workstation. |
| Stage 6 | + cluster-scale headline result with named-state discovery — top-venue claim. |

Stages 1–4 are sequential and gate-locked. Stages 5–6 are the production ramp — they prove the operational claim ("STRIDE could drive a real cluster") but do not change the science contribution. If A100 access falls through, ship Stage 4 and write a smaller real paper.

---

## What doesn't go in this plan

- Toy systems beyond alanine dipeptide. Alanine is the MSM "hello world"; everything else is full proteins.
- Custom MD. We download trajectories. The only MD we run is what the trained scheduler launches in Stages 5–6.
- Half-implementation of skipped stages. If you skip Stage 5, you skip Stage 5 — don't half-write `OpenMMLocalDriver` and leave it. Either finish or delete.
- Per-stage refactors of earlier stages. Each completion gate locks the artifact. Stage 4 doesn't reach back into Stage 2's encoder; if Stage 2 was wrong, you reopen Stage 2 and re-pass its gate.
