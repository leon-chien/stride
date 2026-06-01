# STRIDE

**Adaptive Trajectory Learning and Scheduling for Protein Molecular Dynamics.** Predicts imminent conformational transitions in protein MD trajectories and uses those predictions plus a coverage-driven novelty signal to allocate scarce simulation compute toward trajectories most likely to discover new conformational states.

- **[AGENTS.md](./AGENTS.md)** — architecture contract: what we're building and why, including decisions explicitly rejected.
- **[IMPLEMENTATION.md](./IMPLEMENTATION.md)** — operational guide: stages, completion gates, commands, exit ramps.

## Quick start

```bash
# Laptop (CPU only — macOS arm64 or Linux x86_64):
uv sync --group dev --group cpu
uv run stride --help

# Server (Linux x86_64 + CUDA 12.4):
uv sync --group dev --group cuda
./scripts/install_server_extras.sh   # mamba-ssm, faiss-gpu, torch-cluster, torch-scatter
uv run stride --help
```

`uv.lock` is committed — `uv sync` reproduces the exact same dep graph on every machine. The two-step server install exists because a handful of GPU/CUDA-bound packages need wheels matched to torch + CUDA + Python and don't survive uv's universal lockfile.

Copy `.env.example` to `.env.local` on each machine and fill in machine-specific paths and W&B credentials. `.env.local` is ignored by git.

## Stage A pilot

On the 4080 server, download the Tier 1 seed subset and convert it to training-ready Zarr:

```bash
scripts/download_mdcath.sh \
  --domains configs/mdcath_tier1_domains.txt \
  --out "$STRIDE_DATA_ROOT/mdcath_raw"

uv run stride preprocess \
  --domains configs/mdcath_tier1_domains.txt \
  --input-root "$STRIDE_DATA_ROOT/mdcath_raw" \
  --output-root "$STRIDE_DATA_ROOT/stride-data"
```

For a fast smoke run, use `configs/mdcath_smoke_domains.txt` instead of the Tier 1 list. The converter writes `coords.zarr`, `features.zarr`, `residue_mask.zarr`, `metadata.parquet`, `manifest.json`, and `splits/by_topology.json`.

Validate a completed Stage A output with real-data round-trip, manifest determinism, and random-window read checks:

```bash
uv run stride validate-stage-a \
  --data "$STRIDE_DATA_ROOT/stride-data-tier1" \
  --input-root "$STRIDE_DATA_ROOT/mdcath_raw" \
  --domains configs/mdcath_tier1_domains.txt
```

Per-sub-stage commit/push ritual (run this every time you finish a numbered sub-stage like §1.1, §1.2, §2.1, …) is documented in IMPLEMENTATION.md → "Per-step ritual" section.
