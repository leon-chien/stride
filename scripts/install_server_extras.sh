#!/usr/bin/env bash
set -euo pipefail

uv run python - <<'PY'
import sys

if sys.platform != "linux":
    raise SystemExit("Server extras are Linux/CUDA-only.")
if sys.version_info[:2] != (3, 11):
    raise SystemExit("STRIDE server extras require Python 3.11.")
PY

uv pip install \
  --no-build-isolation \
  "mamba-ssm==1.2.*" \
  "causal-conv1d==1.2.*"

uv pip install \
  "torch-cluster" \
  "torch-scatter" \
  -f "https://data.pyg.org/whl/torch-2.4.0+cu124.html"

uv pip install "faiss-gpu-cu12>=1.8"
