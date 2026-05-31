#!/usr/bin/env bash
set -euo pipefail

DOMAINS="configs/mdcath_tier1_domains.txt"
OUT="${STRIDE_DATA_ROOT:-stride-data}/mdcath_raw"
REVISION="main"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domains)
      DOMAINS="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    --revision)
      REVISION="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$DOMAINS" ]]; then
  echo "domain list not found: $DOMAINS" >&2
  exit 1
fi

mkdir -p "$OUT"
INCLUDES=("--include" "mdcath_source.h5")
while IFS= read -r domain; do
  [[ -z "$domain" || "$domain" =~ ^# ]] && continue
  INCLUDES+=("--include" "data/mdcath_dataset_${domain}.h5")
done < "$DOMAINS"

download_rc=0
uv run hf download compsciencelab/mdCATH \
  --repo-type dataset \
  --revision "$REVISION" \
  --local-dir "$OUT" \
  "${INCLUDES[@]}" || download_rc=$?

missing=0
[[ -f "$OUT/mdcath_source.h5" ]] || missing=1
while IFS= read -r domain; do
  [[ -z "$domain" || "$domain" =~ ^# ]] && continue
  [[ -f "$OUT/data/mdcath_dataset_${domain}.h5" ]] || missing=1
done < "$DOMAINS"

if [[ "$missing" -ne 0 ]]; then
  exit "$download_rc"
fi

{
  echo "# Data Provenance"
  echo
  echo "- Dataset: compsciencelab/mdCATH"
  echo "- Revision: ${REVISION}"
  echo "- Domain list: ${DOMAINS}"
  echo "- Local dir: ${OUT}"
  echo "- Downloaded at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${OUT}/data_provenance.md"
