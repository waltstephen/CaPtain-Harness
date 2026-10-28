#!/usr/bin/env bash
# Download the model weights CaPtain-Harness uses (idempotent; finished downloads are skipped, FORCE=1 re-downloads).
#
#   scripts/download_models.sh [--dry-run] [ASSET ...]
#
# ASSET (default: sam3 molmo2):
#   sam3      facebook/sam3 sam3.pt                  segmentation service :8114   required   ~3.5 GB  GATED: accept the SAM License, export HF_TOKEN
#   molmo2    allenai/Molmo2-8B @ e28fa28            pointing service :8122       required   ~35 GB
#   moge      Ruicheng/moge-2-vitl-normal @ cb0e8bb  depth service :8130          optional: only hardware_real cameras without depth
#   policy    YijiaFan/CaPtain-4B                   policy (vLLM) :8600          optional; any OpenAI-compatible endpoint works
#   policy-sft YijiaFan/CaPtain-4B-SFT              policy (vLLM) :8600          optional
#   all       everything above
#
# Environment:
#   CAPTAIN_MODELS_DIR  target directory                     [<repo>/models]
#   HF_TOKEN            read by the HF CLI from the environment (gated repos); never stored by these scripts
#   HF_CLI              Hugging Face CLI                     [hf, else huggingface-cli]
#   CAPTAIN_HF_USER     owner of the CaPtain-4B repos        [placeholder until the release]
# One script per asset: scripts/models/download_{sam3,molmo2,moge,policy}.sh (same flags).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
flags=(); assets=()
for a in "$@"; do
  case "$a" in
    --dry-run|-n) flags+=(--dry-run) ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    all) assets+=(sam3 molmo2 moge policy policy-sft) ;;
    sam3|molmo2|moge|policy|policy-sft) assets+=("$a") ;;
    *) echo "unknown asset or flag: $a (see --help)" >&2; exit 2 ;;
  esac
done
[ ${#assets[@]} -gt 0 ] || assets=(sam3 molmo2)
echo "[download_models] target: ${CAPTAIN_MODELS_DIR:-$(cd "$HERE/.." && pwd)/models}  assets: ${assets[*]}${flags:+  (dry run)}"
exports=()
for a in "${assets[@]}"; do
  case "$a" in
    sam3)       out=$("$HERE/models/download_sam3.sh" "${flags[@]}" | tee /dev/stderr | tail -n 1) ;;
    molmo2)     out=$("$HERE/models/download_molmo2.sh" "${flags[@]}" | tee /dev/stderr | tail -n 1) ;;
    moge)       out=$("$HERE/models/download_moge.sh" "${flags[@]}" | tee /dev/stderr | tail -n 1) ;;
    policy)     out=$("$HERE/models/download_policy.sh" 4b "${flags[@]}" | tee /dev/stderr | tail -n 1) ;;
    policy-sft) out=$("$HERE/models/download_policy.sh" 4b-sft "${flags[@]}" | tee /dev/stderr | tail -n 1) ;;
  esac
  case "$out" in export*|services/*) exports+=("$out") ;; esac
done
if [ ${#exports[@]} -gt 0 ]; then
  echo; echo "[download_models] use them with:"; printf '  %s\n' "${exports[@]}"
fi
