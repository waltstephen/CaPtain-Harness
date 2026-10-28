#!/usr/bin/env bash
# The policy model: CaPtain-4B (SFT + RL) or CaPtain-4B-SFT, both post-trained from Qwen3.5-4B (Apache-2.0).
# Weights: https://huggingface.co/YijiaFan/CaPtain-4B and https://huggingface.co/YijiaFan/CaPtain-4B-SFT.
# You do not need these weights to use the harness: any OpenAI-compatible chat endpoint works.
#
# Usage: scripts/models/download_policy.sh [4b|4b-sft] [--dry-run]
#   CAPTAIN_HF_USER     Hugging Face user/org that hosts the weights   [YijiaFan]
#   CAPTAIN_POLICY_REPO full repo id, overrides the two above          [$CAPTAIN_HF_USER/CaPtain-4B or .../CaPtain-4B-SFT]
# Result: $CAPTAIN_MODELS_DIR/CaPtain-4B[-SFT]  ->  services/serve_policy.sh $CAPTAIN_MODELS_DIR/CaPtain-4B 8600
#
# Using another model instead: serve it (or use a hosted API) behind an OpenAI-compatible /v1/chat/completions endpoint and
#   export POLICY_URL=http://<host>:<port>/v1 POLICY_MODEL=<served model name>
#   for endpoints that need a key, export CAPTAIN_API_KEY (or OPENAI_API_KEY) and append --auth bearer to
#   eval/run_episode.sh (extra arguments after the 4 positional ones go to scripts/react_episode.py).
source "$(dirname "$0")/_common.sh"
variant=4b
for a in "$@"; do case "$a" in 4b|4b-sft) variant=$a ;; esac; done
name=CaPtain-4B; [ "$variant" = 4b-sft ] && name=CaPtain-4B-SFT
HF_USER=${CAPTAIN_HF_USER:-YijiaFan}
REPO=${CAPTAIN_POLICY_REPO:-$HF_USER/$name}
DEST=$MODELS_DIR/$name
if is_done "$DEST"; then log "already present: $DEST (FORCE=1 to re-download)"; else
  log "downloading $REPO -> $DEST (~10 GB, bf16)"
  run mkdir -p "$DEST"
  run "$(hf_cli)" download "$REPO" --local-dir "$DEST"
  mark_done "$DEST"
fi
echo "services/serve_policy.sh $DEST 8600"
