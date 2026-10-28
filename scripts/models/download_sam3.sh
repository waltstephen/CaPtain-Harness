#!/usr/bin/env bash
# SAM 3 (facebook/sam3) for the segmentation service (services/start_perception.sh, port 8114).
#
# facebook/sam3 is GATED on Hugging Face: open https://huggingface.co/facebook/sam3 with your account, accept the
# SAM License, then export HF_TOKEN=<a read token of that account> (or run `hf auth login`). The weights are not
# redistributed by this repository. Only sam3.pt (the checkpoint, ~3.5 GB) and config.json are downloaded.
#
# Usage: scripts/models/download_sam3.sh [--dry-run]
# Result: $CAPTAIN_MODELS_DIR/sam3/sam3.pt  ->  export SAM3_CHECKPOINT=$CAPTAIN_MODELS_DIR/sam3/sam3.pt
source "$(dirname "$0")/_common.sh"
DEST=$MODELS_DIR/sam3
SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e   # sam3.pt used for the paper
if is_done "$DEST"; then log "already present: $DEST/sam3.pt (FORCE=1 to re-download)"; else
  [ -n "${HF_TOKEN:-}" ] || log "note: HF_TOKEN is not set; this works only if the HF CLI is already logged in with an account that accepted the SAM License"
  log "downloading facebook/sam3 (gated, SAM License) -> $DEST"
  run mkdir -p "$DEST"
  run "$(hf_cli)" download facebook/sam3 sam3.pt config.json --local-dir "$DEST"
  verify_sha256 "$DEST/sam3.pt" "$SHA256"
  mark_done "$DEST"
fi
echo "export SAM3_CHECKPOINT=$DEST/sam3.pt"
