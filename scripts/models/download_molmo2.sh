#!/usr/bin/env bash
# Molmo2-8B (allenai/Molmo2-8B, Apache-2.0) for the pointing service (services/start_perception.sh, port 8122,
# served with vLLM; the harness requests it under the served name allenai/Molmo2-8B).
# Pinned to the revision used for the paper. ~35 GB.
#
# Usage: scripts/models/download_molmo2.sh [--dry-run]
#   MOLMO_REVISION   [e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b]
# Result: $CAPTAIN_MODELS_DIR/Molmo2-8B  ->  export MOLMO_MODEL=$CAPTAIN_MODELS_DIR/Molmo2-8B
source "$(dirname "$0")/_common.sh"
DEST=$MODELS_DIR/Molmo2-8B
REV=${MOLMO_REVISION:-e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b}
if is_done "$DEST"; then log "already present: $DEST (FORCE=1 to re-download)"; else
  log "downloading allenai/Molmo2-8B @ $REV -> $DEST"
  run mkdir -p "$DEST"
  run "$(hf_cli)" download allenai/Molmo2-8B --revision "$REV" --local-dir "$DEST"
  mark_done "$DEST"
fi
echo "export MOLMO_MODEL=$DEST"
