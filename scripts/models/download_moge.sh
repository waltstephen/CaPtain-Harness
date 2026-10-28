#!/usr/bin/env bash
# MoGe-2 (Ruicheng/moge-2-vitl-normal, ViT-L) for monocular metric depth. OPTIONAL: needed only by the hardware backend
# (hardware_real) for cameras without a depth sensor (`"depth": "mono"` in the robot config,
# adapters/hardware/mono_depth.py, model name "moge2"). The simulator backends use rendered depth and do not need it.
# Served by services/depth_server.py (port 8130). Pinned to the revision used for the real-robot runs. ~1.3 GB.
#
# The depth service also needs the MoGe code (MIT) in its Python environment, e.g.
#   pip install "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" fastapi uvicorn
#
# Usage: scripts/models/download_moge.sh [--dry-run]
#   MOGE_REVISION   [cb0e8bbd6b1e243589717c78e750b1ba4c093acf]
# Result: $CAPTAIN_MODELS_DIR/moge-2-vitl-normal/model.pt -> export MOGE_CHECKPOINT=$CAPTAIN_MODELS_DIR/moge-2-vitl-normal/model.pt
source "$(dirname "$0")/_common.sh"
DEST=$MODELS_DIR/moge-2-vitl-normal
REV=${MOGE_REVISION:-cb0e8bbd6b1e243589717c78e750b1ba4c093acf}
SHA256=280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01   # model.pt at that revision
if is_done "$DEST"; then log "already present: $DEST/model.pt (FORCE=1 to re-download)"; else
  log "downloading Ruicheng/moge-2-vitl-normal @ $REV -> $DEST"
  run mkdir -p "$DEST"
  run "$(hf_cli)" download Ruicheng/moge-2-vitl-normal model.pt --revision "$REV" --local-dir "$DEST"
  verify_sha256 "$DEST/model.pt" "$SHA256"
  mark_done "$DEST"
fi
echo "export MOGE_CHECKPOINT=$DEST/model.pt"
