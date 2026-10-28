# Shared helpers for scripts/models/*.sh (sourced, not executed).
#   CAPTAIN_MODELS_DIR  where weights go                 [<repo>/models]
#   HF_CLI              Hugging Face CLI executable      [hf, else huggingface-cli]
#   HF_TOKEN            read from the environment by the HF CLI (needed for gated repos); never written anywhere
#   DRY_RUN=1 or --dry-run   print the commands without running them
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODELS_DIR=${CAPTAIN_MODELS_DIR:-$REPO_ROOT/models}
DRY_RUN=${DRY_RUN:-0}
for _arg in "$@"; do
  case "$_arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
  esac
done

log() { echo "[$(basename "$0" .sh)] $*"; }

run() {  # print, then execute unless dry-run
  echo "+ $*"
  if [ "$DRY_RUN" != 1 ]; then "$@"; fi
}

hf_cli() {
  if [ -n "${HF_CLI:-}" ]; then echo "$HF_CLI"; return; fi
  if command -v hf >/dev/null 2>&1; then echo hf; return; fi
  if command -v huggingface-cli >/dev/null 2>&1; then echo huggingface-cli; return; fi
  if [ "$DRY_RUN" = 1 ]; then echo hf; return; fi
  echo "error: no Hugging Face CLI found; pip install -U 'huggingface_hub[cli]' or set HF_CLI" >&2
  exit 1
}

# verify_sha256 <file> <expected>: warn (do not fail) on mismatch; the expected value is the file used for the paper
verify_sha256() {
  [ "$DRY_RUN" = 1 ] && { echo "+ sha256sum $1  (expect $2)"; return 0; }
  [ -f "$1" ] || { log "warning: $1 not found"; return 0; }
  local got; got=$(sha256sum "$1" | cut -d' ' -f1)
  if [ "$got" = "$2" ]; then log "sha256 ok: $(basename "$1")"
  else log "warning: sha256 of $1 is $got, the paper used $2 (a newer upstream file?)"; fi
}

# done-marker helpers make every script idempotent: a finished download is skipped unless FORCE=1
is_done() { [ "${FORCE:-0}" != 1 ] && [ -f "$1/.captain_download_complete" ]; }
mark_done() { [ "$DRY_RUN" = 1 ] || date -u +%FT%TZ > "$1/.captain_download_complete"; }
