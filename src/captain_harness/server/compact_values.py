"""Compact capability results for the `compact_v1` prompt profile (task_markdown.COMPACT_PROFILE).

Some results are very large: a move_to receipt carries its full measured trajectory (median about 11 k tokens, up to
100 k), and inspect_contact_geometry and recover results are similar. When an agent prints one, the output is
truncated, and the large tool output forces a context compaction. Under the profile, env_server returns a summary
instead. The summary keeps every scalar and short list; it shortens lists of more than LIST_MAX items (and lists of more
than SHORT_NESTED nested objects) to their length plus the first and last element, cuts strings longer than STR_MAX
characters, and replaces objects nested deeper than DEPTH with a key count. The complete result is written to
`values/<call index>_<capability>.json` in the workspace, and the summary names that file in `details_file`. The host
records (calls.jsonl, history) keep the full result. Only results longer than MIN_CHARS characters of JSON are
summarised. The same code runs for the simulator and the real arm, so both present identical tool outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

MIN_CHARS = 6000
LIST_MAX = 12
SHORT_NESTED = 4
STR_MAX = 160
DEPTH = 4
VALUES_DIR = "values"
KEEP = ("success", "executed", "failure_category", "message", "budget", "receipt")


def _summarise(value: Any, depth: int) -> Any:
    if isinstance(value, Mapping):
        if depth >= DEPTH:
            return f"<{len(value)} keys: see details_file>"
        return {str(k): _summarise(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        nested = any(isinstance(v, (Mapping, list, tuple)) for v in value)
        if len(value) > LIST_MAX or (nested and len(value) > SHORT_NESTED):
            return {"list_len": len(value), "first": _summarise(value[0], DEPTH - 1),
                    "last": _summarise(value[-1], DEPTH - 1), "full": "see details_file"}
        return [_summarise(v, depth + 1) for v in value]
    if isinstance(value, str) and len(value) > STR_MAX:
        return value[:STR_MAX - 24] + " ... [see details_file]"
    return value


def compact_result(payload: Mapping[str, Any], *, workspace: Path, call_index: int, name: str) -> dict[str, Any]:
    """The agent-facing result: unchanged when small, else a summary plus `details_file` (full result on disk)."""
    text = json.dumps(payload, sort_keys=True, default=repr)
    if len(text) <= MIN_CHARS:
        return dict(payload)
    relative = f"{VALUES_DIR}/{call_index:04d}_{name}.json"
    target = Path(workspace) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1, sort_keys=True, default=repr), encoding="utf-8")
    out: dict[str, Any] = {k: payload[k] for k in KEEP if k in payload}
    for key, value in payload.items():
        if key not in out:
            out[key] = _summarise(value, 0)
    out["details_file"] = relative
    out["summarised"] = f"full result ({len(text)} characters of JSON) in {relative}"
    return out


__all__ = ["DEPTH", "LIST_MAX", "MIN_CHARS", "STR_MAX", "VALUES_DIR", "compact_result"]
