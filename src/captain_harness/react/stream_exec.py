"""Stream-exec (opt-in speed option, docs/harness.md "Speed options"): run a `python` cell while it is still being decoded.

The chat client streams the completion and hands the growing `python` call input to a ``StreamingCell``. As soon as the
text has a prefix of complete top-level statements, those statements run in the persistent session (``PythonRepl``
begin_stream / run_part / close_stream) while the model keeps decoding; when the response ends the rest runs. The tool
output is the whole-cell output (one cell, same number, output in order, same error text).

A prefix is complete when it ends at a line boundary, compiles on its own (``compile(..., "exec")``: parses, and no
``return``/``break``/... outside their block), and the next non-blank, non-comment line has started at column 0 with a
token that cannot continue the previous statement (not ``else``, ``elif``, ``except``, ``finally``; a decorator line
does not compile without its target). An indented next line, an open bracket, a backslash continuation or an open
string never yields a cut (the prefix does not compile or the next line is indented). A partly decoded next line is
waited for. An incomplete statement is never executed.

A prefix also ends right after a top-level ``;`` (one-line cells such as ``r = f(); print(r); g()``) when it compiles,
the ``;`` is a real token (not in a string or comment) and the statement before it is simple: in ``if c: a; b`` the
``;`` belongs to the suite and never cuts. Nothing after a top-level ``;`` can extend the statement before it.

``overlap_s`` (execution done before the response ended) is the wall time saved: after the response ends the rest of
the cell runs back to back, so the cell ends ``overlap_s`` earlier than whole-cell execution would.

Semantics changes against the release (recorded per cell):
- a cell that does not compile as a whole runs nothing in the release; here the statements cut before the error have
  already run. The tool text still reports the release's SyntaxError (plus the output of what ran) and the cell's
  ``stream_exec.partial_before_syntax_error`` is true.
- a request retried after statements already ran whose new decoding no longer extends them (``diverged``): the early
  statements stay executed in a cell of their own (its output is not shown to the model); the final input then runs
  as the next cell.
- the early statements of a response whose `python` call the loop never runs (``orphaned``) stay executed.
"""
from __future__ import annotations

import ast
import io
import queue
import re
import threading
import time
import tokenize
from typing import Any, Callable

CONTINUATION_KEYWORDS = frozenset({"else", "elif", "except", "finally"})
_FIRST_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_CANDIDATE = re.compile(r"[\n;]")
_COMPOUND = tuple(node for node in (getattr(ast, name, None) for name in (
    "If", "For", "AsyncFor", "While", "With", "AsyncWith", "Try", "TryStar", "FunctionDef", "AsyncFunctionDef", "ClassDef",
    "Match")) if node is not None)


class StatementCutter:
    """Cuts a growing source into consecutive chunks of complete top-level statements (offsets into the source)."""

    def __init__(self) -> None:
        self.start = 0          # end of the last cut
        self._settled = 0       # candidate cut points up to here were rejected for good

    def cuts(self, text: str) -> list[tuple[int, int]]:
        """New complete chunks ``(start, end)`` of ``text`` (each extends the previous; ``text`` only ever grows)."""
        found = []
        while True:
            end = self._next_cut(text)
            if end is None:
                return found
            found.append((self.start, end))
            self.start = self._settled = end

    def reset_lookahead(self) -> None:
        """Forget rejected cut points after the last cut (the text after it was replaced)."""
        self._settled = self.start

    def _next_cut(self, text: str) -> int | None:
        position = max(self.start, self._settled)
        while True:
            match = _CANDIDATE.search(text, position)
            if match is None:
                return None
            end = match.end()
            if match.group() == ";":
                verdict = _semicolon_verdict(text, self.start, end)
            else:
                verdict = _cut_verdict(text, self.start, end)
            if verdict is None:
                return None              # the next line has not shown enough yet
            if verdict:
                return end
            self._settled = position = end


def chunk_code(text: str, start: int, end: int) -> str:
    """The source of ``text[start:end]``; a chunk that starts after a `;` mid-line loses that line's leading blanks."""
    chunk = text[start:end]
    return chunk.lstrip(" \t") if start > 0 and text[start - 1] != "\n" else chunk


def _cut_verdict(text: str, start: int, end: int) -> bool | None:
    """Cut at a line end. True: text[start:end] is complete top-level statements; False: never; None: undecided until
    more text arrives."""
    code = chunk_code(text, start, end)
    if not _has_code(code):
        return False                              # blank/comment-only: joins the next statement's chunk
    line_start = end
    while True:                                   # skip blank and comment-only lines (an `else:` may follow them)
        newline = text.find("\n", line_start)
        line = text[line_start:] if newline < 0 else text[line_start:newline]
        stripped = line.strip()
        if newline >= 0 and (not stripped or stripped.startswith("#")):
            line_start = newline + 1
            continue
        if not stripped or stripped.startswith("#"):
            return None
        break
    if line[0] in " \t\f":
        return False                              # indented: continues the block (or is an error the whole cell reports)
    word = _FIRST_WORD.match(line)
    if word is not None:
        if word.end() == len(line) and newline < 0:
            return None                           # "el" could still become "else"
        if word.group() in CONTINUATION_KEYWORDS:
            return False
    return _compiles(code)                        # compile, not parse: also rejects `return`/`break` outside


def _semicolon_verdict(text: str, start: int, end: int) -> bool:
    """Cut after a `;` (one-line cells: ``a = f(); print(a); g(a)``): the chunk compiles, the `;` is a real token (not
    in a comment or string) and it ends a top-level simple statement (not one inside ``if c: a; b``, whose suite
    continues after the `;`). Nothing after a top-level `;` can extend the statement before it."""
    code = chunk_code(text, start, end)
    if not _has_code(code) or not _compiles(code):
        return False
    try:
        tree = ast.parse(code)
        tokens = [t for t in tokenize.generate_tokens(io.StringIO(code).readline)
                  if t.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)]
    except (SyntaxError, ValueError, tokenize.TokenError):
        return False
    return (bool(tokens) and tokens[-1].type == tokenize.OP and tokens[-1].string == ";"
            and bool(tree.body) and not isinstance(tree.body[-1], _COMPOUND))


_INPUT_KEY = re.compile(r'\s*\{\s*"input"\s*:\s*"')
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def partial_input(arguments: str) -> str:
    """The decoded prefix of the ``input`` string of a streamed ``{"input": "..."}`` arguments text ("" until it starts)."""
    match = _INPUT_KEY.match(arguments)
    if match is None:
        return ""
    out: list[str] = []
    index, size = match.end(), len(arguments)
    while index < size:
        char = arguments[index]
        if char == '"':
            break
        if char != "\\":
            out.append(char)
            index += 1
            continue
        if index + 1 >= size:
            break
        kind = arguments[index + 1]
        if kind != "u":
            out.append(_ESCAPES.get(kind, kind))
            index += 2
            continue
        if index + 6 > size:
            break
        code = int(arguments[index + 2:index + 6], 16)
        if 0xD800 <= code < 0xDC00:                       # surrogate pair
            if index + 12 > size:
                break
            low = int(arguments[index + 8:index + 12], 16)
            out.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
            index += 12
            continue
        out.append(chr(code))
        index += 6
    return "".join(out)


class StreamingCell:
    """One `python` call executed while it streams. ``feed`` runs on the client's thread, statements run on this cell's
    executor thread (which owns the session lock from begin to close), ``complete`` runs on the loop's thread."""

    def __init__(self, repl: Any, timeout_s: Callable[[], float | None], before_start: Callable[[], None] | None = None) -> None:
        self.repl = repl
        self.timeout_s = timeout_s          # cell budget when the first statement is ready; None = do not start early
        self.before_start = before_start    # runs on the executor thread before the first statement (e.g. the request mark)
        self.cutter = StatementCutter()
        self.text = ""
        self.started = False                # begin was queued: the call must be completed through this cell
        self.disabled: str | None = None    # why nothing more runs early (fence, diverged, budget)
        self.response_end: float | None = None
        self.parts: list[dict[str, Any]] = []
        self._queue: queue.Queue = queue.Queue()
        self._done = threading.Event()
        self._result: Any = None
        self._facts: dict[str, Any] = {}
        self._available = True
        self._hook_error: str | None = None

    # ---------------------------------------------------------------- client thread
    def feed(self, text: str) -> None:
        """``text`` is the call's whole input decoded so far (it grows; a retried request starts again from scratch)."""
        if self.disabled or self.response_end is not None:
            return
        done = self.text[:self.cutter.start]
        if not text.startswith(done):
            if done.startswith(text):
                return                       # a retried request is still re-decoding what already ran
            self.disabled = "diverged"      # a retry decoded a different cell: run nothing more early
            return
        if not text.startswith(self.text):
            self.cutter.reset_lookahead()    # a retry changed text after the last cut
        self.text = text
        head = text.lstrip()
        if len(head) < 3:
            return
        if head.startswith("```"):
            self.disabled = "fenced"        # the release strips the fence; line numbers would differ
            return
        for start, end in self.cutter.cuts(text):
            if not self.started:
                budget = self.timeout_s()
                if budget is None:
                    self.disabled = "budget"
                    return
                self.started = True
                threading.Thread(target=self._execute, name="stream-exec", daemon=True).start()
                self._queue.put(("begin", budget))
            self._queue.put(("part", chunk_code(text, start, end), text[:start].count("\n"), text[:end]))

    def end_response(self) -> None:
        if self.response_end is None:
            self.response_end = time.monotonic()

    # ---------------------------------------------------------------- executor thread
    def _execute(self) -> None:
        status = "ok"
        try:
            while True:
                message = self._queue.get()
                if message[0] == "begin":
                    if self.before_start is not None:
                        try:
                            self.before_start()
                        except Exception as exc:  # noqa: BLE001 - nothing runs early; the loop's own hook call reports it
                            self._hook_error = f"{type(exc).__name__}: {exc}"[:300]
                            self._available = False
                            continue
                    if self.repl.begin_stream(timeout_s=message[1]) is None:
                        self._available = False
                elif message[0] == "part":
                    if not self._available or status != "ok":
                        continue
                    started = time.monotonic()
                    status = self.repl.run_part(message[1], line_offset=message[2], source=message[3])
                    self.parts.append({"statements": len(ast.parse(message[1]).body), "start": started, "end": time.monotonic(),
                                       "status": status})
                else:
                    if self._available:
                        self._result, self._facts = self.repl.close_stream(message[1])
                    return
        finally:
            self._done.set()

    # ---------------------------------------------------------------- loop thread
    def complete(self, source: str) -> tuple[Any, dict[str, Any]]:
        """Run the rest of the final input ``source`` and close the cell. Returns (PythonCellResult or None, facts); None
        means the session never ran anything for this call and the caller runs the cell the release way. ``facts["diverged"]``
        (the final input does not extend what ran) means the caller still has to run ``source`` as a cell of its own."""
        self.end_response()
        executed = self.text[:self.cutter.start]
        diverged = self.started and not source.startswith(executed)
        if self.started:
            if not diverged:
                rest = chunk_code(source, self.cutter.start, len(source))
                if _has_code(rest) and _compiles(source):
                    self._queue.put(("part", rest, executed.count("\n"), source))
            self._queue.put(("close", executed if diverged else source))
            self._done.wait()
        end = self.response_end
        parts = self.parts
        facts = {"early_statements": sum(p["statements"] for p in parts if p["start"] < end),
                 "statements": sum(p["statements"] for p in parts), "parts": len(parts),
                 "overlap_s": round(sum(max(0.0, min(p["end"], end) - p["start"]) for p in parts), 3),
                 "lead_s": round(max(0.0, end - parts[0]["start"]), 3) if parts else 0.0,
                 "partial_before_syntax_error": bool(self._facts.get("syntax_error")) and bool(self._facts.get("ran")),
                 "diverged": diverged, "disabled": self.disabled}
        if self._facts.get("syntax_error") and self._facts.get("run_error"):
            facts["run_error"] = self._facts["run_error"]
        if self._hook_error is not None:
            facts["before_start_error"] = self._hook_error
        return self._result, facts


def _has_code(text: str) -> bool:
    return any(line.strip() and not line.strip().startswith("#") for line in text.splitlines())


def _compiles(text: str) -> bool:
    try:
        compile(text, "<stream-exec>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError):
        return False
    return True


class StreamExec:
    """Episode-level glue: the chat client streams the first `python` call of each response into a ``StreamingCell``;
    ``run_python_cell`` takes that cell when the loop runs the call."""

    def __init__(self, repl: Callable[[], Any], timeout_s: Callable[[], float | None],
                 emit: Callable[[dict[str, Any]], None] | None = None, before_start: Callable[[], None] | None = None) -> None:
        self.repl = repl                    # the current PythonRepl (a handoff may replace it)
        self.timeout_s = timeout_s
        self.emit = emit
        self.before_start = before_start    # called once per cell before its first early statement (the request mark)
        self.cell: StreamingCell | None = None
        self.cells: list[dict[str, Any]] = []   # facts of every streamed cell (episode summary)

    # ---------------------------------------------------------------- chat client hooks
    def begin_response(self) -> None:
        self.flush()

    def tool_input(self, index: int, name: str, arguments: str) -> None:
        if index != 0 or name != "python":
            return
        if self.cell is None:
            self.cell = StreamingCell(self.repl(), self.timeout_s, self.before_start)
        try:
            text = partial_input(arguments)
        except ValueError:
            return
        self.cell.feed(text)

    def end_response(self) -> None:
        if self.cell is not None:
            self.cell.end_response()

    # ---------------------------------------------------------------- runner / driver
    def take(self) -> StreamingCell | None:
        cell, self.cell = self.cell, None
        return cell

    def record(self, cell: int | None, facts: dict[str, Any]) -> None:
        row = {"cell": cell, **facts}
        self.cells.append(row)
        if self.emit is not None:
            self.emit({"type": "stream_exec", **row})

    def flush(self) -> None:
        """Close a cell whose response produced no `python` call that the loop ran (its early statements did run)."""
        cell, self.cell = self.cell, None
        if cell is None or not cell.started:
            return
        executed = cell.text[:cell.cutter.start]
        result, facts = cell.complete(executed)
        self.record(getattr(result, "cell", None), {**facts, "orphaned": True})

    def summary(self) -> dict[str, Any]:
        return {"cells": len(self.cells), "early_statements": sum(c.get("early_statements", 0) for c in self.cells),
                "statements": sum(c.get("statements", 0) for c in self.cells),
                "overlap_s": round(sum(c.get("overlap_s", 0.0) for c in self.cells), 3),
                "partial_before_syntax_error": sum(1 for c in self.cells if c.get("partial_before_syntax_error")),
                "diverged": sum(1 for c in self.cells if c.get("diverged")),
                "orphaned": sum(1 for c in self.cells if c.get("orphaned"))}


__all__ = ["CONTINUATION_KEYWORDS", "StatementCutter", "StreamExec", "StreamingCell", "partial_input"]
