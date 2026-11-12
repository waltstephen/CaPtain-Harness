"""Stream-exec (opt-in speed option, docs/harness.md "Speed options"): the statement cutter, the streamed `input`
argument, and early execution whose tool output equals whole-cell execution."""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from captain_harness.react.chat_llm import ChatCompletionsClient
from captain_harness.react.python_repl import PythonRepl, python_tool, run_python_cell
from captain_harness.react.sandbox import Sandbox
from captain_harness.react.stream_exec import StatementCutter, StreamExec, chunk_code, partial_input
from captain_harness.react.tools import ToolRouter
from captain_harness.react.unified_exec import SessionManager, child_environment


# --------------------------------------------------------------------------- statement cutter
def _stream_cuts(text: str) -> tuple[list[tuple[str, int]], int]:
    """Feed ``text`` one character at a time: the cut chunks with the decoded length at which each was cut."""
    cutter = StatementCutter()
    found = []
    for n in range(len(text) + 1):
        found += [(text[start:end], n) for start, end in cutter.cuts(text[:n])]
    return found, cutter.start


CUTS = {
    "multi-line call": ("x = f(1,\n      2)\ny = g(x)\nprint(y)\n", ["x = f(1,\n      2)\n", "y = g(x)\n"]),
    "open bracket with column-0 lines": ("x = [\n1,\n2]\ny = 3\nz = 4\n", ["x = [\n1,\n2]\n", "y = 3\n"]),
    "if/else": ("if a:\n    b = 1\nelse:\n    b = 2\nc = b\nd = 1\n", ["if a:\n    b = 1\nelse:\n    b = 2\n", "c = b\n"]),
    "if/elif": ("if a:\n    b = 1\nelif c:\n    b = 2\nd = b\ne = 1\n", ["if a:\n    b = 1\nelif c:\n    b = 2\n", "d = b\n"]),
    "for/else": ("for i in r:\n    pass\nelse:\n    j = 1\nk = 2\nm = 3\n", ["for i in r:\n    pass\nelse:\n    j = 1\n", "k = 2\n"]),
    "try/except/finally": ("try:\n    x = 1\nexcept E:\n    x = 2\nfinally:\n    y = 3\nz = 4\nw = 5\n",
                           ["try:\n    x = 1\nexcept E:\n    x = 2\nfinally:\n    y = 3\n", "z = 4\n"]),
    "decorators": ("@dec\n@dec2(1)\ndef f():\n    return 1\nclass C:\n    pass\nf()\n",
                   ["@dec\n@dec2(1)\ndef f():\n    return 1\n", "class C:\n    pass\n"]),
    "multi-line string": ('s = """\nx = 1\n\ny = 2\n"""\nt = 1\nu = 2\n', ['s = """\nx = 1\n\ny = 2\n"""\n', "t = 1\n"]),
    "string with column-0 code lines": ("print('''\nif a:\nelse:\n''')\nv = 1\nw = 2\n", ["print('''\nif a:\nelse:\n''')\n", "v = 1\n"]),
    "comments and blank lines": ("x = 1  # set\n# comment\n\nif x:\n    pass\n# between\n\nelse:\n    pass\ny = 2\nz = 3\n",
                                 ["x = 1  # set\n", "# comment\n\nif x:\n    pass\n# between\n\nelse:\n    pass\n", "y = 2\n"]),
    "backslash continuation": ("x = 1 + \\\n2\ny = 3\nz = 4\n", ["x = 1 + \\\n2\n", "y = 3\n"]),
    "indented continuation": ("x = 1 + \\\n    2\ny = 3\nz = 4\n", ["x = 1 + \\\n    2\n", "y = 3\n"]),
    "return outside a function never cuts": ("a = 1\nreturn a\nb = 2\nc = 3\n", ["a = 1\n"]),
    "syntax error stops cutting": ("a = 1\nb = (\nc = 2\nd = 3\n", ["a = 1\n"]),
    "trailing partial line": ("a = 1\nb = foo(1,", ["a = 1\n"]),
    "last line is never cut": ("a = 1\nb = 2\n", ["a = 1\n"]),
    # one-line cells (the SO-101 policy's usual style): cut after each top-level `;` as soon as it is decoded
    "semicolons": ("obs=observe(); image(obs); print(obs['a']); open('N','w').write('x;y')",
                   ["obs=observe();", " image(obs);", " print(obs['a']);"]),
    "semicolon inside a string": ("s = 'a;b'; t = 1; u = 2", ["s = 'a;b';", " t = 1;"]),
    "semicolon inside a comment": ("x = 1  # a; b\ny = 2\nz = 3\n", ["x = 1  # a; b\n", "y = 2\n"]),
    "semicolons inside brackets": ("d = {'k': f(1); 2}\ne = 1\n", []),
    "semicolons in a one-line suite": ("if cid: a=1; print(a); r=2\nz=1; w=2\nv=3\n", ["if cid: a=1; print(a); r=2\n", "z=1;", " w=2\n"]),
    "semicolons in a one-line loop and def": ("for i in r: a; b\ndef f(): return 1; x = 2\ny = f(); z = 0",
                                              ["for i in r: a; b\n", "def f(): return 1; x = 2\n", "y = f();"]),
    "trailing semicolons": ("a = 1;\nb = 2;\n", ["a = 1;", "\nb = 2;"]),
    "semicolon after a multi-line statement": ("x = f(1,\n  2); y = 3; z", ["x = f(1,\n  2);", " y = 3;"]),
}


@pytest.mark.parametrize("name", list(CUTS))
def test_cutter_cuts_only_complete_top_level_statements(name):
    text, expected = CUTS[name]
    found, start = _stream_cuts(text)
    assert [chunk for chunk, _ in found] == expected
    assert "".join(expected) == text[:start]
    offset = 0
    for chunk, decoded in found:
        compile(chunk_code(text, offset, offset + len(chunk)), "<t>", "exec")
        offset += len(chunk)
        if chunk.endswith(";"):
            assert decoded == offset, "a top-level `;` ends its statement at once"
            continue
        nxt = text[offset:].lstrip("\n")
        assert decoded > offset and (decoded > len(text) - len(nxt)), "a cut at a line end needs the next line's first token"


def test_cutter_waits_for_a_word_that_could_still_become_a_continuation_keyword():
    cutter = StatementCutter()
    head = "if a:\n    b = 1\n"
    for partial in ("e", "el", "els", "else"):
        assert cutter.cuts(head + partial) == []
    assert cutter.cuts(head + "else:\n    b = 2\n") == []
    cutter = StatementCutter()
    for partial in ("e", "el", "ela", "elapsed"):
        assert cutter.cuts("x = 1\n" + partial) == []
    assert cutter.cuts("x = 1\nelapsed ") == [(0, 6)]


def test_cuts_are_reported_once_each_in_order():
    cutter = StatementCutter()
    text = "a = 1\nb = (1,\n2)\nc = 3\nd"
    assert cutter.cuts(text) == [(0, 6), (6, 17)]          # "d" could still become a continuation keyword
    assert cutter.cuts(text + " = 4\n") == [(17, 23)]
    assert cutter.cuts(text + " = 4\ne = 5\n") == [(23, 29)]


# --------------------------------------------------------------------------- streamed arguments
@pytest.mark.parametrize("ascii_only", [True, False])
def test_partial_input_decodes_every_prefix_of_the_arguments(ascii_only):
    code = 'print("tab\\t", "quote\\"", \'é\', "😀")\nx = {"k": [1, 2]}  # \\ back\n'
    arguments = json.dumps({"input": code}, ensure_ascii=ascii_only)
    previous = ""
    for n in range(len(arguments) + 1):
        decoded = partial_input(arguments[:n])
        assert code.startswith(decoded) and decoded.startswith(previous)
        previous = decoded
    assert previous == code
    assert partial_input('{"input":"a\\nb') == "a\nb" and partial_input('{"inp') == "" and partial_input("") == ""


# --------------------------------------------------------------------------- whole-cell equivalence
def _repl(tmp_path, name):
    workspace = tmp_path / name / "workspace"
    workspace.mkdir(parents=True)
    sandbox = Sandbox(workspace=workspace.resolve())
    env = child_environment(os.environ)
    repl = PythonRepl(sandbox=sandbox, env=env, python=sys.executable, kill_grace_s=3.0)
    repl.start()
    router = ToolRouter(sessions=SessionManager(sandbox=sandbox, env=env, default_cwd=workspace), workspace=workspace)
    return repl, router, workspace


def _text(parts) -> str:
    return re.sub(r" in \d+\.\d s", " in T s", parts[0]["text"])


def _stream(stream: StreamExec, code: str, *, settle_s: float = 0.4) -> None:
    """A fake decoding of one response: the `python` arguments arrive a few characters at a time."""
    stream.begin_response()
    arguments = json.dumps({"input": code})
    for n in range(0, len(arguments) + 1, 3):
        stream.tool_input(0, "python", arguments[:n])
    stream.tool_input(0, "python", arguments)
    time.sleep(settle_s)                # the cut statements run while "decoding" continues
    stream.end_response()


CELLS = [
    "import math\nx = math.sqrt(16)\nprint('x', x)\ndef sq(v):\n    return v * v\nprint(sq(3))\nfor i in range(2):\n    print(i)\nsq(x)\n",
    "print('before')\ny = 1 / 0\nprint('after')\nz = 2\n",
    "z\n",
    '"""the docstring"""\nimport os\nprint(__doc__)\nos.path.join("a", "b")\n',
    "if x > 3:\n    label = 'big'\nelse:\n    label = 'small'\ntry:\n    int('q')\nexcept ValueError as exc:\n    print('caught', exc)\nfinally:\n    print('finally')\nlabel",
    "def boom():\n    raise RuntimeError('deep')\nprint('a')\nboom()\nprint('never')\n",
    "items = [\n    1,\n    2,\n]\ns = '''\nmulti\n'''\n# trailing comment\nlen(items)",
    "import sys\nprint('to stderr', file=sys.stderr)\nprint('to stdout')\nw = 5\n",
    # one-line cells (`;`-separated), the SO-101 policy's usual style
    "a = 2; print('a', a); b = a * 3; print(b); b",
    "print('x'); y2 = 1 / 0; print('never')",
    "if a: print('in'); c = 5\nprint(c); c + 1;",
    "def g(v): return v + 1; h = g(1); print(h)",
]


def test_streamed_cells_render_exactly_like_whole_cell_execution(tmp_path):
    release_repl, release_router, _ = _repl(tmp_path, "release")
    stream_repl, stream_router, _ = _repl(tmp_path, "stream")
    stream = StreamExec(lambda: stream_repl, lambda: 60.0)
    try:
        for code in CELLS:
            expected = _text(run_python_cell(code, router=release_router, repl=release_repl, timeout_s=60.0))
            _stream(stream, code)
            got = _text(run_python_cell(code, router=stream_router, repl=stream_repl, timeout_s=60.0, stream=stream))
            assert got == expected, code
        facts = [cell for cell in stream.cells]
        assert len(facts) == len(CELLS) and all(not f["diverged"] and not f["partial_before_syntax_error"] for f in facts)
        assert facts[0]["early_statements"] >= 4 and facts[0]["overlap_s"] >= 0.0
        assert facts[1]["statements"] == 2          # the raising statement stops the cell: `after` never ran
        assert facts[2]["early_statements"] == facts[2]["statements"] == 0   # one line: nothing to cut, runs the release way
        assert [c["cell"] for c in stream_router.python_cells] == [c["cell"] for c in release_router.python_cells]
        assert stream_router.python_cells[0]["stream_exec"]["early_statements"] == facts[0]["early_statements"]
        assert stream.summary()["cells"] == len(CELLS)
        assert facts[8]["early_statements"] == 4 and facts[8]["statements"] == 5     # every `;` statement but the last
        assert facts[9]["early_statements"] == facts[9]["statements"] == 2           # 1 / 0 stops the cell
        assert facts[11]["statements"] == 0                                          # `;` inside a def suite never cuts
        assert "stream_exec" not in release_router.python_cells[0]
    finally:
        release_repl.close()
        stream_repl.close()


def test_a_cell_that_does_not_compile_reports_the_release_error_after_its_early_statements_ran(tmp_path):
    release_repl, release_router, _ = _repl(tmp_path, "release")
    stream_repl, stream_router, workspace = _repl(tmp_path, "stream")
    stream = StreamExec(lambda: stream_repl, lambda: 60.0)
    code = "print('ran')\nopen('early.txt', 'w').write('x')\nq = 1\nwhile True print(q)\nprint('end')\n"
    try:
        expected = _text(run_python_cell(code, router=release_router, repl=release_repl, timeout_s=60.0))
        _stream(stream, code)
        got = _text(run_python_cell(code, router=stream_router, repl=stream_repl, timeout_s=60.0, stream=stream))
        assert "Output: (none)" in expected and "Output:\nran" in got
        error = lambda text: text.split("Error:\n")[1].split("\nChanged:")[0]  # noqa: E731
        assert error(expected) == error(got) and "SyntaxError" in got
        assert (workspace / "early.txt").is_file()
        facts = stream.cells[-1]
        assert facts["partial_before_syntax_error"] is True and facts["statements"] == 3 and "print('end')" not in got
        assert stream_router.python_cells[-1]["stream_exec"]["partial_before_syntax_error"] is True
    finally:
        release_repl.close()
        stream_repl.close()


def test_fenced_or_budgetless_cells_run_the_release_way(tmp_path):
    repl, router, _ = _repl(tmp_path, "s")
    try:
        stream = StreamExec(lambda: repl, lambda: 60.0)
        fenced = "```python\na = 1\nb = 2\nprint(a + b)\n```"
        _stream(stream, fenced, settle_s=0.0)
        assert "Cell 1 completed" in _text(run_python_cell(fenced, router=router, repl=repl, stream=stream))
        assert stream.cells[-1]["disabled"] == "fenced" and stream.cells[-1]["statements"] == 0
        stream = StreamExec(lambda: repl, lambda: None)
        _stream(stream, "a = 1\nb = 2\nprint(a + b)\n", settle_s=0.0)
        assert "Cell 2 completed" in _text(run_python_cell("a = 1\nb = 2\nprint(a + b)\n", router=router, repl=repl, stream=stream))
        assert stream.cells[-1]["disabled"] == "budget"
    finally:
        repl.close()


def test_a_response_whose_call_never_runs_is_closed_as_orphaned(tmp_path):
    repl, router, workspace = _repl(tmp_path, "s")
    marks = []
    stream = StreamExec(lambda: repl, lambda: 60.0, before_start=lambda: marks.append(len(marks)))
    try:
        _stream(stream, "open('a.txt', 'w').write('1')\nb = 2\nc = 3\n")
        stream.begin_response()          # the next request: the previous cell is closed
        assert (workspace / "a.txt").is_file() and marks == [0]
        assert stream.cells[-1]["orphaned"] is True and stream.cells[-1]["statements"] == 2
        assert "Cell 2 completed" in _text(run_python_cell("print(b)\n", router=router, repl=repl))
    finally:
        repl.close()


# --------------------------------------------------------------------------- streamed chat completion
class _StreamingServer:
    """Serves one chat completion either as JSON or as server-sent events; while streaming it waits until the file
    created by the cell's first statement exists before decoding the rest (the early statement ran mid-response)."""

    def __init__(self, message: dict, workspace, chunks: list[str]) -> None:
        owner = self
        self.requests: list[dict] = []
        self.seen_early = None

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                return

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(body)
                usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                if not body.get("stream"):
                    data = json.dumps({"id": "c1", "model": "m", "usage": usage,
                                       "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def send(delta, finish=None, **extra):
                    chunk = {"id": "c1", "model": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()

                send({"role": "assistant", "content": ""})
                for piece in ("thinking ", "briefly"):
                    send({"reasoning_content": piece})
                call = message["tool_calls"][0]
                send({"tool_calls": [{"index": 0, "id": call["id"], "type": "function", "function": {"name": "python"}}]})
                for i, piece in enumerate(chunks):
                    if i == 2:
                        deadline = time.monotonic() + 10
                        while not (workspace / "early.txt").is_file() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        owner.seen_early = (workspace / "early.txt").is_file()
                    send({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]})
                send({}, finish="tool_calls")
                self.wfile.write(f"data: {json.dumps({'id': 'c1', 'model': 'm', 'choices': [], 'usage': usage})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"


def _items(result):
    return [{k: v for k, v in item.items() if k != "id"} for item in result.items]


def test_streamed_completion_assembles_the_release_message_and_runs_the_cell_while_decoding(tmp_path):
    release_repl, release_router, _ = _repl(tmp_path, "release")
    stream_repl, stream_router, workspace = _repl(tmp_path, "stream")
    code = "__import__('time').sleep(0.2); open('early.txt', 'w').write('x')\nprint('first')\nvalue = 6 * 7\nvalue"
    arguments = json.dumps({"input": code})
    cut = arguments.index("print")
    chunks = [arguments[:cut - 3], arguments[cut - 3:cut + 6], arguments[cut + 6:cut + 20], arguments[cut + 20:]]
    message = {"role": "assistant", "content": None, "reasoning_content": "thinking briefly",
               "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "python", "arguments": arguments}}]}
    server = _StreamingServer(message, workspace, chunks)
    body = {"model": "m", "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]}],
            "tools": [python_tool("run python")]}
    stream = StreamExec(lambda: stream_repl, lambda: 60.0)
    try:
        release = ChatCompletionsClient(server.url).respond(body)
        streamed = ChatCompletionsClient(server.url, stream_exec=stream).respond(body)
        assert server.requests[0]["stream"] is False and "stream_options" not in server.requests[0]
        assert server.requests[1]["stream"] is True and server.requests[1]["stream_options"] == {"include_usage": True}
        assert _items(streamed) == _items(release) and streamed.usage == release.usage and streamed.status == release.status
        assert streamed.raw["finish_reason"] == release.raw["finish_reason"] == "tool_calls"
        assert server.seen_early is True                          # the first statement ran before the cell was decoded
        source = next(item["input"] for item in streamed.items if item["type"] == "custom_tool_call")
        assert source == code
        expected = _text(run_python_cell(code, router=release_router, repl=release_repl, timeout_s=60.0))
        got = _text(run_python_cell(source, router=stream_router, repl=stream_repl, timeout_s=60.0, stream=stream))
        assert got == expected and "Out: 42" in got
        facts = stream.cells[-1]
        assert facts["early_statements"] >= 2 and facts["statements"] == 5 and facts["overlap_s"] >= 0.15
    finally:
        release_repl.close()
        stream_repl.close()
        server.server.shutdown()
