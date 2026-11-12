"""compact_v1 prompt profile (TASK.md, workspace API docs, summarised capability results) and the table-footprint
plane fit of the monocular depth path. CPU only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

INTERFACE = '''# Every call returns CapabilityResult: .success, .executed, .failure_category, .message, .value.
# Optional .diagnostics retains deadline scope, service attempts and measured partial execution.
# A timed-out operation may have partially executed; inspect its receipt before retrying.

# Control this robot by writing Python from public images and measurements. Long general rules follow here.

def observe() -> CapabilityResult[dict]:
    """Capture current public RGB-D, calibration and robot state. Observation handles can change.
    Frames: {'result': 'robot_base'}
    """

def move_to(position: list[float] = None, quaternion: list[float] = None, gripper: str = None) -> CapabilityResult[dict]:
    """Execute your absolute EEF pose OR one gripper command. Sampled Cartesian IK is checked.
    Argument constraints: {"required": []}
    """
'''


def _md(**overrides):
    from captain_harness.server.task_markdown import task_markdown

    info = {"task": "put the cube in the box", "initial_images": [], "task_markdown_overrides": overrides}
    return task_markdown(info, {"python_interface": INTERFACE}, wall_time_s=3600, max_actions=150, max_tool_calls=600,
                         runtime="python")


def test_default_task_markdown_unchanged_by_profile_code() -> None:
    from captain_harness.server.task_markdown import STANDARDS

    md = _md()
    assert "## Capability API (Python signatures; call them as `robot.<name>(...)`)" in md
    assert "Long general rules follow here" in md and STANDARDS in md and "details_file" not in md


def test_compact_task_markdown_and_api_files() -> None:
    from captain_harness.server.task_markdown import COMPACT_PROFILE, api_files

    md = _md(profile=COMPACT_PROFILE, intro="You are operating an SO-101 arm.", standards="## Standards\n- one rule")
    assert md.startswith("# Robot manipulation task\n\nYou are operating an SO-101 arm.")
    assert "## Standards\n- one rule" in md and "Articulated targets" not in md
    assert "def observe() -> dict  # Capture current public RGB-D, calibration and robot state." in md
    assert "def move_to(position: list[float] = None, quaternion: list[float] = None, gripper: str = None) -> dict  #" in md
    assert "Long general rules follow here" not in md and "Argument constraints" not in md
    assert "STATUS block at the TOP of `NOTES.md`" in md and "details_file" in md
    files = api_files({"python_interface": INTERFACE})
    assert set(files) == {"API.md", "api/observe.md", "api/move_to.md"}
    assert "Long general rules follow here" in files["API.md"] and "api/move_to.md" in files["API.md"]
    assert "Argument constraints" in files["api/move_to.md"] and "def observe" not in files["api/move_to.md"]
    with pytest.raises(ValueError):
        _md(profile="nope")
    with pytest.raises(ValueError):
        _md(standards="## Standards")      # a full replacement only with a profile


def test_build_workspace_writes_api_docs(tmp_path: Path) -> None:
    from captain_harness.server.episode_launch import build_workspace
    from captain_harness.server.task_markdown import COMPACT_PROFILE

    manifest = {"python_interface": INTERFACE, "manifest": {"capabilities": []}}
    info = {"task": "t", "initial_images": [], "url": "http://127.0.0.1:1", "task_markdown_overrides": {"profile": COMPACT_PROFILE}}
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ws = build_workspace(tmp_path / "a", info, manifest, wall_time_s=60, max_actions=5, max_tool_calls=5, runtime="python")
    assert (ws / "API.md").is_file() and (ws / "api" / "move_to.md").is_file()
    plain = build_workspace(tmp_path / "b", {**info, "task_markdown_overrides": {}}, manifest, wall_time_s=60,
                            max_actions=5, max_tool_calls=5, runtime="python")
    assert not (plain / "API.md").exists() and not (plain / "api").exists()


def test_compact_result(tmp_path: Path) -> None:
    from captain_harness.server.compact_values import MIN_CHARS, compact_result

    small = {"success": True, "value": {"a": 1}}
    assert compact_result(small, workspace=tmp_path, call_index=1, name="observe") == small
    traj = [{"t": i, "pose": [0.1 * i, 0.2, 0.3], "force": {"n": float(i)}} for i in range(400)]
    big = {"success": True, "executed": True, "failure_category": "none", "message": "ok", "budget": {"left": 3},
           "receipt": "move_to: ok", "value": {"endpoint_reached": True, "trajectory": traj, "note": "x" * 500,
                                               "deep": {"a": {"b": {"c": {"d": 1}}}}},
           "diagnostics": {"deadline": {"elapsed_s": 1.0}}}
    assert len(json.dumps(big)) > MIN_CHARS
    out = compact_result(big, workspace=tmp_path, call_index=7, name="move_to")
    assert out["details_file"] == "values/0007_move_to.json"
    assert json.loads((tmp_path / out["details_file"]).read_text()) == big
    assert out["success"] is True and out["receipt"] == "move_to: ok" and out["budget"] == {"left": 3}
    assert out["value"]["endpoint_reached"] is True and out["value"]["trajectory"]["list_len"] == 400
    assert out["value"]["note"].endswith("[see details_file]") and len(out["value"]["note"]) <= 160
    assert out["value"]["deep"]["a"]["b"]["c"] == "<1 keys: see details_file>"
    assert len(json.dumps(out)) < 2000


def _scene(h=120, w=160):
    """A camera 0.6 m above the table looking down-forward, a table rectangle and a floor 0.7 m below beyond x > 0.4 m."""
    from captain_harness.adapters.hardware.mono_depth import plane_depth, plane_hit_xy

    k = np.array([[120.0, 0, w / 2], [0, 120.0, h / 2], [0, 0, 1]])
    pitch = np.deg2rad(50)
    # optical frame: z forward (down-tilted toward +x), x right (-y base), y down
    fwd = np.array([np.cos(pitch), 0, -np.sin(pitch)])
    right = np.array([0, -1.0, 0])
    down = np.cross(fwd, right)
    t = np.eye(4)
    t[:3, :3] = np.stack([right, down, fwd], 1)
    t[:3, 3] = [-0.2, 0.0, 0.6]
    table = plane_depth(k, t, 0.0, (h, w))
    floor = plane_depth(k, t, -0.7, (h, w))
    x, y = plane_hit_xy(k, t, 0.0, (h, w))
    true = np.where((x <= 0.4) & (np.abs(y) <= 0.5), table, floor)
    return k, t, true


def test_footprint_fit_ignores_floor_past_the_table_edge() -> None:
    from captain_harness.adapters.hardware.mono_depth import fit_plane_scale, footprint_mask, plane_depth

    k, t, true = _scene()
    mono = true / 4.0                       # monocular depth: right shape, wrong scale (true scale 4)
    plane = plane_depth(k, t, 0.0, true.shape)
    s_far, _ = fit_plane_scale(mono, plane, min_pixels=200)
    inside = footprint_mask(k, t, 0.0, true.shape, {"x": [-0.05, 0.39], "y": [-0.33, 0.33]})
    s_ft, info = fit_plane_scale(mono, plane, min_pixels=200, exclude=~inside, start="densest")
    assert info["status"] == "ok" and abs(s_ft / 4.0 - 1) < 0.01
    assert abs(s_far / 4.0 - 1) > 0.05     # the default start locks onto the floor seen past the edge
    with pytest.raises(ValueError):
        fit_plane_scale(mono, plane, start="nearest")


def test_mono_depth_source_footprint_option() -> None:
    from captain_harness.adapters.hardware.mono_depth import MonoDepthSource

    k, t, true = _scene()
    src = MonoDepthSource("http://unused", align="plane", support_plane_z_m=0.0,
                          table_footprint_m={"x": [-0.05, 0.39], "y": [-0.33, 0.33]})
    src._request = lambda rgb, intr: (true / 4.0, {})
    rgb = np.zeros(true.shape + (3,), np.uint8)
    depth = src(camera="agentview", rgb=rgb, raw_sim_depth=None, intrinsics_3x3=k, camera_to_base_4x4=t)
    assert src.last_record["align"]["footprint"] is True and abs(src.last_record["align"]["scale"] / 4 - 1) < 0.01
    finite = np.isfinite(true)
    assert np.allclose(depth[finite], true[finite], rtol=0.01)
    assert "calibrated table footprint" in " ".join(src.describe()["inputs"])
    with pytest.raises(ValueError):
        MonoDepthSource("http://unused", align="plane", support_plane_z_m=0.0, table_footprint_m={"x": [1, 0], "y": [0, 1]})
