"""Opt-in speed options (docs/harness.md "Speed options"): each is off by default and does exactly what it says when on."""
from __future__ import annotations

import importlib.util
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from captain_harness.adapters.capx.direct_programming import DirectProgramming
from captain_harness.adapters.capx.motion_speed import MotionSpeedOptions
from captain_harness.ports.capability import CapabilityResult
from captain_harness.react.chat_llm import IMAGE_OMITTED_TEXT, ChatCompletionsClient, cap_images
from captain_harness.server.episode_launch import wait_for_server

ROOT = Path(__file__).resolve().parents[1]
PNG = "data:image/png;base64,iVBORw0KGgo="


# --------------------------------------------------------------------------- chunked image cap
def _history(n_images: int) -> list[dict]:
    return [{"role": "tool", "tool_call_id": str(i), "content": [{"type": "image_url", "image_url": {"url": f"{PNG}#{i}"}}]}
            for i in range(n_images)]


def test_chunk_one_is_the_release_cap():
    for n in range(0, 20):
        a, b = _history(n), _history(n)
        assert cap_images(a, 12) == cap_images(b, 12, 1) == max(0, n - 12)
        assert a == b


def test_chunked_cap_drops_multiples_of_chunk_and_keeps_the_newest():
    for n in range(13, 40):
        messages = _history(n)
        dropped = cap_images(messages, 12, 6)
        kept = [m["content"][0] for m in messages if m["content"][0]["type"] == "image_url"]
        assert dropped % 6 == 0 and dropped >= n - 12 and dropped - (n - 12) < 6
        assert 7 <= len(kept) <= 12 and len(kept) == n - dropped
        assert kept[-1]["image_url"]["url"].endswith(f"#{n - 1}")
        assert all(m["content"][0] == {"type": "text", "text": IMAGE_OMITTED_TEXT.format(n=12)} for m in messages[:dropped])
    with pytest.raises(ValueError):
        cap_images(_history(20), 12, 0)


def test_chunked_cap_keeps_the_prompt_append_only_between_drops():
    """Request k+1's capped messages extend request k's unchanged, except when a new chunk is dropped."""
    previous, rewrites = None, 0
    for n in range(1, 40):
        messages = _history(n)
        cap_images(messages, 12, 6)
        if previous is not None and messages[:len(previous)] != previous:
            rewrites += 1
        previous = messages
    assert rewrites == 5          # n = 13, 19, 25, 31, 37
    previous, rewrites = None, 0
    for n in range(1, 40):        # release cap: one rewrite per image beyond the cap
        messages = _history(n)
        cap_images(messages, 12)
        rewrites += previous is not None and messages[:len(previous)] != previous
        previous = messages
    assert rewrites == 39 - 12


def test_client_applies_the_chunk_and_sends_thinking_budget_only_when_set():
    body = {"model": "m", "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "p"}]}]}
    request, _, _ = ChatCompletionsClient("http://x/v1", max_output_tokens=1536, temperature=.7).build_request(body)
    assert "thinking_token_budget" not in request and request["max_tokens"] == 1536 and request["temperature"] == .7
    request, _, _ = ChatCompletionsClient("http://x/v1", thinking_token_budget=1024).build_request(body)
    assert request["thinking_token_budget"] == 1024
    images = {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": PNG}] * 13}
    _, _, stats = ChatCompletionsClient("http://x/v1", max_images=12).build_request({**body, "input": [images]})
    assert stats["images_dropped"] == 1 and stats["images"] == 12
    _, _, stats = ChatCompletionsClient("http://x/v1", max_images=12, image_drop_chunk=6).build_request({**body, "input": [images]})
    assert stats["images_dropped"] == 6 and stats["images"] == 7


def test_sticky_cap_keeps_omitted_images_omitted_after_a_compaction():
    """A compaction removes turns; without sticky the oldest images come back under the cap and are omitted again at the
    next drop (two early rewrites); sticky keeps them omitted, and an identical newer frame stays visible."""
    def img(name):
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{name}"}}
    def msgs(names, last=("z",)):
        return ([{"role": "user", "content": [img(n) for n in names]}]
                + [{"role": "user", "content": [img(n)]} for n in last])
    first = [f"a{i}" for i in range(14)]
    sticky: set[str] = set()
    m = msgs(first)
    assert cap_images(m, 12, 6, sticky=sticky) == 6
    assert sum(p["type"] == "image_url" for p in m[0]["content"]) == 8
    after = msgs(first[:2] + first[10:])           # compaction removed a2..a9; a0, a1 were omitted before
    plain = msgs(first[:2] + first[10:])
    cap_images(plain, 12, 6)
    assert plain[0]["content"][0]["type"] == "image_url"           # release: restored
    cap_images(after, 12, 6, sticky=sticky)
    assert [p["type"] for p in after[0]["content"][:2]] == ["text", "text"]
    assert all(p["type"] == "image_url" for p in after[0]["content"][2:])
    again = msgs(["a0", "b"], last=("a0",))        # leading run only, never the last message
    cap_images(again, 12, 6, sticky=sticky)
    assert [p["type"] for p in again[0]["content"]] == ["text", "image_url"] and again[1]["content"][0]["type"] == "image_url"


# --------------------------------------------------------------------------- env-server start polling
class _Alive:
    def poll(self):
        return None


def test_fast_server_poll_tolerates_a_partially_written_server_json(tmp_path):
    info = {"url": "http://127.0.0.1:1", "task": "t", "padding": "x" * 4000}
    text = json.dumps(info, indent=2)

    def writer():
        time.sleep(.05)
        with open(tmp_path / "server.json", "w", encoding="utf-8") as fh:
            fh.write(text[:len(text) // 2])
            fh.flush()
            time.sleep(.15)
            fh.write(text[len(text) // 2:])

    thread = threading.Thread(target=writer)
    thread.start()
    t0 = time.monotonic()
    assert wait_for_server(tmp_path, 5., _Alive(), poll_s=.01, grace_s=0.) == info
    assert time.monotonic() - t0 < 1.
    thread.join()


def test_release_server_poll_keeps_the_grace(tmp_path):
    (tmp_path / "server.json").write_text(json.dumps({"url": "u"}), encoding="utf-8")
    t0 = time.monotonic()
    assert wait_for_server(tmp_path, 5., _Alive()) == {"url": "u"}
    assert time.monotonic() - t0 >= .5


def test_fast_server_poll_still_reports_a_dead_server(tmp_path):
    class Dead:
        def poll(self):
            return 1
    with pytest.raises(RuntimeError, match="exited"):
        wait_for_server(tmp_path, 5., Dead(), poll_s=.01, grace_s=0.)


# --------------------------------------------------------------------------- motion speed options
def test_motion_speed_bounds():
    assert not MotionSpeedOptions().active
    for kwargs in ({"scale": 0}, {"scale": 11}, {"default_max_speed_m_s": .2}, {"default_max_angular_speed_rad_s": 0},
                   {"scale": float("nan")}):
        with pytest.raises(ValueError):
            MotionSpeedOptions(**kwargs)
    fast = MotionSpeedOptions(3, .1, 1.)
    args, record = fast.resolve({"position": [0, 0, 0], "max_speed_m_s": .02, "max_angular_speed_rad_s": .2})
    assert args["max_speed_m_s"] == pytest.approx(.06) and args["max_angular_speed_rad_s"] == pytest.approx(.6)
    assert record["requested_max_speed_m_s"] == .02 and record["executed_max_speed_m_s"] == pytest.approx(.06)
    args, record = fast.resolve({"position": [0, 0, 0]})          # defaults, then scaled, capped at the API maxima
    assert args["max_speed_m_s"] == .1 and args["max_angular_speed_rad_s"] == 1.
    assert record["requested_max_speed_m_s"] is None and record["executed_max_angular_speed_rad_s"] == 1.
    for bad in (.5, 0, -1, True, "fast"):                          # invalid requests reach the validator unchanged
        args, record = fast.resolve({"max_speed_m_s": bad})
        assert args == {"max_speed_m_s": bad} and record is None
    args, record = MotionSpeedOptions().resolve({"max_speed_m_s": .02})
    assert args == {"max_speed_m_s": .02} and record is None


@dataclass
class _Pose:
    end_effector_position_m: tuple = (0., 0., 1.)
    end_effector_quaternion_wxyz: tuple = (1., 0., 0., 0.)
    joint_positions_rad: tuple = (0.,) * 7
    gripper_open_fraction: float = 1.


class _Robot:
    _control_frequency = 20
    _cartesian_max_steps = 480
    _gripper_command_fraction = 1.
    simulation_steps = 0

    def __init__(self):
        self.sweeps = []

    def _eef_force_norm(self):
        return 0.

    def get_robot_state(self):
        return CapabilityResult.ok(_Pose())

    def _execute_contact_sweep(self, path, **options):
        self.sweeps.append(options)
        self.simulation_steps += 1
        return CapabilityResult.ok({"position_residual_m": .001})


class _Model:
    limits = np.array([[-3., 3.]] * 7)
    last_ik = {}

    def collision(self, joints, aperture, points):
        return {"collision": False, "observed_collisions": {}, "self_collisions": []}

    def solve_ik(self, pose, seed, *, aperture, position_tolerance, orientation_tolerance):
        return np.zeros(7)


class _SpyDirect(DirectProgramming):
    def __init__(self, tools):
        super().__init__(tools)
        self.prepared = []

    def _prepare_move(self, position, quaternion, max_speed_m_s=.05, max_angular_speed_rad_s=.5, *a, **k):
        self.prepared.append((max_speed_m_s, max_angular_speed_rad_s))
        return super()._prepare_move(position, quaternion, max_speed_m_s, max_angular_speed_rad_s, *a, **k)


def _direct(options=None):
    robot = _Robot()
    model = _Model()
    tools = SimpleNamespace(robot=robot, _motion_model=lambda: model)
    if options is not None:
        tools._motion_speed = options
    return _SpyDirect(tools), robot


def test_move_to_is_scaled_once_and_the_receipt_reports_requested_and_executed_speeds():
    direct, robot = _direct(MotionSpeedOptions(3, .1, 1.))
    goal = {"position": [0, 0, 1.02], "quaternion": [1, 0, 0, 0], "max_speed_m_s": .02, "max_angular_speed_rad_s": .2}
    result = direct.invoke("move_to", goal)
    assert result.success, result.message
    # the nested planning call (_move -> _prepare_move) sees the executed limits, not a second scaling (.18 -> .1)
    assert direct.prepared == [(pytest.approx(.06), pytest.approx(.6))]
    assert robot.sweeps[0]["max_speed_m_s"] == pytest.approx(.06)
    assert robot.sweeps[0]["max_angular_speed_rad_s"] == pytest.approx(.6)
    assert result.value["requested"]["max_speed_m_s"] == .02
    speed = result.value["motion_speed"]
    assert speed["requested_max_speed_m_s"] == .02 and speed["executed_max_speed_m_s"] == pytest.approx(.06)
    assert direct._recent_motion_timing[-1]["executed_max_speed_m_s"] == pytest.approx(.06)
    preview = direct.invoke("preview_motion", {"action": "move_to", "arguments": {"position": [0, 0, 1.02],
                                                                                  "quaternion": [1, 0, 0, 0]}})
    assert preview.success and direct.prepared[-1] == (.1, 1.)
    assert preview.value["motion_speed"]["executed_max_speed_m_s"] == .1


def test_default_motion_speed_leaves_requests_and_receipts_untouched():
    direct, robot = _direct()
    result = direct.invoke("move_to", {"position": [0, 0, 1.02], "quaternion": [1, 0, 0, 0], "max_speed_m_s": .02})
    assert result.success and direct.prepared == [(.02, .5)]
    assert robot.sweeps[0]["max_speed_m_s"] == .02 and "motion_speed" not in result.value
    assert "executed_max_speed_m_s" not in direct._recent_motion_timing[-1]
    preview = direct.invoke("preview_motion", {"action": "move_to", "arguments": {"position": [0, 0, 1.02],
                                                                                  "quaternion": [1, 0, 0, 0]}})
    assert "motion_speed" not in preview.value and direct.prepared[-1] == (.05, .5)


def test_tool_backend_options_carry_the_motion_speed():
    from captain_harness.adapters.capx.backend import ToolCapabilityBackend
    common = dict(robot=SimpleNamespace(), services=SimpleNamespace(), registry=SimpleNamespace())
    assert ToolCapabilityBackend(**common)._motion_speed == MotionSpeedOptions()
    tools = ToolCapabilityBackend(**common, motion_speed_scale=3, default_max_speed_m_s=.1,
                                  default_max_angular_speed_rad_s=1.)
    assert tools._motion_speed == MotionSpeedOptions(3, .1, 1.)
    with pytest.raises(ValueError):
        ToolCapabilityBackend(**common, motion_speed_scale=0)


# --------------------------------------------------------------------------- hardware step profile
def test_step_profile_is_per_backend_instance(monkeypatch):
    pytest.importorskip("mujoco")
    from captain_harness.adapters.hardware import backend as hw
    monkeypatch.setattr(hw.ClosingMonitoredPossessionActionBackend, "__init__", lambda self, env, **kwargs: None)
    fast = hw.XR1ActionBackend(None, step_profile={"speed_scale": 3, "slow_zone_m": .03, "settle_samples_needed": 1,
                                                   "gripper_still_needed": 2})
    other = hw.XR1ActionBackend(None, step_profile={"speed_scale": 1.5, "velocity_samples": 1})
    plain = hw.XR1ActionBackend(None)
    assert (fast._speed_scale, fast._slow_zone_m, fast._settle_samples_needed, fast._gripper_still_needed) == (3., .03, 1, 2)
    assert other._speed_scale == 1.5 and other._velocity_samples is True and not hasattr(other, "_slow_zone_m")
    assert isinstance(fast._settle_samples_needed, int)
    # readers use getattr(self, name, release default): an instance without a profile keeps the release behaviour
    assert getattr(plain, "_speed_scale", 1.0) == 1.0 and getattr(plain, "_gripper_still_needed", 3) == 3
    assert not any(hasattr(hw.XR1ActionBackend, f"_{key}") for key in hw.STEP_PROFILE_KEYS)
    with pytest.raises(ValueError, match="unknown step_profile keys"):
        hw.XR1ActionBackend(None, step_profile={"speed": 3})


def test_robodojo_passes_the_step_profile_as_a_backend_option():
    source = (ROOT / "src/captain_harness/adapters/robodojo/episode.py").read_text()
    assert '"step_profile": dict(config["control"].get("step_profile") or {})' in source
    assert "XR1ActionBackend._speed_scale" not in source


# --------------------------------------------------------------------------- SAM3 image-encoder cache
def _sam3_server():
    spec = importlib.util.spec_from_file_location("sam3_server_under_test", ROOT / "services" / "sam3_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sam3_cache_encodes_each_image_once_and_returns_the_same_state():
    pil = pytest.importorskip("PIL.Image")

    class Processor:
        encodes = 0

        def set_image(self, image, state=None):
            type(self).encodes += 1
            state = {} if state is None else state
            state["original_height"], state["original_width"] = image.size[1], image.size[0]
            state["backbone_out"] = {"vision": np.asarray(image, float).sum(), "fpn": [1, 2]}
            return state

    reference = Processor()
    a = pil.fromarray(np.full((4, 6, 3), 7, np.uint8))
    b = pil.fromarray(np.full((4, 6, 3), 9, np.uint8))
    expected = {id(img): reference.set_image(img) for img in (a, b)}
    Processor.encodes = 0
    stats = _sam3_server().install_image_cache(Processor, 2)
    proc = Processor()
    first = proc.set_image(a)
    first["backbone_out"]["language_features"] = "text prompt"   # what set_text_prompt does to the per-call copy
    second = proc.set_image(a)
    assert second == expected[id(a)] and "language_features" not in second["backbone_out"]
    assert proc.set_image(b) == expected[id(b)]
    assert Processor.encodes == 2 and stats["hits"] == 1 and stats["misses"] == 2
    proc.set_image(pil.fromarray(np.zeros((4, 6, 3), np.uint8)))  # third image evicts the least recently used (a)
    proc.set_image(a)
    assert Processor.encodes == 4
    proc.set_image(a, state={})                                   # explicit state: never cached
    assert Processor.encodes == 5
    with pytest.raises(ValueError):
        _sam3_server().install_image_cache(Processor, 0)
