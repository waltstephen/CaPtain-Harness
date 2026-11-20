"""Opt-in execution speed policy for the coding-control motion tools (move_to, preview_motion of move_to).

The public move_to schema bounds the Cartesian speed limits at 0.1 m/s and 1.0 rad/s, with defaults 0.05 m/s and
0.5 rad/s. A deployment may execute the policy's requests faster without changing the model-visible tool text:

- ``default_max_speed_m_s`` / ``default_max_angular_speed_rad_s`` replace the defaults used when the policy omits the
  argument;
- ``motion_speed_scale`` multiplies the requested (or default) limits; the product is capped at the schema maxima.

Defaults (1.0, 0.05, 0.5) reproduce the release behaviour exactly; DirectProgramming then adds nothing to receipts.
When any option differs, move_to / preview_motion receipts carry ``motion_speed`` with the requested and the executed
limits. Passed as ``tool_backend_options`` (ToolCapabilityBackend keyword arguments).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

MAX_SPEED_M_S = .1                 # move_to schema maxima
MAX_ANGULAR_SPEED_RAD_S = 1.
DEFAULT_SPEED_M_S = .05            # move_to schema defaults
DEFAULT_ANGULAR_SPEED_RAD_S = .5
MAX_SCALE = 10.


def _positive(value: float, upper: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0. < value <= upper:
        raise ValueError(f"{name} must be in (0, {upper}], got {value!r}")
    return value


def _valid_request(value: object, upper: float) -> bool:
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                             and math.isfinite(value) and 0. < value <= upper)


@dataclass(frozen=True)
class MotionSpeedOptions:
    scale: float = 1.
    default_max_speed_m_s: float = DEFAULT_SPEED_M_S
    default_max_angular_speed_rad_s: float = DEFAULT_ANGULAR_SPEED_RAD_S

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", _positive(self.scale, MAX_SCALE, "motion_speed_scale"))
        object.__setattr__(self, "default_max_speed_m_s",
                           _positive(self.default_max_speed_m_s, MAX_SPEED_M_S, "default_max_speed_m_s"))
        object.__setattr__(self, "default_max_angular_speed_rad_s",
                           _positive(self.default_max_angular_speed_rad_s, MAX_ANGULAR_SPEED_RAD_S,
                                     "default_max_angular_speed_rad_s"))

    @property
    def active(self) -> bool:
        return self != MotionSpeedOptions()

    def resolve(self, arguments: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object] | None]:
        """Executed move_to arguments and the receipt record (None when inactive: arguments are returned unchanged).

        Applied once per public call, before any nested planning call, so a request is never scaled twice."""
        if not self.active:
            return dict(arguments), None
        requested = (arguments.get("max_speed_m_s"), arguments.get("max_angular_speed_rad_s"))
        if not (_valid_request(requested[0], MAX_SPEED_M_S) and _valid_request(requested[1], MAX_ANGULAR_SPEED_RAD_S)):
            return dict(arguments), None          # invalid request: the schema validator rejects it unchanged
        base = (float(requested[0]) if requested[0] is not None else self.default_max_speed_m_s,
                float(requested[1]) if requested[1] is not None else self.default_max_angular_speed_rad_s)
        executed = (min(MAX_SPEED_M_S, base[0] * self.scale), min(MAX_ANGULAR_SPEED_RAD_S, base[1] * self.scale))
        record = {"requested_max_speed_m_s": requested[0], "requested_max_angular_speed_rad_s": requested[1],
                  "executed_max_speed_m_s": executed[0], "executed_max_angular_speed_rad_s": executed[1],
                  "speed_scale": self.scale, "default_max_speed_m_s": self.default_max_speed_m_s,
                  "default_max_angular_speed_rad_s": self.default_max_angular_speed_rad_s,
                  "caps": [MAX_SPEED_M_S, MAX_ANGULAR_SPEED_RAD_S]}
        return {**arguments, "max_speed_m_s": executed[0], "max_angular_speed_rad_s": executed[1]}, record
