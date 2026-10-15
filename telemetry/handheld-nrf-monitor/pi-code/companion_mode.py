#!/usr/bin/env python3
"""Safety state machine for the Raspberry Pi companion computer.

This module evaluates whether the companion computer is healthy enough to
monitor the aircraft or become ready for a future, separately guarded assist
feature.  It never sends MAVLink commands.
"""

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Dict, Optional, Tuple


class RequestedCompanionMode(str, Enum):
    DISABLED = "disabled"
    MONITOR = "monitor"
    ASSIST = "assist"


class CompanionState(str, Enum):
    DISABLED = "disabled"
    MONITOR = "monitor"
    ASSIST_READY = "assist_ready"
    DEGRADED = "degraded"
    FAILSAFE = "failsafe"


@dataclass(frozen=True)
class CompanionStatus:
    state: CompanionState
    reasons: Tuple[str, ...]
    commands_allowed: bool = False


@dataclass(frozen=True)
class CompanionLimits:
    telemetry_timeout_s: float = 3.0
    minimum_satellites: int = 6
    minimum_battery_percent: int = 20
    critical_battery_percent: int = 10
    maximum_cpu_temp_c: float = 80.0
    critical_cpu_temp_c: float = 85.0


class CompanionModeController:
    """Evaluates companion readiness without taking control of the aircraft."""

    def __init__(
        self,
        requested_mode: RequestedCompanionMode = RequestedCompanionMode.MONITOR,
        limits: Optional[CompanionLimits] = None,
    ):
        self.requested_mode = requested_mode
        self.limits = limits or CompanionLimits()

    def evaluate(
        self,
        telemetry: Dict[str, Any],
        sbc_metrics: Dict[str, Any],
        now: Optional[float] = None,
    ) -> CompanionStatus:
        """Return the current state and every reason preventing full readiness."""
        if self.requested_mode == RequestedCompanionMode.DISABLED:
            return CompanionStatus(CompanionState.DISABLED, ("disabled by configuration",))

        now = time.time() if now is None else now
        reasons = []
        critical_reasons = []
        connected = bool(telemetry.get("connected", False))
        armed = bool(telemetry.get("armed", False))
        last_packet = self._number(telemetry.get("last_packet_timestamp"), 0.0)
        packet_age = now - last_packet if last_packet > 0 else float("inf")

        if not connected:
            reasons.append("Pixhawk link disconnected")
        if packet_age > self.limits.telemetry_timeout_s:
            reasons.append("telemetry is stale")

        battery = self._number(telemetry.get("battery_remaining"), 0.0)
        if 0 < battery <= self.limits.critical_battery_percent:
            critical_reasons.append("battery is critical")
        elif 0 < battery < self.limits.minimum_battery_percent:
            reasons.append("battery is low")

        cpu_temp = self._number(sbc_metrics.get("cpu_temp_c"), 0.0)
        if cpu_temp >= self.limits.critical_cpu_temp_c:
            critical_reasons.append("companion CPU temperature is critical")
        elif cpu_temp >= self.limits.maximum_cpu_temp_c:
            reasons.append("companion CPU temperature is high")

        if self.requested_mode == RequestedCompanionMode.ASSIST:
            if battery <= 0:
                reasons.append("battery telemetry is unavailable")
            if cpu_temp <= 0:
                reasons.append("companion CPU temperature is unavailable")
            fix_type = str(telemetry.get("gps_fix_type", "NO FIX")).upper()
            satellites = int(self._number(telemetry.get("satellites"), 0.0))
            if fix_type not in {"3D FIX", "DGPS", "RTK FLT", "RTK FIX"}:
                reasons.append("3D GPS fix is unavailable")
            if satellites < self.limits.minimum_satellites:
                reasons.append("too few GPS satellites")

        if critical_reasons or (armed and (not connected or packet_age > self.limits.telemetry_timeout_s)):
            all_reasons = tuple(critical_reasons + reasons)
            return CompanionStatus(CompanionState.FAILSAFE, all_reasons)

        if reasons:
            return CompanionStatus(CompanionState.DEGRADED, tuple(reasons))

        if self.requested_mode == RequestedCompanionMode.ASSIST:
            return CompanionStatus(
                CompanionState.ASSIST_READY,
                ("health checks passed; command output remains disabled",),
            )

        return CompanionStatus(CompanionState.MONITOR, ("telemetry monitoring healthy",))

    @staticmethod
    def _number(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
