import pathlib
import sys
import unittest


PI_CODE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PI_CODE))

from companion_mode import (  # noqa: E402
    CompanionModeController,
    CompanionState,
    RequestedCompanionMode,
)


class CompanionModeControllerTests(unittest.TestCase):
    def setUp(self):
        self.now = 1_000.0
        self.telemetry = {
            "connected": True,
            "armed": False,
            "last_packet_timestamp": self.now - 0.5,
            "battery_remaining": 80,
            "gps_fix_type": "3D FIX",
            "satellites": 12,
        }
        self.sbc = {"cpu_temp_c": 55.0}

    def test_monitor_mode_reports_healthy(self):
        controller = CompanionModeController(RequestedCompanionMode.MONITOR)

        status = controller.evaluate(self.telemetry, self.sbc, now=self.now)

        self.assertEqual(CompanionState.MONITOR, status.state)
        self.assertFalse(status.commands_allowed)

    def test_assist_mode_only_reports_readiness(self):
        controller = CompanionModeController(RequestedCompanionMode.ASSIST)

        status = controller.evaluate(self.telemetry, self.sbc, now=self.now)

        self.assertEqual(CompanionState.ASSIST_READY, status.state)
        self.assertFalse(status.commands_allowed)

    def test_assist_requires_usable_gps(self):
        self.telemetry.update({"gps_fix_type": "2D FIX", "satellites": 4})
        controller = CompanionModeController(RequestedCompanionMode.ASSIST)

        status = controller.evaluate(self.telemetry, self.sbc, now=self.now)

        self.assertEqual(CompanionState.DEGRADED, status.state)
        self.assertIn("3D GPS fix is unavailable", status.reasons)
        self.assertIn("too few GPS satellites", status.reasons)

    def test_assist_fails_closed_when_health_values_are_unavailable(self):
        self.telemetry["battery_remaining"] = 0
        self.sbc["cpu_temp_c"] = 0
        controller = CompanionModeController(RequestedCompanionMode.ASSIST)

        status = controller.evaluate(self.telemetry, self.sbc, now=self.now)

        self.assertEqual(CompanionState.DEGRADED, status.state)
        self.assertIn("battery telemetry is unavailable", status.reasons)
        self.assertIn("companion CPU temperature is unavailable", status.reasons)

    def test_stale_telemetry_while_armed_reports_failsafe(self):
        self.telemetry.update({"armed": True, "last_packet_timestamp": self.now - 10.0})
        controller = CompanionModeController(RequestedCompanionMode.MONITOR)

        status = controller.evaluate(self.telemetry, self.sbc, now=self.now)

        self.assertEqual(CompanionState.FAILSAFE, status.state)
        self.assertFalse(status.commands_allowed)

    def test_critical_battery_reports_failsafe(self):
        self.telemetry["battery_remaining"] = 9
        controller = CompanionModeController(RequestedCompanionMode.MONITOR)

        status = controller.evaluate(self.telemetry, self.sbc, now=self.now)

        self.assertEqual(CompanionState.FAILSAFE, status.state)
        self.assertIn("battery is critical", status.reasons)

    def test_disabled_mode_ignores_health(self):
        controller = CompanionModeController(RequestedCompanionMode.DISABLED)

        status = controller.evaluate({}, {}, now=self.now)

        self.assertEqual(CompanionState.DISABLED, status.state)
        self.assertFalse(status.commands_allowed)


if __name__ == "__main__":
    unittest.main()
