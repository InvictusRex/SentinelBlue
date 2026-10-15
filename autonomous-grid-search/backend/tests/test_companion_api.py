import json
import pathlib
import sys
import unittest
import time
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PI_CODE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PI_CODE))

from companion_api import CompanionAPIServer  # noqa: E402
from test_auto_grid_planner import survey_payload


class FakeMAVLinkManager:
    def __init__(self, simulate: bool):
        self.simulate = simulate
        self.calls = []
        self.armed = False
        self.uploaded = []

    def get_telemetry_snapshot(self):
        return {"connected": True, "latitude": 12.9700, "longitude": 77.5900, "armed": self.armed,
                "gps_fix_type": "3D FIX", "last_position_timestamp": time.time(), "last_gps_timestamp": time.time()}

    def clear_all_missions(self):
        self.calls.append("clear")
        return True

    def upload_mission_items(self, items):
        self.calls.append(("upload", len(items)))
        self.uploaded = items
        return True

    def set_arm(self, arm):
        self.calls.append(("arm", arm))
        self.armed = arm
        return True

    def set_flight_mode(self, mode):
        self.calls.append(("mode", mode))
        return True


class CompanionAPITests(unittest.TestCase):
    def setUp(self):
        self.payload = json.dumps({
            "center": {"latitude": 12.9716, "longitude": 77.5946},
            "radius_km": 0.01,
            "altitude_m": 30,
        }).encode("utf-8")

    def _start(self, simulate):
        manager = FakeMAVLinkManager(simulate)
        server = CompanionAPIServer(manager, host="127.0.0.1", port=0)
        server.start()
        self.addCleanup(server.stop)
        return manager, server.server.server_address[1]

    def _post(self, port, path):
        request = Request(
            f"http://127.0.0.1:{port}{path}",
            data=self.payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urlopen(request, timeout=2)

    def test_plan_endpoint_returns_generated_grid(self):
        _, port = self._start(simulate=False)

        with self._post(port, "/api/mission/plan") as response:
            body = json.load(response)

        self.assertEqual("planned", body["status"])
        self.assertEqual(1, body["plan"]["cell_count"])
        self.assertGreater(body["plan"]["estimate"]["survey_line_count"], 0)

    def test_radius_only_api_request_loads_pi_profile(self):
        _, port = self._start(simulate=True)
        self.payload = json.dumps({"search_area": {"center": {"latitude": 12.9700, "longitude": 77.5900}, "radius_m": 25}}).encode()
        with self._post(port, "/api/mission/plan") as response:
            plan = json.load(response)["plan"]
        self.assertEqual(25, plan["search_area"]["radius_m"])
        self.assertEqual("downward_water_camera", plan["profile_name"])
        self.assertTrue(plan["plan_id"])
        self.assertEqual("disabled", plan["terrain"]["status"])

    def test_live_execution_is_locked(self):
        manager, port = self._start(simulate=False)

        with self.assertRaises(HTTPError) as raised:
            self._post(port, "/api/mission/simulate")

        self.assertEqual(403, raised.exception.code)
        self.assertEqual([], manager.calls)

    def test_simulation_execution_loads_and_starts_mission(self):
        manager, port = self._start(simulate=True)

        with self._post(port, "/api/mission/simulate") as response:
            body = json.load(response)

        self.assertEqual(202, response.status)
        self.assertEqual("simulation_started", body["status"])
        self.assertEqual("clear", manager.calls[0])
        self.assertEqual("upload", manager.calls[1][0])
        self.assertGreater(manager.calls[1][1], 8)
        self.assertEqual([("arm", True), ("mode", "AUTO")], manager.calls[2:])

    def test_reviewed_sortie_runs_exact_cached_items_and_rejects_replacement(self):
        manager, port = self._start(simulate=True)
        self.payload = json.dumps(survey_payload()).encode()
        with self._post(port, "/api/mission/plan") as response:
            plan = json.load(response)["plan"]
        self.payload = json.dumps({"plan_id": plan["plan_id"], "sortie_id": 1, "review_confirmed": True,
                                   "altitude_m": 100}).encode()
        with self._post(port, "/api/mission/simulate") as response:
            self.assertEqual(202, response.status)
        self.assertEqual(plan["sorties"][0]["mission_items"], manager.uploaded)
        with self.assertRaises(HTTPError) as raised:
            self._post(port, "/api/mission/simulate")
        self.assertEqual(409, raised.exception.code)

    def test_cache_expiry_and_review_confirmation(self):
        manager, port = self._start(simulate=True)
        self.payload = json.dumps(survey_payload()).encode()
        with self._post(port, "/api/mission/plan") as response:
            plan = json.load(response)["plan"]
        self.payload = json.dumps({"plan_id": plan["plan_id"], "sortie_id": 1}).encode()
        with self.assertRaises(HTTPError) as raised:
            self._post(port, "/api/mission/simulate")
        self.assertEqual(400, raised.exception.code)
        with patch("companion_api.time.monotonic", return_value=float("inf")):
            with self.assertRaises(HTTPError) as raised:
                self._post(port, "/api/mission/simulate")
        self.assertEqual(409, raised.exception.code)
        self.assertEqual([], manager.calls)


if __name__ == "__main__":
    unittest.main()
