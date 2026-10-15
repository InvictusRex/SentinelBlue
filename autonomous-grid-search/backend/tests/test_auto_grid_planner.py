import math
import pathlib
import sys
import unittest
import time
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "telemetry" / "handheld-nrf-monitor" / "pi-code"))
from auto_grid_planner import build_auto_grid_mission, build_radius_survey
from mavlink_manager import MAVLinkManager


def survey_payload(width=50, height=50):
    lat, lon = 12.971598, 77.594562
    dx, dy = width / 111320 / math.cos(math.radians(lat)), height / 111320
    return {"boundary": [{"latitude": lat, "longitude": lon}, {"latitude": lat, "longitude": lon + dx},
                         {"latitude": lat + dy, "longitude": lon + dx}, {"latitude": lat + dy, "longitude": lon}],
            "home": {"latitude": lat, "longitude": lon}, "altitude_m": 5, "cruise_speed_m_s": 2,
            "survey_options": {"angle_mode": "manual", "grid_angle_deg": 90}}


class AutoGridTests(unittest.TestCase):
    def test_radius_request_uses_pi_settings_and_vehicle_home(self):
        center = {"latitude": 12.971598, "longitude": 77.594562}
        request = {"search_area": {"center": center, "radius_m": 25}, "altitude_m": 100,
                   "home": {"latitude": 0, "longitude": 0}, "survey_options": {"spacing_mode": "manual", "line_spacing_m": 100}}
        telemetry = {**center, "connected": True, "armed": False, "gps_fix_type": "3D FIX",
                     "last_position_timestamp": time.time(), "last_gps_timestamp": time.time()}
        plan = build_radius_survey(request, telemetry).to_dict()
        self.assertEqual(5, plan["altitude_m"])
        self.assertEqual(center, plan["home"])
        self.assertEqual("vehicle_position", plan["home_source"])
        self.assertEqual(4, len(plan["boundary"]))
        self.assertEqual(2500, plan["area_m2"])
        self.assertEqual("square", plan["search_area"]["shape"])
        self.assertLess(plan["camera"]["line_spacing_m"], 3)
        self.assertTrue(all(sortie["ready"] for sortie in plan["sorties"]))
        stale = build_radius_survey(request, {**telemetry, "last_gps_timestamp": 0}).to_dict()
        self.assertFalse(stale["sorties"][0]["ready"])
        self.assertIn("fresh 3D GPS", stale["estimate"]["blocking_reasons"][0])
        with self.assertRaisesRegex(ValueError, "current flight"):
            build_radius_survey(request, {**telemetry, "armed": True})

    def test_square_cells_cover_area_once_and_assign_every_line(self):
        from pyproj import CRS, Transformer
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        center = {"latitude": 12.971598, "longitude": 77.594562}
        telemetry = {**center, "connected": True, "gps_fix_type": "3D FIX",
                     "last_position_timestamp": time.time(), "last_gps_timestamp": time.time()}
        for extent, count in [(50, 4), (60, 9)]:
            plan = build_radius_survey({"search_area": {"center": center, "radius_m": extent}}, telemetry).to_dict()
            self.assertEqual(count, plan["cell_count"])
            self.assertLessEqual(plan["cell_side_m"], 50)
            crs = CRS.from_proj4("+proj=aeqd +lat_0=12.971598 +lon_0=77.594562 +datum=WGS84 +units=m")
            projection = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            polygons = [Polygon([projection.transform(p["longitude"], p["latitude"]) for p in cell["boundary"]]) for cell in plan["cells"]]
            self.assertAlmostEqual(sum(p.area for p in polygons), (2 * extent) ** 2, places=2)
            self.assertAlmostEqual(unary_union(polygons).area, (2 * extent) ** 2, places=2)
            assignments = [cell for drone in plan["swarm"]["drones"] for cell in drone["cell_ids"]]
            self.assertEqual(list(range(1, count + 1)), sorted(assignments))
            for cell in plan["cells"]:
                tracks = [line for line in plan["survey_lines"] if line["cell_id"] == cell["cell_id"]]
                self.assertEqual(cell["line_count"], len(tracks))
                self.assertGreater(len(tracks), 1)
                self.assertGreaterEqual(cell["coverage_percent"], 99.5)
                for track in tracks:
                    self.assertEqual(cell["drone_id"], track["drone_id"])
                    self.assertIn(track["sortie_id"], cell["sortie_ids"])
            for drone in plan["swarm"]["drones"]:
                segments = drone["segments"]
                self.assertEqual("takeoff", segments[0]["phase"])
                self.assertEqual("landing", segments[-1]["phase"])
                self.assertEqual(plan["home"], segments[-1]["end"])
                for a, b in zip(segments, segments[1:]):
                    self.assertEqual(a["end_s"], b["start_s"])
                    self.assertEqual(a["end"], b["start"])
            self.assertTrue(plan["swarm"]["simulation_only"])

    def test_square_cell_limit_rejects_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "100 square cells"):
            build_radius_survey({"search_area": {"center": {"latitude": 12, "longitude": 77}, "radius_m": 1000}}, {})

    def plan(self, payload):
        return build_auto_grid_mission(payload, {}).to_dict()

    def test_camera_axes_coverage_and_export_sequence(self):
        plan = self.plan(survey_payload())
        camera = plan["camera"]
        self.assertAlmostEqual(camera["cross_track_m"], 10 * math.tan(math.radians(39)), places=3)
        self.assertAlmostEqual(camera["trigger_distance_m"], 10 * math.tan(math.radians(30)) * .25, places=3)
        self.assertLessEqual(camera["line_spacing_m"], camera["cross_track_m"] * .3 + .001)
        self.assertGreaterEqual(plan["coverage_percent"], 99.5)
        for sortie in plan["sorties"]:
            self.assertTrue(sortie["ready"], sortie["blocking_reasons"])
            self.assertLessEqual(sortie["outside_boundary_m"], .1)
            items = sortie["mission_items"]
            self.assertEqual([16, 22, 178], [item["command"] for item in items[:3]])
            self.assertEqual(2, items[2]["param2"])
            self.assertEqual([16, 206, 16, 206], [item["command"] for item in items[3:7]])
            self.assertGreater(items[4]["param1"], 0)
            self.assertEqual(0, items[6]["param1"])
            self.assertEqual(21, items[-1]["command"])
            self.assertEqual(list(range(len(items))), [item["seq"] for item in items])
            self.assertEqual(items[0]["x"], items[-1]["x"])

    def test_sensor_orientation_and_gsd(self):
        payload = survey_payload()
        payload["survey_options"].update(camera_model="sensor", sensor_width_mm=6, sensor_height_mm=4,
                                         focal_length_mm=5, camera_orientation="portrait", minimum_trigger_interval_s=.1)
        camera = self.plan(payload)["camera"]
        self.assertEqual(4, camera["cross_track_m"])
        self.assertEqual(6, camera["along_track_m"])
        self.assertEqual(.15, camera["gsd_width_cm"])

    def test_invalid_geometry_and_numeric_values(self):
        payload = survey_payload()
        payload["boundary"][1], payload["boundary"][2] = payload["boundary"][2], payload["boundary"][1]
        with self.assertRaisesRegex(ValueError, "Invalid boundary"):
            self.plan(payload)
        for value in [float("nan"), float("inf"), True, -10]:
            payload = survey_payload(); payload["altitude_m"] = value
            with self.assertRaises(ValueError):
                self.plan(payload)

    def test_camera_rate_manual_gaps_and_missing_home_block(self):
        payload = survey_payload(); payload["cruise_speed_m_s"] = 15
        plan = self.plan(payload)
        self.assertIn("Camera cannot keep up", " ".join(plan["estimate"]["blocking_reasons"]))
        self.assertEqual([], plan["sorties"][0]["mission_items"])
        payload = survey_payload(); payload["survey_options"].update(spacing_mode="manual", line_spacing_m=20)
        plan = self.plan(payload)
        self.assertLess(plan["coverage_percent"], 99.5)
        self.assertIn("Manual line spacing", " ".join(plan["estimate"]["blocking_reasons"]))
        payload = survey_payload(); del payload["home"]
        self.assertFalse(self.plan(payload)["sorties"][0]["ready"])

    def test_split_keeps_every_line_and_limits_each_sortie(self):
        plan = self.plan(survey_payload(150, 150))
        self.assertGreater(len(plan["sorties"]), 1)
        self.assertEqual(len(plan["survey_lines"]), sum(sortie["line_count"] for sortie in plan["sorties"]))
        for sortie in plan["sorties"]:
            self.assertLessEqual(sortie["mission_item_count"], 500)
            self.assertLessEqual(sortie["estimated_minutes"], plan["sortie_budget_minutes"] + .05)
            self.assertTrue(sortie["ready"])

    def test_rotation_reverse_crosshatch_and_concave_boundary(self):
        payload = survey_payload(60, 30); payload["survey_options"]["grid_angle_deg"] = 35
        first = self.plan(payload)
        payload["survey_options"]["entry"] = "reverse"
        reversed_plan = self.plan(payload)
        self.assertEqual(first["survey_lines"][0]["start_latitude"], reversed_plan["survey_lines"][-1]["end_latitude"])
        payload["survey_options"]["pattern"] = "crosshatch"
        self.assertGreater(self.plan(payload)["estimate"]["survey_line_count"], first["estimate"]["survey_line_count"])
        payload = survey_payload(); a, b, c, d = payload["boundary"]
        middle = {"latitude": (a["latitude"] + c["latitude"]) / 2, "longitude": (a["longitude"] + c["longitude"]) / 2}
        payload["boundary"] = [a, b, middle, c, d]
        plan = self.plan(payload)
        self.assertGreater(plan["area_m2"], 1000)
        self.assertTrue(plan["survey_lines"])
        payload["survey_options"]["angle_mode"] = "auto"
        self.assertGreaterEqual(self.plan(payload)["grid_angle_deg"], 0)

    def test_indivisible_line_exceeding_budget_is_blocked(self):
        payload = survey_payload(500, 10)
        payload["survey_options"].update(endurance_minutes=3, reserve_percent=50)
        plan = self.plan(payload)
        self.assertFalse(plan["sorties"][0]["ready"])
        self.assertEqual([], plan["sorties"][0]["mission_items"])

    def test_simulator_completes_at_home_and_stays_landed(self):
        payload = survey_payload(5, 5)
        plan = self.plan(payload)
        manager = MAVLinkManager(simulate=True)
        manager.running = True
        ticks, snapshots = 0, []

        def tick(_):
            nonlocal ticks
            ticks += 1
            if ticks == 1:
                manager.upload_mission_items(plan["sorties"][0]["mission_items"])
                manager.set_arm(True)
                manager.set_flight_mode("AUTO")
            snapshots.append(manager.get_telemetry_snapshot())
            if ticks >= 600:
                manager.running = False

        with patch("mavlink_manager.time.sleep", side_effect=tick):
            manager._run_simulation()
        final = manager.get_telemetry_snapshot()
        self.assertEqual("MISSION COMPLETED", final["mission_state"])
        self.assertFalse(final["armed"])
        self.assertEqual(0, final["altitude_relative"])
        self.assertEqual(0, final["ground_speed"])
        self.assertEqual(100, final["mission_progress_percent"])
        self.assertAlmostEqual(payload["home"]["latitude"], final["latitude"], places=6)
        self.assertTrue(any(1.9 <= item["ground_speed"] <= 2.1 for item in snapshots))
        self.assertEqual(0, snapshots[-50]["altitude_relative"])


if __name__ == "__main__":
    unittest.main()
