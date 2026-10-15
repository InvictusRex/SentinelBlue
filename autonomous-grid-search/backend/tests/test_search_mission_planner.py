import pathlib
import sys
import unittest


PI_CODE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PI_CODE))

from search_mission_planner import build_search_mission  # noqa: E402


class SearchMissionPlannerTests(unittest.TestCase):
    def test_one_kilometre_radius_creates_four_cells(self):
        plan = build_search_mission(12.9716, 77.5946, 1, 5)

        self.assertEqual(4, plan.cell_count)
        self.assertEqual((2, 2), (plan.rows, plan.columns))
        self.assertEqual((1, 2, 4, 3), plan.route_cell_ids)
        self.assertGreater(plan.estimate.survey_line_count, 2_000)
        self.assertFalse(plan.estimate.simulation_ready)

    def test_two_kilometre_radius_creates_eight_cells(self):
        plan = build_search_mission(12.9716, 77.5946, 2, 40)

        self.assertEqual(8, plan.cell_count)
        self.assertEqual((2, 4), (plan.rows, plan.columns))
        self.assertEqual((1, 2, 3, 4, 8, 7, 6, 5), plan.route_cell_ids)

    def test_mission_contains_takeoff_search_return_and_land(self):
        plan = build_search_mission(12.9716, 77.5946, 0.01, 30, 12.9700, 77.5900)
        commands = [item["command"] for item in plan.mission_items]

        self.assertEqual(16, commands[0])
        self.assertEqual(22, commands[1])
        self.assertEqual(206, commands[2])
        self.assertEqual(206, commands[-3])
        self.assertEqual(16, commands[-2])
        self.assertEqual(21, commands[-1])
        self.assertTrue(plan.estimate.simulation_ready)

    def test_camera_footprint_controls_line_and_photo_spacing(self):
        plan = build_search_mission(
            12.9716, 77.5946, 0.01, 5,
            camera_horizontal_fov_deg=78,
            camera_vertical_fov_deg=60,
            side_overlap_percent=70,
            forward_overlap_percent=75,
        )

        self.assertAlmostEqual(8.1, plan.camera.footprint_width_m, places=1)
        self.assertAlmostEqual(5.77, plan.camera.footprint_height_m, places=1)
        self.assertAlmostEqual(1.73, plan.camera.line_spacing_m, places=1)
        self.assertAlmostEqual(2.02, plan.camera.trigger_distance_m, places=1)
        self.assertEqual(12, plan.estimate.survey_line_count)

    def test_rejects_unsafe_or_invalid_ranges(self):
        invalid_values = [
            (91, 77, 1, 30),
            (12, 181, 1, 30),
            (12, 77, 0.001, 30),
            (12, 77, 1, 121),
        ]
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    build_search_mission(*values)

    def test_rejects_plan_too_large_to_render_safely(self):
        with self.assertRaisesRegex(ValueError, "planning limit 10000 lines"):
            build_search_mission(12.9716, 77.5946, 10, 5)


if __name__ == "__main__":
    unittest.main()
