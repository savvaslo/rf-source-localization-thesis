import unittest

from simulator_qt import (
    MainWindow,
    destination_point,
    distance_meters,
    reset_target_route_progress,
)


class TargetRouteTests(unittest.TestCase):
    def setUp(self):
        self.start = (35.1856, 33.3823)
        self.east = destination_point(*self.start, 90.0, 100.0)
        self.north = destination_point(*self.east, 0.0, 100.0)
        self.target = {
            "name": "Target-1",
            "active": True,
            "moving": True,
            "speed_mps": 10.0,
            "heading_degrees": 0.0,
            "loop_route": False,
            "route": [list(self.start), list(self.east), list(self.north)],
            "latitude": self.north[0],
            "longitude": self.north[1],
        }

    def advance(self, distance: float):
        MainWindow._advance_target_on_route(None, self.target, distance)

    def test_route_reset_discards_stale_progress_and_returns_to_start(self):
        self.target["_route_index"] = 99
        self.target["_route_finished"] = True

        reset_target_route_progress(self.target, move_to_start=True)

        self.assertEqual(self.target["_route_index"], 1)
        self.assertFalse(self.target["_route_finished"])
        self.assertLess(
            distance_meters(
                self.target["latitude"],
                self.target["longitude"],
                *self.start,
            ),
            0.01,
        )

    def test_open_route_visits_waypoints_in_order_and_stops_at_the_end(self):
        reset_target_route_progress(self.target, move_to_start=True)

        self.advance(150.0)

        self.assertEqual(self.target["_route_index"], 2)
        self.assertAlmostEqual(
            distance_meters(
                *self.east,
                self.target["latitude"],
                self.target["longitude"],
            ),
            50.0,
            delta=0.2,
        )
        self.advance(50.0)
        self.advance(1.0)
        self.assertTrue(self.target["_route_finished"])
        self.assertLess(
            distance_meters(
                self.target["latitude"],
                self.target["longitude"],
                *self.north,
            ),
            0.01,
        )

    def test_loop_route_uses_the_closing_leg_without_finishing(self):
        self.target["loop_route"] = True
        reset_target_route_progress(self.target, move_to_start=True)

        self.advance(250.0)

        self.assertFalse(self.target["_route_finished"])
        self.assertEqual(self.target["_route_index"], 0)
        self.assertAlmostEqual(
            distance_meters(
                *self.north,
                self.target["latitude"],
                self.target["longitude"],
            ),
            50.0,
            delta=0.2,
        )


if __name__ == "__main__":
    unittest.main()
