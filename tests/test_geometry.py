from __future__ import annotations

import math
import unittest

from physcensis.geometry import (
    convex_overlap_area,
    point_in_convex_polygon,
    polygon_area,
    rectangle_polygon,
)


class GeometryTest(unittest.TestCase):
    def test_rotated_rectangle_preserves_area(self) -> None:
        polygon = rectangle_polygon((0.0, 0.0), (0.4, 0.2), math.pi / 3.0)
        self.assertAlmostEqual(polygon_area(polygon), 0.08, places=8)

    def test_convex_overlap_is_exact_for_axis_aligned_rectangles(self) -> None:
        first = rectangle_polygon((0.0, 0.0), (0.4, 0.2), 0.0)
        second = rectangle_polygon((0.1, 0.0), (0.4, 0.2), 0.0)
        self.assertAlmostEqual(convex_overlap_area(first, second), 0.06, places=8)

    def test_point_in_rotated_polygon(self) -> None:
        polygon = rectangle_polygon((0.0, 0.0), (0.4, 0.2), math.pi / 4.0)
        self.assertTrue(point_in_convex_polygon((0.0, 0.0), polygon))
        self.assertFalse(point_in_convex_polygon((1.0, 1.0), polygon))


if __name__ == "__main__":
    unittest.main()
