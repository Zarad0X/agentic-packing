"""Dependency-light convex geometry used by the spatial and physical solvers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from physcensis.types import SceneObject, Vec2

EPSILON = 1.0e-9


def rectangle_polygon(center: Vec2, size: Vec2, yaw_rad: float) -> tuple[Vec2, ...]:
    """Return counter-clockwise corners for a yaw-rotated rectangle."""
    half_x, half_y = size[0] / 2.0, size[1] / 2.0
    local = ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y))
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    return tuple(
        (
            center[0] + cosine * x - sine * y,
            center[1] + sine * x + cosine * y,
        )
        for x, y in local
    )


def object_polygon(obj: SceneObject) -> tuple[Vec2, ...]:
    physical_size = obj.asset.physical_size_m
    return rectangle_polygon(
        (obj.position_m[0], obj.position_m[1]),
        (physical_size[0], physical_size[1]),
        obj.yaw_rad,
    )


def signed_polygon_area(vertices: Sequence[Vec2]) -> float:
    if len(vertices) < 3:
        return 0.0
    return 0.5 * sum(
        vertices[index][0] * vertices[(index + 1) % len(vertices)][1]
        - vertices[(index + 1) % len(vertices)][0] * vertices[index][1]
        for index in range(len(vertices))
    )


def polygon_area(vertices: Sequence[Vec2]) -> float:
    return abs(signed_polygon_area(vertices))


def ensure_counter_clockwise(vertices: Sequence[Vec2]) -> tuple[Vec2, ...]:
    result = tuple(vertices)
    if signed_polygon_area(result) < 0:
        return tuple(reversed(result))
    return result


def _inside(point: Vec2, edge_start: Vec2, edge_end: Vec2) -> bool:
    cross = (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    )
    return cross >= -EPSILON


def _line_intersection(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2) -> Vec2:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < EPSILON:
        return p2
    first = x1 * y2 - y1 * x2
    second = x3 * y4 - y3 * x4
    return (
        (first * (x3 - x4) - (x1 - x2) * second) / denominator,
        (first * (y3 - y4) - (y1 - y2) * second) / denominator,
    )


def convex_intersection(subject: Sequence[Vec2], clip: Sequence[Vec2]) -> tuple[Vec2, ...]:
    """Clip one convex polygon by another via Sutherland-Hodgman."""
    output = list(ensure_counter_clockwise(subject))
    clip_vertices = ensure_counter_clockwise(clip)
    for index, edge_end in enumerate(clip_vertices):
        edge_start = clip_vertices[index - 1]
        input_vertices = output
        output = []
        if not input_vertices:
            break
        previous = input_vertices[-1]
        for current in input_vertices:
            current_inside = _inside(current, edge_start, edge_end)
            previous_inside = _inside(previous, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, edge_start, edge_end))
            previous = current
    return tuple(output)


def convex_overlap_area(first: Sequence[Vec2], second: Sequence[Vec2]) -> float:
    return polygon_area(convex_intersection(first, second))


def point_in_convex_polygon(point: Vec2, polygon: Sequence[Vec2]) -> bool:
    vertices = ensure_counter_clockwise(polygon)
    return all(_inside(point, vertices[index - 1], vertices[index]) for index in range(len(vertices)))


def polygon_bounds(vertices: Sequence[Vec2]) -> tuple[float, float, float, float]:
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    return min(xs), max(xs), min(ys), max(ys)


def outside_distance_squared(
    polygon: Sequence[Vec2], bounds: tuple[float, float, float, float]
) -> float:
    """Squared axis distance accumulated for vertices outside a rectangle."""
    min_x, max_x, min_y, max_y = bounds
    penalty = 0.0
    for x, y in polygon:
        if x < min_x:
            penalty += (min_x - x) ** 2
        elif x > max_x:
            penalty += (x - max_x) ** 2
        if y < min_y:
            penalty += (min_y - y) ** 2
        elif y > max_y:
            penalty += (y - max_y) ** 2
    return penalty


def projected_support_ratio(subject: SceneObject, supporter: SceneObject) -> float:
    bottom = object_polygon(subject)
    support = object_polygon(supporter)
    denominator = polygon_area(bottom)
    if denominator <= EPSILON:
        return 0.0
    return convex_overlap_area(bottom, support) / denominator


def objects_overlap_3d(first: SceneObject, second: SceneObject, tolerance: float = 1.0e-6) -> bool:
    vertical_overlap = min(first.top_z, second.top_z) - max(first.bottom_z, second.bottom_z)
    if vertical_overlap <= tolerance:
        return False
    return convex_overlap_area(object_polygon(first), object_polygon(second)) > tolerance


def total_pairwise_overlap(objects: Iterable[SceneObject]) -> float:
    values = list(objects)
    return sum(
        convex_overlap_area(object_polygon(values[i]), object_polygon(values[j]))
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )
