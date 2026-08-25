"""Occupancy-grid candidate generation for physical predicates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from physcensis.geometry import (
    convex_overlap_area,
    object_polygon,
    objects_overlap_3d,
    point_in_convex_polygon,
    polygon_area,
)
from physcensis.types import SceneObject, SceneState, Vec3


@dataclass(frozen=True)
class PlacementCandidate:
    position_m: Vec3
    support_id: str
    support_ratio: float
    center_margin_m: float
    score: float = 0.0


@dataclass(frozen=True)
class ContainerPlacementCandidate:
    """One collision-free pose inside an open container."""

    position_m: Vec3
    yaw_rad: float
    local_xy_m: tuple[float, float]
    support_ratio: float
    support_ids: tuple[str, ...]
    layer_index: int


def cuboid_occupancy(size_m: Vec3, yaw_rad: float, resolution_m: float) -> np.ndarray:
    """Voxelize an oriented cuboid around its local origin."""
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    extent_x = abs(cosine) * size_m[0] + abs(sine) * size_m[1]
    extent_y = abs(sine) * size_m[0] + abs(cosine) * size_m[1]
    nx = max(1, int(np.ceil(extent_x / resolution_m)))
    ny = max(1, int(np.ceil(extent_y / resolution_m)))
    nz = max(1, int(np.ceil(size_m[2] / resolution_m)))
    xs = (np.arange(nx) + 0.5 - nx / 2.0) * resolution_m
    ys = (np.arange(ny) + 0.5 - ny / 2.0) * resolution_m
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    local_x = cosine * xx + sine * yy
    local_y = -sine * xx + cosine * yy
    footprint = (np.abs(local_x) <= size_m[0] / 2.0) & (np.abs(local_y) <= size_m[1] / 2.0)
    return np.repeat(footprint[:, :, None], nz, axis=2)


def surface_candidates(
    scene: SceneState,
    subject: SceneObject,
    supporter_id: str,
    resolution_m: float,
) -> list[PlacementCandidate]:
    """Enumerate supported, non-penetrating placements on one top surface."""
    if supporter_id == "root":
        min_x, max_x, min_y, max_y = scene.root_bounds_xy
        top_z = scene.root_top_z
        support_polygon = ((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y))
    else:
        supporter = scene.get(supporter_id)
        bounds = object_polygon(supporter)
        xs = [value[0] for value in bounds]
        ys = [value[1] for value in bounds]
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        top_z = supporter.top_z
        support_polygon = bounds

    candidates: list[PlacementCandidate] = []
    x_values = np.arange(min_x, max_x + resolution_m * 0.5, resolution_m)
    y_values = np.arange(min_y, max_y + resolution_m * 0.5, resolution_m)
    for x in x_values:
        for y in y_values:
            candidate = subject.moved(
                position_m=(
                    float(x),
                    float(y),
                    top_z + subject.asset.physical_size_m[2] / 2.0,
                )
            )
            com = (
                candidate.position_m[0] + candidate.asset.com_shift_m[0],
                candidate.position_m[1] + candidate.asset.com_shift_m[1],
            )
            if not point_in_convex_polygon(com, support_polygon):
                continue
            collision = any(
                existing.object_id not in {subject.object_id, supporter_id}
                and objects_overlap_3d(candidate, existing)
                for existing in scene.objects.values()
            )
            if collision:
                continue
            subject_polygon = object_polygon(candidate)
            try:
                from physcensis.geometry import convex_overlap_area, polygon_area

                ratio = convex_overlap_area(subject_polygon, support_polygon) / max(
                    polygon_area(subject_polygon), 1.0e-9
                )
            except ZeroDivisionError:
                ratio = 0.0
            center_margin = min(x - min_x, max_x - x, y - min_y, max_y - y)
            candidates.append(
                PlacementCandidate(candidate.position_m, supporter_id, ratio, float(center_margin))
            )
    return candidates


def candidate_supports(scene: SceneState, minimum_probability: float = 0.5) -> Iterable[str]:
    yield "root"
    for object_id, obj in scene.objects.items():
        if obj.asset.supporting_probability >= minimum_probability:
            yield object_id


def container_candidates(
    scene: SceneState,
    subject: SceneObject,
    container: SceneObject,
    resolution_m: float,
    *,
    allow_protrusion_m: float = 0.0,
    yaw_offsets_rad: tuple[float, ...] = (0.0, np.pi / 2.0),
    floor_only: bool = False,
) -> list[ContainerPlacementCandidate]:
    """Enumerate supported poses on the floor and on existing packed objects.

    The search is expressed in the container's local frame. Candidate heights
    come from the inner floor and every existing top surface, enabling dense
    multi-layer packing while retaining deterministic collision checks.
    """
    inner = container.asset.container_inner_size_m
    if inner is None:
        raise ValueError(f"{container.object_id} is not a container")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")

    wall = max(0.005, (container.asset.size_m[2] - inner[2]) / 2.0)
    floor_z = container.bottom_z + wall
    ceiling_z = floor_z + inner[2] + max(0.0, allow_protrusion_m)
    packed = [
        existing
        for existing in scene.objects.values()
        if existing.object_id not in {subject.object_id, container.object_id}
    ]
    support_levels = [floor_z]
    support_levels.extend(
        existing.top_z
        for existing in packed
        if existing.support_id == container.object_id and existing.top_z <= ceiling_z + 1.0e-6
    )
    levels = sorted({round(value, 6) for value in support_levels})
    if floor_only:
        levels = levels[:1]

    container_cosine = float(np.cos(container.yaw_rad))
    container_sine = float(np.sin(container.yaw_rad))
    candidates: list[ContainerPlacementCandidate] = []
    for yaw_offset in yaw_offsets_rad:
        yaw = container.yaw_rad + float(yaw_offset)
        relative_yaw = yaw - container.yaw_rad
        cosine, sine = abs(float(np.cos(relative_yaw))), abs(float(np.sin(relative_yaw)))
        physical_size = subject.asset.physical_size_m
        extent_x = cosine * physical_size[0] + sine * physical_size[1]
        extent_y = sine * physical_size[0] + cosine * physical_size[1]
        max_dx = (inner[0] - extent_x) / 2.0
        max_dy = (inner[1] - extent_y) / 2.0
        if max_dx < -1.0e-9 or max_dy < -1.0e-9:
            continue
        x_offsets = np.arange(-max_dx, max_dx + 1.0e-9, resolution_m)
        y_offsets = np.arange(-max_dy, max_dy + 1.0e-9, resolution_m)
        for layer_index, support_z in enumerate(levels):
            center_z = support_z + physical_size[2] / 2.0
            if center_z + physical_size[2] / 2.0 > ceiling_z + 1.0e-6:
                continue
            for local_y in y_offsets:
                for local_x in x_offsets:
                    world_x = (
                        container.position_m[0]
                        + container_cosine * float(local_x)
                        - container_sine * float(local_y)
                    )
                    world_y = (
                        container.position_m[1]
                        + container_sine * float(local_x)
                        + container_cosine * float(local_y)
                    )
                    candidate = subject.moved(
                        position_m=(world_x, world_y, center_z),
                        yaw_rad=yaw,
                    )
                    if any(objects_overlap_3d(candidate, existing) for existing in packed):
                        continue

                    if abs(support_z - floor_z) <= 1.0e-6:
                        support_ratio = 1.0
                        support_ids = (container.object_id,)
                    else:
                        footprint_area = max(polygon_area(object_polygon(candidate)), 1.0e-9)
                        supporters = [
                            existing
                            for existing in packed
                            if abs(existing.top_z - support_z) <= max(0.004, resolution_m * 0.55)
                            and convex_overlap_area(
                                object_polygon(candidate), object_polygon(existing)
                            )
                            > 1.0e-8
                        ]
                        support_area = sum(
                            convex_overlap_area(
                                object_polygon(candidate), object_polygon(existing)
                            )
                            for existing in supporters
                        )
                        support_ratio = min(1.0, support_area / footprint_area)
                        support_ids = tuple(existing.object_id for existing in supporters)
                        if support_ratio < 0.55:
                            continue
                    candidates.append(
                        ContainerPlacementCandidate(
                            candidate.position_m,
                            yaw,
                            (float(local_x), float(local_y)),
                            support_ratio,
                            support_ids,
                            layer_index,
                        )
                    )
    return candidates
