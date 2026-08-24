"""Paper-aligned planar predicate solver with convex overlap penalties."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from physcensis.config import ReproductionConfig
from physcensis.geometry import (
    convex_overlap_area,
    object_polygon,
    outside_distance_squared,
    polygon_bounds,
)
from physcensis.types import (
    AssetRecord,
    GroupState,
    Issue,
    PlacementProgram,
    Predicate,
    PredicateKind,
    SceneObject,
    SceneState,
    SolveReport,
)


@dataclass
class _SolvedAxes:
    x: bool = False
    y: bool = False
    yaw: bool = False
    base: bool = False
    physical: bool = False


class SpatialSolver:
    """Solve base-surface placement using paper-style coordinate refinement."""

    def __init__(self, config: ReproductionConfig):
        self.config = config
        self._rng = random.Random(config.random_seed)

    def solve(
        self,
        program: PlacementProgram,
        scene: SceneState,
        assets: Mapping[str, AssetRecord],
    ) -> SolveReport:
        working = scene.clone()
        issues: list[Issue] = []
        axes = {object_id: _SolvedAxes() for object_id in program.descriptions}
        for object_id in program.descriptions:
            asset = assets.get(object_id)
            if asset is None:
                issues.append(Issue("asset_missing", f"No asset resolved for {object_id}", object_id))
                continue
            if object_id not in working.objects:
                working.add_object(
                    SceneObject(
                        object_id=object_id,
                        asset=asset,
                        position_m=(0.0, 0.0, working.root_top_z + asset.size_m[2] / 2.0),
                    )
                )
        if issues:
            return SolveReport(False, working, issues)

        self._materialize_groups(program, working, axes, issues)
        spatial_predicates = [
            predicate
            for predicate in program.predicates
            if not predicate.kind.is_physical
            and predicate.kind not in {PredicateKind.GROUP, PredicateKind.COPY_GROUP}
        ]
        # Repeated propagation resolves relations whose geometry changes after yaw updates.
        for _ in range(4):
            for predicate in spatial_predicates:
                self._apply_predicate(predicate, working, axes, issues)

        physical_subjects = {
            predicate.subject
            for predicate in program.predicates
            if predicate.kind.is_physical and isinstance(predicate.subject, str)
        }
        for object_id, solved in axes.items():
            if object_id in physical_subjects:
                solved.physical = True
                continue
            missing = [name for name in ("x", "y", "yaw", "base") if not getattr(solved, name)]
            if missing:
                issues.append(
                    Issue(
                        "spatially_unsolved",
                        f"{object_id} remains unsolved for: {', '.join(missing)}",
                        object_id=object_id,
                    )
                )
        if issues:
            return SolveReport(False, working, issues)

        base_ids = [object_id for object_id, solved in axes.items() if solved.base and not solved.physical]
        initial_penalty = self._penalty(working, base_ids, spatial_predicates)
        if initial_penalty > self.config.spatial.acceptance_threshold:
            self._coordinate_refine(working, base_ids, spatial_predicates)
        final_penalty = self._penalty(working, base_ids, spatial_predicates)
        success = final_penalty <= self.config.spatial.acceptance_threshold
        if not success:
            issues.append(
                Issue(
                    "spatial_penalty_above_threshold",
                    f"Spatial penalty {final_penalty:.6g} exceeds threshold "
                    f"{self.config.spatial.acceptance_threshold:.6g}",
                    details={"penalty": final_penalty},
                )
            )
        return SolveReport(
            success,
            working,
            issues,
            metrics={"initial_spatial_penalty": initial_penalty, "spatial_penalty": final_penalty},
            solved_object_ids=base_ids if success else [],
        )

    def _materialize_groups(
        self,
        program: PlacementProgram,
        scene: SceneState,
        axes: dict[str, _SolvedAxes],
        issues: list[Issue],
    ) -> None:
        for predicate in program.predicates:
            if predicate.kind is PredicateKind.GROUP:
                if not isinstance(predicate.subject, str) or not isinstance(predicate.reference, tuple):
                    continue
                anchor = str(predicate.params["anchor"])
                scene.groups[predicate.subject] = GroupState(predicate.subject, predicate.reference, anchor)
            elif predicate.kind is PredicateKind.COPY_GROUP:
                if not isinstance(predicate.subject, str) or not isinstance(predicate.reference, str):
                    continue
                source = scene.groups.get(predicate.reference)
                if source is None:
                    issues.append(Issue("copy_group_missing", f"Unknown group {predicate.reference}"))
                    continue
                new_ids: list[str] = []
                new_anchor = ""
                for source_id in source.object_ids:
                    source_obj = scene.get(source_id)
                    new_id = f"{source_id}-{predicate.subject}"
                    scene.add_object(
                        SceneObject(
                            object_id=new_id,
                            asset=source_obj.asset,
                            position_m=source_obj.position_m,
                            yaw_rad=source_obj.yaw_rad,
                            support_id=source_obj.support_id,
                        )
                    )
                    axes[new_id] = _SolvedAxes(x=True, y=True, yaw=True, base=True)
                    new_ids.append(new_id)
                    if source_id == source.anchor_id:
                        new_anchor = new_id
                scene.groups[predicate.subject] = GroupState(
                    predicate.subject, tuple(new_ids), new_anchor
                )

    def _apply_predicate(
        self,
        predicate: Predicate,
        scene: SceneState,
        axes: dict[str, _SolvedAxes],
        issues: list[Issue],
    ) -> None:
        if not isinstance(predicate.subject, str) or not isinstance(predicate.reference, str):
            return
        subject_id = predicate.subject
        kind = predicate.kind
        if kind is PredicateKind.PLACE_ON_BASE:
            if subject_id in scene.groups:
                issues.append(Issue("group_on_base_unsupported", "Groups are positioned through relations"))
                return
            obj = scene.get(subject_id)
            x = float(predicate.params.get("x", obj.position_m[0]))
            y = float(predicate.params.get("y", obj.position_m[1]))
            obj.position_m = (x, y, scene.root_top_z + obj.asset.size_m[2] / 2.0)
            axes[subject_id].base = True
            axes[subject_id].x |= "x" in predicate.params
            axes[subject_id].y |= "y" in predicate.params
            return

        if kind in {
            PredicateKind.FACING_FRONT,
            PredicateKind.FACING_BACK,
            PredicateKind.FACING_LEFT,
            PredicateKind.FACING_RIGHT,
            PredicateKind.RANDOM_ROT,
        }:
            yaw_by_kind = {
                PredicateKind.FACING_FRONT: 0.0,
                PredicateKind.FACING_BACK: math.pi,
                PredicateKind.FACING_LEFT: math.pi / 2.0,
                PredicateKind.FACING_RIGHT: -math.pi / 2.0,
            }
            yaw = yaw_by_kind.get(kind)
            if yaw is None:
                yaw = self._stable_random_yaw(subject_id)
            self._set_yaw(scene, subject_id, yaw)
            self._mark_group_axes(axes, scene, subject_id, "yaw")
            return

        reference_bounds = self._entity_bounds(scene, predicate.reference)
        subject_bounds = self._entity_bounds(scene, subject_id)
        sx0, sx1, sy0, sy1 = subject_bounds
        rx0, rx1, ry0, ry1 = reference_bounds
        subject_center = ((sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0)
        reference_center = ((rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0)
        distance = float(predicate.params.get("distance", 0.0))

        if kind is PredicateKind.LEFT_OF:
            self._set_center(scene, subject_id, axis=1, value=ry1 + distance + (sy1 - sy0) / 2.0)
            self._mark_group_axes(axes, scene, subject_id, "y")
        elif kind is PredicateKind.RIGHT_OF:
            self._set_center(scene, subject_id, axis=1, value=ry0 - distance - (sy1 - sy0) / 2.0)
            self._mark_group_axes(axes, scene, subject_id, "y")
        elif kind is PredicateKind.FRONT_OF:
            self._set_center(scene, subject_id, axis=0, value=rx1 + distance + (sx1 - sx0) / 2.0)
            self._mark_group_axes(axes, scene, subject_id, "x")
        elif kind is PredicateKind.BACK_OF:
            self._set_center(scene, subject_id, axis=0, value=rx0 - distance - (sx1 - sx0) / 2.0)
            self._mark_group_axes(axes, scene, subject_id, "x")
        elif kind is PredicateKind.ALIGN_CENTER_LR:
            self._set_center(scene, subject_id, axis=1, value=reference_center[1])
            self._mark_group_axes(axes, scene, subject_id, "y")
        elif kind is PredicateKind.ALIGN_CENTER_FB:
            self._set_center(scene, subject_id, axis=0, value=reference_center[0])
            self._mark_group_axes(axes, scene, subject_id, "x")
        elif kind is PredicateKind.ALIGN_LEFT:
            self._set_center(scene, subject_id, axis=1, value=ry1 - (sy1 - sy0) / 2.0)
            self._mark_group_axes(axes, scene, subject_id, "y")
        elif kind is PredicateKind.ALIGN_RIGHT:
            self._set_center(scene, subject_id, axis=1, value=ry0 + (sy1 - sy0) / 2.0)
            self._mark_group_axes(axes, scene, subject_id, "y")
        elif kind is PredicateKind.ALIGN_FRONT:
            self._set_center(scene, subject_id, axis=0, value=rx1 - (sx1 - sx0) / 2.0)
            self._mark_group_axes(axes, scene, subject_id, "x")
        elif kind is PredicateKind.ALIGN_BACK:
            self._set_center(scene, subject_id, axis=0, value=rx0 + (sx1 - sx0) / 2.0)
            self._mark_group_axes(axes, scene, subject_id, "x")
        elif kind is PredicateKind.SYMMETRY_ALONG:
            center_id = str(predicate.params.get("C", ""))
            cx0, cx1, cy0, cy1 = self._entity_bounds(scene, center_id)
            self._set_center(scene, subject_id, axis=0, value=(cx0 + cx1) - reference_center[0])
            self._set_center(scene, subject_id, axis=1, value=(cy0 + cy1) - reference_center[1])
            self._mark_group_axes(axes, scene, subject_id, "x")
            self._mark_group_axes(axes, scene, subject_id, "y")
        elif kind is PredicateKind.FACING_TO:
            yaw = math.atan2(reference_center[1] - subject_center[1], reference_center[0] - subject_center[0])
            self._set_yaw(scene, subject_id, yaw)
            self._mark_group_axes(axes, scene, subject_id, "yaw")
        elif kind in {PredicateKind.FACING_SAME_AS, PredicateKind.FACING_OPPOSITE_TO}:
            reference_yaw = self._entity_yaw(scene, predicate.reference)
            if kind is PredicateKind.FACING_OPPOSITE_TO:
                reference_yaw = (reference_yaw + math.pi) % (2.0 * math.pi)
            self._set_yaw(scene, subject_id, reference_yaw)
            self._mark_group_axes(axes, scene, subject_id, "yaw")
        elif kind is PredicateKind.ORIENT_BY_RELATIVE_SIDE:
            obj = scene.get(subject_id) if subject_id not in scene.groups else scene.get(scene.groups[subject_id].anchor_id)
            default = obj.asset.front_yaw_rad
            width_x, width_y = sx1 - sx0, sy1 - sy0
            if abs(subject_center[0] - reference_center[0]) > abs(subject_center[1] - reference_center[1]):
                yaw = default if width_y >= width_x else default + math.pi / 2.0
            else:
                yaw = default if width_x >= width_y else default + math.pi / 2.0
            self._set_yaw(scene, subject_id, yaw)
            self._mark_group_axes(axes, scene, subject_id, "yaw")

    def _coordinate_refine(
        self, scene: SceneState, object_ids: Sequence[str], predicates: Sequence[Predicate]
    ) -> None:
        if not object_ids:
            return
        radius = min(scene.root_size_m[:2]) * self.config.spatial.distance_radius_fraction
        yaw_radius = math.radians(self.config.spatial.rotation_radius_degrees)
        for _ in range(self.config.spatial.iterations):
            for object_id in object_ids:
                obj = scene.get(object_id)
                for axis, sample_radius in ((0, radius), (1, radius), (3, yaw_radius)):
                    current = obj.yaw_rad if axis == 3 else obj.position_m[axis]
                    candidates = [current] + [
                        self._rng.uniform(current - sample_radius, current + sample_radius)
                        for _ in range(self.config.spatial.candidates_per_parameter - 1)
                    ]
                    best_value = current
                    best_penalty = self._penalty(scene, object_ids, predicates)
                    for candidate in candidates[1:]:
                        self._assign_scalar(obj, axis, candidate)
                        penalty = self._penalty(scene, object_ids, predicates)
                        if penalty < best_penalty:
                            best_penalty = penalty
                            best_value = candidate
                    self._assign_scalar(obj, axis, best_value)

    @staticmethod
    def _assign_scalar(obj: SceneObject, axis: int, value: float) -> None:
        if axis == 3:
            obj.yaw_rad = value
            return
        position = list(obj.position_m)
        position[axis] = value
        obj.position_m = tuple(position)  # type: ignore[assignment]

    def _penalty(
        self, scene: SceneState, object_ids: Sequence[str], predicates: Sequence[Predicate]
    ) -> float:
        objects = [scene.get(object_id) for object_id in object_ids]
        overlap = sum(
            convex_overlap_area(object_polygon(objects[i]), object_polygon(objects[j]))
            for i in range(len(objects))
            for j in range(i + 1, len(objects))
        )
        boundary = sum(
            outside_distance_squared(object_polygon(obj), scene.root_bounds_xy) for obj in objects
        )
        relation = sum(self._relation_residual(predicate, scene) for predicate in predicates)
        return (
            self.config.spatial.collision_weight * overlap
            + self.config.spatial.boundary_weight * boundary
            + self.config.spatial.relation_weight * relation
        )

    def _relation_residual(self, predicate: Predicate, scene: SceneState) -> float:
        if not isinstance(predicate.subject, str) or not isinstance(predicate.reference, str):
            return 0.0
        if predicate.kind in {
            PredicateKind.PLACE_ON_BASE,
            PredicateKind.RANDOM_ROT,
        }:
            if predicate.kind is PredicateKind.PLACE_ON_BASE and predicate.subject in scene.objects:
                obj = scene.get(predicate.subject)
                return sum(
                    (obj.position_m[axis] - float(predicate.params[key])) ** 2
                    for axis, key in ((0, "x"), (1, "y"))
                    if key in predicate.params
                )
            return 0.0
        if predicate.subject not in scene.objects and predicate.subject not in scene.groups:
            return 0.0
        if predicate.reference != "root" and predicate.reference not in scene.objects and predicate.reference not in scene.groups:
            return 0.0
        sx0, sx1, sy0, sy1 = self._entity_bounds(scene, predicate.subject)
        rx0, rx1, ry0, ry1 = self._entity_bounds(scene, predicate.reference)
        scx, scy = (sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0
        rcx, rcy = (rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0
        distance = float(predicate.params.get("distance", 0.0))
        residuals: dict[PredicateKind, Callable[[], float]] = {
            PredicateKind.LEFT_OF: lambda: (sy0 - ry1 - distance) ** 2,
            PredicateKind.RIGHT_OF: lambda: (ry0 - sy1 - distance) ** 2,
            PredicateKind.FRONT_OF: lambda: (sx0 - rx1 - distance) ** 2,
            PredicateKind.BACK_OF: lambda: (rx0 - sx1 - distance) ** 2,
            PredicateKind.ALIGN_CENTER_LR: lambda: (scy - rcy) ** 2,
            PredicateKind.ALIGN_CENTER_FB: lambda: (scx - rcx) ** 2,
            PredicateKind.ALIGN_LEFT: lambda: (sy1 - ry1) ** 2,
            PredicateKind.ALIGN_RIGHT: lambda: (sy0 - ry0) ** 2,
            PredicateKind.ALIGN_FRONT: lambda: (sx1 - rx1) ** 2,
            PredicateKind.ALIGN_BACK: lambda: (sx0 - rx0) ** 2,
        }
        if predicate.kind in residuals:
            return residuals[predicate.kind]()
        if predicate.kind in {
            PredicateKind.FACING_FRONT,
            PredicateKind.FACING_BACK,
            PredicateKind.FACING_LEFT,
            PredicateKind.FACING_RIGHT,
            PredicateKind.FACING_SAME_AS,
            PredicateKind.FACING_OPPOSITE_TO,
        }:
            targets = {
                PredicateKind.FACING_FRONT: 0.0,
                PredicateKind.FACING_BACK: math.pi,
                PredicateKind.FACING_LEFT: math.pi / 2.0,
                PredicateKind.FACING_RIGHT: -math.pi / 2.0,
                PredicateKind.FACING_SAME_AS: self._entity_yaw(scene, predicate.reference),
                PredicateKind.FACING_OPPOSITE_TO: self._entity_yaw(scene, predicate.reference) + math.pi,
            }
            return self._angle_difference(self._entity_yaw(scene, predicate.subject), targets[predicate.kind]) ** 2
        if predicate.kind is PredicateKind.FACING_TO:
            target = math.atan2(rcy - scy, rcx - scx)
            return self._angle_difference(self._entity_yaw(scene, predicate.subject), target) ** 2
        return 0.0

    @staticmethod
    def _angle_difference(first: float, second: float) -> float:
        return (first - second + math.pi) % (2.0 * math.pi) - math.pi

    def _entity_bounds(self, scene: SceneState, entity_id: str) -> tuple[float, float, float, float]:
        if entity_id == "root":
            return scene.root_bounds_xy
        if entity_id in scene.groups:
            bounds = [self._entity_bounds(scene, member) for member in scene.groups[entity_id].object_ids]
            return (
                min(value[0] for value in bounds),
                max(value[1] for value in bounds),
                min(value[2] for value in bounds),
                max(value[3] for value in bounds),
            )
        return polygon_bounds(object_polygon(scene.get(entity_id)))

    def _entity_yaw(self, scene: SceneState, entity_id: str) -> float:
        if entity_id == "root":
            return 0.0
        if entity_id in scene.groups:
            return scene.get(scene.groups[entity_id].anchor_id).yaw_rad
        return scene.get(entity_id).yaw_rad

    def _set_center(self, scene: SceneState, entity_id: str, *, axis: int, value: float) -> None:
        bounds = self._entity_bounds(scene, entity_id)
        current = (bounds[0] + bounds[1]) / 2.0 if axis == 0 else (bounds[2] + bounds[3]) / 2.0
        delta = value - current
        object_ids = scene.groups[entity_id].object_ids if entity_id in scene.groups else (entity_id,)
        for object_id in object_ids:
            obj = scene.get(object_id)
            position = list(obj.position_m)
            position[axis] += delta
            obj.position_m = tuple(position)  # type: ignore[assignment]

    def _set_yaw(self, scene: SceneState, entity_id: str, yaw: float) -> None:
        if entity_id not in scene.groups:
            scene.get(entity_id).yaw_rad = yaw
            return
        group = scene.groups[entity_id]
        anchor = scene.get(group.anchor_id)
        delta = self._angle_difference(yaw, anchor.yaw_rad)
        cosine, sine = math.cos(delta), math.sin(delta)
        ax, ay = anchor.position_m[:2]
        for object_id in group.object_ids:
            obj = scene.get(object_id)
            dx, dy = obj.position_m[0] - ax, obj.position_m[1] - ay
            obj.position_m = (
                ax + cosine * dx - sine * dy,
                ay + sine * dx + cosine * dy,
                obj.position_m[2],
            )
            obj.yaw_rad += delta

    @staticmethod
    def _mark_group_axes(
        axes: dict[str, _SolvedAxes], scene: SceneState, entity_id: str, axis_name: str
    ) -> None:
        object_ids = scene.groups[entity_id].object_ids if entity_id in scene.groups else (entity_id,)
        for object_id in object_ids:
            if object_id in axes:
                setattr(axes[object_id], axis_name, True)

    def _stable_random_yaw(self, object_id: str) -> float:
        seed = self.config.random_seed + sum((index + 1) * ord(char) for index, char in enumerate(object_id))
        return random.Random(seed).random() * 2.0 * math.pi
