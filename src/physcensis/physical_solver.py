"""Physical predicate realization with grid search and backend validation."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import replace

from physcensis.assets import AssetCatalog, AssetNotFoundError
from physcensis.config import ReproductionConfig
from physcensis.inventory import InventoryArrangementPlan
from physcensis.occupancy import (
    ContainerPlacementCandidate,
    PlacementCandidate,
    candidate_supports,
    container_candidates,
    surface_candidates,
)
from physcensis.physics import PhysicsBackend
from physcensis.stability import StabilityEstimator
from physcensis.storage_semantics import semantically_compatible_support, storage_profile
from physcensis.types import (
    Issue,
    PlacementProgram,
    Predicate,
    PredicateKind,
    SceneObject,
    SceneState,
    SolveReport,
)
from physcensis.visual_contact import build_visual_contact_layout


class PhysicalSolver:
    def __init__(
        self,
        config: ReproductionConfig,
        backend: PhysicsBackend,
        catalog: AssetCatalog,
    ):
        self.config = config
        self.backend = backend
        self.catalog = catalog
        self.stability = StabilityEstimator(config, backend)
        self._rng = random.Random(config.random_seed)

    def solve(self, program: PlacementProgram, scene: SceneState) -> SolveReport:
        working = scene.clone()
        physical_predicates = [predicate for predicate in program.predicates if predicate.kind.is_physical]
        pending_ids = {
            predicate.subject
            for predicate in physical_predicates
            if isinstance(predicate.subject, str) and predicate.subject in working.objects
        }
        pending = {object_id: working.objects.pop(object_id) for object_id in pending_ids}
        issues: list[Issue] = []
        solved: list[str] = []
        stability_values: list[float] = []

        for predicate in physical_predicates:
            if predicate.kind is PredicateKind.PLACE_ON:
                result_ids = self._place_on(predicate, working, pending, issues)
            elif predicate.kind is PredicateKind.PLACE_ANYWHERE:
                result_ids = self._place_anywhere(predicate, working, pending, issues)
            elif predicate.kind is PredicateKind.PLACE_IN:
                result_ids = self._place_in(predicate, working, pending, issues)
            else:  # pragma: no cover - exhaustive enum guard
                result_ids = []
            if not result_ids:
                return SolveReport(False, working, issues, solved_object_ids=solved)
            solved.extend(result_ids)
            if predicate.kind is PredicateKind.PLACE_ON and self.config.stability.sample_count > 0:
                estimate = self.stability.estimate(working, result_ids[-1])
                stability_values.append(1.0 - estimate.local_failure_probability)

        if pending:
            for object_id in pending:
                issues.append(
                    Issue("physical_subject_unplaced", f"Physical subject was never placed: {object_id}", object_id)
                )
            return SolveReport(False, working, issues, solved_object_ids=solved)

        return self._finalize_scene(working, solved, issues, stability_values)

    def arrange_inventory(
        self,
        scene: SceneState,
        container_id: str,
        objects: list[SceneObject],
        *,
        allow_protrusion_m: float = 0.0,
        arrangement_plan: InventoryArrangementPlan | None = None,
    ) -> SolveReport:
        """Arrange a fixed set of object instances without replacing or dropping any."""
        working = scene.clone()
        if container_id not in working.objects:
            issue = Issue("inventory_container_missing", f"Unknown container: {container_id}")
            return SolveReport(False, working, [issue])
        container = working.get(container_id)
        if container.asset.container_inner_size_m is None:
            issue = Issue(
                "inventory_not_a_container",
                f"Asset {container_id} has no inner volume",
                container_id,
            )
            return SolveReport(False, working, [issue])

        supplied_ids = [obj.object_id for obj in objects]
        working.metadata.update(
            {
                "arrangement_mode": "fixed_inventory",
                "inventory_container_id": container_id,
                "inventory_input_object_ids": supplied_ids,
                "inventory_object_count": len(supplied_ids),
                "inventory_asset_ids": {
                    obj.object_id: obj.asset.asset_id for obj in objects
                },
            }
        )
        planned_objects = [replace(obj) for obj in objects]
        if arrangement_plan is not None:
            stacked_ids = {
                object_id
                for group in arrangement_plan.stack_groups
                for object_id in group.bottom_to_top_object_ids
            }
            planned_objects = [
                self._with_nesting_collision(obj) if obj.object_id in stacked_ids else obj
                for obj in planned_objects
            ]
            working.metadata["inventory_arrangement_plan"] = arrangement_plan.to_dict()
        accepted = self._place_organized_objects(
            working,
            planned_objects,
            container,
            {
                "allow_protrusion_m": allow_protrusion_m,
                "placement_order": None
                if arrangement_plan is None
                else arrangement_plan.placement_order,
                "stack_groups": ()
                if arrangement_plan is None
                else tuple(
                    group.bottom_to_top_object_ids
                    for group in arrangement_plan.stack_groups
                ),
                "adjacency_groups": ()
                if arrangement_plan is None
                else tuple(group.object_ids for group in arrangement_plan.adjacency_groups),
            },
        )
        if len(accepted) != len(objects):
            rejected_ids = list(working.metadata.pop("organized_failure_ids", supplied_ids))
            issue = Issue(
                "inventory_no_complete_arrangement",
                "The complete inventory does not fit under the current geometry and support rules",
                object_id=rejected_ids[0] if rejected_ids else None,
                details={
                    "unplaced_object_ids": rejected_ids,
                    "supplied_object_count": len(supplied_ids),
                    "placed_object_count": 0,
                },
            )
            return SolveReport(False, working, [issue])
        if arrangement_plan is not None:
            stacks = list(working.metadata.get("semantic_stacks", []))
            for group in arrangement_plan.stack_groups:
                members = [working.get(object_id) for object_id in group.bottom_to_top_object_ids]
                stacks.append(
                    {
                        "stack_id": group.group_id,
                        "category": members[0].asset.description,
                        "container_id": container_id,
                        "member_ids": list(group.bottom_to_top_object_ids),
                        "asset_variant": self._asset_variant_key(members[0]),
                        "mode": "llm_planned_nested",
                        "stacking_step_ratio": members[0].asset.stacking_step_ratio,
                    }
                )
            working.metadata["semantic_stacks"] = stacks
        return self._finalize_scene(working, accepted, [])

    def _finalize_scene(
        self,
        working: SceneState,
        solved: list[str],
        issues: list[Issue],
        stability_values: list[float] | None = None,
    ) -> SolveReport:
        """Run the one authoritative assembled-scene simulation and metrics pass."""
        simulation = self.backend.simulate(working, self.config.physical.settle_steps)
        for object_id, position in simulation.final_positions_m.items():
            if object_id in working.objects:
                working.objects[object_id].position_m = position
        mean_displacement = min(simulation.mean_displacement_m, self.config.physical.settle_clamp_m)
        success = simulation.success and mean_displacement <= self.config.physical.displacement_threshold_m
        if not success:
            issues.append(
                Issue(
                    "final_simulation_failed",
                    "Final physics validation reported falling, penetration, or excessive displacement",
                    details={
                        "mean_displacement_m": mean_displacement,
                        "fallen": list(simulation.fallen_object_ids),
                        "penetrations": [list(pair) for pair in simulation.penetrations],
                    },
                )
            )
        metrics = {
            "settle_distance_m": mean_displacement,
            "physical_object_count": float(len(solved)),
        }
        metrics.update(self._container_metrics(working))
        if metrics.get("container_item_count", 0.0) >= 10:
            working.metadata["presentation_mode"] = "dense_container"
        working.metadata["physical_placement_order"] = solved
        if stability_values:
            metrics["mean_stability"] = sum(stability_values) / len(stability_values)
        return SolveReport(success, working, issues, metrics, solved)

    def _place_on(
        self,
        predicate: Predicate,
        scene: SceneState,
        pending: dict[str, SceneObject],
        issues: list[Issue],
    ) -> list[str]:
        if not isinstance(predicate.subject, str) or not isinstance(predicate.reference, str):
            issues.append(Issue("invalid_place_on", "PLACE-ON requires object ids"))
            return []
        obj = pending.get(predicate.subject)
        if obj is None:
            issues.append(Issue("place_on_subject_missing", f"Missing pending object {predicate.subject}"))
            return []
        if predicate.reference not in scene.objects:
            issues.append(Issue("place_on_support_missing", f"Support object not placed: {predicate.reference}"))
            return []
        candidates = surface_candidates(
            scene, obj, predicate.reference, self.config.physical.scene_resolution_m
        )
        ranked = sorted(candidates, key=lambda candidate: self._place_on_score(candidate, predicate, scene))
        accepted = self._first_physically_valid(obj, ranked, scene)
        if accepted is None:
            issues.append(
                Issue(
                    "place_on_no_solution",
                    f"No supported physical placement found for {predicate.subject} on {predicate.reference}",
                    predicate_index=predicate.source_index,
                )
            )
            return []
        obj.position_m = accepted.position_m
        obj.support_id = predicate.reference
        scene.objects[obj.object_id] = obj
        pending.pop(obj.object_id)
        return [obj.object_id]

    def _place_anywhere(
        self,
        predicate: Predicate,
        scene: SceneState,
        pending: dict[str, SceneObject],
        issues: list[Issue],
    ) -> list[str]:
        if not isinstance(predicate.subject, str):
            issues.append(Issue("invalid_place_anywhere", "PLACE-ANYWHERE requires an object id"))
            return []
        obj = pending.get(predicate.subject)
        if obj is None:
            issues.append(Issue("place_anywhere_subject_missing", f"Missing object {predicate.subject}"))
            return []
        all_candidates: list[PlacementCandidate] = []
        for support_id in candidate_supports(scene):
            if support_id == obj.object_id:
                continue
            all_candidates.extend(
                surface_candidates(scene, obj, support_id, self.config.physical.scene_resolution_m)
            )
        # Prefer compact placements and high support while preserving seeded diversity.
        self._rng.shuffle(all_candidates)
        ranked = sorted(
            all_candidates,
            key=lambda candidate: (
                -candidate.support_ratio,
                abs(candidate.position_m[0]) + abs(candidate.position_m[1]),
            ),
        )
        accepted = self._first_physically_valid(obj, ranked, scene)
        if accepted is None:
            issues.append(
                Issue(
                    "place_anywhere_no_solution",
                    f"No physically valid free placement found for {predicate.subject}",
                    predicate_index=predicate.source_index,
                )
            )
            return []
        obj.position_m = accepted.position_m
        obj.support_id = accepted.support_id
        scene.objects[obj.object_id] = obj
        pending.pop(obj.object_id)
        return [obj.object_id]

    def _place_in(
        self,
        predicate: Predicate,
        scene: SceneState,
        pending: dict[str, SceneObject],
        issues: list[Issue],
    ) -> list[str]:
        if not isinstance(predicate.reference, str) or predicate.reference not in scene.objects:
            issues.append(Issue("place_in_container_missing", f"Unknown container: {predicate.reference}"))
            return []
        container = scene.get(predicate.reference)
        inner = container.asset.container_inner_size_m
        if inner is None:
            issues.append(Issue("not_a_container", f"Asset {predicate.reference} has no inner volume"))
            return []

        objects: list[SceneObject] = []
        if isinstance(predicate.subject, str):
            obj = pending.get(predicate.subject)
            if obj is None:
                issues.append(Issue("place_in_subject_missing", f"Missing object: {predicate.subject}"))
                return []
            objects.append(obj)
        else:
            for category, quantity in predicate.subject:
                for index in range(quantity):
                    object_id = self._unique_batch_id(scene, pending, category, index)
                    try:
                        asset = self.catalog.resolve_category(category, object_id)
                    except AssetNotFoundError as exc:
                        issues.append(Issue("batch_asset_missing", str(exc), object_id))
                        return []
                    objects.append(SceneObject(object_id, asset, yaw_rad=self._rng.random() * 2.0 * math.pi))

        strategy = str(predicate.params.get("strategy", "dense")).lower()
        if strategy == "nested":
            not_stackable = [obj.object_id for obj in objects if not obj.asset.stackable]
            if not_stackable:
                issues.append(
                    Issue(
                        "asset_not_stackable",
                        "Nested placement requires assets explicitly marked stackable: "
                        + ", ".join(not_stackable),
                        predicate_index=predicate.source_index,
                        details={"rejected_object_ids": not_stackable},
                    )
                )
                return []
            objects = [self._with_nesting_collision(obj) for obj in objects]

        if strategy == "organized":
            accepted_ids = self._place_organized_objects(
                scene,
                objects,
                container,
                predicate.params,
            )
            if len(accepted_ids) != len(objects):
                rejected_ids = list(scene.metadata.pop("organized_failure_ids", []))
                rejected = rejected_ids[0] if rejected_ids else objects[0].object_id
                issues.append(
                    Issue(
                        "place_in_no_organized_solution",
                        f"No household-like position in {container.object_id} for {rejected}",
                        object_id=rejected,
                        predicate_index=predicate.source_index,
                        details={"unplaced_object_ids": rejected_ids},
                    )
                )
                return []
            for object_id in accepted_ids:
                pending.pop(object_id, None)
        else:
            accepted_ids = []
            for obj in objects:
                placement = self._find_container_position(
                    scene,
                    obj,
                    container,
                    predicate.params,
                    accepted_ids,
                )
                if placement is None:
                    issues.append(
                        Issue(
                            "place_in_no_solution",
                            f"No collision-free position in {container.object_id} for {obj.object_id}",
                            object_id=obj.object_id,
                            predicate_index=predicate.source_index,
                        )
                    )
                    return []
                self._accept_container_placement(scene, obj, container, placement)
                pending.pop(obj.object_id, None)
                accepted_ids.append(obj.object_id)
        if strategy == "nested":
            self._record_semantic_stacks(
                scene,
                objects,
                container.object_id,
                predicate,
            )
        if getattr(self.backend, "defer_incremental_validation", False):
            return accepted_ids
        result = self.backend.simulate(scene, self.config.physical.settle_steps)
        invalid_ids = [
            object_id
            for object_id in accepted_ids
            if object_id in result.fallen_object_ids
            or result.displacement_m.get(object_id, math.inf)
            > self.config.physical.displacement_threshold_m
        ]
        if invalid_ids or result.penetrations:
            for object_id in accepted_ids:
                scene.objects.pop(object_id, None)
                self._forget_container_support(scene, object_id)
            issues.append(
                Issue(
                    "place_in_physics_failed",
                    f"Physics rejected container placement for: {invalid_ids or accepted_ids}",
                    predicate_index=predicate.source_index,
                )
            )
            return []
        return accepted_ids

    def _place_organized_objects(
        self,
        scene: SceneState,
        objects: list[SceneObject],
        container: SceneObject,
        params: Mapping[str, object],
    ) -> list[str]:
        """Fill the floor globally before allowing load-aware upper layers."""
        placement_order = params.get("placement_order")
        if isinstance(placement_order, (list, tuple)):
            rank = {str(object_id): index for index, object_id in enumerate(placement_order)}
            remaining = sorted(objects, key=lambda obj: rank.get(obj.object_id, len(rank)))
        else:
            remaining = sorted(objects, key=self._organized_object_priority)
        stack_groups = [
            tuple(str(object_id) for object_id in group)
            for group in params.get("stack_groups", ())
            if isinstance(group, (list, tuple))
        ]
        forced_upper_ids = {
            object_id for group in stack_groups for object_id in group[1:]
        }
        preferred_support = {
            object_id: group[index - 1]
            for group in stack_groups
            for index, object_id in enumerate(group)
            if index > 0
        }
        adjacency_lookup: dict[str, set[str]] = {}
        for group in params.get("adjacency_groups", ()):
            if not isinstance(group, (list, tuple)):
                continue
            members = {str(object_id) for object_id in group}
            for object_id in members:
                adjacency_lookup.setdefault(object_id, set()).update(members - {object_id})
        accepted_ids: list[str] = []
        bottom_ids: list[str] = []

        while remaining:
            selected: tuple[SceneObject, ContainerPlacementCandidate] | None = None
            for obj in remaining:
                if obj.object_id in forced_upper_ids:
                    continue
                candidates = container_candidates(
                    scene,
                    obj,
                    container,
                    self.config.physical.scene_resolution_m,
                    allow_protrusion_m=float(params.get("allow_protrusion_m", 0.0)),
                    yaw_offsets_rad=self._container_yaw_offsets(obj),
                    floor_only=True,
                )
                if candidates:
                    placement = min(
                        candidates,
                        key=lambda candidate: self._organized_floor_score(
                            scene,
                            obj,
                            container,
                            candidate,
                            adjacency_lookup.get(obj.object_id, set()),
                        ),
                    )
                    selected = (obj, placement)
                    break
            if selected is None:
                break
            obj, placement = selected
            self._accept_container_placement(scene, obj, container, placement)
            remaining.remove(obj)
            accepted_ids.append(obj.object_id)
            bottom_ids.append(obj.object_id)

        while remaining:
            selected = None
            for obj in remaining:
                candidates = container_candidates(
                    scene,
                    obj,
                    container,
                    self.config.physical.scene_resolution_m,
                    allow_protrusion_m=float(params.get("allow_protrusion_m", 0.0)),
                    yaw_offsets_rad=self._container_yaw_offsets(obj),
                )
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate.layer_index > 0
                    and self._reasonable_container_support(scene, obj, candidate)
                ]
                required_support = preferred_support.get(obj.object_id)
                if required_support is not None:
                    candidates = [
                        candidate
                        for candidate in candidates
                        if required_support in candidate.support_ids
                    ]
                if candidates:
                    placement = min(
                        candidates,
                        key=lambda candidate: self._organized_upper_score(
                            scene, obj, candidate, required_support
                        ),
                    )
                    selected = (obj, placement)
                    break
            if selected is None:
                scene.metadata["organized_failure_ids"] = [
                    obj.object_id for obj in remaining
                ]
                for object_id in accepted_ids:
                    scene.objects.pop(object_id, None)
                    self._forget_container_support(scene, object_id)
                return []
            obj, placement = selected
            self._accept_container_placement(scene, obj, container, placement)
            remaining.remove(obj)
            accepted_ids.append(obj.object_id)

        scene.metadata.setdefault("storage_plans", []).append(
            {
                "container_id": container.object_id,
                "mode": "organized",
                "placement_order": accepted_ids,
                "bottom_layer_ids": bottom_ids,
                "upper_layer_ids": [
                    object_id for object_id in accepted_ids if object_id not in bottom_ids
                ],
                "llm_planned": isinstance(placement_order, (list, tuple)),
                "planned_stack_groups": [list(group) for group in stack_groups],
            }
        )
        return accepted_ids

    @staticmethod
    def _organized_object_priority(obj: SceneObject) -> tuple[float, float, float, str]:
        size = obj.asset.physical_size_m
        footprint = size[0] * size[1]
        load_value = obj.asset.mass_kg * (0.5 + obj.asset.supporting_probability)
        return (-footprint, -load_value, -max(size[0], size[1]), obj.object_id)

    @staticmethod
    def _container_yaw_offsets(obj: SceneObject) -> tuple[float, ...]:
        if obj.asset.visual_shape in {"bottle", "bowl", "can", "cup", "jar", "plate"}:
            return (0.0,)
        return (0.0, math.pi / 2.0)

    @staticmethod
    def _accept_container_placement(
        scene: SceneState,
        obj: SceneObject,
        container: SceneObject,
        placement: ContainerPlacementCandidate,
    ) -> None:
        obj.position_m = placement.position_m
        obj.yaw_rad = placement.yaw_rad
        obj.support_id = container.object_id
        scene.objects[obj.object_id] = obj
        supports = scene.metadata.setdefault("container_supports", {})
        supports[obj.object_id] = list(placement.support_ids)

    @staticmethod
    def _forget_container_support(scene: SceneState, object_id: str) -> None:
        supports = scene.metadata.get("container_supports", {})
        if isinstance(supports, dict):
            supports.pop(object_id, None)

    def _organized_floor_score(
        self,
        scene: SceneState,
        obj: SceneObject,
        container: SceneObject,
        candidate: ContainerPlacementCandidate,
        adjacency_ids: set[str] | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        placed = replace(obj, position_m=candidate.position_m, yaw_rad=candidate.yaw_rad)
        supports = scene.metadata.get("container_supports", {})
        floor_objects = [
            existing
            for existing in scene.objects.values()
            if isinstance(supports, dict)
            and supports.get(existing.object_id) == [container.object_id]
        ]
        bounds = [self._container_local_bounds(existing, container) for existing in floor_objects]
        candidate_bounds = self._container_local_bounds(placed, container)
        all_bounds = bounds + [candidate_bounds]
        min_x = min(bound[0] for bound in all_bounds)
        max_x = max(bound[1] for bound in all_bounds)
        min_y = min(bound[2] for bound in all_bounds)
        max_y = max(bound[3] for bound in all_bounds)
        used_area = sum((bound[1] - bound[0]) * (bound[3] - bound[2]) for bound in all_bounds)
        compactness = used_area / max((max_x - min_x) * (max_y - min_y), 1.0e-9)

        inner = container.asset.container_inner_size_m
        assert inner is not None
        wall_gap = min(
            candidate_bounds[0] + inner[0] / 2.0,
            inner[0] / 2.0 - candidate_bounds[1],
            candidate_bounds[2] + inner[1] / 2.0,
            inner[1] / 2.0 - candidate_bounds[3],
        )
        neighbor_gap = min(
            (self._rectangle_gap(candidate_bounds, existing) for existing in bounds),
            default=float("inf"),
        )
        contact_gap = min(wall_gap, neighbor_gap)
        adjacent = [
            scene.get(object_id)
            for object_id in (adjacency_ids or set())
            if object_id in scene.objects
        ]
        adjacency_gap = min(
            (
                (candidate.position_m[0] - existing.position_m[0]) ** 2
                + (candidate.position_m[1] - existing.position_m[1]) ** 2
                for existing in adjacent
            ),
            default=0.0,
        )
        return (
            round(1.0 - compactness, 8),
            round(adjacency_gap, 8),
            round(contact_gap, 8),
            -round(candidate.local_xy_m[1], 8),
            round(abs(candidate.local_xy_m[0]), 8),
            round(candidate.yaw_rad, 8),
        )

    @staticmethod
    def _container_local_bounds(
        obj: SceneObject, container: SceneObject
    ) -> tuple[float, float, float, float]:
        dx = obj.position_m[0] - container.position_m[0]
        dy = obj.position_m[1] - container.position_m[1]
        cosine = math.cos(container.yaw_rad)
        sine = math.sin(container.yaw_rad)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        relative_yaw = obj.yaw_rad - container.yaw_rad
        yaw_cosine = abs(math.cos(relative_yaw))
        yaw_sine = abs(math.sin(relative_yaw))
        size = obj.asset.physical_size_m
        extent_x = yaw_cosine * size[0] + yaw_sine * size[1]
        extent_y = yaw_sine * size[0] + yaw_cosine * size[1]
        return (
            local_x - extent_x / 2.0,
            local_x + extent_x / 2.0,
            local_y - extent_y / 2.0,
            local_y + extent_y / 2.0,
        )

    @staticmethod
    def _rectangle_gap(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        gap_x = max(second[0] - first[1], first[0] - second[1], 0.0)
        gap_y = max(second[2] - first[3], first[2] - second[3], 0.0)
        return math.hypot(gap_x, gap_y)

    @staticmethod
    def _reasonable_container_support(
        scene: SceneState,
        obj: SceneObject,
        candidate: ContainerPlacementCandidate,
    ) -> bool:
        supporters = [
            scene.get(object_id)
            for object_id in candidate.support_ids
            if object_id in scene.objects
        ]
        strong = [
            supporter
            for supporter in supporters
            if supporter.asset.supporting_probability >= 0.10
        ]
        if not strong:
            return False
        if not semantically_compatible_support(obj, supporters):
            return False
        capacity = sum(supporter.asset.mass_kg * 2.0 for supporter in strong)
        return obj.asset.mass_kg <= capacity + 1.0e-9

    def _organized_upper_score(
        self,
        scene: SceneState,
        obj: SceneObject,
        candidate: ContainerPlacementCandidate,
        required_support: str | None = None,
    ) -> tuple[float, float, float, float, float]:
        supporters = [
            scene.get(object_id)
            for object_id in candidate.support_ids
            if object_id in scene.objects
        ]
        same_kind = sum(
            storage_profile(supporter.asset).group == storage_profile(obj.asset).group
            for supporter in supporters
        )
        if supporters:
            center_x = sum(supporter.position_m[0] for supporter in supporters) / len(supporters)
            center_y = sum(supporter.position_m[1] for supporter in supporters) / len(supporters)
            alignment = (candidate.position_m[0] - center_x) ** 2 + (
                candidate.position_m[1] - center_y
            ) ** 2
        else:
            alignment = float("inf")
        return (
            -float(required_support is not None and required_support in candidate.support_ids),
            -float(same_kind),
            round(candidate.position_m[2], 8),
            -round(candidate.support_ratio, 8),
            round(alignment, 8),
        )

    def _find_container_position(
        self,
        scene: SceneState,
        obj: SceneObject,
        container: SceneObject,
        params: Mapping[str, object],
        accepted_ids: list[str],
    ) -> ContainerPlacementCandidate | None:
        yaw_offsets = self._container_yaw_offsets(obj)
        candidates = container_candidates(
            scene,
            obj,
            container,
            self.config.physical.scene_resolution_m,
            allow_protrusion_m=float(params.get("allow_protrusion_m", 0.0)),
            yaw_offsets_rad=yaw_offsets,
        )
        strategy = str(params.get("strategy", "dense")).lower()
        if strategy == "floor":
            candidates = [candidate for candidate in candidates if candidate.layer_index == 0]
        elif strategy == "spread":
            floor_candidates = [
                candidate for candidate in candidates if candidate.layer_index == 0
            ]
            candidates = floor_candidates or candidates
            if not candidates:
                return None
            existing_xy = [
                (existing.position_m[0], existing.position_m[1])
                for existing in scene.objects.values()
                if existing.support_id == container.object_id
            ]
            if not existing_xy:
                return min(
                    candidates,
                    key=lambda candidate: (
                        candidate.local_xy_m[1],
                        candidate.local_xy_m[0],
                    ),
                )
            return max(
                candidates,
                key=lambda candidate: (
                    min(
                        (candidate.position_m[0] - x) ** 2
                        + (candidate.position_m[1] - y) ** 2
                        for x, y in existing_xy
                    ),
                    candidate.support_ratio,
                ),
            )
        elif strategy in {"stacked", "nested"}:
            columns = max(1, int(params.get("columns", 2)))
            if len(accepted_ids) < columns:
                candidates = [candidate for candidate in candidates if candidate.layer_index == 0]
                if not candidates:
                    return None
                inner = container.asset.container_inner_size_m
                assert inner is not None
                target_index = len(accepted_ids)
                target_x = inner[0] * (
                    (target_index + 0.5) / columns - 0.5
                ) + inner[0] * float(params.get("center_x_fraction", 0.0))
                target_y = inner[1] * float(params.get("center_y_fraction", 0.0))
                return min(
                    candidates,
                    key=lambda candidate: (
                        (candidate.local_xy_m[0] - target_x) ** 2
                        + (candidate.local_xy_m[1] - target_y) ** 2,
                        -candidate.support_ratio,
                    ),
                )
            accepted = set(accepted_ids)
            target_column = len(accepted_ids) % columns
            column_ids = {
                object_id
                for index, object_id in enumerate(accepted_ids)
                if index % columns == target_column
            }
            candidates = [
                candidate
                for candidate in candidates
                if candidate.layer_index > 0
                and accepted.intersection(candidate.support_ids)
                and column_ids.intersection(candidate.support_ids)
            ]
            if bool(params.get("group_by_asset", strategy == "nested")):
                variant = self._asset_variant_key(obj)
                candidates = [
                    candidate
                    for candidate in candidates
                    if any(
                        support_id in scene.objects
                        and self._asset_variant_key(scene.get(support_id)) == variant
                        for support_id in candidate.support_ids
                    )
                ]
            if not candidates:
                return None
            highest_support = max(
                (scene.get(object_id) for object_id in column_ids),
                key=lambda support: support.top_z,
            )
            return min(
                candidates,
                key=lambda candidate: (
                    (candidate.position_m[0] - highest_support.position_m[0]) ** 2
                    + (candidate.position_m[1] - highest_support.position_m[1]) ** 2,
                    -candidate.support_ratio,
                    candidate.position_m[2],
                ),
            )
        if not candidates:
            return None

        # Dense bottom-left packing is deterministic and naturally fills gaps;
        # supported upper layers are considered only when lower layers are full.
        return min(
            candidates,
            key=lambda candidate: (
                round(candidate.position_m[2], 6),
                round(candidate.local_xy_m[1], 6),
                round(candidate.local_xy_m[0], 6),
                -candidate.support_ratio,
                round(candidate.yaw_rad, 6),
            ),
        )

    @staticmethod
    def _with_nesting_collision(obj: SceneObject) -> SceneObject:
        size = obj.asset.size_m
        collision_size = (size[0], size[1], size[2] * obj.asset.stacking_step_ratio)
        return replace(
            obj,
            asset=replace(
                obj.asset,
                collision_size_m=collision_size,
                friction=max(2.0, obj.asset.friction),
            ),
        )

    @staticmethod
    def _asset_variant_key(obj: SceneObject) -> str:
        parts = obj.asset.asset_id.split(":")
        if len(parts) >= 2 and parts[0] == "objaverse":
            return f"objaverse:{parts[1]}"
        return ":".join(parts[:2])

    def _record_semantic_stacks(
        self,
        scene: SceneState,
        objects: list[SceneObject],
        container_id: str,
        predicate: Predicate,
    ) -> None:
        tolerance = max(1.0e-5, self.config.physical.scene_resolution_m * 1.1)
        columns: list[list[SceneObject]] = []
        for obj in sorted(objects, key=lambda item: item.position_m[2]):
            matching = next(
                (
                    column
                    for column in columns
                    if abs(column[0].position_m[0] - obj.position_m[0]) <= tolerance
                    and abs(column[0].position_m[1] - obj.position_m[1]) <= tolerance
                ),
                None,
            )
            if matching is None:
                columns.append([obj])
            else:
                matching.append(obj)

        stacks = list(scene.metadata.get("semantic_stacks", []))
        category = (
            predicate.subject[0][0]
            if isinstance(predicate.subject, tuple) and len(predicate.subject) == 1
            else "mixed"
        )
        for column in columns:
            if len(column) < 2:
                continue
            variant_keys = {self._asset_variant_key(obj) for obj in column}
            if len(variant_keys) != 1:
                continue
            ordered = sorted(column, key=lambda item: item.position_m[2])
            stacks.append(
                {
                    "stack_id": f"{category.replace(' ', '_')}_stack_{len(stacks)}",
                    "category": category,
                    "container_id": container_id,
                    "member_ids": [obj.object_id for obj in ordered],
                    "asset_variant": next(iter(variant_keys)),
                    "mode": "nested",
                    "stacking_step_ratio": ordered[0].asset.stacking_step_ratio,
                }
            )
        scene.metadata["semantic_stacks"] = stacks
        scene.metadata.setdefault("stackability_checks", []).append(
            {
                "category": category,
                "requested_mode": "nested",
                "accepted": True,
                "object_count": len(objects),
            }
        )

    def _container_metrics(self, scene: SceneState) -> dict[str, float]:
        containers = [
            obj for obj in scene.objects.values() if obj.asset.container_inner_size_m is not None
        ]
        packed_items = [
            obj
            for obj in scene.objects.values()
            if obj.support_id in {container.object_id for container in containers}
        ]
        inner_volume = sum(
            inner[0] * inner[1] * inner[2]
            for container in containers
            if (inner := container.asset.container_inner_size_m) is not None
        )
        packed_volume = sum(
            obj.asset.size_m[0] * obj.asset.size_m[1] * obj.asset.size_m[2]
            for obj in packed_items
        )
        layer_count = len({round(obj.bottom_z, 2) for obj in packed_items})
        support_map = scene.metadata.get("container_supports", {})
        bottom_items: list[SceneObject] = []
        floor_area = 0.0
        occupied_floor_area = 0.0
        compactness_weighted = 0.0
        load_violations = 0
        semantic_support_violations = 0
        supported_item_count = 0
        for container in containers:
            inner = container.asset.container_inner_size_m
            assert inner is not None
            container_items = [
                obj for obj in packed_items if obj.support_id == container.object_id
            ]
            wall = max(0.005, (container.asset.size_m[2] - inner[2]) / 2.0)
            floor_z = container.bottom_z + wall
            local_bottom: list[SceneObject] = []
            for obj in container_items:
                recorded = (
                    support_map.get(obj.object_id)
                    if isinstance(support_map, dict)
                    else None
                )
                on_floor = recorded == [container.object_id] or (
                    recorded is None
                    and abs(obj.bottom_z - floor_z)
                    <= max(0.018, self.config.physical.scene_resolution_m * 1.8)
                )
                if on_floor:
                    local_bottom.append(obj)
                    continue
                if not isinstance(recorded, list):
                    continue
                supporters = [
                    scene.get(object_id)
                    for object_id in recorded
                    if object_id in scene.objects and object_id != container.object_id
                ]
                if not supporters:
                    continue
                supported_item_count += 1
                strong_capacity = sum(
                    supporter.asset.mass_kg * 2.0
                    for supporter in supporters
                    if supporter.asset.supporting_probability >= 0.10
                )
                if obj.asset.mass_kg > strong_capacity + 1.0e-9:
                    load_violations += 1
                if not semantically_compatible_support(obj, supporters):
                    semantic_support_violations += 1

            bottom_items.extend(local_bottom)
            container_floor_area = inner[0] * inner[1]
            local_occupied = sum(
                obj.asset.physical_size_m[0] * obj.asset.physical_size_m[1]
                for obj in local_bottom
            )
            floor_area += container_floor_area
            occupied_floor_area += min(container_floor_area, local_occupied)
            if local_bottom:
                bounds = [
                    self._container_local_bounds(obj, container) for obj in local_bottom
                ]
                bbox_area = (
                    max(bound[1] for bound in bounds) - min(bound[0] for bound in bounds)
                ) * (
                    max(bound[3] for bound in bounds) - min(bound[2] for bound in bounds)
                )
                local_compactness = min(1.0, local_occupied / max(bbox_area, 1.0e-9))
                compactness_weighted += local_compactness * container_floor_area

        floor_coverage = occupied_floor_area / max(floor_area, 1.0e-9)
        bottom_fraction = len(bottom_items) / max(len(packed_items), 1)
        floor_compactness = compactness_weighted / max(floor_area, 1.0e-9)
        load_order_score = 1.0 - load_violations / max(supported_item_count, 1)
        semantic_support_score = (
            1.0 - semantic_support_violations / max(supported_item_count, 1)
        )
        organization_score = (
            0.45 * floor_coverage
            + 0.15 * floor_compactness
            + 0.15 * bottom_fraction
            + 0.10 * load_order_score
            + 0.15 * semantic_support_score
        )
        stacks = scene.metadata.get("semantic_stacks", [])
        stacked_ids = {
            object_id
            for stack in stacks
            for object_id in stack.get("member_ids", [])
        }
        metrics = {
            "container_item_count": float(len(packed_items)),
            "packing_fraction": packed_volume / max(inner_volume, 1.0e-9),
            "packing_layer_count": float(layer_count),
            "floor_coverage": floor_coverage,
            "floor_void_fraction": 1.0 - floor_coverage,
            "floor_compactness": floor_compactness,
            "bottom_layer_item_fraction": bottom_fraction,
            "load_bearing_violation_count": float(load_violations),
            "semantic_support_violation_count": float(semantic_support_violations),
            "organization_score": organization_score,
            "semantic_stack_count": float(len(stacks)),
            "nested_object_count": float(len(stacked_ids)),
            "same_asset_stack_fraction": (
                1.0 if stacks else 0.0
            ),
        }
        metrics.update(build_visual_contact_layout(scene).metrics())
        return metrics

    def _first_physically_valid(
        self,
        obj: SceneObject,
        candidates: Iterable[PlacementCandidate],
        scene: SceneState,
    ) -> PlacementCandidate | None:
        for candidate in candidates:
            if getattr(self.backend, "defer_incremental_validation", False):
                return candidate
            trial = scene.clone()
            placed = replace(obj, position_m=candidate.position_m, support_id=candidate.support_id)
            trial.objects[obj.object_id] = placed
            simulation = self.backend.simulate(trial, self.config.physical.settle_steps)
            if simulation.penetrations or obj.object_id in simulation.fallen_object_ids:
                continue
            if simulation.displacement_m.get(obj.object_id, math.inf) > self.config.physical.displacement_threshold_m:
                continue
            return candidate
        return None

    @staticmethod
    def _place_on_score(candidate: PlacementCandidate, predicate: Predicate, scene: SceneState) -> float:
        supporter = scene.get(str(predicate.reference))
        x_offset = candidate.position_m[0] - supporter.position_m[0]
        y_offset = candidate.position_m[1] - supporter.position_m[1]
        score = 0.0
        if "x_offset" in predicate.params:
            score += (x_offset - float(predicate.params["x_offset"])) ** 2
        if "y_offset" in predicate.params:
            score += (y_offset - float(predicate.params["y_offset"])) ** 2
        if "overlap" in predicate.params:
            score += (candidate.support_ratio - float(predicate.params["overlap"])) ** 2
        stability = predicate.params.get("stability")
        if stability is not None:
            if isinstance(stability, str):
                target = 1.0 if stability.lower() == "stable" else 0.0
            else:
                target = float(stability)
            scale = max(supporter.asset.size_m[0], supporter.asset.size_m[1], 1.0e-9)
            normalized_margin = max(0.0, min(1.0, candidate.center_margin_m / scale * 4.0))
            score += (normalized_margin - target) ** 2
        return score

    @staticmethod
    def _unique_batch_id(
        scene: SceneState, pending: dict[str, SceneObject], category: str, initial_index: int
    ) -> str:
        index = initial_index
        while f"{category}_{index}" in scene.objects or f"{category}_{index}" in pending:
            index += 1
        return f"{category}_{index}"
