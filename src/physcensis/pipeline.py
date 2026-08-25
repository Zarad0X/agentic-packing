"""End-to-end orchestration for deterministic and agent-driven generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physcensis.agent import PredicateAgent
from physcensis.assets import AssetCatalog, PrimitiveAssetCatalog
from physcensis.config import ReproductionConfig
from physcensis.feedback import FeedbackEngine
from physcensis.inventory import InventoryObjectSpec, InventoryParser, InventoryValidationError
from physcensis.physical_solver import PhysicalSolver
from physcensis.physics import PhysicsBackend
from physcensis.predicates import PredicateParser, ProgramValidationError
from physcensis.render import RenderArtifacts, SceneRenderer
from physcensis.spatial_solver import SpatialSolver
from physcensis.types import (
    AssetRecord,
    Feedback,
    Issue,
    PlacementProgram,
    SceneObject,
    SceneState,
    SolveReport,
)


@dataclass(frozen=True)
class GenerationResult:
    success: bool
    scene: SceneState
    feedback: Feedback
    program: PlacementProgram | None
    artifacts: RenderArtifacts | None = None
    rounds: int = 1


class ScenePipeline:
    def __init__(
        self,
        config: ReproductionConfig,
        backend: PhysicsBackend,
        *,
        catalog: AssetCatalog | None = None,
        renderer: SceneRenderer | None = None,
    ):
        self.config = config
        self.backend = backend
        self.catalog = catalog or PrimitiveAssetCatalog()
        self.renderer = renderer or SceneRenderer(backend)
        self.parser = PredicateParser()
        self.inventory_parser = InventoryParser()
        self.spatial = SpatialSolver(config)
        self.physical = PhysicalSolver(config, backend, self.catalog)
        self.feedback_engine = FeedbackEngine()

    def run_payload(
        self, payload: list[Any], *, output_dir: str | Path | None = None
    ) -> GenerationResult:
        scene = SceneState(self.config.root_size_m, self.config.root_height_m)
        try:
            program = self.parser.parse(payload)
        except ProgramValidationError as exc:
            report = SolveReport(False, scene, list(exc.issues))
            return GenerationResult(False, scene, self.feedback_engine.build(report), None)
        try:
            assets = self.catalog.resolve_program(program)
        except LookupError as exc:
            report = SolveReport(False, scene, [Issue("asset_resolution_failed", str(exc))])
            return GenerationResult(False, scene, self.feedback_engine.build(report), program)

        spatial_report = self.spatial.solve(program, scene, assets)
        if not spatial_report.success:
            return GenerationResult(
                False, spatial_report.scene, self.feedback_engine.build(spatial_report), program
            )
        physical_report = self.physical.solve(program, spatial_report.scene)
        physical_report.metrics = {**spatial_report.metrics, **physical_report.metrics}
        feedback = self.feedback_engine.build(physical_report)
        artifacts = None
        if physical_report.success and output_dir is not None:
            artifacts = self.renderer.render(physical_report.scene, Path(output_dir), feedback)
        return GenerationResult(
            physical_report.success, physical_report.scene, feedback, program, artifacts
        )

    def run_inventory(
        self,
        payload: Any,
        *,
        base_dir: str | Path = ".",
        output_dir: str | Path | None = None,
    ) -> GenerationResult:
        """Arrange an explicit fixed inventory with the organized storage planner."""
        scene = SceneState(self.config.root_size_m, self.config.root_height_m)
        try:
            inventory = self.inventory_parser.parse(payload, base_dir=base_dir)
        except (InventoryValidationError, TypeError, ValueError) as exc:
            report = SolveReport(False, scene, [Issue("invalid_inventory", str(exc))])
            return GenerationResult(False, scene, self.feedback_engine.build(report), None)

        try:
            container_asset = self._resolve_inventory_asset(inventory.container)
            object_assets = {
                spec.object_id: self._resolve_inventory_asset(spec) for spec in inventory.objects
            }
        except LookupError as exc:
            report = SolveReport(
                False,
                scene,
                [Issue("inventory_asset_resolution_failed", str(exc))],
            )
            return GenerationResult(False, scene, self.feedback_engine.build(report), None)

        if container_asset.container_inner_size_m is None:
            report = SolveReport(
                False,
                scene,
                [
                    Issue(
                        "inventory_not_a_container",
                        f"Asset {inventory.container.object_id} has no inner volume",
                        inventory.container.object_id,
                    )
                ],
            )
            return GenerationResult(False, scene, self.feedback_engine.build(report), None)

        x, y = inventory.container_position_xy_m
        container = SceneObject(
            inventory.container.object_id,
            container_asset,
            position_m=(
                x,
                y,
                scene.root_top_z + container_asset.physical_size_m[2] / 2.0,
            ),
            yaw_rad=math.radians(inventory.container_yaw_deg),
            fixed=True,
            support_id="root",
        )
        scene.add_object(container)
        objects = [
            SceneObject(spec.object_id, object_assets[spec.object_id])
            for spec in inventory.objects
        ]
        inventory_categories = {
            spec.object_id: spec.category for spec in inventory.objects
        }
        inventory_categories[inventory.container.object_id] = inventory.container.category
        scene.metadata["inventory_categories"] = inventory_categories
        requested_uids = {
            spec.object_id: spec.asset_uid
            for spec in inventory.objects
            if spec.asset_uid is not None
        }
        if inventory.container.asset_uid is not None:
            requested_uids[inventory.container.object_id] = inventory.container.asset_uid
        scene.metadata["inventory_requested_asset_uids"] = requested_uids
        report = self.physical.arrange_inventory(
            scene,
            container.object_id,
            objects,
            allow_protrusion_m=inventory.allow_protrusion_m,
        )
        feedback = self.feedback_engine.build(report)
        artifacts = None
        if report.success and output_dir is not None:
            artifacts = self.renderer.render(report.scene, Path(output_dir), feedback)
        return GenerationResult(report.success, report.scene, feedback, None, artifacts)

    def _resolve_inventory_asset(self, spec: InventoryObjectSpec) -> AssetRecord:
        if spec.asset is not None:
            return spec.asset
        if spec.asset_uid is not None:
            resolver = getattr(self.catalog, "resolve_uid", None)
            if not callable(resolver):
                raise LookupError(
                    f"Exact asset_uid for {spec.object_id} requires a licensed manifest catalog"
                )
            return resolver(spec.object_id, spec.asset_uid)
        return self.catalog.resolve(spec.object_id, spec.category)

    def generate(
        self,
        prompt: str,
        agent: PredicateAgent,
        *,
        output_dir: str | Path | None = None,
    ) -> GenerationResult:
        previous: list[Any] | None = None
        feedback: Feedback | None = None
        last: GenerationResult | None = None
        for round_index in range(1, self.config.generation_round_limit + 1):
            payload = agent.propose(prompt, previous_payload=previous, feedback=feedback)
            last = self.run_payload(payload, output_dir=output_dir)
            if last.success:
                return GenerationResult(
                    True, last.scene, last.feedback, last.program, last.artifacts, round_index
                )
            previous = payload
            feedback = last.feedback
        assert last is not None
        return GenerationResult(
            False,
            last.scene,
            last.feedback,
            last.program,
            last.artifacts,
            self.config.generation_round_limit,
        )
