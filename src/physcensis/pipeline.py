"""End-to-end orchestration for deterministic and agent-driven generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physcensis.agent import PredicateAgent
from physcensis.assets import AssetCatalog, PrimitiveAssetCatalog
from physcensis.config import ReproductionConfig
from physcensis.feedback import FeedbackEngine
from physcensis.physical_solver import PhysicalSolver
from physcensis.physics import PhysicsBackend
from physcensis.predicates import PredicateParser, ProgramValidationError
from physcensis.render import RenderArtifacts, SceneRenderer
from physcensis.spatial_solver import SpatialSolver
from physcensis.types import Feedback, Issue, PlacementProgram, SceneState, SolveReport


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
            False, last.scene, last.feedback, last.program, last.artifacts, self.config.generation_round_limit
        )
