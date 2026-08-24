"""Structured feedback corresponding to the paper's three feedback modes."""

from __future__ import annotations

from typing import ClassVar

from physcensis.geometry import object_polygon, polygon_area
from physcensis.types import Feedback, SceneState, SolveReport


class FeedbackEngine:
    _GRAMMAR_CODES: ClassVar[set[str]] = {
        "not_a_list",
        "invalid_record",
        "invalid_arity",
        "unknown_predicate",
        "invalid_params",
        "missing_description",
        "unknown_reference",
        "height_unsolved",
        "yaw_unsolved",
    }

    def build(self, report: SolveReport) -> Feedback:
        if report.success:
            measurements = dict(report.metrics)
            measurements.update(self.scene_measurements(report.scene))
            return Feedback(
                "success",
                "Scene solved successfully. Measurements can be used to decide whether to enrich it.",
                tuple(report.issues),
                measurements,
            )
        grammar = any(issue.code in self._GRAMMAR_CODES for issue in report.issues)
        category = "grammar_error" if grammar else "solver_failure"
        if grammar:
            summary = "The predicate program is invalid; correct the listed objects and relationships."
        else:
            empty = self.empty_regions(report.scene)
            summary = "The solver could not realize the program."
            if empty:
                summary += f" Candidate empty regions: {', '.join(empty)}."
        return Feedback(category, summary, tuple(report.issues), report.metrics)

    @staticmethod
    def scene_measurements(scene: SceneState) -> dict[str, float]:
        root_area = scene.root_size_m[0] * scene.root_size_m[1]
        footprint = sum(polygon_area(object_polygon(obj)) for obj in scene.objects.values())
        xs = [obj.position_m[0] for obj in scene.objects.values()]
        ys = [obj.position_m[1] for obj in scene.objects.values()]
        compactness = 0.0
        if xs and ys:
            span = max(xs) - min(xs) + max(ys) - min(ys)
            compactness = 1.0 / (1.0 + span)
        return {
            "object_count": float(len(scene.objects)),
            "surface_coverage": min(1.0, footprint / max(root_area, 1.0e-9)),
            "compactness": compactness,
        }

    @staticmethod
    def empty_regions(scene: SceneState) -> list[str]:
        regions = {
            "front-left": (0.25, 0.25),
            "front-right": (0.25, -0.25),
            "back-left": (-0.25, 0.25),
            "back-right": (-0.25, -0.25),
        }
        scale_x, scale_y = scene.root_size_m[0], scene.root_size_m[1]
        empty = []
        for name, (fx, fy) in regions.items():
            target = (fx * scale_x, fy * scale_y)
            if all(
                (obj.position_m[0] - target[0]) ** 2 + (obj.position_m[1] - target[1]) ** 2
                > (0.15 * min(scale_x, scale_y)) ** 2
                for obj in scene.objects.values()
            ):
                empty.append(name)
        return empty
