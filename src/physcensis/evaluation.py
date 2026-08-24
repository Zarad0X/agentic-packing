"""Reproducible acceptance gates for the core solver and demo families."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from physcensis.assets import AssetCatalog, PrimitiveAssetCatalog
from physcensis.config import ReproductionConfig
from physcensis.physics import PhysicsBackend
from physcensis.pipeline import ScenePipeline

SCENE_FAMILIES = (
    "dining_table.json",
    "office_desk.json",
    "workbench.json",
    "coffee_table.json",
    "physical_showcase.json",
)

DENSE_SCENE_FAMILIES = (
    "dense_grocery_basket.json",
    "dense_kitchen_sink.json",
)


def run_core_gate(
    config: ReproductionConfig,
    backend: PhysicsBackend,
    *,
    repetitions_per_family: int = 20,
    examples_dir: str | Path = "examples",
    catalog: AssetCatalog | None = None,
) -> dict[str, Any]:
    return _run_gate(
        config,
        backend,
        scene_families=SCENE_FAMILIES,
        gate_name="core_solver",
        repetitions_per_family=repetitions_per_family,
        examples_dir=examples_dir,
        catalog=catalog,
    )


def run_dense_gate(
    config: ReproductionConfig,
    backend: PhysicsBackend,
    *,
    repetitions_per_family: int = 10,
    examples_dir: str | Path = "examples",
    catalog: AssetCatalog | None = None,
) -> dict[str, Any]:
    """Validate high-count, multi-layer container arrangements."""
    return _run_gate(
        config,
        backend,
        scene_families=DENSE_SCENE_FAMILIES,
        gate_name="dense_container",
        repetitions_per_family=repetitions_per_family,
        examples_dir=examples_dir,
        catalog=catalog,
    )


def _run_gate(
    config: ReproductionConfig,
    backend: PhysicsBackend,
    *,
    scene_families: tuple[str, ...],
    gate_name: str,
    repetitions_per_family: int,
    examples_dir: str | Path,
    catalog: AssetCatalog | None,
) -> dict[str, Any]:
    if repetitions_per_family <= 0:
        raise ValueError("repetitions_per_family must be positive")
    examples_path = Path(examples_dir)
    runs: list[dict[str, Any]] = []
    for family_index, filename in enumerate(scene_families):
        payload = json.loads((examples_path / filename).read_text(encoding="utf-8"))
        for repetition in range(repetitions_per_family):
            run_config = replace(
                config,
                random_seed=config.random_seed + family_index * 10_000 + repetition,
            )
            result = ScenePipeline(
                run_config,
                backend,
                catalog=catalog or PrimitiveAssetCatalog(),
            ).run_payload(payload)
            runs.append(
                {
                    "family": filename.removesuffix(".json"),
                    "repetition": repetition,
                    "success": result.success,
                    "object_count": len(result.scene.objects),
                    "settle_distance_m": result.feedback.measurements.get(
                        "settle_distance_m", 0.0
                    ),
                    "packing_fraction": result.feedback.measurements.get(
                        "packing_fraction", 0.0
                    ),
                    "packing_layer_count": result.feedback.measurements.get(
                        "packing_layer_count", 0.0
                    ),
                    "issue_codes": [issue.code for issue in result.feedback.issues],
                }
            )

    total = len(runs)
    successful = [run for run in runs if run["success"]]
    gate = config.raw.get("acceptance_gates", {}).get(gate_name, {})
    required_success_rate = float(gate.get("minimum_success_rate", 0.90))
    minimum_objects = int(gate.get("minimum_objects", 15))
    required_object_fraction = float(gate.get("fraction_runs_meeting_object_count", 0.80))
    maximum_settle = float(gate.get("maximum_mean_settle_distance_m", 0.01))
    minimum_packing_fraction = float(gate.get("minimum_packing_fraction", 0.0))
    minimum_layers = float(gate.get("minimum_packing_layers", 0.0))
    success_rate = len(successful) / total
    object_fraction = sum(run["object_count"] >= minimum_objects for run in runs) / total
    mean_settle = (
        sum(float(run["settle_distance_m"]) for run in successful) / len(successful)
        if successful
        else float("inf")
    )
    packing_fraction = min(float(run["packing_fraction"]) for run in runs)
    packing_layers = min(float(run["packing_layer_count"]) for run in runs)
    family_summary = {}
    for family in (name.removesuffix(".json") for name in scene_families):
        family_runs = [run for run in runs if run["family"] == family]
        family_summary[family] = {
            "runs": len(family_runs),
            "success_rate": sum(run["success"] for run in family_runs) / len(family_runs),
            "minimum_object_count": min(run["object_count"] for run in family_runs),
            "minimum_packing_fraction": min(
                run["packing_fraction"] for run in family_runs
            ),
            "minimum_packing_layers": min(
                run["packing_layer_count"] for run in family_runs
            ),
        }
    return {
        "backend": backend.name,
        "evidence_level": "real_physics" if backend.name == "genesis" else "geometry_only",
        "total_runs": total,
        "success_rate": success_rate,
        "object_count_fraction": object_fraction,
        "mean_settle_distance_m": mean_settle,
        "minimum_packing_fraction": packing_fraction,
        "minimum_packing_layers": packing_layers,
        "thresholds": {
            "minimum_success_rate": required_success_rate,
            "minimum_objects": minimum_objects,
            "fraction_runs_meeting_object_count": required_object_fraction,
            "maximum_mean_settle_distance_m": maximum_settle,
            "minimum_packing_fraction": minimum_packing_fraction,
            "minimum_packing_layers": minimum_layers,
        },
        "passed": (
            success_rate >= required_success_rate
            and object_fraction >= required_object_fraction
            and mean_settle <= maximum_settle
            and packing_fraction >= minimum_packing_fraction
            and packing_layers >= minimum_layers
        ),
        "families": family_summary,
        "failed_runs": [run for run in runs if not run["success"]],
    }
