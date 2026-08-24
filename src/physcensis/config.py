"""Configuration loading with explicit paper/reproduction provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the reproduction configuration is incomplete or invalid."""


def _nested(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigError(f"Missing required configuration key: {path}")
        value = value[key]
    return value


@dataclass(frozen=True)
class SpatialSolverConfig:
    iterations: int
    candidates_per_parameter: int
    distance_radius_fraction: float
    rotation_radius_degrees: float
    acceptance_threshold: float
    relation_weight: float
    collision_weight: float
    boundary_weight: float


@dataclass(frozen=True)
class PhysicalSolverConfig:
    engine: str
    ground_resolution_m: float
    scene_resolution_m: float
    organized_bottom_voxels: int
    organized_search_voxels: int
    messy_bottom_voxels: int
    messy_search_voxels: int
    displacement_threshold_m: float
    settle_steps: int
    settle_clamp_m: float
    per_predicate_time_limit_s: float


@dataclass(frozen=True)
class StabilityConfig:
    sample_count: int
    position_std_m: float
    rotation_std_rad: float
    center_of_mass_std_m: float
    friction_std: float
    mass_std_kg: float


@dataclass(frozen=True)
class ReproductionConfig:
    """Runtime configuration with paper constants and declared choices."""

    random_seed: int
    root_size_m: tuple[float, float, float]
    root_height_m: float
    generation_round_limit: int
    perturbation_sample_count: int
    local_geometry_backend: str
    spatial: SpatialSolverConfig
    physical: PhysicalSolverConfig
    stability: StabilityConfig
    raw: Mapping[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> ReproductionConfig:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised by doctor command
            raise ConfigError("PyYAML is required to read YAML configuration") from exc
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, Mapping):
            raise ConfigError(f"Expected a mapping in {config_path}")
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ReproductionConfig:
        choices = _nested(data, "reproduction_choices")
        paper_spatial = _nested(data, "paper_reported.spatial_solver")
        paper_physical = _nested(data, "paper_reported.physical_solver")
        evaluation = _nested(data, "paper_reported.evaluation")
        if not all(isinstance(value, Mapping) for value in (choices, paper_spatial, paper_physical)):
            raise ConfigError("Configuration sections must be mappings")

        root_size = tuple(float(v) for v in _nested(data, "reproduction_choices.default_root_size_m"))
        if len(root_size) != 3 or any(v <= 0 for v in root_size):
            raise ConfigError("default_root_size_m must contain three positive values")

        spatial = SpatialSolverConfig(
            iterations=int(paper_spatial["iterations"]),
            candidates_per_parameter=int(paper_spatial["candidates_per_parameter"]),
            distance_radius_fraction=float(
                paper_spatial["distance_sampling_radius_fraction_of_short_scene_side"]
            ),
            rotation_radius_degrees=float(paper_spatial["rotation_sampling_radius_degrees"]),
            acceptance_threshold=float(choices["spatial_penalty_acceptance_threshold"]),
            relation_weight=float(choices["relation_weight"]),
            collision_weight=float(choices["collision_weight"]),
            boundary_weight=float(choices["boundary_weight"]),
        )
        resolutions = paper_physical["occupancy_grid_resolution_m"]
        physical = PhysicalSolverConfig(
            engine=str(paper_physical["engine"]),
            ground_resolution_m=float(resolutions["ground_scene"]),
            scene_resolution_m=float(resolutions["other_scene"]),
            organized_bottom_voxels=int(paper_physical["organized_bottom_voxels"]),
            organized_search_voxels=int(paper_physical["organized_search_voxels"]),
            messy_bottom_voxels=int(paper_physical["messy_bottom_voxels"]),
            messy_search_voxels=int(paper_physical["messy_search_voxels"]),
            displacement_threshold_m=float(choices["physical_displacement_acceptance_threshold_m"]),
            settle_steps=int(evaluation["settle_steps"]),
            settle_clamp_m=float(evaluation["settle_distance_clamp_m"]),
            per_predicate_time_limit_s=float(choices["per_predicate_time_limit_s"]),
        )
        stability_values = choices["stability_perturbation_std"]
        stability = StabilityConfig(
            sample_count=int(choices["perturbation_sample_count"]),
            position_std_m=float(stability_values["position_m"]),
            rotation_std_rad=float(stability_values["rotation_rad"]),
            center_of_mass_std_m=float(stability_values["center_of_mass_m"]),
            friction_std=float(stability_values["friction"]),
            mass_std_kg=float(stability_values["mass_kg"]),
        )
        if spatial.iterations <= 0 or spatial.candidates_per_parameter <= 1:
            raise ConfigError("Spatial solver requires positive iterations and multiple candidates")
        if physical.scene_resolution_m <= 0 or physical.ground_resolution_m <= 0:
            raise ConfigError("Occupancy grid resolutions must be positive")

        return cls(
            random_seed=int(choices["random_seed"]),
            root_size_m=root_size,
            root_height_m=float(choices["default_root_height_m"]),
            generation_round_limit=int(choices["generation_round_limit"]),
            perturbation_sample_count=int(choices["perturbation_sample_count"]),
            local_geometry_backend=str(choices["local_geometry_backend"]),
            spatial=spatial,
            physical=physical,
            stability=stability,
            raw=data,
        )
