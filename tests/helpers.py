from __future__ import annotations

from dataclasses import replace

from physcensis.config import ReproductionConfig


def make_test_config() -> ReproductionConfig:
    return ReproductionConfig.from_mapping(
        {
            "paper_reported": {
                "spatial_solver": {
                    "iterations": 2,
                    "candidates_per_parameter": 8,
                    "distance_sampling_radius_fraction_of_short_scene_side": 0.1,
                    "rotation_sampling_radius_degrees": 10.0,
                },
                "physical_solver": {
                    "engine": "Genesis",
                    "occupancy_grid_resolution_m": {"ground_scene": 0.03, "other_scene": 0.02},
                    "organized_bottom_voxels": 1,
                    "organized_search_voxels": 1,
                    "messy_bottom_voxels": 5,
                    "messy_search_voxels": 5,
                },
                "evaluation": {"settle_steps": 20, "settle_distance_clamp_m": 1.0},
            },
            "reproduction_choices": {
                "random_seed": 20260823,
                "spatial_penalty_acceptance_threshold": 1.0e-6,
                "physical_displacement_acceptance_threshold_m": 0.01,
                "perturbation_sample_count": 8,
                "stability_perturbation_std": {
                    "position_m": 0.005,
                    "rotation_rad": 0.034906585,
                    "center_of_mass_m": 0.002,
                    "friction": 0.05,
                    "mass_kg": 0.05,
                },
                "generation_round_limit": 2,
                "per_predicate_time_limit_s": 2.0,
                "local_geometry_backend": "quasistatic",
                "default_root_size_m": [1.6, 0.9, 0.08],
                "default_root_height_m": 0.75,
                "relation_weight": 1000.0,
                "collision_weight": 10000.0,
                "boundary_weight": 10000.0,
            },
            "acceptance_gates": {
                "dense_container": {
                    "minimum_success_rate": 1.0,
                    "minimum_objects": 20,
                    "fraction_runs_meeting_object_count": 1.0,
                    "maximum_mean_settle_distance_m": 0.01,
                    "minimum_packing_fraction": 0.30,
                    "minimum_packing_layers": 2,
                },
                "organized_container": {
                    "minimum_success_rate": 1.0,
                    "minimum_objects": 24,
                    "fraction_runs_meeting_object_count": 1.0,
                    "maximum_mean_settle_distance_m": 0.01,
                    "minimum_floor_coverage": 0.78,
                    "minimum_floor_compactness": 0.78,
                    "minimum_bottom_layer_item_fraction": 0.45,
                    "maximum_load_bearing_violation_count": 0,
                    "maximum_semantic_support_violation_count": 0,
                    "minimum_organization_score": 0.75,
                    "maximum_visual_contact_gap_m": 0.005,
                    "maximum_visual_contact_violation_count": 0,
                    "maximum_unresolved_visual_support_count": 0,
                }
            },
        }
    )


def make_dense_test_config() -> ReproductionConfig:
    config = make_test_config()
    return replace(
        config,
        physical=replace(config.physical, scene_resolution_m=0.01),
    )
