"""Probabilistic stability estimation from the paper's 11-D perturbations."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from physcensis.config import ReproductionConfig
from physcensis.physics import PhysicsBackend
from physcensis.types import SceneState, StabilityResult


class StabilityEstimator:
    def __init__(self, config: ReproductionConfig, backend: PhysicsBackend):
        self.config = config
        self.backend = backend

    def estimate(self, scene: SceneState, object_id: str) -> StabilityResult:
        cfg = self.config.stability
        sigmas = np.asarray(
            [cfg.position_std_m] * 3
            + [cfg.rotation_std_rad] * 3
            + [cfg.center_of_mass_std_m] * 3
            + [cfg.friction_std, cfg.mass_std_kg],
            dtype=float,
        )
        if np.any(sigmas <= 0):
            raise ValueError("All stability perturbation standard deviations must be positive")
        seed = self.config.random_seed + sum(ord(char) for char in object_id)
        rng = np.random.default_rng(seed)
        samples = rng.normal(size=(cfg.sample_count, 11)) * sigmas
        failures = np.zeros(cfg.sample_count, dtype=float)
        stable_indices: list[int] = []
        for index, sample in enumerate(samples):
            perturbed = scene.clone()
            obj = perturbed.get(object_id)
            obj.position_m = tuple(obj.position_m[i] + float(sample[i]) for i in range(3))
            obj.yaw_rad += float(sample[5])
            obj.asset = replace(
                obj.asset,
                com_shift_m=tuple(
                    obj.asset.com_shift_m[i] + float(sample[6 + i]) for i in range(3)
                ),
                friction=max(0.0, obj.asset.friction + float(sample[9])),
                mass_kg=max(1.0e-3, obj.asset.mass_kg + float(sample[10])),
            )
            result = self.backend.simulate(perturbed, self.config.physical.settle_steps)
            failed = (
                object_id in result.fallen_object_ids
                or result.displacement_m.get(object_id, math.inf)
                > self.config.physical.displacement_threshold_m
            )
            failures[index] = float(failed)
            if not failed:
                stable_indices.append(index)

        normalized_squared = np.sum((samples / sigmas) ** 2, axis=1)
        weights = np.exp(-0.5 * normalized_squared)
        local_failure = float(np.sum(weights * failures) / max(np.sum(weights), 1.0e-12))
        most_unstable = None
        if stable_indices:
            # The kernel failure estimate around each stable sample is used as the instability score.
            scores = []
            normalized = samples / sigmas
            for index in stable_indices:
                distances = np.sum((normalized - normalized[index]) ** 2, axis=1)
                local_weights = np.exp(-0.5 * distances)
                scores.append(float(np.sum(local_weights * failures) / max(np.sum(local_weights), 1.0e-12)))
            selected = stable_indices[int(np.argmax(scores))]
            most_unstable = tuple(float(value) for value in samples[selected])
        return StabilityResult(
            object_id=object_id,
            sample_count=cfg.sample_count,
            local_failure_probability=local_failure,
            stable_sample_fraction=1.0 - float(np.mean(failures)),
            most_unstable_stable_offset=most_unstable,
        )
