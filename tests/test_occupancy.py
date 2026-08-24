from __future__ import annotations

import unittest

from physcensis.assets import PrimitiveAssetCatalog
from physcensis.occupancy import container_candidates
from physcensis.types import SceneObject, SceneState


class ContainerOccupancyTest(unittest.TestCase):
    def test_candidates_include_supported_second_layer(self) -> None:
        catalog = PrimitiveAssetCatalog()
        scene = SceneState((1.6, 0.9, 0.08), 0.75)
        basket = SceneObject(
            "grocery_basket_0",
            catalog.resolve("grocery_basket_0", "a large grocery basket"),
            position_m=(0.0, 0.0, scene.root_top_z + 0.15),
            fixed=True,
        )
        scene.add_object(basket)
        lower = SceneObject(
            "pantry_box_0",
            catalog.resolve_category("pantry_box", "pantry_box_0"),
            position_m=(0.0, 0.0, basket.bottom_z + 0.0225 + 0.0575),
            support_id=basket.object_id,
        )
        scene.add_object(lower)
        upper = SceneObject(
            "pantry_box_1",
            catalog.resolve_category("pantry_box", "pantry_box_1"),
        )
        candidates = container_candidates(scene, upper, basket, 0.02)
        self.assertTrue(any(candidate.layer_index > 0 for candidate in candidates))
        self.assertTrue(
            all(candidate.support_ratio >= 0.55 for candidate in candidates)
        )


if __name__ == "__main__":
    unittest.main()
