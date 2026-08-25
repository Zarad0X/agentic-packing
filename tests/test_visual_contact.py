from __future__ import annotations

import unittest

from physcensis.types import AssetRecord, SceneObject, SceneState
from physcensis.visual_contact import build_visual_contact_layout


def asset(asset_id: str, physical_height: float, visual_height: float) -> AssetRecord:
    return AssetRecord(
        asset_id=asset_id,
        description=asset_id,
        size_m=(0.10, 0.10, visual_height),
        collision_size_m=(0.10, 0.10, physical_height),
        visual_size_m=(0.10, 0.10, visual_height),
    )


class VisualContactLayoutTest(unittest.TestCase):
    def test_supported_visual_meets_shorter_visible_support_surface(self) -> None:
        scene = SceneState(root_size_m=(1.0, 1.0, 0.1), root_height_m=0.0)
        lower = SceneObject(
            "lower",
            asset("lower", physical_height=0.18, visual_height=0.156),
            position_m=(0.0, 0.0, 0.14),
        )
        upper = SceneObject(
            "upper",
            asset("upper", physical_height=0.05, visual_height=0.027),
            position_m=(0.0, 0.0, 0.255),
        )
        scene.add_object(lower)
        scene.add_object(upper)
        scene.metadata["container_supports"] = {"upper": ["lower"]}

        layout = build_visual_contact_layout(scene)

        self.assertAlmostEqual(layout.bottoms_m["lower"], lower.bottom_z)
        self.assertAlmostEqual(layout.bottoms_m["upper"], layout.tops_m["lower"])
        self.assertAlmostEqual(layout.correction_m["upper"], -0.024)
        self.assertAlmostEqual(
            layout.metrics()["maximum_visual_contact_gap_before_alignment_m"],
            0.024,
        )
        self.assertEqual(layout.metrics()["maximum_visual_contact_gap_m"], 0.0)

    def test_multiple_supports_use_highest_visible_surface(self) -> None:
        scene = SceneState(root_size_m=(1.0, 1.0, 0.1), root_height_m=0.0)
        low = SceneObject(
            "low", asset("low", 0.10, 0.08), position_m=(-0.1, 0.0, 0.10)
        )
        high = SceneObject(
            "high", asset("high", 0.14, 0.13), position_m=(0.1, 0.0, 0.12)
        )
        upper = SceneObject(
            "upper", asset("upper", 0.04, 0.03), position_m=(0.0, 0.0, 0.21)
        )
        for obj in (low, high, upper):
            scene.add_object(obj)
        scene.metadata["container_supports"] = {"upper": ["low", "high"]}

        layout = build_visual_contact_layout(scene)

        self.assertAlmostEqual(layout.bottoms_m["upper"], layout.tops_m["high"])
        self.assertGreater(layout.tops_m["high"], layout.tops_m["low"])

    def test_container_support_is_treated_as_floor_contact(self) -> None:
        container_asset = AssetRecord(
            asset_id="crate",
            description="crate",
            size_m=(0.5, 0.4, 0.2),
            container_inner_size_m=(0.46, 0.36, 0.17),
        )
        scene = SceneState(root_size_m=(1.0, 1.0, 0.1), root_height_m=0.0)
        crate = SceneObject("crate", container_asset, position_m=(0.0, 0.0, 0.15))
        item = SceneObject(
            "item", asset("item", 0.10, 0.07), position_m=(0.0, 0.0, 0.12)
        )
        scene.add_object(crate)
        scene.add_object(item)
        scene.metadata["container_supports"] = {"item": ["crate"]}

        layout = build_visual_contact_layout(scene)

        self.assertAlmostEqual(layout.bottoms_m["item"], item.bottom_z)
        self.assertEqual(layout.metrics()["visual_contact_count"], 0.0)

    def test_support_cycle_is_reported_without_recursing_forever(self) -> None:
        scene = SceneState(root_size_m=(1.0, 1.0, 0.1), root_height_m=0.0)
        first = SceneObject(
            "first", asset("first", 0.10, 0.08), position_m=(0.0, 0.0, 0.10)
        )
        second = SceneObject(
            "second", asset("second", 0.10, 0.08), position_m=(0.0, 0.0, 0.20)
        )
        scene.add_object(first)
        scene.add_object(second)
        scene.metadata["container_supports"] = {
            "first": ["second"],
            "second": ["first"],
        }

        layout = build_visual_contact_layout(scene)

        self.assertEqual(layout.unresolved_object_ids, ("first", "second"))
        self.assertEqual(layout.metrics()["unresolved_visual_support_count"], 2.0)
