from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from physcensis.physics import QuasiStaticBackend
from physcensis.pipeline import ScenePipeline
from tests.helpers import make_dense_test_config, make_test_config


class PipelineTest(unittest.TestCase):
    def test_dining_table_program_renders(self) -> None:
        config = make_test_config()
        pipeline = ScenePipeline(config, QuasiStaticBackend())
        payload = json.loads(Path("examples/dining_table.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            result = pipeline.run_payload(payload, output_dir=directory)
            self.assertTrue(result.success, [issue.message for issue in result.feedback.issues])
            self.assertEqual(len(result.scene.objects), 16)
            self.assertTrue((Path(directory) / "overview.svg").exists())
            self.assertTrue((Path(directory) / "scene.json").exists())

    def test_physical_showcase_places_stack_and_container_batch(self) -> None:
        config = make_test_config()
        pipeline = ScenePipeline(config, QuasiStaticBackend())
        payload = json.loads(Path("examples/physical_showcase.json").read_text(encoding="utf-8"))
        result = pipeline.run_payload(payload)
        self.assertTrue(result.success, [issue.message for issue in result.feedback.issues])
        self.assertEqual(result.scene.get("book_2").support_id, "book_1")
        self.assertGreaterEqual(len([key for key in result.scene.objects if key.startswith("can_")]), 6)

    def test_all_curated_demo_families_solve(self) -> None:
        pipeline = ScenePipeline(make_test_config(), QuasiStaticBackend())
        for filename in (
            "dining_table.json",
            "office_desk.json",
            "workbench.json",
            "coffee_table.json",
            "physical_showcase.json",
        ):
            payload = json.loads((Path("examples") / filename).read_text(encoding="utf-8"))
            result = pipeline.run_payload(payload)
            self.assertTrue(result.success, f"{filename}: {result.feedback.summary}")

    def test_dense_container_scenes_are_multilayer_and_high_count(self) -> None:
        pipeline = ScenePipeline(make_dense_test_config(), QuasiStaticBackend())
        expected_minimum = {
            "dense_grocery_basket.json": 23,
            "dense_kitchen_sink.json": 24,
        }
        for filename, minimum_objects in expected_minimum.items():
            payload = json.loads((Path("examples") / filename).read_text(encoding="utf-8"))
            result = pipeline.run_payload(payload)
            self.assertTrue(result.success, f"{filename}: {result.feedback.summary}")
            self.assertGreaterEqual(len(result.scene.objects), minimum_objects)
            self.assertGreaterEqual(
                result.feedback.measurements["packing_fraction"], 0.30
            )
            self.assertGreaterEqual(
                result.feedback.measurements["packing_layer_count"], 2
            )
            self.assertEqual(result.scene.metadata["presentation_mode"], "dense_container")
            if filename == "dense_kitchen_sink.json":
                self.assertGreaterEqual(
                    result.feedback.measurements["semantic_stack_count"], 4
                )
                self.assertGreaterEqual(
                    result.feedback.measurements["nested_object_count"], 14
                )
                self.assertEqual(
                    result.feedback.measurements["same_asset_stack_fraction"], 1.0
                )
                categories = {
                    stack["category"]
                    for stack in result.scene.metadata["semantic_stacks"]
                }
                self.assertEqual(categories, {"plate", "bowl", "stackable cup"})

    def test_nested_strategy_rejects_non_stackable_mugs(self) -> None:
        pipeline = ScenePipeline(make_dense_test_config(), QuasiStaticBackend())
        payload = [
            ["sink_0", "a kitchen sink"],
            ["sink_0", "PLACE-ON-BASE", "root", {"x": 0.0, "y": 0.0}],
            ["sink_0", "FACING-FRONT", "root", {}],
            [[["cup", 2]], "PLACE-IN", "sink_0", {"strategy": "nested"}],
        ]
        result = pipeline.run_payload(payload)
        self.assertFalse(result.success)
        self.assertIn("asset_not_stackable", {issue.code for issue in result.feedback.issues})

    def test_complex_dense_demo_families_solve(self) -> None:
        pipeline = ScenePipeline(make_dense_test_config(), QuasiStaticBackend())
        expected_minimum = {
            "dense_dishwashing_station.json": 31,
            "dense_tool_crate.json": 24,
            "dense_office_tote.json": 28,
        }
        results = {}
        for filename, minimum_objects in expected_minimum.items():
            payload = json.loads((Path("examples") / filename).read_text(encoding="utf-8"))
            result = pipeline.run_payload(payload)
            results[filename] = result
            self.assertTrue(result.success, f"{filename}: {result.feedback.summary}")
            self.assertGreaterEqual(len(result.scene.objects), minimum_objects)
            self.assertGreaterEqual(
                result.feedback.measurements["container_item_count"],
                minimum_objects - 1,
            )
            self.assertGreaterEqual(
                result.feedback.measurements["packing_layer_count"], 2
            )
            self.assertEqual(result.scene.metadata["presentation_mode"], "dense_container")
        dish_result = results["dense_dishwashing_station.json"]
        self.assertGreaterEqual(dish_result.feedback.measurements["semantic_stack_count"], 4)
        self.assertGreaterEqual(dish_result.feedback.measurements["nested_object_count"], 18)


if __name__ == "__main__":
    unittest.main()
