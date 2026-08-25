from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from physcensis.inventory import InventoryParser, InventoryValidationError
from physcensis.physics import QuasiStaticBackend
from physcensis.pipeline import ScenePipeline
from tests.helpers import make_dense_test_config


class InventoryTest(unittest.TestCase):
    def test_example_preserves_twenty_explicit_ids(self) -> None:
        payload = json.loads(Path("examples/inventory_tool_crate.json").read_text(encoding="utf-8"))
        expected_ids = [value["object_id"] for value in payload["objects"]]
        pipeline = ScenePipeline(make_dense_test_config(), QuasiStaticBackend())

        result = pipeline.run_inventory(payload, base_dir="examples")

        self.assertTrue(result.success, [issue.message for issue in result.feedback.issues])
        self.assertEqual(result.scene.metadata["inventory_input_object_ids"], expected_ids)
        self.assertEqual(set(result.scene.metadata["physical_placement_order"]), set(expected_ids))
        self.assertEqual(result.scene.metadata["inventory_object_count"], 20)
        self.assertEqual(len(result.scene.objects), 21)
        self.assertEqual(result.feedback.measurements["container_item_count"], 20.0)
        self.assertEqual(result.scene.metadata["arrangement_mode"], "fixed_inventory")
        self.assertEqual(result.scene.metadata["presentation_mode"], "dense_container")
        self.assertEqual(result.feedback.measurements["load_bearing_violation_count"], 0.0)
        self.assertEqual(result.feedback.measurements["semantic_support_violation_count"], 0.0)

    def test_duplicate_ids_are_rejected(self) -> None:
        payload = {
            "container": {"object_id": "crate", "category": "tool crate"},
            "objects": [
                {"object_id": "same", "category": "can"},
                {"object_id": "same", "category": "wrench"},
            ],
        }
        with self.assertRaisesRegex(InventoryValidationError, "must be unique"):
            InventoryParser().parse(payload)

    def test_inline_user_mesh_is_resolved_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "my_can.glb"
            mesh.write_bytes(b"user-owned-mesh")
            payload = {
                "container": {"object_id": "crate", "category": "tool crate"},
                "objects": [
                    {
                        "object_id": "my_can",
                        "category": "can",
                        "asset": {
                            "size_m": [0.07, 0.07, 0.11],
                            "mass_kg": 0.3,
                            "mesh_path": "my_can.glb",
                            "visual_shape": "can",
                        },
                    }
                ],
            }

            spec = InventoryParser().parse(payload, base_dir=root)
            asset = spec.objects[0].asset

            assert asset is not None
            self.assertEqual(asset.mesh_path, str(mesh.resolve()))
            self.assertEqual(asset.sha256, hashlib.sha256(b"user-owned-mesh").hexdigest())
            self.assertEqual(asset.source, "user_inventory")
            self.assertEqual(asset.license, "user-provided")

    def test_impossible_inventory_reports_every_unplaced_id(self) -> None:
        payload = {
            "container": {
                "object_id": "tiny_bin",
                "category": "custom bin",
                "asset": {
                    "size_m": [0.32, 0.32, 0.20],
                    "container_inner_size_m": [0.28, 0.28, 0.16],
                },
            },
            "objects": [
                {
                    "object_id": "too_large_a",
                    "category": "box",
                    "asset": {"size_m": [0.40, 0.40, 0.10]},
                },
                {
                    "object_id": "too_large_b",
                    "category": "box",
                    "asset": {"size_m": [0.35, 0.35, 0.10]},
                },
            ],
        }
        result = ScenePipeline(make_dense_test_config(), QuasiStaticBackend()).run_inventory(
            payload
        )

        self.assertFalse(result.success)
        issue = result.feedback.issues[0]
        self.assertEqual(issue.code, "inventory_no_complete_arrangement")
        self.assertEqual(
            set(issue.details["unplaced_object_ids"]),
            {"too_large_a", "too_large_b"},
        )
        self.assertEqual(set(result.scene.objects), {"tiny_bin"})


if __name__ == "__main__":
    unittest.main()
