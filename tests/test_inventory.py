from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from physcensis.agent import DeterministicInventoryAgent
from physcensis.assets import PrimitiveAssetCatalog
from physcensis.inventory import (
    InventoryParser,
    InventoryPlanParser,
    InventoryPlanValidationError,
    InventoryValidationError,
)
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

    def test_plan_cannot_force_ordinary_cups_into_a_stack(self) -> None:
        payload = {
            "container": {"object_id": "basket", "category": "basket"},
            "objects": [
                {"object_id": "mug_a", "category": "cup"},
                {"object_id": "mug_b", "category": "cup"},
            ],
        }
        inventory = InventoryParser().parse(payload)
        catalog = PrimitiveAssetCatalog()
        assets = {
            spec.object_id: catalog.resolve(spec.object_id, spec.category)
            for spec in inventory.objects
        }
        plan = {
            "placement_order": ["mug_a", "mug_b"],
            "stack_groups": [
                {
                    "group_id": "unsafe_mugs",
                    "bottom_to_top_object_ids": ["mug_a", "mug_b"],
                }
            ],
            "adjacency_groups": [],
            "rationale": "unsafe on purpose",
        }

        with self.assertRaisesRegex(InventoryPlanValidationError, "marked stackable"):
            InventoryPlanParser().parse(plan, inventory, assets)

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

    def test_agent_loop_rejects_identity_change_then_replans(self) -> None:
        payload = {
            "container": {"object_id": "basket", "category": "basket"},
            "objects": [
                {"object_id": "can_a", "category": "can"},
                {"object_id": "can_b", "category": "can"},
            ],
        }

        class RepairingAgent:
            def __init__(self):
                self.calls = 0
                self.last_call_metadata = {"provider": "test"}
                self.fallback = DeterministicInventoryAgent()

            def propose_inventory(
                self, prompt, inventory_context, *, previous_plan=None, feedback=None
            ):
                self.calls += 1
                if self.calls == 1:
                    return {
                        "placement_order": ["can_a", "invented_can"],
                        "stack_groups": [],
                        "adjacency_groups": [],
                        "rationale": "invalid on purpose",
                    }
                self.assert_feedback = feedback
                return self.fallback.propose_inventory(prompt, inventory_context)

        agent = RepairingAgent()
        result = ScenePipeline(make_dense_test_config(), QuasiStaticBackend()).generate_inventory(
            "keep both cans",
            payload,
            agent,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.agent_trace[0]["status"], "plan_rejected")
        self.assertEqual(agent.assert_feedback.category, "grammar_error")
        self.assertEqual(
            set(result.scene.metadata["inventory_input_object_ids"]), {"can_a", "can_b"}
        )

    def test_agent_plan_realizes_repeated_dish_stacks_and_trace(self) -> None:
        payload = json.loads(Path("examples/inventory_dish_sink.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            result = ScenePipeline(
                make_dense_test_config(), QuasiStaticBackend()
            ).generate_inventory(
                "Stack identical dishes and use all supplied objects.",
                payload,
                DeterministicInventoryAgent(),
                base_dir="examples",
                output_dir=directory,
            )

            self.assertTrue(result.success, [issue.message for issue in result.feedback.issues])
            self.assertEqual(result.feedback.measurements["semantic_stack_count"], 3.0)
            self.assertEqual(result.feedback.measurements["nested_object_count"], 12.0)
            self.assertEqual(len(result.scene.metadata["physical_placement_order"]), 20)
            self.assertTrue(Path(directory, "llm_trace.json").is_file())


if __name__ == "__main__":
    unittest.main()
