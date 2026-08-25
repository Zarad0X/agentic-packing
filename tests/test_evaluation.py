from __future__ import annotations

import unittest

from physcensis.evaluation import run_core_gate, run_dense_gate, run_organized_gate
from physcensis.physics import QuasiStaticBackend
from tests.helpers import make_dense_test_config, make_test_config


class EvaluationTest(unittest.TestCase):
    def test_curated_core_gate_passes_and_labels_geometry_evidence(self) -> None:
        report = run_core_gate(
            make_test_config(),
            QuasiStaticBackend(),
            repetitions_per_family=2,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["total_runs"], 10)
        self.assertEqual(report["evidence_level"], "geometry_only")

    def test_dense_gate_requires_count_fill_and_layers(self) -> None:
        report = run_dense_gate(
            make_dense_test_config(),
            QuasiStaticBackend(),
            repetitions_per_family=1,
        )
        self.assertTrue(report["passed"], report["failed_runs"])
        self.assertEqual(report["total_runs"], 2)
        self.assertGreaterEqual(report["minimum_packing_fraction"], 0.30)
        self.assertGreaterEqual(report["minimum_packing_layers"], 2)

    def test_organized_gate_requires_floor_use_and_safe_load_order(self) -> None:
        report = run_organized_gate(
            make_dense_test_config(),
            QuasiStaticBackend(),
            repetitions_per_family=1,
        )
        self.assertTrue(report["passed"], report["failed_runs"])
        self.assertEqual(report["total_runs"], 2)
        self.assertGreaterEqual(report["minimum_floor_coverage"], 0.78)
        self.assertGreaterEqual(report["minimum_floor_compactness"], 0.78)
        self.assertEqual(report["maximum_load_bearing_violation_count"], 0.0)
        self.assertEqual(report["maximum_semantic_support_violation_count"], 0.0)
        self.assertGreaterEqual(report["minimum_organization_score"], 0.75)
        self.assertLessEqual(report["maximum_visual_contact_gap_m"], 0.005)
        self.assertEqual(report["maximum_visual_contact_violation_count"], 0.0)
        self.assertEqual(report["maximum_unresolved_visual_support_count"], 0.0)
