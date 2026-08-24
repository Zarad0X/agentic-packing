from __future__ import annotations

import unittest

from physcensis.evaluation import run_core_gate, run_dense_gate
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
