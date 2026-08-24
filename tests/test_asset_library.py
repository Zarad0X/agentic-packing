from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from physcensis.asset_library import (
    AssetManifestError,
    LicensedAssetManifest,
    ManifestAssetCatalog,
)


def _manifest(license_name: str = "CC-BY-4.0") -> dict:
    payload = b"licensed-mesh"
    return {
        "schema_version": 1,
        "name": "test-assets",
        "assets": [
            {
                "category": "cup",
                "uid": "a" * 32,
                "title": "Test cup",
                "author": "Test author",
                "license": license_name,
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "source_url": f"https://sketchfab.com/3d-models/{'a' * 32}",
                "download_url": (
                    "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"
                    f"glbs/000-000/{'a' * 32}.glb"
                ),
                "filename": f"{'a' * 32}.glb",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_up_axis": "y",
                "raw_bounds_min": [-1.0, 0.0, -1.0],
                "raw_bounds_max": [1.0, 4.0, 1.0],
                "face_count": 100,
                "vertex_count": 60,
                "stackable": True,
                "stacking_step_ratio": 0.45,
            }
        ],
    }


class AssetLibraryTest(unittest.TestCase):
    def test_manifest_catalog_requires_hash_and_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
            cache = root / "cache"
            cache.mkdir()
            mesh = cache / f"{'a' * 32}.glb"
            mesh.write_bytes(b"licensed-mesh")

            catalog = ManifestAssetCatalog.load(manifest_path, cache)
            asset = catalog.resolve("cup_0", "ceramic cup")

            self.assertEqual(asset.source, "Objaverse 1.0 / Sketchfab")
            self.assertEqual(asset.license, "CC-BY-4.0")
            self.assertEqual(asset.author, "Test author")
            self.assertEqual(asset.mesh_path, str(mesh.resolve()))
            self.assertAlmostEqual(asset.mesh_scale[0], 0.085 / 2.0)
            self.assertAlmostEqual(asset.mesh_scale[1], 0.10 / 4.0)
            self.assertAlmostEqual(asset.mesh_scale[2], 0.085 / 2.0)
            self.assertTrue(asset.stackable)
            self.assertEqual(asset.stacking_step_ratio, 0.45)

            mesh.write_bytes(b"tampered")
            with self.assertRaisesRegex(AssetManifestError, "SHA-256 mismatch"):
                catalog.manifest.validate_files(cache)

    def test_manifest_rejects_license_outside_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(_manifest("CC-BY-NC-4.0")), encoding="utf-8")
            with self.assertRaisesRegex(AssetManifestError, "disallowed license"):
                LicensedAssetManifest.load(path)

    def test_manifest_enforces_quality_and_opacity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = _manifest()
            manifest["quality_gate"] = {
                "minimum_score": 3,
                "require_opaque": True,
                "source_url": (
                    "https://huggingface.co/datasets/cindyxl/ObjaversePlusPlus"
                ),
            }
            manifest["assets"][0]["quality_score"] = 2
            manifest["assets"][0]["is_transparent"] = False
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(AssetManifestError, "below the required 3"):
                LicensedAssetManifest.load(path)

            manifest["assets"][0]["quality_score"] = 3
            manifest["assets"][0]["is_transparent"] = True
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(AssetManifestError, "opaque-only gate"):
                LicensedAssetManifest.load(path)

            manifest["assets"][0]["is_transparent"] = False
            manifest["assets"][0]["visual_qa_status"] = "thumbnail_pass"
            manifest["quality_gate"]["required_visual_qa_status"] = "genesis_pass"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(AssetManifestError, "Genesis visual QA"):
                LicensedAssetManifest.load(path)


if __name__ == "__main__":
    unittest.main()
