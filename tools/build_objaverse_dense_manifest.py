#!/usr/bin/env python3
"""Merge a validated base manifest with dense-scene audited Objaverse assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _new_entry(asset: dict[str, Any], *, finalized_genesis: bool = False) -> dict[str, Any]:
    fit = asset["uniform_fit"]
    return {
        "category": asset["category"],
        "uid": asset["uid"],
        "title": asset["title"],
        "author": asset["author"],
        "license": asset["license"],
        "license_url": asset["license_url"],
        "source_url": asset["source_url"],
        "download_url": asset["download_url"],
        "filename": asset["filename"],
        "sha256": asset["sha256"],
        "source_up_axis": fit["source_up_axis"],
        "scale_mode": "uniform_fit",
        "mesh_euler_deg": fit["mesh_euler_deg"],
        "raw_bounds_min": asset["raw_bounds_min"],
        "raw_bounds_max": asset["raw_bounds_max"],
        "fit_bounds_min": fit["fit_bounds_min"],
        "fit_bounds_max": fit["fit_bounds_max"],
        "face_count": asset["actual_face_count"],
        "vertex_count": asset["actual_vertex_count"],
        "file_size_bytes": asset["actual_glb_size"],
        "quality_score": int(asset["quality"]["score"]),
        "is_transparent": False,
        "visual_qa_status": ("genesis_pass" if finalized_genesis else "thumbnail_pass"),
        "dense_scene_fit_status": ("genesis_pass" if finalized_genesis else "audit_pass"),
        "dense_instance_count": asset["dense_instance_count"],
        "repeated_face_count": asset["repeated_face_count"],
        "repeated_glb_size": asset["repeated_glb_size"],
        "connected_component_count": asset["connected_component_count"],
        "minimum_axis_fill": fit["minimum_axis_fill"],
        "volume_fill_ratio": fit["volume_fill_ratio"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--finalize-genesis-pass",
        action="store_true",
        help=(
            "Mark audit-pass assets as visually and physically accepted only after "
            "the representative dense scenes have passed final Genesis review"
        ),
    )
    args = parser.parse_args()

    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    new_assets = [
        _new_entry(asset, finalized_genesis=args.finalize_genesis_pass)
        for asset in audit["assets"]
        if asset["dense_scene_fit_status"] == "audit_pass"
    ]
    old_assets = []
    for asset in base["assets"]:
        old_assets.append(
            {
                **asset,
                "dense_scene_fit_status": asset.get("dense_scene_fit_status", "genesis_pass"),
            }
        )
    assets = sorted(old_assets + new_assets, key=lambda item: (item["category"], item["uid"]))
    manifest = {
        "schema_version": 1,
        "name": "objaverse_cc_by_v3",
        "description": (
            "Licensed opaque Objaverse++ score-3 visual meshes selected for repeated "
            "dense-container placement. Uniform-fit entries preserve aspect ratio; "
            "procedural proxies remain the physics authority."
        ),
        "license_allowlist": sorted({asset["license"] for asset in assets}),
        "quality_gate": {
            "minimum_score": 3,
            "require_opaque": True,
            "source_url": "https://huggingface.co/datasets/cindyxl/ObjaversePlusPlus",
            **(
                {
                    "required_visual_qa_status": "genesis_pass",
                    "required_dense_scene_fit_status": "genesis_pass",
                }
                if args.finalize_genesis_pass
                else {}
            ),
            "dense_scene_gate": {
                **audit["gate"],
                "requires_single_object": True,
                "requires_non_scene": True,
                "requires_non_figure": True,
                "requires_thumbnail_review": True,
                "requires_final_genesis_dense_scene_review": True,
            },
        },
        "assets": assets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "base_assets": len(old_assets),
                "new_audit_pass_assets": len(new_assets),
                "rejected_assets": len(audit["assets"]) - len(new_assets),
                "manifest_assets": len(assets),
                "categories": sorted({asset["category"] for asset in assets}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
