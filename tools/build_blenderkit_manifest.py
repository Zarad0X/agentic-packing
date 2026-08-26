#!/usr/bin/env python3
"""Build a credential-free BlenderKit provenance manifest from QA reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _index(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["asset_base_id"]): item for item in items}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--render-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    render = json.loads(args.render_report.read_text(encoding="utf-8"))
    audits = _index(audit["assets"])
    renders = _index(render["assets"])
    assets = []
    for selected in selection["assets"]:
        asset_base_id = selected["asset_base_id"]
        audited = audits[asset_base_id]
        rendered = renders[asset_base_id]
        if not audited.get("geometry_pass") or not rendered.get("render_pass"):
            raise SystemExit(f"selected asset has not passed QA: {asset_base_id}")
        metadata = json.loads(
            (args.metadata_dir / f"{asset_base_id}.json").read_text(encoding="utf-8")
        )
        author = metadata.get("author") or {}
        assets.append(
            {
                **selected,
                "name": metadata.get("name"),
                "author": author.get("fullName") or author.get("firstName") or "unknown",
                "license": metadata.get("license"),
                "source_url": metadata.get("source_url"),
                "sha256": metadata.get("sha256"),
                "size_bytes": metadata.get("size_bytes"),
                "geometry": {
                    key: audited[key]
                    for key in (
                        "mesh_count",
                        "node_count",
                        "primitive_count",
                        "triangle_count",
                        "material_count",
                        "image_count",
                        "dimensions",
                    )
                },
                "verification_status": metadata.get("verification_status"),
                "is_free": metadata.get("is_free"),
                "geometry_qa_status": "pass",
                "visual_qa_status": selection["visual_review"],
                "dense_scene_fit_status": "pending",
            }
        )
    output = {
        "schema_version": 1,
        "source": "BlenderKit",
        "distribution_policy": (
            "Raw royalty-free BlenderKit files stay in the user-authorized private cache. "
            "This manifest contains provenance and hashes only."
        ),
        "renderer": render["renderer"],
        "asset_count": len(assets),
        "assets": assets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"asset_count": len(assets), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
