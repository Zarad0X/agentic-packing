#!/usr/bin/env python3
"""Stage an authenticated BlenderKit manifest for transfer between private hosts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _install(source: Path, destination: Path, hardlink: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if hardlink:
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy instead of hard-linking files.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for asset in manifest["assets"]:
        asset_base_id = asset["asset_base_id"]
        metadata_path = args.metadata_dir / f"{asset_base_id}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_path = Path(metadata["local_path"])
        preview_path = args.preview_dir / f"{asset_base_id}.png"
        _install(raw_path, args.output_dir / "raw" / raw_path.name, not args.copy)
        _install(metadata_path, args.output_dir / "metadata" / metadata_path.name, not args.copy)
        _install(preview_path, args.output_dir / "previews" / preview_path.name, not args.copy)
    shutil.copy2(args.manifest, args.output_dir / "manifest.json")
    print(
        json.dumps(
            {
                "asset_count": len(manifest["assets"]),
                "output_dir": str(args.output_dir),
                "mode": "copy" if args.copy else "hardlink",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
