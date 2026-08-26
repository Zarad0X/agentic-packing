#!/usr/bin/env python3
"""Download one user-authorized BlenderKit model and write provenance metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SEARCH_URL = "https://www.blenderkit.com/api/v1/search/"


def request_json(url: str, token: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "User-Agent": "agentic-packing/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return cleaned[:80] or "asset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-base-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()

    credentials = json.loads(args.credentials.read_text(encoding="utf-8"))
    token = credentials["access_token"]
    search = request_json(
        SEARCH_URL,
        token,
        {
            "query": f"asset_base_id:{args.asset_base_id}",
            "addon_version": "3.21.1",
            "dict_parameters": 1,
            "page_size": 1,
        },
    )
    if not search.get("results"):
        raise SystemExit(f"asset not found: {args.asset_base_id}")
    asset = search["results"][0]
    if asset.get("assetType") != "model":
        raise SystemExit(f"not a model: {asset.get('assetType')}")
    if asset.get("verificationStatus") != "validated":
        raise SystemExit(f"not validated: {asset.get('verificationStatus')}")
    if not asset.get("isFree") and not args.allow_paid:
        raise SystemExit("paid asset rejected; pass --allow-paid only after explicit authorization")

    preferred_types = ("gltf", "blend", "resolution_2K", "resolution_1K")
    files = [item for item in asset.get("files", []) if item.get("fileType") in preferred_types]
    files.sort(key=lambda item: preferred_types.index(item["fileType"]))
    if not files:
        raise SystemExit("no supported downloadable file")
    selected = files[0]
    resolution = request_json(
        selected["downloadUrl"], token, {"scene_uuid": str(uuid.uuid4())}
    )
    file_url = resolution.get("filePath") or resolution.get("download_url") or resolution.get("url")
    if not file_url:
        raise SystemExit("download endpoint returned no file URL")

    extension = ".glb" if selected["fileType"] == "gltf" else ".blend"
    filename = f"{args.asset_base_id}--{safe_slug(asset.get('name', 'asset'))}{extension}"
    raw_dir = args.root / "raw"
    metadata_dir = args.root / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / filename
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        file_url,
        headers={"Authorization": f"Bearer {token}", "User-Agent": "agentic-packing/0.1"},
    )
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    temporary.replace(destination)

    metadata = {
        "source": "blenderkit",
        "asset_base_id": asset.get("assetBaseId"),
        "asset_id": asset.get("id"),
        "name": asset.get("name"),
        "description": asset.get("description"),
        "asset_type": asset.get("assetType"),
        "verification_status": asset.get("verificationStatus"),
        "is_free": asset.get("isFree"),
        "license": asset.get("license"),
        "author": asset.get("author"),
        "source_url": asset.get("url"),
        "selected_file_type": selected.get("fileType"),
        "source_file_id": selected.get("id"),
        "source_file_size": selected.get("fileUploadSize"),
        "local_path": str(destination),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "raw_asset": asset,
    }
    metadata_path = metadata_dir / f"{args.asset_base_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.chmod(metadata_path, 0o640)
    print(
        json.dumps(
            {
                "downloaded": True,
                "asset_base_id": args.asset_base_id,
                "name": asset.get("name"),
                "license": asset.get("license"),
                "is_free": asset.get("isFree"),
                "file_type": selected.get("fileType"),
                "size_bytes": size,
                "sha256": digest.hexdigest(),
                "path": str(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
