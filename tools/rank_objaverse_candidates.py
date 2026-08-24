#!/usr/bin/env python3
"""Apply license, texture, thumbnail, and popularity gates to Objaverse pools."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

LICENSES = {
    "cc0": ("CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/"),
    "by": ("CC-BY-4.0", "https://creativecommons.org/licenses/by/4.0/"),
    "by-sa": ("CC-BY-SA-4.0", "https://creativecommons.org/licenses/by-sa/4.0/"),
}

DENSE_INSTANCE_COUNTS = {
    "hammer": 4,
    "wrench": 6,
    "drill": 3,
    "saw": 2,
    "motor": 2,
    "book": 6,
    "notebook": 6,
    "laptop": 2,
    "keyboard": 1,
    "phone": 3,
    "mouse": 3,
    "pen": 3,
    "pencil": 3,
}
MAX_REPEATED_FACE_BUDGET = 2_500_000
MAX_REPEATED_GLB_BYTES = 300 * 1024 * 1024


def _load_shard(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Invalid metadata shard: {path}")
    return data


def _thumbnail(metadata: dict[str, Any]) -> str | None:
    images = metadata.get("thumbnails", {}).get("images", [])
    valid = [image for image in images if image.get("url") and image.get("width")]
    if not valid:
        return None
    return min(valid, key=lambda image: (abs(int(image["width"]) - 720), -int(image["width"])))[
        "url"
    ]


def _rank(metadata: dict[str, Any], glb: dict[str, Any]) -> float:
    texture_count = int(glb.get("textureCount") or 0)
    texture_resolution = int(glb.get("textureMaxResolution") or 0)
    likes = int(metadata.get("likeCount") or 0)
    views = int(metadata.get("viewCount") or 0)
    faces = int(glb.get("faceCount") or metadata.get("faceCount") or 0)
    staffpicked = metadata.get("staffpickedAt") is not None
    face_bonus = 8.0 if 500 <= faces <= 500_000 else 0.0
    return (
        (100.0 if staffpicked else 0.0)
        + min(texture_count, 8) * 8.0
        + min(math.log2(max(texture_resolution, 1)), 13.0) * 2.0
        + math.log1p(likes) * 7.0
        + math.log1p(views) * 1.5
        + face_bonus
    )


def rank_candidates(
    candidate_path: Path,
    metadata_dir: Path,
    limit: int,
) -> dict[str, Any]:
    source = json.loads(candidate_path.read_text(encoding="utf-8"))
    shard_cache: dict[str, dict[str, Any]] = {}
    categories: dict[str, Any] = {}

    for category, pool in source["categories"].items():
        accepted = []
        rejection_counts: dict[str, int] = {}
        instance_count = DENSE_INSTANCE_COUNTS[category]
        for candidate in pool["candidates"]:
            shard = candidate["object_path"].split("/")[1]
            if shard not in shard_cache:
                shard_cache[shard] = _load_shard(metadata_dir / f"{shard}.json.gz")
            metadata = shard_cache[shard].get(candidate["uid"])
            if metadata is None:
                reason = "missing_metadata"
            elif metadata.get("license") not in LICENSES:
                reason = "license"
            elif not metadata.get("isDownloadable"):
                reason = "not_downloadable"
            elif metadata.get("isAgeRestricted"):
                reason = "age_restricted"
            elif not (thumbnail_url := _thumbnail(metadata)):
                reason = "thumbnail"
            elif not (glb := metadata.get("archives", {}).get("glb")):
                reason = "glb_metadata"
            elif int(glb.get("textureCount") or 0) < 1:
                reason = "no_embedded_texture"
            elif int(glb.get("size") or 0) > 80 * 1024 * 1024:
                reason = "glb_too_large"
            elif int(glb.get("faceCount") or 0) * instance_count > MAX_REPEATED_FACE_BUDGET:
                reason = "dense_face_budget"
            elif int(glb.get("size") or 0) * instance_count > MAX_REPEATED_GLB_BYTES:
                reason = "dense_memory_budget"
            else:
                license_name, license_url = LICENSES[metadata["license"]]
                user = metadata.get("user") or {}
                accepted.append(
                    {
                        **candidate,
                        "title": str(metadata.get("name") or candidate["uid"]),
                        "author": str(user.get("displayName") or user.get("username") or "unknown"),
                        "license": license_name,
                        "license_url": license_url,
                        "source_url": str(metadata.get("viewerUrl")),
                        "download_url": (
                            "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"
                            + candidate["object_path"]
                        ),
                        "thumbnail_url": thumbnail_url,
                        "staffpicked": metadata.get("staffpickedAt") is not None,
                        "like_count": int(metadata.get("likeCount") or 0),
                        "view_count": int(metadata.get("viewCount") or 0),
                        "face_count": int(glb.get("faceCount") or 0),
                        "vertex_count": int(glb.get("vertexCount") or 0),
                        "texture_count": int(glb.get("textureCount") or 0),
                        "texture_max_resolution": int(glb.get("textureMaxResolution") or 0),
                        "glb_size": int(glb.get("size") or 0),
                        "metadata_rank": _rank(metadata, glb),
                    }
                )
                continue
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        accepted.sort(key=lambda item: (-item["metadata_rank"], item["uid"]))
        categories[category] = {
            "input_count": len(pool["candidates"]),
            "accepted_count": len(accepted),
            "dense_instance_count": instance_count,
            "rejection_counts": rejection_counts,
            "candidates": accepted[:limit],
        }

    return {
        "schema_version": 1,
        "gate": {
            **source["gate"],
            "licenses": sorted(name for name, _ in LICENSES.values()),
            "minimum_embedded_texture_count": 1,
            "maximum_glb_size_mib": 80,
            "maximum_repeated_face_budget": MAX_REPEATED_FACE_BUDGET,
            "maximum_repeated_glb_mib": MAX_REPEATED_GLB_BYTES // (1024 * 1024),
            "requires_thumbnail": True,
            "requires_downloadable": True,
        },
        "categories": categories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    result = rank_candidates(args.candidates, args.metadata_dir, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                category: {
                    "accepted": pool["accepted_count"],
                    "shortlisted": len(pool["candidates"]),
                    "rejected": pool["rejection_counts"],
                }
                for category, pool in result["categories"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
