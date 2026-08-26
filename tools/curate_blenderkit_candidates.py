#!/usr/bin/env python3
"""Search and rank free BlenderKit models for dense object-packing scenes.

This stage downloads metadata and public thumbnails only. Raw model downloads are
handled separately after visual review. BlenderKit credentials are therefore not
needed for curation and are never written into the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEARCH_URL = "https://www.blenderkit.com/api/v1/search/"
ALLOWED_LICENSES = {"royalty_free", "cc_zero"}
NEGATIVE_PHRASES = {
    "assortment",
    "bundle",
    "collection",
    "dinner set",
    "dining set",
    "coffee set",
    "kitchen scene",
    "pack of",
    "pile",
    "scene",
    "set of",
    "table setting",
}
QUALITY_WORDS = {"pbr", "photoreal", "photorealistic", "realistic", "scan", "scanned"}


@dataclass(frozen=True)
class Category:
    queries: tuple[str, ...]
    max_dimension: float
    max_objects: int = 12
    stackable: bool = False
    exclude_words: tuple[str, ...] = ()


# Deliberately biased toward rigid, everyday objects that can be packed densely.
# Flexible objects, open laptops, articulated tools, and multi-object sets are out
# of scope for this pass because they need stronger collision/semantic treatment.
CATEGORIES: dict[str, Category] = {
    "basket": Category(("plastic basket", "shopping basket"), 1.2, max_objects=20),
    "crate": Category(("plastic crate", "storage crate"), 1.2, max_objects=20),
    "storage_bin": Category(("plastic storage bin", "storage box"), 1.2, max_objects=20),
    "sink": Category(("kitchen sink", "utility sink"), 2.0, max_objects=25),
    "tray": Category(("serving tray", "plastic tray"), 1.2, max_objects=12),
    "plate": Category(("ceramic plate", "dinner plate"), 0.6, stackable=True),
    "bowl": Category(("ceramic bowl", "kitchen bowl"), 0.6, stackable=True),
    "handleless_cup": Category(
        ("handleless cup", "tea cup no handle", "tumbler cup"),
        0.4,
        stackable=True,
        exclude_words=("mug", "handle"),
    ),
    "mug": Category(("ceramic mug", "coffee mug"), 0.4),
    "tumbler": Category(("drinking tumbler", "plastic tumbler"), 0.4, stackable=True),
    "jar": Category(("food jar", "glass jar"), 0.6),
    "can": Category(("food can", "tin can"), 0.5),
    "bottle": Category(("plastic bottle", "drink bottle"), 0.7),
    "carton": Category(("milk carton", "juice carton"), 0.6),
    "food_box": Category(("cereal box", "food package box"), 0.8),
    "pan": Category(("frying pan", "sauce pan"), 0.8),
    "drill": Category(("cordless drill", "power drill"), 0.8, exclude_words=("bit set",)),
    "hammer": Category(("claw hammer", "hand hammer"), 0.8),
    "wrench": Category(("combination wrench", "open end wrench"), 0.7),
    "pliers": Category(("hand pliers", "combination pliers"), 0.6),
    "screwdriver": Category(("screwdriver tool",), 0.6, exclude_words=("set",)),
    "hand_saw": Category(("hand saw tool",), 1.0),
    "tape_measure": Category(("tape measure",), 0.4),
    "spray_can": Category(("spray can", "aerosol can"), 0.5),
    "book": Category(("closed book", "single book"), 0.7, exclude_words=("stack",)),
    "notebook": Category(("closed notebook", "single notebook"), 0.6, exclude_words=("stack",)),
    "keyboard": Category(("computer keyboard",), 0.8),
    "mouse": Category(("computer mouse",), 0.4),
    "phone": Category(("smartphone", "mobile phone"), 0.4),
    "pen": Category(("ballpoint pen",), 0.4, max_objects=8, exclude_words=("set",)),
    "pencil": Category(("single pencil",), 0.4, max_objects=8, exclude_words=("set",)),
    "marker": Category(("marker pen",), 0.4, max_objects=8, exclude_words=("set",)),
    "stapler": Category(("office stapler",), 0.4),
    "calculator": Category(("desk calculator",), 0.5),
    "closed_laptop": Category(("closed laptop",), 0.8, exclude_words=("open",)),
    "headphones": Category(("over ear headphones",), 0.6),
}

CATEGORY_MATCHES: dict[str, tuple[str, ...]] = {
    "basket": ("basket",),
    "crate": ("crate",),
    "storage_bin": ("storage bin", "storage box", "plastic container"),
    "sink": ("sink",),
    "tray": ("tray",),
    "plate": ("plate", "platter"),
    "bowl": ("bowl",),
    "handleless_cup": ("cup", "glass", "tumbler"),
    "mug": ("mug",),
    "tumbler": ("tumbler", "drinking glass", "cup"),
    "jar": ("jar",),
    "can": ("can", "tin"),
    "bottle": ("bottle",),
    "carton": ("carton",),
    "food_box": ("cereal box", "food box", "food package"),
    "pan": ("pan", "skillet"),
    "drill": ("drill",),
    "hammer": ("hammer",),
    "wrench": ("wrench", "spanner"),
    "pliers": ("plier",),
    "screwdriver": ("screwdriver",),
    "hand_saw": ("hand saw", "handsaw"),
    "tape_measure": ("tape measure", "measuring tape"),
    "spray_can": ("spray can", "aerosol"),
    "book": ("book",),
    "notebook": ("notebook",),
    "keyboard": ("keyboard",),
    "mouse": ("mouse",),
    "phone": ("phone", "smartphone"),
    "pen": ("pen",),
    "pencil": ("pencil",),
    "marker": ("marker",),
    "stapler": ("stapler",),
    "calculator": ("calculator",),
    "closed_laptop": ("laptop",),
    "headphones": ("headphone", "headset"),
}


def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": "agentic-packing-asset-curator/0.1"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=90) as response:
        return json.loads(response.read())


def _model_file(asset: dict[str, Any]) -> dict[str, Any] | None:
    files = [item for item in asset.get("files", []) if item.get("fileType") == "gltf"]
    return min(files, key=lambda item: int(item.get("fileUploadSize") or 0), default=None)


def _text(asset: dict[str, Any]) -> str:
    values = [asset.get("name", ""), asset.get("description", ""), *asset.get("tags", [])]
    return " ".join(str(value) for value in values).casefold()


def _identity_text(asset: dict[str, Any]) -> str:
    values = [asset.get("name", ""), *asset.get("tags", [])]
    return " ".join(str(value) for value in values).casefold()


def _reason(asset: dict[str, Any], category_name: str, category: Category) -> str | None:
    parameters = asset.get("dictParameters") or {}
    text = _text(asset)
    identity_text = _identity_text(asset)
    gltf = _model_file(asset)
    dimensions = [float(parameters.get(f"dimension{axis}") or 0) for axis in "XYZ"]
    object_count = int(parameters.get("objectCount") or 1)
    face_count = int(parameters.get("faceCountRender") or parameters.get("faceCount") or 0)

    if asset.get("assetType") != "model":
        return "not_model"
    if asset.get("verificationStatus") != "validated":
        return "not_validated"
    if not asset.get("isFree"):
        return "not_free"
    if asset.get("license") not in ALLOWED_LICENSES:
        return "license"
    if asset.get("canDownload") is False:
        return "not_downloadable"
    if gltf is None:
        return "no_glb"
    if int(gltf.get("fileUploadSize") or 0) > 80 * 1024 * 1024:
        return "glb_too_large"
    if object_count > category.max_objects:
        return "multi_object"
    if face_count > 1_000_000:
        return "dense_face_budget"
    if any(phrase in text for phrase in NEGATIVE_PHRASES):
        return "set_or_scene"
    if not any(word in identity_text for word in CATEGORY_MATCHES[category_name]):
        return "semantic_mismatch"
    if any(word.casefold() in text for word in category.exclude_words):
        return "category_exclusion"
    if all(dimensions) and max(dimensions) > category.max_dimension:
        return "implausible_scale"
    if parameters.get("modelStyle") not in (None, "realistic"):
        return "not_realistic"
    if parameters.get("rig") or parameters.get("animated") or parameters.get("simulation"):
        return "non_rigid"
    if not asset.get("thumbnailMiddleUrl"):
        return "no_thumbnail"
    return None


def _rating(asset: dict[str, Any], name: str) -> float:
    return float((asset.get("ratingsAverage") or {}).get(name) or 0.0)


def _count(asset: dict[str, Any], name: str) -> int:
    return int((asset.get("ratingsCount") or {}).get(name) or 0)


def _rank(asset: dict[str, Any]) -> float:
    parameters = asset.get("dictParameters") or {}
    gltf = _model_file(asset) or {}
    text_tokens = set(re.findall(r"[a-z]+", _text(asset)))
    size = int(gltf.get("fileUploadSize") or 0)
    texture_count = int(parameters.get("textureCount") or 0)
    texture_resolution = int(parameters.get("textureResolutionMax") or 0)
    face_count = int(parameters.get("faceCountRender") or parameters.get("faceCount") or 0)
    object_count = int(parameters.get("objectCount") or 1)

    return (
        min(float(asset.get("score") or 0.0), 300.0) * 0.12
        + _rating(asset, "quality") * 8.0
        + math.log1p(_count(asset, "quality")) * 5.0
        + math.log1p(_count(asset, "bookmarks")) * 6.0
        + min(texture_count, 8) * 3.0
        + min(math.log2(max(texture_resolution, 1)), 13.0) * 1.5
        + (12.0 if parameters.get("purePbr") else 0.0)
        + (8.0 if parameters.get("manifold") else 0.0)
        + (8.0 if QUALITY_WORDS & text_tokens else 0.0)
        + (6.0 if 50_000 <= size <= 30 * 1024 * 1024 else 0.0)
        + (5.0 if 200 <= face_count <= 500_000 else 0.0)
        - max(object_count - 4, 0) * 1.5
    )


def _summary(asset: dict[str, Any], category_name: str, stackable: bool) -> dict[str, Any]:
    parameters = asset.get("dictParameters") or {}
    gltf = _model_file(asset) or {}
    author = asset.get("author") or {}
    return {
        "category": category_name,
        "asset_base_id": asset.get("assetBaseId"),
        "asset_id": asset.get("id"),
        "name": asset.get("name"),
        "author": author.get("fullName") or author.get("firstName") or "unknown",
        "license": asset.get("license"),
        "source_url": asset.get("url"),
        "thumbnail_url": asset.get("thumbnailMiddleUrlNonsquared")
        or asset.get("thumbnailMiddleUrl"),
        "rank": round(_rank(asset), 4),
        "stackable_candidate": stackable,
        "dimensions_m": [parameters.get(f"dimension{axis}") for axis in "XYZ"],
        "face_count": int(parameters.get("faceCountRender") or parameters.get("faceCount") or 0),
        "object_count": int(parameters.get("objectCount") or 1),
        "texture_count": int(parameters.get("textureCount") or 0),
        "texture_resolution_max": int(parameters.get("textureResolutionMax") or 0),
        "model_style": parameters.get("modelStyle"),
        "pure_pbr": bool(parameters.get("purePbr")),
        "manifold": parameters.get("manifold"),
        "gltf_file_id": gltf.get("id"),
        "gltf_size": int(gltf.get("fileUploadSize") or 0),
        "tags": asset.get("tags") or [],
    }


def curate(per_query: int, limit: int, selected: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": 1,
        "source": "BlenderKit",
        "download_policy": "metadata_and_public_thumbnails_only",
        "gate": {
            "free_only": True,
            "verification_status": "validated",
            "licenses": sorted(ALLOWED_LICENSES),
            "requires_glb": True,
            "maximum_glb_mib": 80,
            "maximum_face_count": 1_000_000,
            "rejects_sets_scenes_nonrigid_and_nonrealistic": True,
        },
        "categories": {},
    }
    for category_name in selected:
        spec = CATEGORIES[category_name]
        unique: dict[str, dict[str, Any]] = {}
        rejected: dict[str, int] = {}
        searched = 0
        for query in spec.queries:
            data = _request_json(
                SEARCH_URL,
                {
                    "query": f"asset_type:model {query}",
                    "dict_parameters": 1,
                    "page_size": per_query,
                },
            )
            for asset in data.get("results", []):
                searched += 1
                reason = _reason(asset, category_name, spec)
                if reason:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                unique[str(asset["assetBaseId"])] = asset
            time.sleep(0.08)

        ranked = sorted(unique.values(), key=lambda asset: (-_rank(asset), str(asset["assetBaseId"])))
        output["categories"][category_name] = {
            "queries": list(spec.queries),
            "searched": searched,
            "accepted": len(ranked),
            "rejected": rejected,
            "candidates": [
                _summary(asset, category_name, spec.stackable) for asset in ranked[:limit]
            ],
        }
        print(
            f"{category_name:16s} searched={searched:3d} "
            f"accepted={len(ranked):3d} shortlisted={min(len(ranked), limit):2d}"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-query", type=int, default=30)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--categories",
        nargs="*",
        choices=sorted(CATEGORIES),
        default=sorted(CATEGORIES),
        help="Category subset; defaults to every supported category.",
    )
    args = parser.parse_args()
    if args.per_query <= 0 or args.limit <= 0:
        parser.error("--per-query and --limit must be positive")

    result = curate(args.per_query, args.limit, args.categories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
