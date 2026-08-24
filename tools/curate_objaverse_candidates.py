#!/usr/bin/env python3
"""Build strict Objaverse++ candidate pools for the dense-scene demos."""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DEFAULT_CATEGORY_GROUPS = {
    "hammer": ("hammer",),
    "wrench": ("wrench",),
    "drill": ("drill",),
    "saw": ("handsaw",),
    "motor": ("motor",),
    "book": ("book", "hardback_book", "paperback_book", "comic_book"),
    "notebook": ("notebook",),
    "laptop": ("laptop_computer",),
    "keyboard": ("computer_keyboard",),
    "phone": ("cellular_telephone",),
    "mouse": ("mouse_(computer_equipment)",),
    "pen": ("pen",),
    "pencil": ("pencil",),
}


def _load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _strict_quality_rows(path: Path) -> dict[str, dict[str, Any]]:
    columns = (
        "UID",
        "style",
        "score",
        "is_multi_object",
        "is_scene",
        "is_figure",
        "is_transparent",
        "is_single_color",
        "density",
    )
    table = pq.read_table(path, columns=list(columns))
    result: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        if (
            row["score"] == 3
            and row["is_multi_object"] == "false"
            and row["is_scene"] == "false"
            and row["is_figure"] == "false"
            and row["is_transparent"] == "false"
        ):
            result[row["UID"]] = {key: value for key, value in row.items() if key != "UID"}
    return result


def _ordered_union(groups: Iterable[str], annotations: dict[str, list[str]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for uid in annotations.get(group, []):
            if uid not in seen:
                seen.add(uid)
                result.append(uid)
    return result


def build_pools(
    quality_path: Path,
    lvis_path: Path,
    object_paths_path: Path,
    limit: int,
) -> dict[str, Any]:
    quality = _strict_quality_rows(quality_path)
    lvis = _load_json_gz(lvis_path)
    object_paths = _load_json_gz(object_paths_path)

    pools: dict[str, Any] = {}
    for category, groups in DEFAULT_CATEGORY_GROUPS.items():
        candidates = []
        for uid in _ordered_union(groups, lvis):
            if uid not in quality or uid not in object_paths:
                continue
            candidates.append(
                {
                    "uid": uid,
                    "object_path": object_paths[uid],
                    "lvis_groups": [group for group in groups if uid in lvis.get(group, [])],
                    "quality": quality[uid],
                }
            )
            if len(candidates) >= limit:
                break
        pools[category] = {
            "lvis_groups": list(groups),
            "candidate_count": len(candidates),
            "candidates": candidates,
        }

    return {
        "schema_version": 1,
        "gate": {
            "score": 3,
            "is_transparent": False,
            "is_scene": False,
            "is_multi_object": False,
            "is_figure": False,
        },
        "categories": pools,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--lvis", type=Path, required=True)
    parser.add_argument("--object-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("--limit must be positive")
    result = build_pools(args.quality, args.lvis, args.object_paths, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {category: pool["candidate_count"] for category, pool in result["categories"].items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
