#!/usr/bin/env python3
"""Audit downloaded BlenderKit GLBs without requiring Blender or GPU packages."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
TRIANGLES = 4


def _identity() -> list[float]:
    return [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _multiply(left: list[float], right: list[float]) -> list[float]:
    # glTF matrices are column-major.
    return [
        sum(left[row + k * 4] * right[k + column * 4] for k in range(4))
        for column in range(4)
        for row in range(4)
    ]


def _node_matrix(node: dict[str, Any]) -> list[float]:
    if "matrix" in node:
        return [float(value) for value in node["matrix"]]
    translation = [float(value) for value in node.get("translation", [0, 0, 0])]
    scale = [float(value) for value in node.get("scale", [1, 1, 1])]
    x, y, z, w = [float(value) for value in node.get("rotation", [0, 0, 0, 1])]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        (1 - 2 * (yy + zz)) * scale[0],
        (2 * (xy + wz)) * scale[0],
        (2 * (xz - wy)) * scale[0],
        0.0,
        (2 * (xy - wz)) * scale[1],
        (1 - 2 * (xx + zz)) * scale[1],
        (2 * (yz + wx)) * scale[1],
        0.0,
        (2 * (xz + wy)) * scale[2],
        (2 * (yz - wx)) * scale[2],
        (1 - 2 * (xx + yy)) * scale[2],
        0.0,
        translation[0],
        translation[1],
        translation[2],
        1.0,
    ]


def _transform(matrix: list[float], point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
        matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
        matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
    )


def _read_glb(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError("file too small")
    magic, version, declared_length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2 or declared_length != len(data):
        raise ValueError("invalid GLB header")
    offset = 12
    chunks: list[tuple[int, bytes]] = []
    while offset < len(data):
        length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunks.append((chunk_type, data[offset : offset + length]))
        offset += length
    json_chunks = [payload for chunk_type, payload in chunks if chunk_type == JSON_CHUNK]
    if len(json_chunks) != 1:
        raise ValueError("GLB must contain exactly one JSON chunk")
    return json.loads(json_chunks[0].rstrip(b" \t\r\n\0"))


def _material_transparent(material: dict[str, Any]) -> bool:
    pbr = material.get("pbrMetallicRoughness") or {}
    alpha = float((pbr.get("baseColorFactor") or [1, 1, 1, 1])[3])
    extensions = material.get("extensions") or {}
    transmission = extensions.get("KHR_materials_transmission") or {}
    transmission_factor = float(transmission.get("transmissionFactor") or 0.0)
    return material.get("alphaMode", "OPAQUE") != "OPAQUE" or alpha < 0.999 or transmission_factor > 0


def audit(path: Path, metadata_dir: Path) -> dict[str, Any]:
    document = _read_glb(path)
    nodes = document.get("nodes", [])
    meshes = document.get("meshes", [])
    accessors = document.get("accessors", [])
    materials = document.get("materials", [])
    images = document.get("images", [])
    primitive_count = 0
    triangle_count = 0
    missing_position_bounds = 0
    bounds_min = [math.inf, math.inf, math.inf]
    bounds_max = [-math.inf, -math.inf, -math.inf]

    scene_index = int(document.get("scene", 0))
    scenes = document.get("scenes", [])
    roots = scenes[scene_index].get("nodes", []) if scenes else list(range(len(nodes)))

    def visit(node_index: int, parent: list[float]) -> None:
        nonlocal primitive_count, triangle_count, missing_position_bounds
        node = nodes[node_index]
        world = _multiply(parent, _node_matrix(node))
        if "mesh" in node:
            for primitive in meshes[int(node["mesh"])].get("primitives", []):
                primitive_count += 1
                if int(primitive.get("mode", TRIANGLES)) == TRIANGLES and "indices" in primitive:
                    triangle_count += int(accessors[int(primitive["indices"])].get("count", 0)) // 3
                position_index = (primitive.get("attributes") or {}).get("POSITION")
                if position_index is None:
                    continue
                accessor = accessors[int(position_index)]
                if "min" not in accessor or "max" not in accessor:
                    missing_position_bounds += 1
                    continue
                lower = [float(value) for value in accessor["min"]]
                upper = [float(value) for value in accessor["max"]]
                for x in (lower[0], upper[0]):
                    for y in (lower[1], upper[1]):
                        for z in (lower[2], upper[2]):
                            point = _transform(world, (x, y, z))
                            for axis, value in enumerate(point):
                                bounds_min[axis] = min(bounds_min[axis], value)
                                bounds_max[axis] = max(bounds_max[axis], value)
        for child in node.get("children", []):
            visit(int(child), world)

    for root in roots:
        visit(int(root), _identity())

    dimensions = [
        bounds_max[axis] - bounds_min[axis] if math.isfinite(bounds_min[axis]) else None
        for axis in range(3)
    ]
    external_images = [image.get("uri") for image in images if image.get("uri") and not str(image["uri"]).startswith("data:")]
    transparent_materials = [index for index, material in enumerate(materials) if _material_transparent(material)]
    errors = []
    if not primitive_count:
        errors.append("no_geometry")
    if primitive_count > 64:
        errors.append("too_many_primitives")
    if len(nodes) > 256:
        errors.append("too_many_nodes")
    if triangle_count > 1_000_000:
        errors.append("triangle_budget")
    if missing_position_bounds:
        errors.append("missing_position_bounds")
    if any(value is None or value <= 0 for value in dimensions):
        errors.append("invalid_bounds")
    if external_images:
        errors.append("external_textures")
    if transparent_materials:
        errors.append("transparent_material")

    asset_base_id = path.name.split("--", 1)[0]
    metadata_path = metadata_dir / f"{asset_base_id}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "asset_base_id": asset_base_id,
        "name": metadata.get("name"),
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": metadata.get("sha256"),
        "license": metadata.get("license"),
        "mesh_count": len(meshes),
        "node_count": len(nodes),
        "primitive_count": primitive_count,
        "triangle_count": triangle_count,
        "material_count": len(materials),
        "image_count": len(images),
        "external_images": external_images,
        "transparent_material_indices": transparent_materials,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "dimensions": dimensions,
        "errors": errors,
        "geometry_pass": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for path in sorted(args.raw_dir.glob("*.glb")):
        try:
            result = audit(path, args.metadata_dir)
        except Exception as exc:
            result = {
                "asset_base_id": path.name.split("--", 1)[0],
                "name": path.name,
                "path": str(path),
                "errors": [f"parse_error:{type(exc).__name__}:{exc}"],
                "geometry_pass": False,
            }
        results.append(result)
        status = "PASS" if result["geometry_pass"] else "FAIL"
        print(f"{status:4s} {result['asset_base_id']} {result['name']} {result['errors']}")
    report = {
        "schema_version": 1,
        "gate": {
            "maximum_primitives": 64,
            "maximum_nodes": 256,
            "maximum_triangles": 1_000_000,
            "requires_position_bounds": True,
            "requires_embedded_textures": True,
            "requires_opaque_materials": True,
        },
        "passed": sum(result["geometry_pass"] for result in results),
        "failed": sum(not result["geometry_pass"] for result in results),
        "assets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "failed": report["failed"]}, indent=2))


if __name__ == "__main__":
    main()
