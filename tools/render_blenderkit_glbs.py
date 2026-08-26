#!/usr/bin/env python3
"""Render audited BlenderKit GLBs in a neutral studio using Blender.

Invoke through Blender, for example:

    blender --background --python tools/render_blenderkit_glbs.py -- \
      --audit glb-audit.json --output-dir previews --report render-report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=420)
    parser.add_argument("--samples", type=int, default=24)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.images):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        raise ValueError("import produced no mesh bounds")
    return (
        Vector(tuple(min(point[axis] for point in points) for axis in range(3))),
        Vector(tuple(max(point[axis] for point in points) for axis in range(3))),
    )


def _studio(scene: bpy.types.Scene, center: Vector, extent: Vector) -> None:
    largest = max(extent)
    floor_size = max(largest * 4.0, 1.0)
    bpy.ops.mesh.primitive_plane_add(size=floor_size, location=(center.x, center.y, 0.0))
    floor = bpy.context.object
    floor.name = "QA_Floor"
    material = bpy.data.materials.new("QA_Floor_Material")
    material.diffuse_color = (0.18, 0.19, 0.21, 1.0)
    floor.data.materials.append(material)

    for name, location, energy, size in (
        ("QA_Key", (1.8, -2.1, 2.8), 900.0, 4.0),
        ("QA_Fill", (-2.4, -0.6, 1.8), 500.0, 3.0),
        ("QA_Rim", (0.5, 2.4, 2.2), 700.0, 2.5),
    ):
        bpy.ops.object.light_add(type="AREA")
        light = bpy.context.object
        light.name = name
        light.data.energy = energy * max(largest, 0.25) ** 2
        light.data.shape = "DISK"
        light.data.size = size * max(largest, 0.25)
        light.location = center + Vector(location) * max(largest, 0.25)
        _look_at(light, center)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    horizontal = max(extent.x, extent.y)
    vertical = max(extent.z, largest * 0.25)
    camera.location = center + Vector((1.6, -2.0, 1.25)) * max(horizontal, vertical)
    _look_at(camera, center + Vector((0.0, 0.0, extent.z * 0.04)))
    camera.data.lens = 58


def _render_asset(asset: dict, output_dir: Path, resolution: int, samples: int) -> dict:
    _clear()
    path = Path(asset["path"])
    bpy.ops.import_scene.gltf(filepath=str(path), merge_vertices=True)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    lower, upper = _bounds(meshes)
    shift = Vector((-(lower.x + upper.x) / 2, -(lower.y + upper.y) / 2, -lower.z))
    roots = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    for obj in roots:
        obj.location += shift
    bpy.context.view_layer.update()
    lower, upper = _bounds(meshes)
    center = (lower + upper) * 0.5
    extent = upper - lower

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.filepath = str(output_dir / f"{asset['asset_base_id']}.png")
    scene.world.color = (0.055, 0.06, 0.07)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    _studio(scene, center, extent)
    bpy.ops.render.render(write_still=True)
    return {
        "asset_base_id": asset["asset_base_id"],
        "name": asset["name"],
        "source_path": str(path),
        "preview_path": scene.render.filepath,
        "imported_meshes": len(meshes),
        "render_bounds": {"min": list(lower), "max": list(upper), "dimensions": list(extent)},
        "render_pass": True,
    }


def main() -> None:
    args = _arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    results = []
    for asset in audit["assets"]:
        if not asset.get("geometry_pass"):
            continue
        try:
            result = _render_asset(asset, args.output_dir, args.resolution, args.samples)
            print(f"PASS {asset['asset_base_id']} {asset['name']}")
        except Exception as exc:
            result = {
                "asset_base_id": asset["asset_base_id"],
                "name": asset.get("name"),
                "render_pass": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"FAIL {asset['asset_base_id']} {asset['name']} {result['error']}")
        results.append(result)
    report = {
        "schema_version": 1,
        "renderer": bpy.app.version_string,
        "engine": "BLENDER_EEVEE",
        "resolution": args.resolution,
        "passed": sum(result["render_pass"] for result in results),
        "failed": sum(not result["render_pass"] for result in results),
        "assets": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "failed": report["failed"]}, indent=2))


if __name__ == "__main__":
    main()
