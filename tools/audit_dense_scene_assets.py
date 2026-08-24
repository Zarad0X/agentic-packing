#!/usr/bin/env python3
"""Audit reviewed GLBs for opaque, undistorted, repeated dense-scene use."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

TARGET_SIZES_M = {
    "book": (0.25, 0.18, 0.035),
    "notebook": (0.24, 0.17, 0.025),
    "laptop": (0.34, 0.24, 0.025),
    "keyboard": (0.16, 0.44, 0.025),
    "mouse": (0.11, 0.065, 0.04),
    "phone": (0.15, 0.075, 0.009),
    "pen": (0.15, 0.012, 0.012),
    "pencil": (0.18, 0.01, 0.01),
    "drill": (0.27, 0.09, 0.22),
    "wrench": (0.25, 0.045, 0.02),
    "hammer": (0.30, 0.11, 0.04),
    "motor": (0.24, 0.18, 0.18),
    "saw": (0.38, 0.12, 0.05),
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

MIN_AXIS_FILL = 0.45
MIN_VOLUME_FILL = 0.20
MAX_GEOMETRIES = 64
MAX_CONNECTED_COMPONENTS = 256
MAX_REPEATED_FACES = 2_500_000
MAX_REPEATED_BYTES = 300 * 1024 * 1024


def _round_vec(values: np.ndarray, digits: int = 9) -> list[float]:
    return [round(float(value), digits) for value in values]


def _material_audit(scene: trimesh.Scene) -> dict[str, Any]:
    alpha_modes: set[str] = set()
    alpha_factor_min = 1.0
    alpha_texture_min = 255
    textured_materials = 0
    material_count = 0
    for geometry in scene.geometry.values():
        material = getattr(geometry.visual, "material", None)
        if material is None:
            continue
        material_count += 1
        alpha_mode = getattr(material, "alphaMode", None)
        alpha_modes.add("OPAQUE" if alpha_mode is None else str(alpha_mode).upper())
        base_color_factor = getattr(material, "baseColorFactor", None)
        if base_color_factor is not None and len(base_color_factor) >= 4:
            alpha_factor_min = min(alpha_factor_min, float(base_color_factor[3]) / 255.0)
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        if image is None:
            continue
        textured_materials += 1
        if "A" in image.getbands():
            alpha_texture_min = min(alpha_texture_min, int(image.getchannel("A").getextrema()[0]))
    opaque = (
        alpha_modes.issubset({"OPAQUE"}) and alpha_factor_min >= 0.999 and alpha_texture_min >= 250
    )
    return {
        "material_count": material_count,
        "textured_material_count": textured_materials,
        "alpha_modes": sorted(alpha_modes),
        "minimum_base_color_alpha": round(alpha_factor_min, 6),
        "minimum_texture_alpha": alpha_texture_min,
        "opaque": opaque,
    }


def _connected_components(mesh: trimesh.Trimesh) -> int:
    # GLBs commonly duplicate vertices at UV/material seams. Merge by spatial
    # position before counting so seams are not misclassified as loose parts.
    merged = mesh.copy()
    merged.merge_vertices(
        merge_tex=True,
        merge_norm=True,
        digits_vertex=6,
    )
    return int(merged.body_count)


def _best_uniform_fit(
    mesh: trimesh.Trimesh, target_size: tuple[float, float, float]
) -> dict[str, Any]:
    to_origin, obb_extents = trimesh.bounds.oriented_bounds(mesh, angle_digits=1, ordered=False)
    target = np.asarray(target_size, dtype=float)
    best: tuple[tuple[float, float], tuple[int, int, int], np.ndarray] | None = None
    for permutation in itertools.permutations(range(3)):
        extents = obb_extents[list(permutation)]
        scale = float(np.min(target / extents))
        fill = extents * scale / target
        score = (float(np.min(fill)), float(np.prod(fill)))
        if best is None or score > best[0]:
            best = (score, permutation, fill)
    assert best is not None
    _, permutation, _ = best

    rotation = np.asarray(to_origin[:3, :3], dtype=float)[list(permutation)]
    if np.linalg.det(rotation) < 0.0:
        rotation[2] *= -1.0
    transform = np.eye(4)
    transform[:3, :3] = rotation
    aligned = trimesh.transform_points(mesh.vertices, transform)
    bounds_min = np.min(aligned, axis=0)
    bounds_max = np.max(aligned, axis=0)
    extents = bounds_max - bounds_min
    scale = float(np.min(target / extents))
    visual_size = extents * scale
    fill = visual_size / target
    center = (bounds_min + bounds_max) / 2.0
    euler_deg = np.degrees(trimesh.transformations.euler_from_matrix(transform, axes="sxyz"))
    reconstructed = trimesh.transformations.euler_matrix(*np.radians(euler_deg), axes="sxyz")[
        :3, :3
    ]
    if not np.allclose(reconstructed, rotation, atol=1.0e-6):
        raise ValueError("Euler reconstruction mismatch")
    return {
        "source_up_axis": "z",
        "mesh_euler_deg": _round_vec(euler_deg, 6),
        "fit_bounds_min": _round_vec(bounds_min),
        "fit_bounds_max": _round_vec(bounds_max),
        "uniform_scale": round(scale, 12),
        "visual_size_m": _round_vec(visual_size),
        "mesh_offset_m": _round_vec(-center * scale),
        "axis_fill_ratio": _round_vec(fill, 6),
        "minimum_axis_fill": round(float(np.min(fill)), 6),
        "volume_fill_ratio": round(float(np.prod(fill)), 6),
    }


def audit_asset(candidate: dict[str, Any]) -> dict[str, Any]:
    category = candidate["category"]
    target = TARGET_SIZES_M[category]
    path = Path(candidate["local_path"])
    scene = trimesh.load(path, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene):
        raise TypeError(f"Expected a scene for {path}")
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"No triangle geometry in {path}")

    actual_faces = sum(len(geometry.faces) for geometry in scene.geometry.values())
    actual_vertices = sum(len(geometry.vertices) for geometry in scene.geometry.values())
    geometry_count = len(scene.geometry)
    component_count = _connected_components(mesh)
    material = _material_audit(scene)
    fit = _best_uniform_fit(mesh, target)
    instance_count = DENSE_INSTANCE_COUNTS[category]
    repeated_faces = actual_faces * instance_count
    repeated_bytes = path.stat().st_size * instance_count
    raw_bounds = np.asarray(scene.bounds, dtype=float)

    rejection_reasons = []
    if not np.isfinite(mesh.vertices).all():
        rejection_reasons.append("nonfinite_vertices")
    if actual_faces <= 0:
        rejection_reasons.append("no_faces")
    if geometry_count > MAX_GEOMETRIES:
        rejection_reasons.append("too_many_geometries")
    if component_count > MAX_CONNECTED_COMPONENTS:
        rejection_reasons.append("too_many_connected_components")
    if not material["opaque"]:
        rejection_reasons.append("transparent_material")
    if material["textured_material_count"] < 1:
        rejection_reasons.append("missing_embedded_base_color_texture")
    if fit["minimum_axis_fill"] < MIN_AXIS_FILL:
        rejection_reasons.append("poor_axis_fill")
    if fit["volume_fill_ratio"] < MIN_VOLUME_FILL:
        rejection_reasons.append("poor_volume_fill")
    if repeated_faces > MAX_REPEATED_FACES:
        rejection_reasons.append("repeated_face_budget")
    if repeated_bytes > MAX_REPEATED_BYTES:
        rejection_reasons.append("repeated_byte_budget")

    return {
        **candidate,
        "target_size_m": list(target),
        "actual_face_count": actual_faces,
        "actual_vertex_count": actual_vertices,
        "geometry_count": geometry_count,
        "connected_component_count": component_count,
        "actual_glb_size": path.stat().st_size,
        "raw_bounds_min": _round_vec(raw_bounds[0]),
        "raw_bounds_max": _round_vec(raw_bounds[1]),
        "dense_instance_count": instance_count,
        "repeated_face_count": repeated_faces,
        "repeated_glb_size": repeated_bytes,
        "material_audit": material,
        "uniform_fit": fit,
        "dense_scene_fit_status": "audit_pass" if not rejection_reasons else "rejected",
        "dense_scene_rejection_reasons": rejection_reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.selection.read_text(encoding="utf-8"))
    audited = [audit_asset(candidate) for candidate in source["assets"]]
    output = {
        "schema_version": 1,
        "gate": {
            "scale_mode": "uniform_fit",
            "minimum_axis_fill": MIN_AXIS_FILL,
            "minimum_volume_fill": MIN_VOLUME_FILL,
            "maximum_geometries": MAX_GEOMETRIES,
            "maximum_connected_components": MAX_CONNECTED_COMPONENTS,
            "maximum_repeated_faces": MAX_REPEATED_FACES,
            "maximum_repeated_glb_bytes": MAX_REPEATED_BYTES,
            "requires_opaque_materials": True,
            "requires_embedded_base_color_texture": True,
        },
        "assets": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    "category": asset["category"],
                    "uid": asset["uid"],
                    "status": asset["dense_scene_fit_status"],
                    "min_fill": asset["uniform_fit"]["minimum_axis_fill"],
                    "volume_fill": asset["uniform_fit"]["volume_fill_ratio"],
                    "components": asset["connected_component_count"],
                    "reasons": asset["dense_scene_rejection_reasons"],
                }
                for asset in audited
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
