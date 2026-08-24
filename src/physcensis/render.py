"""Dependency-free scene export and isometric SVG renderer."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physcensis.types import AssetRecord, Feedback, SceneObject, SceneState


@dataclass(frozen=True)
class RenderArtifacts:
    scene_json: Path
    overview_svg: Path
    report_json: Path
    overview_png: Path | None = None


class SceneRenderer:
    width = 1200
    height = 760
    scale = 360.0

    def __init__(self, physics_backend: Any | None = None):
        self.physics_backend = physics_backend

    def render(self, scene: SceneState, output_dir: Path, feedback: Feedback) -> RenderArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        scene_json = output_dir / "scene.json"
        overview_svg = output_dir / "overview.svg"
        report_json = output_dir / "report.json"
        scene_json.write_text(json.dumps(scene.to_dict(), indent=2), encoding="utf-8")
        report_json.write_text(
            json.dumps(
                {
                    "category": feedback.category,
                    "summary": feedback.summary,
                    "measurements": dict(feedback.measurements),
                    "issues": [
                        {
                            "code": issue.code,
                            "message": issue.message,
                            "object_id": issue.object_id,
                            "predicate_index": issue.predicate_index,
                            "details": dict(issue.details),
                        }
                        for issue in feedback.issues
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        overview_svg.write_text(self._svg(scene), encoding="utf-8")
        overview_png = None
        render_rgb = getattr(self.physics_backend, "render_rgb", None)
        if callable(render_rgb):
            overview_png = render_rgb(scene, output_dir / "overview.png")
        return RenderArtifacts(scene_json, overview_svg, report_json, overview_png)

    def _svg(self, scene: SceneState) -> str:
        faces: list[tuple[float, str]] = []
        self._dense_render = scene.metadata.get("presentation_mode") == "dense_container"
        if not self._dense_render:
            root = SceneObject(
                "root",
                asset=self._root_asset(scene),
                position_m=(0.0, 0.0, scene.root_height_m),
                fixed=True,
            )
            faces.extend(self._box_faces(root, "#6b4f3a"))
        objects = sorted(
            scene.objects.values(),
            key=lambda obj: (obj.position_m[0] + obj.position_m[1], obj.position_m[2]),
        )
        for obj in objects:
            if obj.asset.container_inner_size_m is not None:
                faces.extend(self._container_faces(obj))
            else:
                faces.extend(self._box_faces(obj, self._asset_color(obj)))
        face_markup = "\n".join(markup for _, markup in sorted(faces, key=lambda item: item[0]))
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">
<defs><radialGradient id="bg" cx="50%" cy="44%" r="75%"><stop offset="0" stop-color="#14242d"/><stop offset="1" stop-color="#071119"/></radialGradient></defs>
<rect width="100%" height="100%" fill="url(#bg)"/>
<ellipse cx="600" cy="620" rx="480" ry="90" fill="#000000" opacity="0.28"/>
{face_markup}
</svg>'''

    @staticmethod
    def _root_asset(scene: SceneState):
        from physcensis.types import AssetRecord

        return AssetRecord("root", "supporting surface", scene.root_size_m, mass_kg=50.0)

    def _box_faces(self, obj: SceneObject, color: str) -> list[tuple[float, str]]:
        vertices = self._vertices(obj)
        face_indices = (
            (0, 1, 2, 3, 0.80),
            (4, 5, 6, 7, 1.08),
            (0, 1, 5, 4, 0.68),
            (1, 2, 6, 5, 0.88),
            (2, 3, 7, 6, 0.62),
            (3, 0, 4, 7, 0.76),
        )
        output = []
        for *indices, shade in face_indices:
            points_3d = [vertices[index] for index in indices]
            points_2d = [self._project(point) for point in points_3d]
            depth = sum(point[0] + point[1] + point[2] * 0.1 for point in points_3d) / 4.0
            fill = self._shade(color, shade)
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points_2d)
            output.append(
                (depth, f'<polygon points="{path}" fill="{fill}" stroke="#26323d" stroke-width="0.8"/>')
            )
        return output

    def _container_faces(self, obj: SceneObject) -> list[tuple[float, str]]:
        """Render open compound walls so packed objects remain visible."""
        inner = obj.asset.container_inner_size_m
        assert inner is not None
        sx, sy, sz = obj.asset.size_m
        x_wall = max(0.008, (sx - inner[0]) / 2.0)
        y_wall = max(0.008, (sy - inner[1]) / 2.0)
        bottom = max(0.008, (sz - inner[2]) / 2.0)
        color = self._asset_color(obj)
        parts = [
            self._container_part(obj, (0.0, 0.0, -sz / 2.0 + bottom / 2.0), (sx, sy, bottom)),
        ]
        if obj.asset.visual_shape == "grocery_basket":
            rail = 0.018
            for dz in (-sz * 0.42, sz * 0.42):
                parts.extend(
                    (
                        self._container_part(obj, (0.0, sy / 2.0, dz), (sx, rail, rail)),
                        self._container_part(obj, (0.0, -sy / 2.0, dz), (sx, rail, rail)),
                        self._container_part(obj, (sx / 2.0, 0.0, dz), (rail, sy, rail)),
                        self._container_part(obj, (-sx / 2.0, 0.0, dz), (rail, sy, rail)),
                    )
                )
            for index in range(-7, 8):
                dx = index * sx / 16.0
                parts.extend(
                    (
                        self._container_part(obj, (dx, sy / 2.0, 0.0), (0.009, rail, sz * 0.78)),
                        self._container_part(obj, (dx, -sy / 2.0, 0.0), (0.009, rail, sz * 0.78)),
                    )
                )
        else:
            parts.extend(
                (
                    self._container_part(
                        obj, (sx / 2.0 - x_wall / 2.0, 0.0, 0.0), (x_wall, sy, sz)
                    ),
                    self._container_part(
                        obj, (-sx / 2.0 + x_wall / 2.0, 0.0, 0.0), (x_wall, sy, sz)
                    ),
                    self._container_part(
                        obj, (0.0, sy / 2.0 - y_wall / 2.0, 0.0), (sx, y_wall, sz)
                    ),
                    self._container_part(
                        obj, (0.0, -sy / 2.0 + y_wall / 2.0, 0.0), (sx, y_wall, sz)
                    ),
                )
            )
        faces: list[tuple[float, str]] = []
        for part in parts:
            faces.extend(self._box_faces(part, color))
        return faces

    @staticmethod
    def _container_part(
        container: SceneObject,
        local_position: tuple[float, float, float],
        size_m: tuple[float, float, float],
    ) -> SceneObject:
        cosine, sine = math.cos(container.yaw_rad), math.sin(container.yaw_rad)
        dx, dy, dz = local_position
        return SceneObject(
            object_id=f"{container.object_id}:part",
            asset=AssetRecord(
                asset_id=f"{container.asset.asset_id}:part",
                description="container presentation part",
                size_m=size_m,
            ),
            position_m=(
                container.position_m[0] + cosine * dx - sine * dy,
                container.position_m[1] + sine * dx + cosine * dy,
                container.position_m[2] + dz,
            ),
            yaw_rad=container.yaw_rad,
            fixed=True,
        )

    @staticmethod
    def _vertices(obj: SceneObject) -> list[tuple[float, float, float]]:
        hx, hy, hz = (value / 2.0 for value in obj.asset.size_m)
        visual_position = obj.visual_position_m
        cosine, sine = math.cos(obj.yaw_rad), math.sin(obj.yaw_rad)
        vertices = []
        for z in (-hz, hz):
            for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
                vertices.append(
                    (
                        visual_position[0] + cosine * x - sine * y,
                        visual_position[1] + sine * x + cosine * y,
                        visual_position[2] + z,
                    )
                )
        return vertices

    def _project(self, point: tuple[float, float, float]) -> tuple[float, float]:
        x, y, z = point
        if getattr(self, "_dense_render", False):
            scale = 720.0
            return (
                self.width / 2.0 + (y - x) * scale * 0.72,
                self.height * 0.74 + (x + y) * scale * 0.38 - z * scale * 0.24,
            )
        return (
            self.width / 2.0 + (y - x) * self.scale * 0.72,
            self.height * 0.70 + (x + y) * self.scale * 0.32 - z * self.scale * 0.72,
        )

    @staticmethod
    def _color(key: str) -> str:
        digest = hashlib.sha256(key.split("_")[0].encode("utf-8")).hexdigest()
        hue = int(digest[:4], 16) % 360
        saturation = 48 + int(digest[4:6], 16) % 25
        lightness = 48 + int(digest[6:8], 16) % 18
        # Convert HSL to RGB without requiring a rendering dependency.
        import colorsys

        red, green, blue = colorsys.hls_to_rgb(hue / 360.0, lightness / 100.0, saturation / 100.0)
        return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"

    @staticmethod
    def _asset_color(obj: SceneObject) -> str:
        red, green, blue, _ = obj.asset.color_rgba
        return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"

    @staticmethod
    def _shade(color: str, factor: float) -> str:
        values = [int(color[index : index + 2], 16) for index in (1, 3, 5)]
        values = [max(0, min(255, int(value * factor))) for value in values]
        return "#" + "".join(f"{value:02x}" for value in values)
