"""Physics backend contract plus deterministic and Genesis implementations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from physcensis.geometry import (
    object_polygon,
    objects_overlap_3d,
    point_in_convex_polygon,
)
from physcensis.types import SceneState, SimulationResult


class PhysicsBackend(Protocol):
    name: str

    def simulate(self, scene: SceneState, steps: int) -> SimulationResult:
        """Simulate one scene and return final poses and physical failures."""


@dataclass
class QuasiStaticBackend:
    """Fast support/penetration model for deterministic local development.

    It is deliberately labeled a geometry backend and is never accepted as
    evidence for the paper's real-physics completion gate.
    """

    displacement_threshold_m: float = 0.01
    name: str = "quasistatic"
    defer_incremental_validation: bool = False

    def simulate(self, scene: SceneState, steps: int) -> SimulationResult:
        del steps
        penetrations: list[tuple[str, str]] = []
        objects = list(scene.objects.values())
        for i, first in enumerate(objects):
            for second in objects[i + 1 :]:
                if self._container_pair(first.object_id, second.object_id, scene):
                    continue
                if objects_overlap_3d(first, second):
                    penetrations.append((first.object_id, second.object_id))

        final_positions = {obj.object_id: obj.position_m for obj in objects}
        displacement: dict[str, float] = {}
        fallen: list[str] = []
        for obj in objects:
            supported = self._is_supported(scene, obj.object_id)
            if supported:
                displacement[obj.object_id] = 0.0
                continue
            target_z = scene.root_top_z + obj.asset.size_m[2] / 2.0
            drop = max(0.0, obj.position_m[2] - target_z)
            displacement[obj.object_id] = drop
            final_positions[obj.object_id] = (obj.position_m[0], obj.position_m[1], target_z)
            if drop > self.displacement_threshold_m:
                fallen.append(obj.object_id)

        success = not penetrations and not fallen
        return SimulationResult(
            success=success,
            final_positions_m=final_positions,
            displacement_m=displacement,
            fallen_object_ids=tuple(fallen),
            penetrations=tuple(penetrations),
        )

    @staticmethod
    def _container_pair(first_id: str, second_id: str, scene: SceneState) -> bool:
        first = scene.get(first_id)
        second = scene.get(second_id)
        return (
            first.support_id == second_id and second.asset.container_inner_size_m is not None
        ) or (
            second.support_id == first_id and first.asset.container_inner_size_m is not None
        )

    @staticmethod
    def _is_supported(scene: SceneState, object_id: str) -> bool:
        obj = scene.get(object_id)
        if obj.fixed:
            return True
        if obj.support_id is None or obj.support_id == "root":
            root_polygon = (
                (scene.root_bounds_xy[0], scene.root_bounds_xy[2]),
                (scene.root_bounds_xy[1], scene.root_bounds_xy[2]),
                (scene.root_bounds_xy[1], scene.root_bounds_xy[3]),
                (scene.root_bounds_xy[0], scene.root_bounds_xy[3]),
            )
            com = (
                obj.position_m[0] + obj.asset.com_shift_m[0],
                obj.position_m[1] + obj.asset.com_shift_m[1],
            )
            expected_bottom = scene.root_top_z
            return point_in_convex_polygon(com, root_polygon) and abs(obj.bottom_z - expected_bottom) <= 0.015
        supporter = scene.get(obj.support_id)
        if supporter.asset.container_inner_size_m is not None:
            inner = supporter.asset.container_inner_size_m
            cosine = math.cos(supporter.yaw_rad)
            sine = math.sin(supporter.yaw_rad)
            local_corners = []
            for x, y in object_polygon(obj):
                dx = x - supporter.position_m[0]
                dy = y - supporter.position_m[1]
                local_corners.append(
                    (cosine * dx + sine * dy, -sine * dx + cosine * dy)
                )
            return all(
                abs(x) <= inner[0] / 2.0 + 1.0e-6
                and abs(y) <= inner[1] / 2.0 + 1.0e-6
                for x, y in local_corners
            )
        com = (
            obj.position_m[0] + obj.asset.com_shift_m[0],
            obj.position_m[1] + obj.asset.com_shift_m[1],
        )
        correct_height = abs(obj.bottom_z - supporter.top_z) <= 0.015
        return correct_height and point_in_convex_polygon(com, object_polygon(supporter))


class GenesisBackend:
    """Rigid-box Genesis backend matching the shared scene contract."""

    name = "genesis"
    # Genesis recompiles kernels when entity counts change. Candidate geometry
    # is filtered deterministically, then the complete scene is validated once.
    defer_incremental_validation = True

    def __init__(self, *, headless: bool = True):
        try:
            import genesis as gs
        except ImportError as exc:
            raise RuntimeError("Genesis is not installed; install the 'genesis' extra") from exc
        self._gs = gs
        self._headless = headless
        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.gpu)

    def simulate(self, scene: SceneState, steps: int) -> SimulationResult:
        simulation, entities = self._build_scene(scene)
        initial = {}
        for object_id, obj in scene.objects.items():
            initial[object_id] = obj.position_m
        simulation.build()
        for _ in range(steps):
            simulation.step()
        final_positions = {}
        displacement = {}
        fallen = []
        for object_id, (entity, offset) in entities.items():
            position_value = entity.get_pos()
            if hasattr(position_value, "detach"):
                position_value = position_value.detach().cpu().numpy()
            position = tuple(
                float(position_value[index]) + offset[index] for index in range(3)
            )
            final_positions[object_id] = position
            delta = math.sqrt(sum((position[i] - initial[object_id][i]) ** 2 for i in range(3)))
            displacement[object_id] = delta
            if position[2] < 0.0:
                fallen.append(object_id)
        return SimulationResult(
            success=not fallen,
            final_positions_m=final_positions,
            displacement_m=displacement,
            fallen_object_ids=tuple(fallen),
        )

    def render_rgb(self, scene: SceneState, output_path: str | Path) -> Path:
        """Render one headless RGB view of an already solved scene."""
        from PIL import Image

        simulation, _ = self._build_scene(scene, visual_context=True)
        dense_container = scene.metadata.get("presentation_mode") == "dense_container"
        camera_pose = (
            {
                "pos": (0.82, -1.12, 2.65),
                "lookat": (0.0, 0.0, scene.root_top_z + 0.08),
                "fov": 26,
            }
            if dense_container
            else {
                "pos": (2.20, -2.38, 1.76),
                "lookat": (0.0, 0.0, scene.root_top_z - 0.12),
                "fov": 36,
            }
        )
        camera = simulation.add_camera(
            res=(1280, 720),
            pos=camera_pose["pos"],
            lookat=camera_pose["lookat"],
            fov=camera_pose["fov"],
            GUI=False,
        )
        simulation.build()
        rgb, _, _, _ = camera.render(rgb=True)
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(path)
        return path

    def _build_scene(self, scene: SceneState, *, visual_context: bool = False):
        gs = self._gs
        simulation = gs.Scene(
            vis_options=gs.options.VisOptions(
                ambient_light=(0.62, 0.62, 0.62),
                background_color=(0.72, 0.69, 0.63),
                shadow=False,
                plane_reflection=False,
            ),
            show_viewer=not self._headless,
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        )
        if visual_context:
            self._add_visual_context(simulation, scene)
        dense_container = scene.metadata.get("presentation_mode") == "dense_container"
        root_color = (0.64, 0.62, 0.58, 1.0) if dense_container else (0.42, 0.24, 0.12, 1.0)
        simulation.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, scene.root_height_m),
                size=scene.root_size_m,
                fixed=True,
            ),
            material=gs.materials.Rigid(
                rho=650.0,
                friction=0.65,
            ),
            surface=gs.surfaces.Default(
                color=root_color,
                roughness=0.72,
            ),
        )
        entities = self._add_semantic_stack_entities(
            simulation,
            scene,
            visual_context=visual_context,
        )
        stacked_ids = {
            object_id
            for stack in scene.metadata.get("semantic_stacks", [])
            for object_id in stack.get("member_ids", [])
        }
        for object_id, obj in scene.objects.items():
            if object_id in stacked_ids:
                if visual_context:
                    has_mesh_visual = (
                        obj.asset.mesh_path is not None
                        and Path(obj.asset.mesh_path).is_file()
                    )
                    if has_mesh_visual:
                        self._add_mesh_visual(simulation, obj)
                    else:
                        self._add_object_details(simulation, obj)
                continue
            physical_size = obj.asset.physical_size_m
            volume = physical_size[0] * physical_size[1] * physical_size[2]
            density = obj.asset.mass_kg / max(volume, 1.0e-9)
            is_container = obj.asset.container_inner_size_m is not None
            has_mesh_visual = (
                visual_context
                and obj.asset.mesh_path is not None
                and Path(obj.asset.mesh_path).is_file()
            )
            custom_visual = visual_context and obj.asset.visual_shape in {
                "drill",
                "hammer",
                "lamp",
                "monitor",
                "motor",
                "plant",
                "saw",
            }
            if obj.asset.visual_shape in {
                "bottle",
                "bowl",
                "can",
                "cup",
                "cylinder",
                "jar",
                "plate",
            }:
                morph = gs.morphs.Cylinder(
                    pos=obj.position_m,
                    euler=(0.0, 0.0, math.degrees(obj.yaw_rad)),
                    height=physical_size[2],
                    radius=min(physical_size[0], physical_size[1]) / 2.0,
                    fixed=obj.fixed,
                    visualization=not has_mesh_visual,
                )
            else:
                morph = gs.morphs.Box(
                    pos=obj.position_m,
                    euler=(0.0, 0.0, math.degrees(obj.yaw_rad)),
                    size=physical_size,
                    fixed=True if is_container else obj.fixed,
                    visualization=(
                        True if is_container else not (custom_visual or has_mesh_visual)
                    ),
                    collision=not is_container,
                )
            entity = simulation.add_entity(
                morph=morph,
                material=gs.materials.Rigid(
                    rho=density,
                    friction=max(0.01, min(5.0, obj.asset.friction)),
                ),
                surface=gs.surfaces.Default(
                    color=obj.asset.color_rgba,
                    opacity=0.0 if is_container else None,
                    roughness=0.48,
                    metallic=0.35 if obj.object_id.startswith(("fork", "knife", "spoon")) else 0.0,
                ),
            )
            entities[object_id] = (entity, (0.0, 0.0, 0.0))
            if is_container:
                self._add_container_walls(simulation, obj, visual_context=visual_context)
                if visual_context:
                    self._add_container_visuals(simulation, obj)
            if has_mesh_visual:
                self._add_mesh_visual(simulation, obj)
            elif visual_context and not is_container:
                self._add_object_details(simulation, obj)
        return simulation, entities

    def _add_semantic_stack_entities(
        self,
        simulation,
        scene: SceneState,
        *,
        visual_context: bool,
    ):
        """Represent each nested column as one freely simulated rigid proxy."""
        gs = self._gs
        bindings = {}
        for stack in scene.metadata.get("semantic_stacks", []):
            member_ids = [
                object_id
                for object_id in stack.get("member_ids", [])
                if object_id in scene.objects
            ]
            if len(member_ids) < 2:
                continue
            members = [scene.get(object_id) for object_id in member_ids]
            center_x = sum(obj.position_m[0] for obj in members) / len(members)
            center_y = sum(obj.position_m[1] for obj in members) / len(members)
            bottom = min(obj.bottom_z for obj in members)
            top = max(obj.top_z for obj in members)
            center = (center_x, center_y, (bottom + top) / 2.0)
            height = max(1.0e-4, top - bottom)
            radius = min(
                min(obj.asset.physical_size_m[0], obj.asset.physical_size_m[1])
                for obj in members
            ) / 2.0
            volume = math.pi * radius * radius * height
            density = sum(obj.asset.mass_kg for obj in members) / max(volume, 1.0e-9)
            entity = simulation.add_entity(
                morph=gs.morphs.Cylinder(
                    pos=center,
                    height=height,
                    radius=radius,
                    fixed=False,
                    visualization=not visual_context,
                ),
                material=gs.materials.Rigid(
                    rho=density,
                    friction=max(obj.asset.friction for obj in members),
                ),
                surface=gs.surfaces.Default(
                    color=members[0].asset.color_rgba,
                    opacity=0.0 if visual_context else None,
                    roughness=0.48,
                ),
            )
            for obj in members:
                bindings[obj.object_id] = (
                    entity,
                    tuple(obj.position_m[index] - center[index] for index in range(3)),
                )
        return bindings

    def _add_mesh_visual(self, simulation, obj) -> None:
        """Overlay a licensed textured mesh on the stable proxy collision body."""
        gs = self._gs
        assert obj.asset.mesh_path is not None
        dx, dy, dz = obj.asset.mesh_offset_m
        cosine, sine = math.cos(obj.yaw_rad), math.sin(obj.yaw_rad)
        visual_position = obj.visual_position_m
        position = (
            visual_position[0] + cosine * dx - sine * dy,
            visual_position[1] + sine * dx + cosine * dy,
            visual_position[2] + dz,
        )
        mesh_euler = obj.asset.mesh_euler_deg
        simulation.add_entity(
            morph=gs.morphs.Mesh(
                file=obj.asset.mesh_path,
                scale=obj.asset.mesh_scale,
                pos=position,
                euler=(
                    mesh_euler[0],
                    mesh_euler[1],
                    mesh_euler[2] + math.degrees(obj.yaw_rad),
                ),
                file_meshes_are_zup=obj.asset.mesh_file_is_zup,
                fixed=True,
                collision=False,
                align=False,
            )
        )

    def _add_visual_context(self, simulation, scene: SceneState) -> None:
        """Add fixed room and furniture context to presentation renders only."""
        gs = self._gs

        def fixed_box(pos, size, color, *, roughness=0.75):
            simulation.add_entity(
                morph=gs.morphs.Box(pos=pos, size=size, fixed=True, collision=False),
                surface=gs.surfaces.Default(color=color, roughness=roughness),
            )

        if scene.metadata.get("presentation_mode") == "dense_container":
            fixed_box((0.0, 0.0, -0.035), (6.0, 6.0, 0.07), (0.58, 0.57, 0.54, 1.0))
            fixed_box((0.0, 2.50, 5.0), (30.0, 0.06, 10.0), (0.72, 0.71, 0.68, 1.0))
            return

        fixed_box((0.0, 0.0, -0.035), (6.0, 6.0, 0.07), (0.34, 0.31, 0.28, 1.0))
        fixed_box((0.0, 2.50, 5.0), (30.0, 0.06, 10.0), (0.72, 0.69, 0.63, 1.0))
        fixed_box((0.0, 0.10, 0.006), (3.15, 2.30, 0.012), (0.18, 0.25, 0.27, 1.0))

        tabletop_bottom = scene.root_height_m - scene.root_size_m[2] / 2.0
        leg_height = max(0.05, tabletop_bottom)
        inset_x = max(0.08, scene.root_size_m[0] / 2.0 - 0.11)
        inset_y = max(0.08, scene.root_size_m[1] / 2.0 - 0.11)
        for x in (-inset_x, inset_x):
            for y in (-inset_y, inset_y):
                fixed_box(
                    (x, y, leg_height / 2.0),
                    (0.085, 0.085, leg_height),
                    (0.32, 0.16, 0.075, 1.0),
                    roughness=0.62,
                )

    def _add_container_walls(self, simulation, obj, *, visual_context: bool = False) -> None:
        """Represent open containers as fixed compound walls, never as solid boxes."""
        gs = self._gs
        sx, sy, sz = obj.asset.size_m
        inner = obj.asset.container_inner_size_m
        assert inner is not None
        x_wall = max(0.006, (sx - inner[0]) / 2.0)
        y_wall = max(0.006, (sy - inner[1]) / 2.0)
        bottom = max(0.006, (sz - inner[2]) / 2.0)
        x, y, z = obj.visual_position_m
        cosine, sine = math.cos(obj.yaw_rad), math.sin(obj.yaw_rad)

        def position(dx: float, dy: float, dz: float):
            return (x + cosine * dx - sine * dy, y + sine * dx + cosine * dy, z + dz)

        density = obj.asset.mass_kg / max(sx * sy * sz, 1.0e-9)
        walls = (
            (position(0.0, 0.0, -sz / 2.0 + bottom / 2.0), (sx, sy, bottom)),
            (position(sx / 2.0 - x_wall / 2.0, 0.0, 0.0), (x_wall, sy, sz)),
            (position(-sx / 2.0 + x_wall / 2.0, 0.0, 0.0), (x_wall, sy, sz)),
            (position(0.0, sy / 2.0 - y_wall / 2.0, 0.0), (sx, y_wall, sz)),
            (position(0.0, -sy / 2.0 + y_wall / 2.0, 0.0), (sx, y_wall, sz)),
        )
        open_slatted_visual = visual_context and obj.asset.visual_shape in {
            "grocery_basket",
            "office_tote",
            "tool_crate",
        }
        for wall_index, (wall_position, wall_size) in enumerate(walls):
            simulation.add_entity(
                morph=gs.morphs.Box(
                    pos=wall_position,
                    euler=(0.0, 0.0, math.degrees(obj.yaw_rad)),
                    size=wall_size,
                    fixed=True,
                ),
                material=gs.materials.Rigid(
                    rho=density,
                    friction=max(0.01, min(5.0, obj.asset.friction)),
                ),
                surface=gs.surfaces.Default(
                    color=obj.asset.color_rgba,
                    opacity=0.0 if open_slatted_visual and wall_index > 0 else None,
                    roughness=0.56,
                ),
            )

    def _add_container_visuals(self, simulation, obj) -> None:
        """Add recognizable non-colliding basket slats or a sink rim/faucet."""
        gs = self._gs
        sx, sy, sz = obj.asset.size_m
        x, y, z = obj.position_m
        yaw_deg = math.degrees(obj.yaw_rad)
        cosine, sine = math.cos(obj.yaw_rad), math.sin(obj.yaw_rad)

        def position(dx: float, dy: float, dz: float):
            return (x + cosine * dx - sine * dy, y + sine * dx + cosine * dy, z + dz)

        def box(pos, size, color, *, metallic=0.0):
            simulation.add_entity(
                morph=gs.morphs.Box(
                    pos=pos,
                    euler=(0.0, 0.0, yaw_deg),
                    size=size,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Default(
                    color=color,
                    roughness=0.48,
                    metallic=metallic,
                ),
            )

        shape = obj.asset.visual_shape
        if shape in {"grocery_basket", "office_tote", "tool_crate"}:
            frame_color = obj.asset.color_rgba
            rail = 0.026 if shape == "tool_crate" else 0.018
            for dz in (-sz * 0.42, sz * 0.42):
                box(position(0.0, sy / 2.0, dz), (sx, rail, rail), frame_color)
                box(position(0.0, -sy / 2.0, dz), (sx, rail, rail), frame_color)
                box(position(sx / 2.0, 0.0, dz), (rail, sy, rail), frame_color)
                box(position(-sx / 2.0, 0.0, dz), (rail, sy, rail), frame_color)
            front_divisions = 8 if shape == "tool_crate" else 15
            for index in range(-(front_divisions // 2), front_divisions // 2 + 1):
                dx = index * sx / (front_divisions + 1)
                upright = 0.016 if shape == "tool_crate" else 0.010
                box(
                    position(dx, sy / 2.0, 0.0),
                    (upright, rail, sz * 0.78),
                    frame_color,
                )
                box(
                    position(dx, -sy / 2.0, 0.0),
                    (upright, rail, sz * 0.78),
                    frame_color,
                )
            side_divisions = 6 if shape == "tool_crate" else 9
            for index in range(-(side_divisions // 2), side_divisions // 2 + 1):
                dy = index * sy / (side_divisions + 1)
                upright = 0.016 if shape == "tool_crate" else 0.010
                box(
                    position(sx / 2.0, dy, 0.0),
                    (rail, upright, sz * 0.78),
                    frame_color,
                )
                box(
                    position(-sx / 2.0, dy, 0.0),
                    (rail, upright, sz * 0.78),
                    frame_color,
                )
        elif shape == "sink":
            steel = obj.asset.color_rgba
            rim = 0.028
            box(position(0.0, sy / 2.0, sz / 2.0), (sx, rim, 0.022), steel, metallic=0.75)
            box(position(0.0, -sy / 2.0, sz / 2.0), (sx, rim, 0.022), steel, metallic=0.75)
            box(position(sx / 2.0, 0.0, sz / 2.0), (rim, sy, 0.022), steel, metallic=0.75)
            box(position(-sx / 2.0, 0.0, sz / 2.0), (rim, sy, 0.022), steel, metallic=0.75)
            faucet_color = (0.36, 0.34, 0.31, 1.0)
            simulation.add_entity(
                morph=gs.morphs.Cylinder(
                    pos=position(0.0, sy * 0.62, sz * 0.95),
                    height=sz * 1.65,
                    radius=0.018,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Default(
                    color=faucet_color,
                    roughness=0.25,
                    metallic=0.8,
                ),
            )

    def _add_object_details(self, simulation, obj) -> None:
        """Build lightweight procedural presentation geometry around collisions."""
        gs = self._gs
        color = obj.asset.color_rgba
        x, y, z = obj.visual_position_m
        sx, sy, sz = obj.asset.size_m
        yaw_deg = math.degrees(obj.yaw_rad)

        def offset(dx: float, dy: float, dz: float):
            cosine, sine = math.cos(obj.yaw_rad), math.sin(obj.yaw_rad)
            return (x + cosine * dx - sine * dy, y + sine * dx + cosine * dy, z + dz)

        def box(pos, size, shade=color, *, yaw=yaw_deg, metallic=0.0):
            simulation.add_entity(
                morph=gs.morphs.Box(
                    pos=pos,
                    euler=(0.0, 0.0, yaw),
                    size=size,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Default(
                    color=shade,
                    roughness=0.42,
                    metallic=metallic,
                ),
            )

        def cylinder(pos, height, radius, shade=color, *, euler=(0.0, 0.0, 0.0)):
            simulation.add_entity(
                morph=gs.morphs.Cylinder(
                    pos=pos,
                    euler=euler,
                    height=height,
                    radius=radius,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Default(color=shade, roughness=0.44),
            )

        def sphere(pos, radius, shade=color):
            simulation.add_entity(
                morph=gs.morphs.Sphere(pos=pos, radius=radius, fixed=True, collision=False),
                surface=gs.surfaces.Default(color=shade, roughness=0.58),
            )

        shape = obj.asset.visual_shape
        if shape == "cup":
            handle = (max(0.008, sx * 0.10), max(0.008, sy * 0.09), sz * 0.34)
            box(offset(0.0, sy * 0.56, sz * 0.18), handle)
            box(offset(0.0, sy * 0.56, -sz * 0.18), handle)
            box(offset(0.0, sy * 0.68, 0.0), (handle[0], handle[1], sz * 0.48))
        elif shape in {"can", "jar", "bottle"}:
            lid = (0.70, 0.73, 0.72, 1.0)
            cylinder(offset(0.0, 0.0, sz * 0.505), sz * 0.035, sx * 0.48, lid)
            if shape != "bottle":
                label = (0.86, 0.82, 0.70, 1.0)
                cylinder(offset(0.0, 0.0, 0.0), sz * 0.48, sx * 0.505, label)
        elif shape == "plate":
            rim = tuple(min(1.0, value * 1.07) for value in color[:3]) + (1.0,)
            cylinder(offset(0.0, 0.0, sz * 0.52), sz * 0.18, sx * 0.48, rim)
            cylinder(offset(0.0, 0.0, sz * 0.62), sz * 0.08, sx * 0.34, color)
        elif shape == "bowl":
            rim = tuple(min(1.0, value * 1.08) for value in color[:3]) + (1.0,)
            cylinder(offset(0.0, 0.0, sz * 0.48), sz * 0.10, sx * 0.51, rim)
        elif shape in {"carton", "package"}:
            label = (0.96, 0.91, 0.72, 1.0) if shape == "carton" else (0.25, 0.34, 0.38, 1.0)
            box(offset(0.0, 0.0, sz * 0.505), (sx * 0.76, sy * 0.72, 0.003), label)
            box(offset(sx * 0.505, 0.0, 0.0), (0.003, sy * 0.70, sz * 0.42), label)
        elif shape == "plant":
            pot = (0.55, 0.25, 0.12, 1.0)
            cylinder(offset(0.0, 0.0, -sz * 0.34), sz * 0.32, sx * 0.34, pot)
            cylinder(offset(0.0, 0.0, 0.0), sz * 0.48, sx * 0.08, (0.18, 0.35, 0.16, 1.0))
            for dx, dy, dz in ((0.05, 0.0, 0.09), (-0.04, 0.02, 0.12), (0.0, -0.05, 0.10), (0.025, 0.04, 0.15), (-0.03, -0.025, 0.16)):
                sphere(offset(dx, dy, dz), sx * 0.22, color)
        elif shape == "lamp":
            cylinder(offset(0.0, 0.0, -sz * 0.45), sz * 0.07, sx * 0.48, color)
            cylinder(offset(0.0, 0.0, -sz * 0.06), sz * 0.72, sx * 0.055, color)
            cylinder(offset(0.0, 0.0, sz * 0.34), sz * 0.19, sx * 0.46, (0.72, 0.62, 0.38, 1.0))
        elif shape == "monitor":
            box(offset(0.0, 0.0, sz * 0.12), (sx * 0.42, sy, sz * 0.76), color)
            box(offset(0.0, 0.0, sz * 0.12), (sx * 0.43, sy * 0.88, sz * 0.64), (0.06, 0.12, 0.15, 1.0))
            box(offset(0.0, 0.0, -sz * 0.31), (sx * 0.34, sy * 0.08, sz * 0.25), color)
            box(offset(0.0, 0.0, -sz * 0.45), (sx, sy * 0.38, sz * 0.04), color)
        elif shape == "drill":
            box(offset(0.02, 0.0, sz * 0.14), (sx * 0.66, sy, sz * 0.42), color)
            box(offset(-sx * 0.10, 0.0, -sz * 0.22), (sx * 0.24, sy * 0.76, sz * 0.48), color)
            cylinder(offset(sx * 0.43, 0.0, sz * 0.15), sx * 0.30, sy * 0.24, (0.20, 0.20, 0.19, 1.0), euler=(0.0, 90.0, yaw_deg))
        elif shape == "hammer":
            box(offset(-sx * 0.05, 0.0, 0.0), (sx * 0.74, sy * 0.24, sz * 0.70), color)
            box(offset(sx * 0.36, 0.0, 0.0), (sx * 0.26, sy, sz), (0.50, 0.52, 0.52, 1.0), metallic=0.65)
        elif shape == "saw":
            box(offset(0.03, 0.0, 0.0), (sx * 0.72, sy * 0.16, sz * 0.62), (0.62, 0.65, 0.65, 1.0), metallic=0.55)
            box(offset(-sx * 0.38, 0.0, 0.0), (sx * 0.23, sy, sz), (0.52, 0.20, 0.09, 1.0))
        elif shape == "motor":
            cylinder(offset(0.0, 0.0, 0.0), sx * 0.82, min(sy, sz) * 0.42, color, euler=(0.0, 90.0, yaw_deg))
            box(offset(-sx * 0.32, 0.0, -sz * 0.38), (sx * 0.18, sy, sz * 0.18), color)
            box(offset(sx * 0.32, 0.0, -sz * 0.38), (sx * 0.18, sy, sz * 0.18), color)
