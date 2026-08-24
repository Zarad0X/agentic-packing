"""Asset catalog interfaces and a deterministic procedural demo catalog."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Protocol

from physcensis.types import AssetRecord, PlacementProgram


class AssetNotFoundError(LookupError):
    pass


class AssetCatalog(Protocol):
    def resolve(self, object_id: str, description: str) -> AssetRecord:
        """Resolve a description to one normalized asset record."""

    def resolve_category(self, category: str, object_id: str) -> AssetRecord:
        """Resolve one member of a batched category."""

    def resolve_program(self, program: PlacementProgram) -> dict[str, AssetRecord]:
        """Resolve all explicitly described program objects."""


def _asset(
    name: str,
    size_m: tuple[float, float, float],
    *,
    mass_kg: float = 0.5,
    friction: float = 0.6,
    supporting_probability: float = 0.4,
    container_inner_size_m: tuple[float, float, float] | None = None,
    visual_shape: str = "box",
    color_rgba: tuple[float, float, float, float] = (0.65, 0.65, 0.65, 1.0),
    stackable: bool = False,
    stacking_step_ratio: float = 1.0,
) -> AssetRecord:
    return AssetRecord(
        asset_id=f"primitive:{name}",
        description=name,
        size_m=size_m,
        mass_kg=mass_kg,
        friction=friction,
        supporting_probability=supporting_probability,
        container_inner_size_m=container_inner_size_m,
        visual_shape=visual_shape,
        color_rgba=color_rgba,
        stackable=stackable,
        stacking_step_ratio=stacking_step_ratio,
    )


_PRIMITIVES: Mapping[str, AssetRecord] = {
    "plate": _asset(
        "plate",
        (0.22, 0.22, 0.022),
        mass_kg=0.38,
        supporting_probability=0.92,
        visual_shape="plate",
        color_rgba=(0.92, 0.91, 0.86, 1.0),
        stackable=True,
        stacking_step_ratio=0.55,
    ),
    "cup": _asset(
        "cup",
        (0.085, 0.085, 0.10),
        mass_kg=0.26,
        supporting_probability=0.7,
        visual_shape="cup",
        color_rgba=(0.82, 0.88, 0.90, 1.0),
    ),
    "fork": _asset("fork", (0.20, 0.025, 0.012), mass_kg=0.06, supporting_probability=0.05, color_rgba=(0.72, 0.75, 0.78, 1.0)),
    "knife": _asset("knife", (0.22, 0.025, 0.012), mass_kg=0.08, supporting_probability=0.05, color_rgba=(0.64, 0.67, 0.70, 1.0)),
    "spoon": _asset("spoon", (0.19, 0.04, 0.015), mass_kg=0.06, supporting_probability=0.05, color_rgba=(0.72, 0.75, 0.78, 1.0)),
    "book": _asset("book", (0.25, 0.18, 0.035), mass_kg=0.55, supporting_probability=0.95, color_rgba=(0.38, 0.16, 0.12, 1.0)),
    "notebook": _asset("notebook", (0.24, 0.17, 0.025), mass_kg=0.35, supporting_probability=0.95),
    "laptop": _asset("laptop", (0.34, 0.24, 0.025), mass_kg=1.4, supporting_probability=0.85),
    "monitor": _asset("monitor", (0.12, 0.48, 0.34), mass_kg=4.0, supporting_probability=0.1, visual_shape="monitor", color_rgba=(0.10, 0.12, 0.13, 1.0)),
    "keyboard": _asset("keyboard", (0.16, 0.44, 0.025), mass_kg=0.7, supporting_probability=0.8, color_rgba=(0.22, 0.24, 0.25, 1.0)),
    "mouse": _asset("mouse", (0.11, 0.065, 0.04), mass_kg=0.1, supporting_probability=0.2, color_rgba=(0.16, 0.17, 0.18, 1.0)),
    "phone": _asset("phone", (0.15, 0.075, 0.009), mass_kg=0.2, supporting_probability=0.8, color_rgba=(0.06, 0.07, 0.08, 1.0)),
    "bottle": _asset(
        "bottle",
        (0.075, 0.075, 0.20),
        mass_kg=0.48,
        supporting_probability=0.2,
        visual_shape="bottle",
        color_rgba=(0.38, 0.68, 0.48, 1.0),
    ),
    "can": _asset(
        "can",
        (0.068, 0.068, 0.105),
        mass_kg=0.32,
        supporting_probability=0.55,
        visual_shape="can",
        color_rgba=(0.78, 0.24, 0.18, 1.0),
    ),
    "jar": _asset(
        "jar",
        (0.078, 0.078, 0.11),
        mass_kg=0.42,
        supporting_probability=0.6,
        visual_shape="jar",
        color_rgba=(0.26, 0.43, 0.54, 1.0),
    ),
    "bowl": _asset(
        "bowl",
        (0.15, 0.15, 0.065),
        mass_kg=0.40,
        supporting_probability=0.78,
        visual_shape="bowl",
        color_rgba=(0.87, 0.87, 0.82, 1.0),
        stackable=True,
        stacking_step_ratio=0.35,
    ),
    "stackable cup": _asset(
        "stackable_cup",
        (0.105, 0.105, 0.085),
        mass_kg=0.24,
        supporting_probability=0.72,
        visual_shape="cup",
        color_rgba=(0.72, 0.78, 0.74, 1.0),
        stackable=True,
        stacking_step_ratio=0.32,
    ),
    "vase": _asset("vase", (0.14, 0.14, 0.28), mass_kg=1.0, supporting_probability=0.2, visual_shape="cylinder", color_rgba=(0.46, 0.58, 0.78, 1.0)),
    "lamp": _asset("lamp", (0.20, 0.20, 0.38), mass_kg=1.5, supporting_probability=0.1, visual_shape="lamp", color_rgba=(0.10, 0.12, 0.12, 1.0)),
    "pen": _asset("pen", (0.15, 0.012, 0.012), mass_kg=0.02, supporting_probability=0.05, color_rgba=(0.08, 0.09, 0.10, 1.0)),
    "pencil": _asset("pencil", (0.18, 0.01, 0.01), mass_kg=0.01, supporting_probability=0.05, color_rgba=(0.82, 0.52, 0.10, 1.0)),
    "basket": _asset(
        "basket",
        (0.50, 0.35, 0.20),
        mass_kg=0.8,
        supporting_probability=0.8,
        container_inner_size_m=(0.44, 0.29, 0.17),
        visual_shape="basket",
        color_rgba=(0.28, 0.46, 0.30, 1.0),
    ),
    "grocery basket": _asset(
        "grocery_basket",
        (0.68, 0.42, 0.30),
        mass_kg=1.15,
        friction=0.75,
        supporting_probability=0.85,
        container_inner_size_m=(0.58, 0.34, 0.255),
        visual_shape="grocery_basket",
        color_rgba=(0.08, 0.30, 0.20, 1.0),
    ),
    "sink": _asset(
        "sink",
        (0.74, 0.54, 0.22),
        mass_kg=8.0,
        friction=0.55,
        supporting_probability=0.9,
        container_inner_size_m=(0.64, 0.44, 0.18),
        visual_shape="sink",
        color_rgba=(0.58, 0.60, 0.60, 1.0),
    ),
    "tool crate": _asset(
        "tool_crate",
        (0.82, 0.52, 0.30),
        mass_kg=3.2,
        friction=0.82,
        supporting_probability=0.9,
        container_inner_size_m=(0.74, 0.44, 0.26),
        visual_shape="tool_crate",
        color_rgba=(0.52, 0.10, 0.075, 1.0),
    ),
    "office tote": _asset(
        "office_tote",
        (0.82, 0.56, 0.30),
        mass_kg=2.1,
        friction=0.72,
        supporting_probability=0.9,
        container_inner_size_m=(0.74, 0.48, 0.26),
        visual_shape="office_tote",
        color_rgba=(0.10, 0.34, 0.52, 1.0),
    ),
    "grocery carton": _asset(
        "grocery_carton",
        (0.20, 0.09, 0.14),
        mass_kg=0.38,
        friction=0.75,
        supporting_probability=0.85,
        visual_shape="carton",
        color_rgba=(0.93, 0.67, 0.18, 1.0),
    ),
    "pantry box": _asset(
        "pantry_box",
        (0.15, 0.075, 0.115),
        mass_kg=0.24,
        friction=0.72,
        supporting_probability=0.86,
        visual_shape="package",
        color_rgba=(0.83, 0.84, 0.80, 1.0),
    ),
    "box": _asset(
        "box",
        (0.24, 0.20, 0.16),
        mass_kg=0.4,
        supporting_probability=0.85,
        container_inner_size_m=(0.21, 0.17, 0.14),
    ),
    "toolbox": _asset(
        "toolbox", (0.34, 0.18, 0.16), mass_kg=2.0, supporting_probability=0.9, color_rgba=(0.64, 0.12, 0.09, 1.0)
    ),
    "drill": _asset("drill", (0.27, 0.09, 0.22), mass_kg=1.8, supporting_probability=0.2, visual_shape="drill", color_rgba=(0.88, 0.58, 0.08, 1.0)),
    "wrench": _asset("wrench", (0.25, 0.045, 0.02), mass_kg=0.35, supporting_probability=0.05),
    "hammer": _asset("hammer", (0.30, 0.11, 0.04), mass_kg=0.65, supporting_probability=0.1, visual_shape="hammer", color_rgba=(0.46, 0.25, 0.10, 1.0)),
    "motor": _asset("motor", (0.24, 0.18, 0.18), mass_kg=4.0, supporting_probability=0.8, visual_shape="motor", color_rgba=(0.23, 0.32, 0.36, 1.0)),
    "saw": _asset("saw", (0.38, 0.12, 0.05), mass_kg=0.8, supporting_probability=0.2, visual_shape="saw", color_rgba=(0.64, 0.66, 0.65, 1.0)),
    "plant": _asset("plant", (0.20, 0.20, 0.32), mass_kg=1.2, supporting_probability=0.2, visual_shape="plant", color_rgba=(0.20, 0.46, 0.25, 1.0)),
}


class PrimitiveAssetCatalog:
    """Resolve common objects to cuboids for deterministic solver development."""

    def resolve(self, object_id: str, description: str) -> AssetRecord:
        searchable = f"{object_id} {description}".lower().replace("_", " ")
        candidates = sorted(
            ((key, record) for key, record in _PRIMITIVES.items() if key in searchable),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        if not candidates:
            raise AssetNotFoundError(f"No procedural asset matches {object_id!r}: {description}")
        _, record = candidates[0]
        color_terms = (
            (("charcoal", "graphite", "black", "dark"), (0.10, 0.12, 0.13, 1.0)),
            (("cream", "ivory", "white"), (0.92, 0.88, 0.72, 1.0)),
            (("rust", "red", "coral"), (0.67, 0.20, 0.13, 1.0)),
            (("blue", "teal", "sea green"), (0.18, 0.48, 0.58, 1.0)),
            (("green",), (0.22, 0.52, 0.30, 1.0)),
            (("brass", "ochre", "yellow"), (0.78, 0.55, 0.18, 1.0)),
            (("silver", "polished", "steel", "aluminum"), (0.66, 0.70, 0.72, 1.0)),
        )
        for terms, color in color_terms:
            if any(term in searchable for term in terms):
                record = replace(record, color_rgba=color)
                break
        return replace(record, asset_id=f"{record.asset_id}:{object_id}", description=description)

    def resolve_category(self, category: str, object_id: str) -> AssetRecord:
        record = self.resolve(object_id, category)
        palettes = {
            "grocery carton": (
                (0.94, 0.65, 0.14, 1.0),
                (0.90, 0.75, 0.28, 1.0),
                (0.85, 0.42, 0.16, 1.0),
            ),
            "pantry box": (
                (0.88, 0.89, 0.85, 1.0),
                (0.76, 0.82, 0.86, 1.0),
                (0.92, 0.83, 0.61, 1.0),
            ),
            "can": (
                (0.76, 0.19, 0.14, 1.0),
                (0.20, 0.48, 0.62, 1.0),
                (0.80, 0.62, 0.18, 1.0),
                (0.39, 0.55, 0.32, 1.0),
            ),
            "jar": (
                (0.16, 0.33, 0.45, 1.0),
                (0.38, 0.23, 0.18, 1.0),
                (0.58, 0.25, 0.18, 1.0),
            ),
            "plate": (
                (0.92, 0.92, 0.88, 1.0),
                (0.82, 0.88, 0.90, 1.0),
                (0.88, 0.84, 0.78, 1.0),
            ),
            "bowl": (
                (0.88, 0.89, 0.86, 1.0),
                (0.66, 0.75, 0.76, 1.0),
                (0.82, 0.78, 0.68, 1.0),
            ),
        }
        palette = palettes.get(category.replace("_", " ").lower())
        if palette:
            index = int(object_id.rsplit("_", 1)[-1]) if "_" in object_id else 0
            record = replace(record, color_rgba=palette[index % len(palette)])
        return record

    def resolve_program(self, program: PlacementProgram) -> dict[str, AssetRecord]:
        return {
            object_id: self.resolve(object_id, description)
            for object_id, description in program.descriptions.items()
        }
