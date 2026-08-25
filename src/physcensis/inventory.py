"""Validated fixed-object inventories for arrangement-only generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physcensis.types import AssetRecord


class InventoryValidationError(ValueError):
    """Raised when a fixed-object inventory is incomplete or ambiguous."""


class InventoryPlanValidationError(ValueError):
    """Raised when an agent plan changes inventory identity or requests unsafe stacks."""


@dataclass(frozen=True)
class InventoryObjectSpec:
    object_id: str
    category: str
    asset_uid: str | None = None
    asset: AssetRecord | None = None


@dataclass(frozen=True)
class InventorySpec:
    container: InventoryObjectSpec
    objects: tuple[InventoryObjectSpec, ...]
    container_position_xy_m: tuple[float, float] = (0.0, 0.0)
    container_yaw_deg: float = 0.0
    allow_protrusion_m: float = 0.0


@dataclass(frozen=True)
class InventoryStackGroup:
    group_id: str
    bottom_to_top_object_ids: tuple[str, ...]


@dataclass(frozen=True)
class InventoryAdjacencyGroup:
    group_id: str
    object_ids: tuple[str, ...]


@dataclass(frozen=True)
class InventoryArrangementPlan:
    """Strict LLM output consumed by the fixed-inventory physical planner."""

    placement_order: tuple[str, ...]
    stack_groups: tuple[InventoryStackGroup, ...]
    adjacency_groups: tuple[InventoryAdjacencyGroup, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "placement_order": list(self.placement_order),
            "stack_groups": [
                {
                    "group_id": group.group_id,
                    "bottom_to_top_object_ids": list(group.bottom_to_top_object_ids),
                }
                for group in self.stack_groups
            ],
            "adjacency_groups": [
                {"group_id": group.group_id, "object_ids": list(group.object_ids)}
                for group in self.adjacency_groups
            ],
            "rationale": self.rationale,
        }


class InventoryPlanParser:
    """Validate that a model plan is complete, identity preserving, and realizable."""

    _fields = frozenset(
        {"placement_order", "stack_groups", "adjacency_groups", "rationale"}
    )

    def parse(
        self,
        payload: Any,
        inventory: InventorySpec,
        assets: dict[str, AssetRecord],
    ) -> InventoryArrangementPlan:
        if not isinstance(payload, dict):
            raise InventoryPlanValidationError("LLM arrangement plan must be a JSON object")
        unknown = sorted(set(payload) - self._fields)
        if unknown:
            raise InventoryPlanValidationError(
                f"Unknown LLM arrangement plan fields: {', '.join(unknown)}"
            )
        expected = [obj.object_id for obj in inventory.objects]
        order = self._id_list(payload.get("placement_order"), "placement_order")
        if len(order) != len(expected) or set(order) != set(expected):
            missing = sorted(set(expected) - set(order))
            invented = sorted(set(order) - set(expected))
            raise InventoryPlanValidationError(
                "placement_order must be an exact permutation of all supplied object ids; "
                f"missing={missing}, invented={invented}, supplied={len(expected)}, planned={len(order)}"
            )

        stack_values = payload.get("stack_groups", [])
        if not isinstance(stack_values, list):
            raise InventoryPlanValidationError("stack_groups must be a JSON array")
        stack_groups: list[InventoryStackGroup] = []
        stacked_ids: set[str] = set()
        group_ids: set[str] = set()
        for index, value in enumerate(stack_values):
            group_id, members = self._group(
                value,
                index,
                id_field="bottom_to_top_object_ids",
                collection="stack_groups",
            )
            if group_id in group_ids:
                raise InventoryPlanValidationError(f"Duplicate group_id: {group_id}")
            group_ids.add(group_id)
            if len(members) < 2:
                raise InventoryPlanValidationError(
                    f"stack_groups[{index}] must contain at least two object ids"
                )
            self._known_ids(members, expected, f"stack_groups[{index}]")
            overlap = sorted(stacked_ids.intersection(members))
            if overlap:
                raise InventoryPlanValidationError(
                    f"An object can belong to only one stack group: {', '.join(overlap)}"
                )
            non_stackable = [object_id for object_id in members if not assets[object_id].stackable]
            if non_stackable:
                raise InventoryPlanValidationError(
                    "Only assets explicitly marked stackable may be forced into a stack; "
                    f"invalid={non_stackable}"
                )
            asset_ids = {assets[object_id].asset_id for object_id in members}
            descriptions = {assets[object_id].description for object_id in members}
            if len(asset_ids) != 1 and len(descriptions) != 1:
                raise InventoryPlanValidationError(
                    f"stack_groups[{index}] must contain the same asset or normalized category"
                )
            order_rank = {object_id: rank for rank, object_id in enumerate(order)}
            if [order_rank[object_id] for object_id in members] != sorted(
                order_rank[object_id] for object_id in members
            ):
                raise InventoryPlanValidationError(
                    f"stack_groups[{index}] bottom-to-top members must appear in that order in placement_order"
                )
            stacked_ids.update(members)
            stack_groups.append(InventoryStackGroup(group_id, tuple(members)))

        adjacency_values = payload.get("adjacency_groups", [])
        if not isinstance(adjacency_values, list):
            raise InventoryPlanValidationError("adjacency_groups must be a JSON array")
        adjacency_groups: list[InventoryAdjacencyGroup] = []
        for index, value in enumerate(adjacency_values):
            group_id, members = self._group(
                value,
                index,
                id_field="object_ids",
                collection="adjacency_groups",
            )
            if group_id in group_ids:
                raise InventoryPlanValidationError(f"Duplicate group_id: {group_id}")
            group_ids.add(group_id)
            if len(members) < 2:
                raise InventoryPlanValidationError(
                    f"adjacency_groups[{index}] must contain at least two object ids"
                )
            self._known_ids(members, expected, f"adjacency_groups[{index}]")
            adjacency_groups.append(InventoryAdjacencyGroup(group_id, tuple(members)))

        rationale = str(payload.get("rationale", "")).strip()
        if not rationale:
            raise InventoryPlanValidationError("rationale must be non-empty")
        return InventoryArrangementPlan(
            tuple(order), tuple(stack_groups), tuple(adjacency_groups), rationale
        )

    @staticmethod
    def _id_list(value: Any, field: str) -> list[str]:
        if not isinstance(value, list):
            raise InventoryPlanValidationError(f"{field} must be a JSON array")
        result = [str(item).strip() for item in value]
        if any(not item for item in result):
            raise InventoryPlanValidationError(f"{field} contains an empty object id")
        if len(result) != len(set(result)):
            raise InventoryPlanValidationError(f"{field} contains duplicate object ids")
        return result

    def _group(
        self,
        value: Any,
        index: int,
        *,
        id_field: str,
        collection: str,
    ) -> tuple[str, list[str]]:
        if not isinstance(value, dict) or set(value) != {"group_id", id_field}:
            raise InventoryPlanValidationError(
                f"{collection}[{index}] must contain exactly group_id and {id_field}"
            )
        group_id = str(value["group_id"]).strip()
        if not group_id:
            raise InventoryPlanValidationError(f"{collection}[{index}].group_id is empty")
        return group_id, self._id_list(value[id_field], f"{collection}[{index}].{id_field}")

    @staticmethod
    def _known_ids(values: list[str], expected: list[str], field: str) -> None:
        unknown = sorted(set(values) - set(expected))
        if unknown:
            raise InventoryPlanValidationError(f"{field} contains unknown ids: {unknown}")


def _vector(value: Any, field: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise InventoryValidationError(f"{field} must contain {length} numbers")
    return tuple(_number(component, field) for component in value)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InventoryValidationError(f"{field} must contain numbers")
    return float(value)


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise InventoryValidationError(f"{field} must be a boolean")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InventoryParser:
    """Parse an explicit inventory without changing object identity or count."""

    _object_fields = frozenset({"object_id", "category", "asset_uid", "asset"})
    _container_fields = _object_fields | {"position_xy_m", "yaw_deg"}
    _arrangement_fields = frozenset({"allow_protrusion_m"})
    _asset_fields = frozenset(
        {
            "asset_id",
            "size_m",
            "mass_kg",
            "friction",
            "com_shift_m",
            "supporting_probability",
            "front_yaw_rad",
            "source",
            "license",
            "source_url",
            "author",
            "sha256",
            "visual_size_m",
            "mesh_path",
            "mesh_scale",
            "mesh_offset_m",
            "mesh_euler_deg",
            "mesh_file_is_zup",
            "stackable",
            "stacking_step_ratio",
            "collision_size_m",
            "container_inner_size_m",
            "visual_shape",
            "color_rgba",
        }
    )

    def parse(self, payload: Any, *, base_dir: str | Path = ".") -> InventorySpec:
        if not isinstance(payload, dict):
            raise InventoryValidationError("Inventory root must be a JSON object")
        unknown = sorted(set(payload) - {"container", "objects", "arrangement"})
        if unknown:
            raise InventoryValidationError(f"Unknown inventory fields: {', '.join(unknown)}")
        if "container" not in payload:
            raise InventoryValidationError("Inventory requires one container")
        objects_value = payload.get("objects")
        if not isinstance(objects_value, list) or not objects_value:
            raise InventoryValidationError("Inventory objects must be a non-empty list")
        if len(objects_value) > 128:
            raise InventoryValidationError("Inventory supports at most 128 objects")

        root = Path(base_dir).resolve()
        container_value = payload["container"]
        container = self._object(container_value, "container", root, is_container=True)
        objects = tuple(
            self._object(value, f"objects[{index}]", root, is_container=False)
            for index, value in enumerate(objects_value)
        )
        identifiers = [container.object_id, *(obj.object_id for obj in objects)]
        if "root" in identifiers:
            raise InventoryValidationError("Object id 'root' is reserved")
        duplicates = sorted(
            {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
        )
        if duplicates:
            raise InventoryValidationError(
                f"Inventory object ids must be unique: {', '.join(duplicates)}"
            )

        arrangement = payload.get("arrangement", {})
        if not isinstance(arrangement, dict):
            raise InventoryValidationError("arrangement must be a JSON object")
        unknown_arrangement = sorted(set(arrangement) - self._arrangement_fields)
        if unknown_arrangement:
            raise InventoryValidationError(
                f"Unknown arrangement fields: {', '.join(unknown_arrangement)}"
            )
        allow_protrusion = _number(
            arrangement.get("allow_protrusion_m", 0.0),
            "arrangement.allow_protrusion_m",
        )
        if allow_protrusion < 0.0:
            raise InventoryValidationError("allow_protrusion_m cannot be negative")

        assert isinstance(container_value, dict)
        position = _vector(
            container_value.get("position_xy_m", [0.0, 0.0]),
            "container.position_xy_m",
            2,
        )
        return InventorySpec(
            container=container,
            objects=objects,
            container_position_xy_m=(position[0], position[1]),
            container_yaw_deg=_number(
                container_value.get("yaw_deg", 0.0),
                "container.yaw_deg",
            ),
            allow_protrusion_m=allow_protrusion,
        )

    def _object(
        self,
        value: Any,
        field: str,
        base_dir: Path,
        *,
        is_container: bool,
    ) -> InventoryObjectSpec:
        if not isinstance(value, dict):
            raise InventoryValidationError(f"{field} must be a JSON object")
        allowed = self._container_fields if is_container else self._object_fields
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise InventoryValidationError(f"Unknown {field} fields: {', '.join(unknown)}")
        object_id = str(value.get("object_id", "")).strip()
        category = str(value.get("category", "")).strip()
        if not object_id:
            raise InventoryValidationError(f"{field}.object_id must be non-empty")
        if not category:
            raise InventoryValidationError(f"{field}.category must be non-empty")
        uid_value = value.get("asset_uid")
        asset_uid = None if uid_value is None else str(uid_value).strip()
        if uid_value is not None and not asset_uid:
            raise InventoryValidationError(f"{field}.asset_uid must be non-empty")
        asset_value = value.get("asset")
        if asset_uid is not None and asset_value is not None:
            raise InventoryValidationError(f"{field} cannot set both asset_uid and asset")
        asset = (
            None
            if asset_value is None
            else self._inline_asset(asset_value, object_id, category, field, base_dir)
        )
        if is_container and asset is not None and asset.container_inner_size_m is None:
            raise InventoryValidationError(
                f"{field}.asset.container_inner_size_m is required for a custom container"
            )
        return InventoryObjectSpec(object_id, category, asset_uid, asset)

    def _inline_asset(
        self,
        value: Any,
        object_id: str,
        category: str,
        field: str,
        base_dir: Path,
    ) -> AssetRecord:
        if not isinstance(value, dict):
            raise InventoryValidationError(f"{field}.asset must be a JSON object")
        unknown = sorted(set(value) - self._asset_fields)
        if unknown:
            raise InventoryValidationError(f"Unknown {field}.asset fields: {', '.join(unknown)}")
        if "size_m" not in value:
            raise InventoryValidationError(f"{field}.asset.size_m is required")

        mesh_path = None
        mesh_hash = value.get("sha256")
        if value.get("mesh_path") is not None:
            candidate = Path(str(value["mesh_path"]))
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise InventoryValidationError(f"Mesh file does not exist: {candidate}")
            mesh_path = str(candidate)
            actual_hash = _sha256(candidate)
            if mesh_hash is not None and str(mesh_hash).lower() != actual_hash:
                raise InventoryValidationError(f"SHA-256 mismatch for {candidate}")
            mesh_hash = actual_hash

        def optional_vec3(key: str) -> tuple[float, ...] | None:
            if value.get(key) is None:
                return None
            return _vector(value[key], f"{field}.asset.{key}", 3)

        return AssetRecord(
            asset_id=str(value.get("asset_id", f"inventory:{object_id}")),
            description=category.lower().replace("_", " "),
            size_m=_vector(value["size_m"], f"{field}.asset.size_m", 3),
            mass_kg=_number(value.get("mass_kg", 1.0), f"{field}.asset.mass_kg"),
            friction=_number(value.get("friction", 0.6), f"{field}.asset.friction"),
            com_shift_m=_vector(
                value.get("com_shift_m", [0.0, 0.0, 0.0]),
                f"{field}.asset.com_shift_m",
                3,
            ),
            supporting_probability=_number(
                value.get("supporting_probability", 0.5),
                f"{field}.asset.supporting_probability",
            ),
            front_yaw_rad=_number(
                value.get("front_yaw_rad", 0.0),
                f"{field}.asset.front_yaw_rad",
            ),
            source=str(value.get("source", "user_inventory")),
            license=str(value.get("license", "user-provided")),
            source_url=None if value.get("source_url") is None else str(value["source_url"]),
            author=None if value.get("author") is None else str(value["author"]),
            sha256=None if mesh_hash is None else str(mesh_hash).lower(),
            visual_size_m=optional_vec3("visual_size_m"),
            mesh_path=mesh_path,
            mesh_scale=_vector(
                value.get("mesh_scale", [1.0, 1.0, 1.0]),
                f"{field}.asset.mesh_scale",
                3,
            ),
            mesh_offset_m=_vector(
                value.get("mesh_offset_m", [0.0, 0.0, 0.0]),
                f"{field}.asset.mesh_offset_m",
                3,
            ),
            mesh_euler_deg=_vector(
                value.get("mesh_euler_deg", [0.0, 0.0, 0.0]),
                f"{field}.asset.mesh_euler_deg",
                3,
            ),
            mesh_file_is_zup=_boolean(
                value.get("mesh_file_is_zup", False),
                f"{field}.asset.mesh_file_is_zup",
            ),
            stackable=_boolean(
                value.get("stackable", False),
                f"{field}.asset.stackable",
            ),
            stacking_step_ratio=_number(
                value.get("stacking_step_ratio", 1.0),
                f"{field}.asset.stacking_step_ratio",
            ),
            collision_size_m=optional_vec3("collision_size_m"),
            container_inner_size_m=optional_vec3("container_inner_size_m"),
            visual_shape=str(value.get("visual_shape", "box")),
            color_rgba=_vector(
                value.get("color_rgba", [0.65, 0.65, 0.65, 1.0]),
                f"{field}.asset.color_rgba",
                4,
            ),
        )
