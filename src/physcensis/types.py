"""Serializable domain types shared by every pipeline stage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


class PredicateKind(str, Enum):
    LEFT_OF = "LEFT-OF"
    RIGHT_OF = "RIGHT-OF"
    FRONT_OF = "FRONT-OF"
    BACK_OF = "BACK-OF"
    ALIGN_CENTER_LR = "ALIGN-CENTER-LR"
    ALIGN_CENTER_FB = "ALIGN-CENTER-FB"
    ALIGN_LEFT = "ALIGN-LEFT"
    ALIGN_RIGHT = "ALIGN-RIGHT"
    ALIGN_FRONT = "ALIGN-FRONT"
    ALIGN_BACK = "ALIGN-BACK"
    SYMMETRY_ALONG = "SYMMETRY-ALONG"
    FACING_TO = "FACING-TO"
    FACING_SAME_AS = "FACING-SAME-AS"
    FACING_OPPOSITE_TO = "FACING-OPPOSITE-TO"
    FACING_FRONT = "FACING-FRONT"
    FACING_BACK = "FACING-BACK"
    FACING_LEFT = "FACING-LEFT"
    FACING_RIGHT = "FACING-RIGHT"
    RANDOM_ROT = "RANDOM-ROT"
    ORIENT_BY_RELATIVE_SIDE = "ORIENT-BY-RELATIVE-SIDE"
    PLACE_ON_BASE = "PLACE-ON-BASE"
    PLACE_ON = "PLACE-ON"
    PLACE_IN = "PLACE-IN"
    PLACE_ANYWHERE = "PLACE-ANYWHERE"
    GROUP = "GROUP"
    COPY_GROUP = "COPY-GROUP"

    @property
    def is_physical(self) -> bool:
        return self in {self.PLACE_ON, self.PLACE_IN, self.PLACE_ANYWHERE}


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    description: str
    size_m: Vec3
    mass_kg: float = 1.0
    friction: float = 0.5
    com_shift_m: Vec3 = (0.0, 0.0, 0.0)
    supporting_probability: float = 0.5
    front_yaw_rad: float = 0.0
    source: str = "procedural"
    license: str = "generated"
    source_url: str | None = None
    author: str | None = None
    sha256: str | None = None
    quality_score: int | None = None
    transparent_visual: bool | None = None
    visual_qa_status: str | None = None
    mesh_path: str | None = None
    mesh_scale: Vec3 = (1.0, 1.0, 1.0)
    mesh_offset_m: Vec3 = (0.0, 0.0, 0.0)
    mesh_euler_deg: Vec3 = (0.0, 0.0, 0.0)
    mesh_file_is_zup: bool = False
    stackable: bool = False
    stacking_step_ratio: float = 1.0
    collision_size_m: Vec3 | None = None
    container_inner_size_m: Vec3 | None = None
    visual_shape: str = "box"
    color_rgba: tuple[float, float, float, float] = (0.65, 0.65, 0.65, 1.0)

    def __post_init__(self) -> None:
        if any(value <= 0 for value in self.size_m):
            raise ValueError(f"Asset {self.asset_id} must have positive dimensions")
        if self.mass_kg <= 0:
            raise ValueError(f"Asset {self.asset_id} must have positive mass")
        if self.friction < 0:
            raise ValueError(f"Asset {self.asset_id} cannot have negative friction")
        if not 0.0 < self.stacking_step_ratio <= 1.0:
            raise ValueError(
                f"Asset {self.asset_id} stacking_step_ratio must be in (0, 1]"
            )
        if self.collision_size_m is not None and any(
            value <= 0 for value in self.collision_size_m
        ):
            raise ValueError(f"Asset {self.asset_id} must have positive collision dimensions")

    @property
    def physical_size_m(self) -> Vec3:
        return self.collision_size_m or self.size_m


@dataclass
class SceneObject:
    object_id: str
    asset: AssetRecord
    position_m: Vec3 = (0.0, 0.0, 0.0)
    yaw_rad: float = 0.0
    fixed: bool = False
    support_id: str | None = None

    @property
    def top_z(self) -> float:
        return self.position_m[2] + self.asset.physical_size_m[2] / 2.0

    @property
    def bottom_z(self) -> float:
        return self.position_m[2] - self.asset.physical_size_m[2] / 2.0

    @property
    def visual_position_m(self) -> Vec3:
        physical_height = self.asset.physical_size_m[2]
        visual_height = self.asset.size_m[2]
        return (
            self.position_m[0],
            self.position_m[1],
            self.position_m[2] + (visual_height - physical_height) / 2.0,
        )

    def moved(self, *, position_m: Vec3 | None = None, yaw_rad: float | None = None) -> SceneObject:
        return replace(
            self,
            position_m=self.position_m if position_m is None else position_m,
            yaw_rad=self.yaw_rad if yaw_rad is None else yaw_rad,
        )


@dataclass(frozen=True)
class GroupState:
    group_id: str
    object_ids: tuple[str, ...]
    anchor_id: str


@dataclass
class SceneState:
    root_size_m: Vec3
    root_height_m: float
    objects: dict[str, SceneObject] = field(default_factory=dict)
    groups: dict[str, GroupState] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def root_top_z(self) -> float:
        return self.root_height_m + self.root_size_m[2] / 2.0

    @property
    def root_bounds_xy(self) -> tuple[float, float, float, float]:
        return (
            -self.root_size_m[0] / 2.0,
            self.root_size_m[0] / 2.0,
            -self.root_size_m[1] / 2.0,
            self.root_size_m[1] / 2.0,
        )

    def add_object(self, obj: SceneObject) -> None:
        if obj.object_id in self.objects or obj.object_id == "root":
            raise ValueError(f"Duplicate or reserved object id: {obj.object_id}")
        self.objects[obj.object_id] = obj

    def get(self, object_id: str) -> SceneObject:
        try:
            return self.objects[object_id]
        except KeyError as exc:
            raise KeyError(f"Unknown scene object: {object_id}") from exc

    def clone(self) -> SceneState:
        return SceneState(
            root_size_m=self.root_size_m,
            root_height_m=self.root_height_m,
            objects={key: replace(value) for key, value in self.objects.items()},
            groups=dict(self.groups),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_size_m": list(self.root_size_m),
            "root_height_m": self.root_height_m,
            "objects": {
                key: {
                    "object_id": obj.object_id,
                    "asset": asdict(obj.asset),
                    "position_m": list(obj.position_m),
                    "yaw_rad": obj.yaw_rad,
                    "fixed": obj.fixed,
                    "support_id": obj.support_id,
                }
                for key, obj in sorted(self.objects.items())
            },
            "groups": {key: asdict(group) for key, group in sorted(self.groups.items())},
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Predicate:
    subject: str | tuple[tuple[str, int], ...]
    kind: PredicateKind
    reference: str | tuple[str, ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    source_index: int = -1


@dataclass(frozen=True)
class PlacementProgram:
    descriptions: Mapping[str, str]
    predicates: tuple[Predicate, ...]
    raw_payload: tuple[Any, ...] = ()

    def predicates_of(self, kinds: Iterable[PredicateKind]) -> tuple[Predicate, ...]:
        accepted = set(kinds)
        return tuple(predicate for predicate in self.predicates if predicate.kind in accepted)


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    object_id: str | None = None
    predicate_index: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SolveReport:
    success: bool
    scene: SceneState
    issues: list[Issue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    solved_object_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SimulationResult:
    success: bool
    final_positions_m: Mapping[str, Vec3]
    displacement_m: Mapping[str, float]
    fallen_object_ids: tuple[str, ...] = ()
    penetrations: tuple[tuple[str, str], ...] = ()

    @property
    def mean_displacement_m(self) -> float:
        if not self.displacement_m:
            return 0.0
        return sum(self.displacement_m.values()) / len(self.displacement_m)


@dataclass(frozen=True)
class StabilityResult:
    object_id: str
    sample_count: int
    local_failure_probability: float
    stable_sample_fraction: float
    most_unstable_stable_offset: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Feedback:
    category: str
    summary: str
    issues: tuple[Issue, ...]
    measurements: Mapping[str, float] = field(default_factory=dict)
