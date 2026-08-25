"""Support-aware alignment for presentation meshes and procedural visuals."""

from __future__ import annotations

from dataclasses import dataclass

from physcensis.types import SceneObject, SceneState, Vec3


@dataclass(frozen=True)
class VisualContactLayout:
    """Resolved visual centers plus measurable contact-correction provenance."""

    centers_m: dict[str, Vec3]
    bottoms_m: dict[str, float]
    tops_m: dict[str, float]
    correction_m: dict[str, float]
    gap_before_alignment_m: dict[str, float]
    contact_gap_m: dict[str, float]
    unresolved_object_ids: tuple[str, ...]

    def metrics(self, *, violation_tolerance_m: float = 0.005) -> dict[str, float]:
        gaps = list(self.contact_gap_m.values())
        corrections = [
            abs(self.correction_m[object_id]) for object_id in self.contact_gap_m
        ]
        gaps_before = list(self.gap_before_alignment_m.values())
        return {
            "visual_contact_count": float(len(gaps)),
            "maximum_visual_contact_gap_m": max(gaps, default=0.0),
            "visual_contact_violation_count": float(
                sum(gap > violation_tolerance_m for gap in gaps)
            ),
            "maximum_visual_contact_gap_before_alignment_m": max(
                gaps_before, default=0.0
            ),
            "maximum_visual_alignment_correction_m": max(corrections, default=0.0),
            "mean_visual_alignment_correction_m": (
                sum(corrections) / len(corrections) if corrections else 0.0
            ),
            "unresolved_visual_support_count": float(len(self.unresolved_object_ids)),
        }


def visual_height_m(obj: SceneObject) -> float:
    """Return the audited visible height, falling back to the authored size."""
    return (obj.asset.visual_size_m or obj.asset.size_m)[2]


def build_visual_contact_layout(scene: SceneState) -> VisualContactLayout:
    """Align visible surfaces along the recorded dense-container support graph.

    Physics proxies remain untouched. Objects on the container floor retain their
    proxy-aligned visible bottom, while supported objects are recursively lowered
    or raised until their visible bottom meets the highest visible supporter top.
    """

    support_map = scene.metadata.get("container_supports", {})
    if not isinstance(support_map, dict):
        support_map = {}

    centers: dict[str, Vec3] = {}
    bottoms: dict[str, float] = {}
    tops: dict[str, float] = {}
    corrections: dict[str, float] = {}
    gaps_before: dict[str, float] = {}
    contact_gaps: dict[str, float] = {}
    unresolved: set[str] = set()
    visiting: set[str] = set()

    def use_proxy_alignment(object_id: str) -> tuple[float, float]:
        obj = scene.get(object_id)
        height = visual_height_m(obj)
        bottom = obj.bottom_z
        center = bottom + height / 2.0
        centers[object_id] = (obj.position_m[0], obj.position_m[1], center)
        bottoms[object_id] = bottom
        tops[object_id] = bottom + height
        corrections[object_id] = center - obj.visual_position_m[2]
        return bottom, bottom + height

    def resolve(object_id: str) -> tuple[float, float]:
        if object_id in bottoms:
            return bottoms[object_id], tops[object_id]
        if object_id in visiting:
            unresolved.update(visiting)
            unresolved.add(object_id)
            return use_proxy_alignment(object_id)

        visiting.add(object_id)
        obj = scene.get(object_id)
        recorded = support_map.get(object_id)
        supporter_ids = (
            [
                supporter_id
                for supporter_id in recorded
                if supporter_id in scene.objects
                and supporter_id != object_id
                and scene.get(supporter_id).asset.container_inner_size_m is None
            ]
            if isinstance(recorded, list)
            else []
        )
        if not supporter_ids:
            result = use_proxy_alignment(object_id)
        else:
            supporter_top = max(
                resolve(supporter_id)[1] for supporter_id in supporter_ids
            )
            height = visual_height_m(obj)
            visible_bottom_before = obj.bottom_z
            visible_bottom = supporter_top
            visible_top = visible_bottom + height
            center = visible_bottom + height / 2.0
            centers[object_id] = (obj.position_m[0], obj.position_m[1], center)
            bottoms[object_id] = visible_bottom
            tops[object_id] = visible_top
            corrections[object_id] = center - obj.visual_position_m[2]
            gaps_before[object_id] = max(0.0, visible_bottom_before - supporter_top)
            contact_gaps[object_id] = abs(visible_bottom - supporter_top)
            result = visible_bottom, visible_top
        visiting.discard(object_id)
        return result

    for object_id in scene.objects:
        resolve(object_id)

    return VisualContactLayout(
        centers_m=centers,
        bottoms_m=bottoms,
        tops_m=tops,
        correction_m=corrections,
        gap_before_alignment_m=gaps_before,
        contact_gap_m=contact_gaps,
        unresolved_object_ids=tuple(sorted(unresolved)),
    )
