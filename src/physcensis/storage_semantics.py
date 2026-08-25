"""Household storage compatibility rules for organized container packing."""

from __future__ import annotations

from dataclasses import dataclass

from physcensis.types import AssetRecord, SceneObject


@dataclass(frozen=True)
class StorageProfile:
    """Coarse semantic properties used by the deterministic organizer."""

    group: str
    family: str
    fragile_support: bool = False
    loose_item: bool = False


_PROFILES = {
    "notebook": StorageProfile("notebook", "paper"),
    "book": StorageProfile("book", "paper"),
    "laptop": StorageProfile("laptop", "electronics", fragile_support=True),
    "keyboard": StorageProfile("keyboard", "electronics", fragile_support=True),
    "monitor": StorageProfile("monitor", "electronics", fragile_support=True),
    "phone": StorageProfile("phone", "electronics", fragile_support=True, loose_item=True),
    "mouse": StorageProfile("mouse", "electronics", fragile_support=True, loose_item=True),
    "pencil": StorageProfile("pencil", "stationery", loose_item=True),
    "pen": StorageProfile("pen", "stationery", loose_item=True),
    # A motor is a heavy workshop base: unlike electronics, it can safely share
    # a support layer with hand tools in a crate.
    "motor": StorageProfile("motor", "hand_tools"),
    "drill": StorageProfile("drill", "hand_tools"),
    "wrench": StorageProfile("wrench", "hand_tools", loose_item=True),
    "hammer": StorageProfile("hammer", "hand_tools"),
    "saw": StorageProfile("saw", "hand_tools"),
    "can": StorageProfile("can", "canned_goods"),
    "jar": StorageProfile("jar", "pantry_goods"),
    "bottle": StorageProfile("bottle", "pantry_goods"),
    "grocery_carton": StorageProfile("carton", "pantry_goods"),
    "pantry_box": StorageProfile("package", "pantry_goods"),
    "plate": StorageProfile("plate", "dishware"),
    "bowl": StorageProfile("bowl", "dishware"),
    "stackable_cup": StorageProfile("stackable_cup", "dishware"),
    "cup": StorageProfile("cup", "dishware"),
}


def storage_profile(asset: AssetRecord) -> StorageProfile:
    """Resolve a stable profile from the normalized primitive description."""
    key = asset.description.lower().replace(" ", "_")
    return _PROFILES.get(key, StorageProfile(key, key))


def semantically_compatible_support(
    subject: SceneObject,
    supporters: list[SceneObject],
) -> bool:
    """Return whether a storage arrangement resembles ordinary use.

    Heavy or broad items may only be placed on the same semantic family. Small
    loose items can use any non-fragile support. A fragile electronic surface
    is never treated as a generic shelf, even if it is geometrically flat.
    """
    if not supporters:
        return False
    subject_profile = storage_profile(subject.asset)
    supporter_profiles = [storage_profile(supporter.asset) for supporter in supporters]
    if any(
        profile.fragile_support and profile.group != subject_profile.group
        for profile in supporter_profiles
    ):
        return False
    if any(profile.group == subject_profile.group for profile in supporter_profiles):
        return True
    if subject_profile.loose_item:
        return True
    return any(profile.family == subject_profile.family for profile in supporter_profiles)
