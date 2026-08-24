"""Licensed external-asset manifests and deterministic catalog overlay."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from physcensis.assets import PrimitiveAssetCatalog
from physcensis.types import AssetRecord, PlacementProgram

ALLOWED_LICENSES = frozenset({"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0"})


class AssetManifestError(ValueError):
    """Raised when provenance, licensing, or installed bytes are invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vec3(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise AssetManifestError(f"{field} must contain three numbers")
    return tuple(float(component) for component in value)


def _optional_bool(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise AssetManifestError(f"{field} must be a boolean")
    return value


@dataclass(frozen=True)
class LicensedAssetEntry:
    category: str
    uid: str
    title: str
    author: str
    license: str
    license_url: str
    source_url: str
    download_url: str
    filename: str
    sha256: str
    raw_bounds_min: tuple[float, float, float]
    raw_bounds_max: tuple[float, float, float]
    source_up_axis: str = "y"
    face_count: int = 0
    vertex_count: int = 0
    quality_score: int | None = None
    is_transparent: bool | None = None
    visual_qa_status: str = "unverified"
    stackable: bool = False
    stacking_step_ratio: float = 1.0

    @classmethod
    def from_mapping(cls, value: Any) -> LicensedAssetEntry:
        if not isinstance(value, dict):
            raise AssetManifestError("Each asset entry must be an object")
        required = {
            "category",
            "uid",
            "title",
            "author",
            "license",
            "license_url",
            "source_url",
            "download_url",
            "filename",
            "sha256",
            "raw_bounds_min",
            "raw_bounds_max",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise AssetManifestError(f"Asset entry is missing: {', '.join(missing)}")
        entry = cls(
            category=str(value["category"]).strip().lower().replace("_", " "),
            uid=str(value["uid"]),
            title=str(value["title"]),
            author=str(value["author"]),
            license=str(value["license"]),
            license_url=str(value["license_url"]),
            source_url=str(value["source_url"]),
            download_url=str(value["download_url"]),
            filename=str(value["filename"]),
            sha256=str(value["sha256"]).lower(),
            raw_bounds_min=_vec3(value["raw_bounds_min"], "raw_bounds_min"),
            raw_bounds_max=_vec3(value["raw_bounds_max"], "raw_bounds_max"),
            source_up_axis=str(value.get("source_up_axis", "y")).lower(),
            face_count=int(value.get("face_count", 0)),
            vertex_count=int(value.get("vertex_count", 0)),
            quality_score=(
                None
                if value.get("quality_score") is None
                else int(value["quality_score"])
            ),
            is_transparent=_optional_bool(
                value.get("is_transparent"), "is_transparent"
            ),
            visual_qa_status=str(value.get("visual_qa_status", "unverified")),
            stackable=_optional_bool(value.get("stackable", False), "stackable") or False,
            stacking_step_ratio=float(value.get("stacking_step_ratio", 1.0)),
        )
        entry.validate()
        return entry

    def validate(self) -> None:
        if self.license not in ALLOWED_LICENSES:
            raise AssetManifestError(
                f"Asset {self.uid} uses disallowed license {self.license!r}"
            )
        if self.source_up_axis not in {"y", "z"}:
            raise AssetManifestError(f"Asset {self.uid} source_up_axis must be y or z")
        if self.filename != f"{self.uid}.glb" or Path(self.filename).name != self.filename:
            raise AssetManifestError(f"Asset {self.uid} has an unsafe or noncanonical filename")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise AssetManifestError(f"Asset {self.uid} has an invalid SHA-256")
        if not self.download_url.startswith("https://huggingface.co/datasets/allenai/objaverse/"):
            raise AssetManifestError(f"Asset {self.uid} download URL is not frozen to Objaverse")
        if not self.source_url.startswith("https://sketchfab.com/3d-models/"):
            raise AssetManifestError(f"Asset {self.uid} source URL is not a Sketchfab model")
        if any(high <= low for low, high in zip(self.raw_bounds_min, self.raw_bounds_max)):
            raise AssetManifestError(f"Asset {self.uid} raw bounds must have positive extents")
        if self.quality_score is not None and self.quality_score not in {0, 1, 2, 3}:
            raise AssetManifestError(f"Asset {self.uid} quality score must be 0 through 3")
        if self.visual_qa_status not in {"unverified", "thumbnail_pass", "genesis_pass"}:
            raise AssetManifestError(f"Asset {self.uid} has an invalid visual QA status")
        if not 0.0 < self.stacking_step_ratio <= 1.0:
            raise AssetManifestError(
                f"Asset {self.uid} stacking_step_ratio must be in (0, 1]"
            )
        if not self.stackable and self.stacking_step_ratio != 1.0:
            raise AssetManifestError(
                f"Asset {self.uid} cannot use a compressed stacking step when stackable is false"
            )

    def installed_path(self, cache_dir: Path) -> Path:
        return cache_dir / self.filename

    def mesh_transform(
        self, target_size_m: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        center = tuple((low + high) / 2.0 for low, high in zip(self.raw_bounds_min, self.raw_bounds_max))
        extent = tuple(high - low for low, high in zip(self.raw_bounds_min, self.raw_bounds_max))
        if self.source_up_axis == "y":
            source_scale = (
                target_size_m[0] / extent[0],
                target_size_m[2] / extent[1],
                target_size_m[1] / extent[2],
            )
            transformed_center = (
                center[0] * source_scale[0],
                -center[2] * source_scale[2],
                center[1] * source_scale[1],
            )
        else:
            source_scale = tuple(target / raw for target, raw in zip(target_size_m, extent))
            transformed_center = tuple(value * scale for value, scale in zip(center, source_scale))
        offset = tuple(-value for value in transformed_center)
        return source_scale, offset


@dataclass(frozen=True)
class LicensedAssetManifest:
    name: str
    entries: tuple[LicensedAssetEntry, ...]
    minimum_quality_score: int | None = None
    require_opaque: bool = False
    quality_source_url: str | None = None
    required_visual_qa_status: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> LicensedAssetManifest:
        manifest_path = Path(path)
        with manifest_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise AssetManifestError("Asset manifest schema_version must be 1")
        entries = tuple(
            LicensedAssetEntry.from_mapping(value) for value in data.get("assets", [])
        )
        if not entries:
            raise AssetManifestError("Asset manifest must contain at least one asset")
        uids = [entry.uid for entry in entries]
        if len(uids) != len(set(uids)):
            raise AssetManifestError("Asset manifest contains duplicate UIDs")
        quality_gate = data.get("quality_gate", {})
        if not isinstance(quality_gate, dict):
            raise AssetManifestError("quality_gate must be an object")
        minimum_quality_score = quality_gate.get("minimum_score")
        manifest = cls(
            name=str(data.get("name", manifest_path.stem)),
            entries=entries,
            minimum_quality_score=(
                None
                if minimum_quality_score is None
                else int(minimum_quality_score)
            ),
            require_opaque=bool(quality_gate.get("require_opaque", False)),
            quality_source_url=(
                None
                if quality_gate.get("source_url") is None
                else str(quality_gate["source_url"])
            ),
            required_visual_qa_status=(
                None
                if quality_gate.get("required_visual_qa_status") is None
                else str(quality_gate["required_visual_qa_status"])
            ),
        )
        manifest.validate_quality_gate()
        return manifest

    def validate_quality_gate(self) -> None:
        if self.minimum_quality_score is not None:
            if self.minimum_quality_score not in {0, 1, 2, 3}:
                raise AssetManifestError("minimum quality score must be 0 through 3")
            for entry in self.entries:
                if entry.quality_score is None:
                    raise AssetManifestError(
                        f"Asset {entry.uid} is missing its required quality score"
                    )
                if entry.quality_score < self.minimum_quality_score:
                    raise AssetManifestError(
                        f"Asset {entry.uid} quality score {entry.quality_score} is below "
                        f"the required {self.minimum_quality_score}"
                    )
        if self.require_opaque:
            for entry in self.entries:
                if entry.is_transparent is not False:
                    raise AssetManifestError(
                        f"Asset {entry.uid} does not satisfy the opaque-only gate"
                    )
        if self.required_visual_qa_status is not None:
            if self.required_visual_qa_status != "genesis_pass":
                raise AssetManifestError(
                    "required visual QA status must be genesis_pass"
                )
            for entry in self.entries:
                if entry.visual_qa_status != self.required_visual_qa_status:
                    raise AssetManifestError(
                        f"Asset {entry.uid} has not passed required Genesis visual QA"
                    )
        if (
            self.minimum_quality_score is not None or self.require_opaque
        ) and not (
            self.quality_source_url
            and self.quality_source_url.startswith(
                "https://huggingface.co/datasets/cindyxl/ObjaversePlusPlus"
            )
        ):
            raise AssetManifestError(
                "quality-gated manifests must cite the Objaverse++ source"
            )

    def validate_files(self, cache_dir: str | Path) -> dict[str, str]:
        root = Path(cache_dir)
        status: dict[str, str] = {}
        for entry in self.entries:
            path = entry.installed_path(root)
            if not path.is_file():
                raise AssetManifestError(f"Missing asset file: {path}")
            actual = _sha256(path)
            if actual != entry.sha256:
                raise AssetManifestError(
                    f"SHA-256 mismatch for {entry.uid}: expected {entry.sha256}, got {actual}"
                )
            status[entry.uid] = actual
        return status

    def fetch(self, cache_dir: str | Path) -> dict[str, str]:
        root = Path(cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        for entry in self.entries:
            path = entry.installed_path(root)
            if path.is_file() and _sha256(path) == entry.sha256:
                continue
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                urllib.request.urlretrieve(entry.download_url, temporary)
                actual = _sha256(temporary)
                if actual != entry.sha256:
                    raise AssetManifestError(
                        f"Downloaded SHA-256 mismatch for {entry.uid}: {actual}"
                    )
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return self.validate_files(root)


class ManifestAssetCatalog:
    """Overlay licensed meshes on the deterministic procedural physics catalog."""

    def __init__(self, manifest: LicensedAssetManifest, cache_dir: str | Path):
        self.manifest = manifest
        self.cache_dir = Path(cache_dir).resolve()
        self.manifest.validate_files(self.cache_dir)
        self.fallback = PrimitiveAssetCatalog()
        by_category: dict[str, list[LicensedAssetEntry]] = {}
        for entry in manifest.entries:
            by_category.setdefault(entry.category, []).append(entry)
        self.by_category = {key: tuple(values) for key, values in by_category.items()}

    @classmethod
    def load(cls, manifest_path: str | Path, cache_dir: str | Path) -> ManifestAssetCatalog:
        return cls(LicensedAssetManifest.load(manifest_path), cache_dir)

    @staticmethod
    def _variant_index(object_id: str, count: int) -> int:
        suffix = object_id.rsplit("_", 1)[-1]
        if suffix.isdigit():
            return int(suffix) % count
        digest = hashlib.sha256(object_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % count

    def resolve(self, object_id: str, description: str) -> AssetRecord:
        base = self.fallback.resolve(object_id, description)
        searchable = f"{object_id} {description}".lower().replace("_", " ")
        matches = sorted(
            (category for category in self.by_category if category in searchable),
            key=len,
            reverse=True,
        )
        if not matches:
            return base
        variants = self.by_category[matches[0]]
        entry = variants[self._variant_index(object_id, len(variants))]
        mesh_scale, mesh_offset = entry.mesh_transform(base.size_m)
        return replace(
            base,
            asset_id=f"objaverse:{entry.uid}:{object_id}",
            source="Objaverse 1.0 / Sketchfab",
            license=entry.license,
            source_url=entry.source_url,
            author=entry.author,
            sha256=entry.sha256,
            quality_score=entry.quality_score,
            transparent_visual=entry.is_transparent,
            visual_qa_status=entry.visual_qa_status,
            mesh_path=str(entry.installed_path(self.cache_dir)),
            mesh_scale=mesh_scale,
            mesh_offset_m=mesh_offset,
            mesh_file_is_zup=entry.source_up_axis == "z",
            stackable=entry.stackable,
            stacking_step_ratio=entry.stacking_step_ratio,
        )

    def resolve_category(self, category: str, object_id: str) -> AssetRecord:
        return self.resolve(object_id, category)

    def resolve_program(self, program: PlacementProgram) -> dict[str, AssetRecord]:
        return {
            object_id: self.resolve(object_id, description)
            for object_id, description in program.descriptions.items()
        }
