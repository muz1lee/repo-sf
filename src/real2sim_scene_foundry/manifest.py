"""Scene manifest data model for real-to-sim reconstruction outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CAMERA_FRAME = "opencv_x_right_y_down_z_forward_meters"
WORLD_FRAME = "z_up_ground_plane_meters"


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    label: str
    mesh_path: str
    mask_path: str
    crop_path: str
    T_object_to_camera: list[list[float]]
    T_object_to_world: list[list[float]]
    scale_m: float
    mass_kg: float
    friction: float
    confidence: float
    source_backend: str = "unknown"
    needs_manual_refine: bool = False

    def validate(self) -> None:
        if not self.object_id:
            raise ValueError("object_id is required")
        if not self.label:
            raise ValueError(f"{self.object_id}: label is required")
        if self.scale_m <= 0.0:
            raise ValueError(f"{self.object_id}: scale_m must be positive")
        if self.mass_kg <= 0.0:
            raise ValueError(f"{self.object_id}: mass_kg must be positive")
        if self.friction <= 0.0:
            raise ValueError(f"{self.object_id}: friction must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"{self.object_id}: confidence must be in [0, 1]")
        _validate_transform(self.T_object_to_camera, f"{self.object_id}: T_object_to_camera")
        _validate_transform(self.T_object_to_world, f"{self.object_id}: T_object_to_world")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneBackground:
    source_backend: str
    bg_only_image_path: str
    foreground_mask_path: str
    point_cloud_path: str
    status: str
    gaussian_splat_path: str | None = None
    gaussian_splat_config_path: str | None = None
    gaussian_splat_checkpoint_path: str | None = None

    def validate(self) -> None:
        if not self.source_backend:
            raise ValueError("background source_backend is required")
        if not self.bg_only_image_path:
            raise ValueError("background bg_only_image_path is required")
        if not self.foreground_mask_path:
            raise ValueError("background foreground_mask_path is required")
        if not self.point_cloud_path:
            raise ValueError("background point_cloud_path is required")
        if not self.status:
            raise ValueError("background status is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneManifest:
    objects: list[SceneObject]
    background: SceneBackground | None = None
    version: int = 1

    def validate(self) -> None:
        if self.background is not None:
            self.background.validate()
        seen: set[str] = set()
        for obj in self.objects:
            obj.validate()
            if obj.object_id in seen:
                raise ValueError(f"duplicate object_id: {obj.object_id}")
            seen.add(obj.object_id)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = {
            "version": int(self.version),
            "coordinate_frames": {
                "camera": CAMERA_FRAME,
                "world": WORLD_FRAME,
            },
            "objects": [obj.to_dict() for obj in self.objects],
        }
        if self.background is not None:
            data["background"] = self.background.to_dict()
        return data


def _validate_transform(value: list[list[float]], context: str) -> None:
    if len(value) != 4:
        raise ValueError(f"{context} must be 4x4")
    for row in value:
        if len(row) != 4:
            raise ValueError(f"{context} must be 4x4")
        for item in row:
            if not isinstance(item, (int, float)):
                raise ValueError(f"{context} entries must be numeric")
