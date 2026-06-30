"""Camera intrinsics and OpenCV-frame projection helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [
                [float(self.fx), 0.0, float(self.cx)],
                [0.0, float(self.fy), float(self.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def k_string(self) -> str:
        return " ".join(str(float(v)) for v in self.K.reshape(-1))

    def backproject_depth(self, depth: np.ndarray) -> np.ndarray:
        z = np.asarray(depth, dtype=np.float32)
        expected = (int(self.height), int(self.width))
        if z.shape != expected:
            raise ValueError(f"depth must have shape {expected}, got {z.shape}")
        yy, xx = np.indices(expected, dtype=np.float32)
        x = (xx - float(self.cx)) * z / float(self.fx)
        y = (yy - float(self.cy)) * z / float(self.fy)
        xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
        xyz[z <= 0.0] = 0.0
        return xyz

    def project(self, points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_xyz must be Nx3, got {points.shape}")
        z = points[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = points[:, 0] / z * float(self.fx) + float(self.cx)
            v = points[:, 1] / z * float(self.fy) + float(self.cy)
        return u, v, z

    @classmethod
    def from_calibration_json(
        cls,
        path: str | Path,
        *,
        camera_name: str | None = None,
    ) -> tuple["CameraIntrinsics", float]:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if camera_name is not None:
            data = data[camera_name]
        elif "head" in data and "fx" not in data:
            data = data["head"]

        width = data.get("width") or data.get("w")
        height = data.get("height") or data.get("h")
        if width is None or height is None:
            raise ValueError("calibration JSON must include width/height")

        baseline = data.get("baseline_m", data.get("baseline"))
        if baseline is None:
            raise ValueError("calibration JSON must include baseline or baseline_m")
        baseline = float(baseline)
        if baseline > 1.0:
            baseline /= 1000.0

        return (
            cls(
                width=int(width),
                height=int(height),
                fx=float(data["fx"]),
                fy=float(data.get("fy", data["fx"])),
                cx=float(data["cx"]),
                cy=float(data["cy"]),
            ),
            baseline,
        )
