"""Similarity alignment helpers for camera-frame object registration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float64)
        return float(self.scale) * (points @ np.asarray(self.rotation, dtype=np.float64).T) + np.asarray(
            self.translation,
            dtype=np.float64,
        )

    def matrix(self) -> np.ndarray:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = float(self.scale) * np.asarray(self.rotation, dtype=np.float64)
        mat[:3, 3] = np.asarray(self.translation, dtype=np.float64)
        return mat


def estimate_similarity_umeyama(source_xyz: np.ndarray, target_xyz: np.ndarray) -> SimilarityTransform:
    source = np.asarray(source_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source and target must both be Nx3 with same shape, got {source.shape} and {target.shape}")
    if source.shape[0] < 3:
        raise ValueError("at least 3 point pairs are required")

    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src_centered = source - src_mean
    dst_centered = target - dst_mean

    covariance = (dst_centered.T @ src_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    d = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        d[-1] = -1.0
    rotation = u @ np.diag(d) @ vt
    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    if variance <= 0.0:
        raise ValueError("source points are degenerate")
    scale = float(np.sum(singular_values * d) / variance)
    scale = float(np.round(scale, 12))
    translation = dst_mean - scale * (rotation @ src_mean)
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)
