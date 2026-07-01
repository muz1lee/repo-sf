"""Background registration metadata for viewer and simulator export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_BACKGROUND_SOURCE_KINDS = {
    "bg_only_cloud",
    "registered_3dgs",
    "3dgs_native_asset_candidate",
    "full_scene_cloud_debug",
    "3dgs_unavailable",
}


@dataclass(frozen=True)
class BackgroundRegistrationResult:
    path: Path
    data: dict[str, Any]


def write_background_registration(
    run_dir: str | Path,
    *,
    source_kind: str | None = None,
    status: str | None = None,
    scale_source: str | None = None,
    blocked_reason: str | None = None,
) -> BackgroundRegistrationResult:
    run = Path(run_dir)
    bg_dir = run / "background"
    bg_dir.mkdir(parents=True, exist_ok=True)
    source = source_kind or _infer_source_kind(run)
    if source not in VALID_BACKGROUND_SOURCE_KINDS:
        raise ValueError(f"unknown background source_kind: {source}")

    gs_status = _load_3dgs_status(run)
    gs_completed = gs_status.get("status") == "completed"
    gs_transform, gs_transform_source, gs_registration_status = _3dgs_transform_record(source, gs_completed)
    effective_status = status or _default_status(source, gs_completed)
    effective_scale_source = scale_source or _default_scale_source(source, gs_completed)
    reason = blocked_reason or _default_blocked_reason(source, effective_status, gs_completed)
    bg_only_transform = _opencv_cloud_to_sim_world(_support_height_m(run))

    data: dict[str, Any] = {
        "version": 1,
        "source_kind": source,
        "status": effective_status,
        "scale_source": effective_scale_source,
        "T_bg_only_cloud_to_sim_world": bg_only_transform,
        "T_3dgs_world_to_sim_world": gs_transform,
        "transforms": {
            "T_bg_only_cloud_to_sim_world": bg_only_transform,
            "T_3dgs_world_to_sim_world": gs_transform,
        },
        "transform_sources": {
            "T_bg_only_cloud_to_sim_world": "opencv_camera_cloud_to_z_up_world_using_support_height",
            "T_3dgs_world_to_sim_world": gs_transform_source,
        },
        "registrations": {
            "bg_only_cloud": {
                "status": "registered" if (run / "background" / "bg_only_cloud.ply").is_file() else "missing",
                "transform": bg_only_transform,
                "transform_source": "opencv_camera_cloud_to_z_up_world_using_support_height",
                "scale_source": "input_metric_depth",
            },
            "3dgs": {
                "status": gs_registration_status,
                "transform": gs_transform,
                "transform_source": gs_transform_source,
                "scale_source": "unknown" if gs_completed else None,
                "blocked_reason": (
                    "No shared camera pose and metric scale evidence is available to derive "
                    "T_3dgs_world_to_sim_world."
                    if gs_completed and gs_transform is None
                    else None
                ),
            },
        },
        "assets": {
            "bg_only_cloud": _rel_if_exists(run, run / "background" / "bg_only_cloud.ply"),
            "3dgs_native_asset_candidate": _rel_if_exists(run, _native_3dgs_asset_path(run)),
            "full_scene_cloud_debug": _rel_if_exists(run, run / "scene_cloud.ply"),
        },
        "gaussian_splat": _gaussian_splat_metadata(gs_status, run, source, gs_registration_status),
    }
    if reason:
        data["blocked_reason"] = reason

    path = bg_dir / "registration.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return BackgroundRegistrationResult(path=path, data=data)


def load_background_registration(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir) / "background" / "registration.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_source_kind(run: Path) -> str:
    if (run / "background" / "bg_only_cloud.ply").is_file():
        return "bg_only_cloud"
    if _native_3dgs_asset_path(run).is_file():
        return "3dgs_native_asset_candidate"
    status = _load_3dgs_status(run)
    if status.get("status") == "completed":
        return "3dgs_unavailable"
    if (run / "scene_cloud.ply").is_file():
        return "full_scene_cloud_debug"
    return "3dgs_unavailable"


def _default_status(source_kind: str, gs_completed: bool) -> str:
    if source_kind == "bg_only_cloud":
        return "partial_external_render_only" if gs_completed else "registered"
    if source_kind == "registered_3dgs":
        return "blocked_missing_3dgs_transform_evidence"
    if source_kind == "3dgs_native_asset_candidate":
        return "blocked_native_3dgs_not_integrated"
    if source_kind == "full_scene_cloud_debug":
        return "blocked_missing_bg_only_background"
    return "blocked_3dgs_unavailable"


def _default_scale_source(source_kind: str, gs_completed: bool) -> str:
    del gs_completed
    if source_kind == "bg_only_cloud":
        return "input_metric_depth"
    if source_kind == "registered_3dgs":
        return "moge_reference_metric_points"
    if source_kind == "3dgs_native_asset_candidate":
        return "unknown_native_asset_candidate"
    return "unknown"


def _default_blocked_reason(source_kind: str, status: str, gs_completed: bool) -> str | None:
    if source_kind == "registered_3dgs":
        return "3DGS registration requested, but no camera pose and metric scale bridge is available."
    if source_kind == "3dgs_native_asset_candidate":
        return "Native 3DGS asset candidate exists, but viewer/simulator runtime loading is not integrated."
    if status == "blocked_missing_bg_only_background":
        return "BG-only cloud is missing; full scene cloud is diagnostic-only because it contains foreground objects."
    if status != "registered" and gs_completed:
        return "3DGS training completed, but native viewer/simulator 3DGS loading is not integrated."
    return None


def _3dgs_transform_record(
    source_kind: str,
    gs_completed: bool,
) -> tuple[list[list[float]] | None, str, str]:
    if not gs_completed:
        return None, "not_available", "not_available"
    del source_kind
    return (
        None,
        "blocked_missing_camera_pose_scale_evidence",
        "blocked_missing_camera_pose_scale_evidence",
    )


def _gaussian_splat_metadata(
    status: dict[str, Any],
    run: Path,
    source_kind: str,
    registration_status: str,
) -> dict[str, Any]:
    outputs = status.get("outputs", {}) if isinstance(status.get("outputs", {}), dict) else {}
    config_path = outputs.get("latest_config")
    checkpoint_path = outputs.get("latest_checkpoint")
    run_path = outputs.get("latest_run_dir") or outputs.get("output_dir")
    native_asset_path = _rel_if_exists(run, _native_3dgs_asset_path(run))
    if status.get("status") == "completed":
        if source_kind == "registered_3dgs":
            return {
                "status": "blocked_missing_camera_pose_scale_evidence",
                "training_status": "completed",
                "run_path": run_path,
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "native_asset_path": native_asset_path,
                "native_rendering": False,
                "registration_status": registration_status,
                "runtime_status": "blocked_native_3dgs_not_integrated",
                "note": "3DGS training completed, but no metric bridge into simulator world is available.",
            }
        if native_asset_path:
            return {
                "status": "native_asset_candidate",
                "training_status": "completed",
                "run_path": run_path,
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "native_asset_path": native_asset_path,
                "native_rendering": False,
                "registration_status": registration_status,
                "runtime_status": "blocked_native_3dgs_not_integrated",
                "note": "3DGS native asset candidate exists, but simulator/viewer runtime loading is not integrated.",
            }
        return {
            "status": "sidecar_not_native",
            "training_status": "completed",
            "run_path": run_path,
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "native_rendering": False,
            "registration_status": registration_status,
            "note": "3DGS is registered as a sidecar artifact; BG-only metric cloud is the simulator background visual.",
        }
    if native_asset_path:
        return {
            "status": "native_asset_candidate",
            "training_status": status.get("status"),
            "run_path": run_path,
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "native_asset_path": native_asset_path,
            "native_rendering": False,
            "registration_status": registration_status,
            "runtime_status": "blocked_native_3dgs_not_integrated",
        }
    if status:
        return {
            "status": str(status.get("status", "unknown")),
            "training_status": status.get("status"),
            "run_path": run_path,
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
        }
    return {"status": "not_available", "training_status": None, "run_path": None, "config_path": None, "checkpoint_path": None}


def _support_height_m(run: Path) -> float:
    manifest_path = run / "scene_manifest.json"
    if not manifest_path.is_file():
        return 0.0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.0
    support = manifest.get("support_plane", {})
    value = support.get("original_height_world_m", support.get("height_world_m", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _opencv_cloud_to_sim_world(support_height_m: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, -float(support_height_m)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _identity4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _load_3dgs_status(run: Path) -> dict[str, Any]:
    path = run / "video" / "3dgs_status.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_status_json"}
    return data if isinstance(data, dict) else {"status": "invalid_status_json"}


def _native_3dgs_asset_path(run: Path) -> Path:
    return run / "background" / "3dgs_native" / "splat_rgb.ply"


def _rel_if_exists(run: Path, path: Path) -> str | None:
    return path.relative_to(run).as_posix() if path.is_file() else None
