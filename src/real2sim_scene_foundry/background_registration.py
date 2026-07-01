"""Background registration metadata for viewer and simulator export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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


def register_3dgs_background(run_dir: str | Path, *, method: str = "camera-sim3", write: bool = True) -> BackgroundRegistrationResult:
    run = Path(run_dir)
    bg_dir = run / "background"
    bg_dir.mkdir(parents=True, exist_ok=True)
    gs_status = _load_3dgs_status(run)
    gs_completed = gs_status.get("status") == "completed"
    if method == "known-phone-sim-world":
        return _register_known_phone_sim_world(run, bg_dir, gs_status, gs_completed, write=write)
    camera = _load_json(run / "camera.json")
    anchor = _load_json(bg_dir / "3dgs_camera_pose.json")
    blocking_reasons: list[str] = []
    t_gs_cam = _transform(anchor.get("T_3dgs_camera_to_world") or anchor.get("T_gs_camera_to_world"))
    t_sim_cam = _transform(camera.get("T_camera_to_world") or camera.get("T_sim_camera_to_world"))
    if t_gs_cam is None:
        blocking_reasons.append("missing_3dgs_anchor_camera_pose")
    if t_sim_cam is None:
        blocking_reasons.append("missing_sim_anchor_camera_pose")
    if not gs_completed:
        blocking_reasons.append("3dgs_training_not_completed")

    bg_only_transform = _opencv_cloud_to_sim_world(_support_height_m(run))
    if blocking_reasons:
        data = {
            "version": 1,
            "source_kind": "registered_3dgs",
            "status": "blocked_missing_camera_pose_scale_evidence",
            "method": method,
            "scale": None,
            "scale_source": "unknown",
            "anchor_frame": anchor.get("anchor_frame"),
            "T_bg_only_cloud_to_sim_world": bg_only_transform,
            "T_3dgs_world_to_sim_world": None,
            "transforms": {
                "T_bg_only_cloud_to_sim_world": bg_only_transform,
                "T_3dgs_world_to_sim_world": None,
            },
            "transform_sources": {
                "T_bg_only_cloud_to_sim_world": "opencv_camera_cloud_to_z_up_world_using_support_height",
                "T_3dgs_world_to_sim_world": "blocked_missing_camera_pose_scale_evidence",
            },
            "registrations": {
                "bg_only_cloud": {
                    "status": "registered" if (run / "background" / "bg_only_cloud.ply").is_file() else "missing",
                    "transform": bg_only_transform,
                    "transform_source": "opencv_camera_cloud_to_z_up_world_using_support_height",
                    "scale_source": "input_metric_depth",
                },
                "3dgs": {
                    "status": "blocked_missing_camera_pose_scale_evidence",
                    "transform": None,
                    "transform_source": "blocked_missing_camera_pose_scale_evidence",
                    "scale_source": None,
                    "blocked_reason": "Missing shared camera pose and metric scale evidence.",
                },
            },
            "assets": {
                "bg_only_cloud": _rel_if_exists(run, run / "background" / "bg_only_cloud.ply"),
                "3dgs_native_asset_candidate": _rel_if_exists(run, _native_3dgs_asset_path(run)),
                "full_scene_cloud_debug": _rel_if_exists(run, run / "scene_cloud.ply"),
            },
            "gaussian_splat": _gaussian_splat_metadata(gs_status, run, "registered_3dgs", "blocked_missing_camera_pose_scale_evidence"),
            "blocking_reasons": _dedupe(blocking_reasons),
            "blocked_reason": "No shared camera pose and metric scale evidence is available to derive T_3dgs_world_to_sim_world.",
        }
    else:
        transform = t_sim_cam @ np.linalg.inv(t_gs_cam)
        transform_list = transform.tolist()
        data = {
            "version": 1,
            "source_kind": "registered_3dgs",
            "status": "registered",
            "method": method,
            "scale": 1.0,
            "scale_source": "shared_anchor_camera_metric_bridge",
            "anchor_frame": anchor.get("anchor_frame"),
            "coordinate_convention": {
                "3dgs": str(anchor.get("coordinate_convention", "3dgs_camera_to_world")),
                "sim": "z_up_meter_camera_to_world",
            },
            "metrics": {
                "anchor_camera_count": 1,
                "camera_center_rmse_m": 0.0,
            },
            "T_bg_only_cloud_to_sim_world": bg_only_transform,
            "T_3dgs_world_to_sim_world": transform_list,
            "transforms": {
                "T_bg_only_cloud_to_sim_world": bg_only_transform,
                "T_3dgs_world_to_sim_world": transform_list,
            },
            "transform_sources": {
                "T_bg_only_cloud_to_sim_world": "opencv_camera_cloud_to_z_up_world_using_support_height",
                "T_3dgs_world_to_sim_world": "camera_anchor_bridge",
            },
            "registrations": {
                "bg_only_cloud": {
                    "status": "registered" if (run / "background" / "bg_only_cloud.ply").is_file() else "missing",
                    "transform": bg_only_transform,
                    "transform_source": "opencv_camera_cloud_to_z_up_world_using_support_height",
                    "scale_source": "input_metric_depth",
                },
                "3dgs": {
                    "status": "registered",
                    "transform": transform_list,
                    "transform_source": "camera_anchor_bridge",
                    "scale_source": "shared_anchor_camera_metric_bridge",
                    "blocked_reason": None,
                },
            },
            "assets": {
                "bg_only_cloud": _rel_if_exists(run, run / "background" / "bg_only_cloud.ply"),
                "3dgs_native_asset_candidate": _rel_if_exists(run, _native_3dgs_asset_path(run)),
                "full_scene_cloud_debug": _rel_if_exists(run, run / "scene_cloud.ply"),
            },
            "gaussian_splat": _gaussian_splat_metadata(gs_status, run, "registered_3dgs", "registered"),
        }
    path = bg_dir / "registration.json"
    if write:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return BackgroundRegistrationResult(path=path, data=data)


def _register_known_phone_sim_world(
    run: Path,
    bg_dir: Path,
    gs_status: dict[str, Any],
    gs_completed: bool,
    *,
    write: bool,
) -> BackgroundRegistrationResult:
    bg_only_transform = _opencv_cloud_to_sim_world(_support_height_m(run))
    evidence, evidence_reasons = _phone_sim_world_identity_evidence(run)
    blocking_reasons = list(evidence_reasons)
    if not gs_completed:
        blocking_reasons.append("3dgs_training_not_completed")
    identity_allowed = not blocking_reasons
    identity = np.eye(4, dtype=np.float64).tolist()
    if identity_allowed:
        data = {
            "version": 1,
            "source_kind": "registered_3dgs",
            "status": "registered",
            "method": "known-phone-sim-world",
            "scale": 1.0,
            "scale_source": "arkit_sceneDepth_meters",
            "identity_allowed": True,
            "evidence": evidence,
            "coordinate_convention": {
                "3dgs": "trained_with_phone_sim_world_camera_poses",
                "sim": "phone_sim_world_meters",
            },
            "metrics": {
                "anchor_camera_count": int(evidence.get("frame_count", 0)),
                "camera_center_rmse_m": 0.0,
            },
            "T_bg_only_cloud_to_sim_world": bg_only_transform,
            "T_3dgs_world_to_sim_world": identity,
            "transforms": {
                "T_bg_only_cloud_to_sim_world": bg_only_transform,
                "T_3dgs_world_to_sim_world": identity,
            },
            "transform_sources": {
                "T_bg_only_cloud_to_sim_world": "opencv_camera_cloud_to_z_up_world_using_support_height",
                "T_3dgs_world_to_sim_world": "identity_allowed_by_phone_sim_world_training",
            },
            "registrations": {
                "bg_only_cloud": {
                    "status": "registered" if (run / "background" / "bg_only_cloud.ply").is_file() else "missing",
                    "transform": bg_only_transform,
                    "transform_source": "opencv_camera_cloud_to_z_up_world_using_support_height",
                    "scale_source": "input_metric_depth",
                },
                "3dgs": {
                    "status": "registered",
                    "transform": identity,
                    "transform_source": "identity_allowed_by_phone_sim_world_training",
                    "scale_source": "arkit_sceneDepth_meters",
                    "identity_allowed": True,
                    "blocked_reason": None,
                },
            },
            "assets": {
                "bg_only_cloud": _rel_if_exists(run, run / "background" / "bg_only_cloud.ply"),
                "3dgs_native_asset_candidate": _rel_if_exists(run, _native_3dgs_asset_path(run)),
                "full_scene_cloud_debug": _rel_if_exists(run, run / "scene_cloud.ply"),
            },
            "gaussian_splat": _gaussian_splat_metadata(gs_status, run, "registered_3dgs", "registered"),
        }
    else:
        data = {
            "version": 1,
            "source_kind": "registered_3dgs",
            "status": "blocked_identity_transform_without_phone_sim_world_evidence",
            "method": "known-phone-sim-world",
            "scale": None,
            "scale_source": "unknown",
            "identity_allowed": False,
            "evidence": evidence,
            "T_bg_only_cloud_to_sim_world": bg_only_transform,
            "T_3dgs_world_to_sim_world": None,
            "transforms": {
                "T_bg_only_cloud_to_sim_world": bg_only_transform,
                "T_3dgs_world_to_sim_world": None,
            },
            "transform_sources": {
                "T_bg_only_cloud_to_sim_world": "opencv_camera_cloud_to_z_up_world_using_support_height",
                "T_3dgs_world_to_sim_world": "blocked_identity_transform_without_phone_sim_world_evidence",
            },
            "registrations": {
                "bg_only_cloud": {
                    "status": "registered" if (run / "background" / "bg_only_cloud.ply").is_file() else "missing",
                    "transform": bg_only_transform,
                    "transform_source": "opencv_camera_cloud_to_z_up_world_using_support_height",
                    "scale_source": "input_metric_depth",
                },
                "3dgs": {
                    "status": "blocked_identity_transform_without_phone_sim_world_evidence",
                    "transform": None,
                    "transform_source": "blocked_identity_transform_without_phone_sim_world_evidence",
                    "scale_source": None,
                    "identity_allowed": False,
                    "blocked_reason": "Identity transform requires phone capture Nerfstudio export with pose_world=sim.",
                },
            },
            "assets": {
                "bg_only_cloud": _rel_if_exists(run, run / "background" / "bg_only_cloud.ply"),
                "3dgs_native_asset_candidate": _rel_if_exists(run, _native_3dgs_asset_path(run)),
                "full_scene_cloud_debug": _rel_if_exists(run, run / "scene_cloud.ply"),
            },
            "gaussian_splat": _gaussian_splat_metadata(
                gs_status,
                run,
                "registered_3dgs",
                "blocked_identity_transform_without_phone_sim_world_evidence",
            ),
            "blocking_reasons": _dedupe(blocking_reasons),
            "blocked_reason": "Identity T_3dgs_world_to_sim_world is forbidden without phone sim-world training evidence.",
        }
    path = bg_dir / "registration.json"
    if write:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return BackgroundRegistrationResult(path=path, data=data)


def qa_background_registration(run_dir: str | Path, *, heldout_frames: int = 16) -> dict[str, Any]:
    run = Path(run_dir)
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    registration = load_background_registration(run) or {}
    reasons: list[str] = []
    if registration.get("status") != "registered":
        reasons.append("background_unregistered")
    if registration.get("T_3dgs_world_to_sim_world") is None:
        reasons.append("missing_T_3dgs_world_to_sim_world")
    transform_source = str((registration.get("transform_sources") or {}).get("T_3dgs_world_to_sim_world") or "")
    if not transform_source or any(marker in transform_source.lower() for marker in ("placeholder", "unknown", "not_available")):
        reasons.append("background_registration_transform_not_evidenced")
    overlay_path = qa_dir / "background_registration_overlay.png"
    Image.new("RGB", (2, 2), color=(0, 0, 0)).save(overlay_path)
    report = {
        "version": 1,
        "status": "passed" if not reasons else "blocked",
        "blocking_reasons": _dedupe(reasons),
        "heldout_frames_requested": int(heldout_frames),
        "registration_path": "background/registration.json" if (run / "background" / "registration.json").is_file() else None,
        "overlay_path": "qa/background_registration_overlay.png",
        "T_3dgs_world_to_sim_world": registration.get("T_3dgs_world_to_sim_world"),
        "transform_source": transform_source or None,
        "metrics": registration.get("metrics", {}),
    }
    (qa_dir / "background_registration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


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


def _phone_sim_world_identity_evidence(run: Path) -> tuple[dict[str, Any], list[str]]:
    transforms_path = run / "video" / "nerfstudio_phone" / "transforms.json"
    gs_status = _load_3dgs_status(run)
    gs_inputs = gs_status.get("inputs", {}) if isinstance(gs_status.get("inputs"), dict) else {}
    gs_claim = gs_status.get("claim", {}) if isinstance(gs_status.get("claim"), dict) else {}
    evidence: dict[str, Any] = {
        "transforms_path": str(transforms_path.relative_to(run)),
        "phone_capture_pose_world": None,
        "identity_allowed": False,
        "frame_count": 0,
        "3dgs_training_status": gs_status.get("status"),
        "3dgs_training_pose_world": gs_inputs.get("pose_world"),
        "3dgs_identity_transform_allowed": bool(gs_claim.get("identity_transform_allowed")),
    }
    if not transforms_path.is_file():
        return evidence, ["missing_phone_sim_world_3dgs_training_evidence"]
    transforms = _load_json(transforms_path)
    phone = transforms.get("phone_capture", {}) if isinstance(transforms.get("phone_capture"), dict) else {}
    frames = transforms.get("frames", []) if isinstance(transforms.get("frames"), list) else []
    evidence.update(
        {
            "phone_capture_pose_world": phone.get("pose_world"),
            "camera_pose_world": phone.get("camera_pose_world"),
            "identity_allowed": bool(phone.get("identity_3dgs_to_sim_allowed_if_trained_with_pose_world")),
            "intrinsics_source": phone.get("intrinsics_source"),
            "extrinsics_source": phone.get("extrinsics_source"),
            "scale_source": phone.get("scale_source"),
            "frame_count": len(frames),
        }
    )
    reasons: list[str] = []
    if phone.get("pose_world") != "sim" or phone.get("camera_pose_world") != "sim":
        reasons.append("phone_capture_pose_world_not_sim")
    if phone.get("identity_3dgs_to_sim_allowed_if_trained_with_pose_world") is not True:
        reasons.append("identity_not_allowed_by_phone_capture_export")
    if phone.get("intrinsics_source") != "arkit_explicit":
        reasons.append("phone_intrinsics_not_arkit_explicit")
    if phone.get("extrinsics_source") != "arkit_explicit":
        reasons.append("phone_extrinsics_not_arkit_explicit")
    if phone.get("scale_source") != "arkit_sceneDepth_meters":
        reasons.append("phone_scale_not_arkit_sceneDepth_meters")
    if not frames:
        reasons.append("phone_nerfstudio_frames_missing")
    if gs_inputs.get("pose_world") != "sim":
        reasons.append("3dgs_training_pose_world_not_sim")
    if gs_claim.get("identity_transform_allowed") is not True:
        reasons.append("3dgs_identity_training_claim_missing")
    return evidence, reasons


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
            if registration_status == "registered":
                return {
                    "status": "registered_3dgs",
                    "training_status": "completed",
                    "run_path": run_path,
                    "config_path": config_path,
                    "checkpoint_path": checkpoint_path,
                    "native_asset_path": native_asset_path,
                    "native_rendering": False,
                    "registration_status": registration_status,
                    "runtime_status": "blocked_native_3dgs_not_integrated",
                    "note": "3DGS has a metric bridge into simulator world; native simulator rendering remains unverified.",
                }
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _transform(value: Any) -> np.ndarray | None:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return None
    return transform


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _native_3dgs_asset_path(run: Path) -> Path:
    return run / "background" / "3dgs_native" / "splat_rgb.ply"


def _rel_if_exists(run: Path, path: Path) -> str | None:
    return path.relative_to(run).as_posix() if path.is_file() else None
