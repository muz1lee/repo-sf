"""Export-level QA report for simulator export bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sim_export_manifest import write_sim_export_manifest


@dataclass(frozen=True)
class SimExportQAResult:
    report_path: Path
    report: dict[str, Any]


def write_sim_export_report(run_dir: str | Path) -> SimExportQAResult:
    run = Path(run_dir)
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    _ensure_table_collision_projection_qa(run)
    manifest_path = write_sim_export_manifest(run)
    _refresh_backend_export_reports(run)
    manifest_path = write_sim_export_manifest(run)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    sections = {
        "pose_alignment": _pose_alignment_section(manifest),
        "background_registration": _background_registration_section(manifest),
        "collision_assets": _collision_assets_section(manifest),
        "table_collision_projection": _table_collision_projection_section(run, manifest),
        "physics_settle": _physics_settle_section(run),
        "usd_export": _usd_export_section(run, manifest),
        "genesis_export": _genesis_export_section(run, manifest),
        "isaac_export": _isaac_export_section(run),
    }
    blocking_reasons = _dedupe(manifest.get("blocking_reasons", []) + _section_blocking_reasons(sections))
    overall_status = "passed" if all(section["status"] == "passed" for section in sections.values()) and not blocking_reasons else "blocked"
    report = {
        "version": 1,
        "overall_status": overall_status,
        "blocking_reasons": blocking_reasons,
        "manifest_status": manifest.get("status"),
        "sections": sections,
        "artifacts": {
            "sim_export_manifest": "sim_export_manifest.json",
            "usd_scene": "exports/scene.usda",
            "genesis_scene": "exports/genesis_scene.py",
            "isaac_scene": "exports/isaac_scene.py",
            "reference_input": _first_existing(run, ["left.png", "video/reference.png"]),
            "render_export": "qa/render_export.png",
            "background_3dgs_render": "qa/background_3dgs_render.png",
            "background_3dgs_render_report": "qa/background_3dgs_render_report.json",
            "render_vs_input_overlay": "qa/render_vs_input_overlay.png",
            "object_projection_overlay": "qa/object_projection_overlay.png",
            "background_registration_overlay": "qa/background_registration_overlay.png",
            "table_collision_overlay": "qa/table_collision_overlay.png",
            "table_collision_report": "qa/table_collision_report.json",
            "genesis_settle_report": "qa/genesis_settle_report.json",
            "isaac_load_report": "qa/isaac_load_report.json",
        },
    }
    report_path = qa_dir / "sim_export_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return SimExportQAResult(report_path=report_path, report=report)



def _ensure_table_collision_projection_qa(run: Path) -> None:
    report_path = run / "qa" / "table_collision_report.json"
    polygon_path = run / "background" / "table_polygon_world.json"
    if not polygon_path.is_file():
        return
    try:
        from .table_collision_qa import write_table_collision_projection_qa

        write_table_collision_projection_qa(run)
    except Exception as exc:  # noqa: BLE001 - QA must fail closed with a report artifact.
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "status": "blocked",
                    "blocking_reasons": ["table_collision_projection_failed"],
                    "error": str(exc),
                    "visual_qa_path": None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _refresh_backend_export_reports(run: Path) -> None:
    if (run / "exports" / "scene.usda").is_file():
        from .usd_export import export_usd_scene

        export_usd_scene(run)
    if (run / "exports" / "genesis_scene.py").is_file():
        from .genesis_export import export_genesis_scene

        export_genesis_scene(run)


def _pose_alignment_section(manifest: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    objects = []
    for obj in manifest.get("objects", []):
        pose = obj["pose"]
        status = "passed" if pose.get("status") == "ready" else "blocked"
        if status != "passed":
            reasons.append(f"pose_not_ready:{obj['object_id']}")
        if pose.get("needs_manual_refine"):
            reasons.append(f"pose_needs_manual_refine:{obj['object_id']}")
        support_alignment = pose.get("support_alignment") or {}
        if support_alignment.get("status") == "penetrating_support":
            reasons.append(f"pose_support_penetration:{obj['object_id']}")
        elif support_alignment.get("status") == "unreadable_visual_geometry":
            reasons.append(f"pose_support_geometry_unreadable:{obj['object_id']}")
        objects.append(
            {
                "object_id": obj["object_id"],
                "status": status,
                "rotation_source": pose.get("rotation_source"),
                "translation_source": pose.get("translation_source"),
                "scale_source": pose.get("scale_source"),
                "pose_refinement_report": pose.get("pose_refinement_report"),
                "support_alignment": support_alignment,
            }
        )
    return {"status": "passed" if not reasons else "blocked", "blocking_reasons": _dedupe(reasons), "objects": objects}


def _background_registration_section(manifest: dict[str, Any]) -> dict[str, Any]:
    background = manifest.get("background", {})
    visual = background.get("visual_asset", {})
    registration = background.get("registration", {})
    reasons = []
    if visual.get("status") != "ready":
        reasons.append(str(visual.get("status") or "missing_background_visual"))
    if registration.get("status") != "registered":
        reasons.append("background_unregistered")
    reasons.extend(str(item) for item in registration.get("blocking_reasons", []))
    return {
        "status": "passed" if not reasons else "blocked",
        "blocking_reasons": _dedupe(reasons),
        "visual_asset": visual,
        "registration": registration,
        "external_render": background.get("external_render"),
    }


def _collision_assets_section(manifest: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    objects = []
    for obj in manifest.get("objects", []):
        collision = obj["collision_asset"]
        visual = obj["visual_asset"]
        if collision.get("status") != "ready":
            reasons.append(f"missing_collision_mesh:{obj['object_id']}")
        if visual.get("status") == "blocked_proxy_visual_asset":
            reasons.append(f"blocked_proxy_visual_asset:{obj['object_id']}")
        elif visual.get("status") != "ready":
            reasons.append(f"visual_asset_not_ready:{obj['object_id']}")
        objects.append({"object_id": obj["object_id"], "visual_asset": visual, "collision_asset": collision})
    return {"status": "passed" if not reasons else "blocked", "blocking_reasons": _dedupe(reasons), "objects": objects}


def _table_collision_projection_section(run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    support = manifest.get("support_surface", {})
    report_path = run / "qa" / "table_collision_report.json"
    report = _load_json(report_path)
    reasons: list[str] = []
    if not report:
        reasons.append("table_collision_missing_visual_qa")
    else:
        if report.get("status") != "passed":
            reasons.append("table_collision_projection_failed")
        if report.get("derived_from_tabletop_mask") is not True:
            reasons.append("table_collision_not_tabletop_mask_derived")
        geometry_type = str(report.get("geometry_type") or support.get("table_collision_geometry_type") or "")
        if geometry_type not in {"polygon_slab", "convex_hull_slab"}:
            reasons.append("table_collision_not_tabletop_mask_derived")
        visual_qa_path = str(report.get("visual_qa_path") or support.get("table_collision_visual_qa_path") or "")
        if not visual_qa_path or not (run / visual_qa_path).is_file():
            reasons.append("table_collision_missing_visual_qa")
        reasons.extend(str(item) for item in report.get("blocking_reasons", []))
    reasons.extend(str(item) for item in support.get("blocking_reasons", []))
    reasons = _dedupe(reasons)
    return {
        "status": "passed" if not reasons else "blocked",
        "blocking_reasons": reasons,
        "report_path": "qa/table_collision_report.json" if report_path.is_file() else None,
        "overlay_path": report.get("visual_qa_path") or support.get("table_collision_visual_qa_path"),
        "projection_iou": report.get("projection_iou"),
        "source_backend": report.get("source_backend") or support.get("table_collision_source_backend"),
        "geometry_type": report.get("geometry_type") or support.get("table_collision_geometry_type"),
        "derived_from_tabletop_mask": report.get("derived_from_tabletop_mask"),
    }


def _physics_settle_section(run: Path) -> dict[str, Any]:
    path = run / "qa" / "genesis_settle_report.json"
    if not path.is_file():
        return {"status": "blocked", "blocking_reasons": ["physics_settle_missing"], "report_path": None}
    report = json.loads(path.read_text(encoding="utf-8"))
    reasons = []
    if report.get("status") != "completed":
        reasons.append("genesis_settle_not_completed")
    if report.get("stability_status") != "passed":
        reasons.append("genesis_settle_failed")
    if report.get("nan_detected"):
        reasons.append("genesis_settle_nan_detected")
    if report.get("fall_out_detected"):
        reasons.append("genesis_settle_fall_out_detected")
    if report.get("fall_below_support_detected"):
        reasons.append("genesis_settle_fall_below_support")
    if report.get("excessive_displacement_detected"):
        reasons.append("genesis_settle_excessive_displacement")
    if float(report.get("max_penetration_depth_m", 0.0)) > 0.005:
        reasons.append("genesis_settle_penetration_too_deep")
    return {
        "status": "passed" if not reasons else "blocked",
        "blocking_reasons": reasons,
        "report_path": "qa/genesis_settle_report.json",
        "stability_status": report.get("stability_status"),
        "max_penetration_depth_m": report.get("max_penetration_depth_m"),
        "max_displacement_m": report.get("max_displacement_m"),
        "fall_below_support_detected": report.get("fall_below_support_detected"),
        "excessive_displacement_detected": report.get("excessive_displacement_detected"),
    }


def _usd_export_section(run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = run / "exports" / "scene.usda"
    report = _load_json(run / "qa" / "usd_export_report.json")
    reasons = []
    if not path.is_file():
        reasons.append("missing_backend_export:usd")
    if report and report.get("status") != "exported":
        reasons.append("usd_export_failed")
    if not all(obj["visual_asset"].get("path") for obj in manifest.get("objects", [])):
        reasons.append("usd_missing_visual_references")
    if not all(obj["collision_asset"].get("path") for obj in manifest.get("objects", [])):
        reasons.append("usd_missing_collision_references")
    return {"status": "passed" if not reasons else "blocked", "blocking_reasons": _dedupe(reasons), "path": "exports/scene.usda" if path.is_file() else None}


def _genesis_export_section(run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    del manifest
    path = run / "exports" / "genesis_scene.py"
    report = _load_json(run / "qa" / "genesis_export_report.json")
    reasons = []
    if not path.is_file():
        reasons.append("missing_backend_export:genesis")
    if report and report.get("status") != "script_written":
        reasons.append("genesis_export_failed")
    return {
        "status": "passed" if not reasons else "blocked",
        "blocking_reasons": _dedupe(reasons),
        "path": "exports/genesis_scene.py" if path.is_file() else None,
        "report_path": "qa/genesis_export_report.json" if report else None,
    }


def _isaac_export_section(run: Path) -> dict[str, Any]:
    path = run / "exports" / "isaac_scene.py"
    report = _load_json(run / "qa" / "isaac_load_report.json")
    repeatability = report.get("repeatability") if isinstance(report.get("repeatability"), dict) else {}
    if report.get("status") == "runtime_unavailable":
        return {
            "status": "runtime_unavailable",
            "blocking_reasons": ["isaac_runtime_unavailable"],
            "path": "exports/isaac_scene.py" if path.is_file() else None,
            "report_path": "qa/isaac_load_report.json",
            "reason": report.get("reason"),
            "runtime_status": report.get("status"),
            "report_source": report.get("report_source"),
            "validation_reused": bool(report.get("validation_reused", False)),
            "direct_runtime_validation": bool(report.get("direct_runtime_validation", False)),
            "repeatability_status": _isaac_repeatability_status(report),
            "repeatability": repeatability,
        }
    reasons = []
    if not path.is_file():
        reasons.append("missing_backend_export:isaac")
    if not report:
        reasons.append("isaac_load_report_missing")
    elif report.get("status") not in {"loaded", "passed"}:
        reasons.append("isaac_load_failed")
    return {
        "status": "passed" if not reasons else "blocked",
        "blocking_reasons": _dedupe(reasons),
        "path": "exports/isaac_scene.py" if path.is_file() else None,
        "report_path": "qa/isaac_load_report.json" if report else None,
        "runtime_status": report.get("status"),
        "report_source": report.get("report_source"),
        "validation_reused": bool(report.get("validation_reused", False)),
        "direct_runtime_validation": bool(report.get("direct_runtime_validation", False)),
        "repeatability_status": _isaac_repeatability_status(report),
        "repeatability": repeatability,
        "native_3dgs_status": report.get("native_3dgs_status"),
        "background_runtime_verified": bool(report.get("background_runtime_verified", False)),
    }


def _isaac_repeatability_status(report: dict[str, Any]) -> str:
    repeatability = report.get("repeatability") if isinstance(report.get("repeatability"), dict) else {}
    if report.get("report_source") == "preserved_existing_worker_report" or report.get("validation_reused"):
        return "partial_preserved_report"
    if report.get("status") == "runtime_unavailable":
        return "runtime_unavailable"
    if repeatability.get("fresh_direct_validation") is True or report.get("direct_runtime_validation") is True:
        return "fresh_direct_report"
    if report:
        return "unknown_report_provenance"
    return "missing_report"


def _section_blocking_reasons(sections: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for section in sections.values():
        reasons.extend(str(item) for item in section.get("blocking_reasons", []))
    return _dedupe(reasons)


def _first_existing(run: Path, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if (run / candidate).is_file():
            return candidate
    return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out



def run_export_qa(run_dir: str | Path) -> Path:
    return write_sim_export_report(run_dir).report_path
