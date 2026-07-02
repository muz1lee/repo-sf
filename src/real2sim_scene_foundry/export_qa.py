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


def write_sim_export_report(run_dir: str | Path, *, strict_claims: bool = False) -> SimExportQAResult:
    run = Path(run_dir)
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    _ensure_table_collision_projection_qa(run, strict_claims=strict_claims)
    manifest_path = write_sim_export_manifest(run)
    _refresh_backend_export_reports(run)
    manifest_path = write_sim_export_manifest(run)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    sections = {
        "pose_alignment": _pose_alignment_section(manifest),
        "background_registration": _background_registration_section(manifest),
        "collision_assets": _collision_assets_section(manifest),
        "table_collision_projection": _table_collision_projection_section(run, manifest, strict_claims=strict_claims),
        "physics_settle": _physics_settle_section(run),
        "usd_export": _usd_export_section(run, manifest),
        "genesis_export": _genesis_export_section(run, manifest),
        "isaac_export": _isaac_export_section(run, strict_claims=strict_claims),
    }
    isaac_fresh_validation = _write_isaac_fresh_validation_report(run, sections["isaac_export"])
    blocking_reasons = _dedupe(manifest.get("blocking_reasons", []) + _section_blocking_reasons(sections))
    overall_status = "passed" if all(section["status"] == "passed" for section in sections.values()) and not blocking_reasons else "blocked"
    claim_gate = _honest_claim_gate(run, manifest, sections, blocking_reasons, strict_claims=strict_claims)
    report = {
        "version": 1,
        "strict_claims": bool(strict_claims),
        "overall_status": overall_status,
        "blocking_reasons": blocking_reasons,
        "manifest_status": manifest.get("status"),
        "claim_gate": claim_gate,
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
            "table_collision_report_v2": "qa/table_collision_report_v2.json",
            "honest_milestone": "qa/honest_milestone.json",
            "genesis_settle_report": "qa/genesis_settle_report.json",
            "isaac_load_report": "qa/isaac_load_report.json",
            "isaac_fresh_validation_report": "qa/isaac_fresh_validation_report.json",
        },
        "isaac_fresh_validation": isaac_fresh_validation,
    }
    report_path = qa_dir / "sim_export_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if strict_claims:
        (qa_dir / "honest_milestone.json").write_text(json.dumps(claim_gate, indent=2), encoding="utf-8")
    return SimExportQAResult(report_path=report_path, report=report)



def _ensure_table_collision_projection_qa(run: Path, *, strict_claims: bool = False) -> None:
    report_path = run / "qa" / "table_collision_report.json"
    polygon_path = run / "background" / "table_polygon_world.json"
    if _write_phone_bg_table_projection_qa(run, strict_claims=strict_claims):
        return
    if not polygon_path.is_file():
        return
    try:
        from .table_collision_qa import write_table_collision_projection_qa, write_table_collision_projection_qa_v2

        write_table_collision_projection_qa(run)
        if strict_claims:
            write_table_collision_projection_qa_v2(run)
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


def _write_phone_bg_table_projection_qa(run: Path, *, strict_claims: bool = False) -> bool:
    scene = _load_json(run / "scene_manifest.phone.json")
    if scene.get("scene_type") != "phone_bg_table_physics_mvp":
        return False
    bg_report = _load_json(run / "qa" / "bg_table_report.json")
    if not bg_report:
        return False
    table = bg_report.get("table_collision") if isinstance(bg_report.get("table_collision"), dict) else {}
    support = scene.get("support_plane") if isinstance(scene.get("support_plane"), dict) else {}
    camera = _load_json(run / "camera.json")
    projection_iou = _float_or_none(table.get("projection_iou") or table.get("raw_projection_iou"))
    tabletop_iou = _float_or_none(table.get("tabletop_iou") or projection_iou)
    visible_iou = _float_or_none(table.get("visible_iou") or projection_iou)
    overreach_ratio = _float_or_none(table.get("overreach_ratio") or table.get("raw_overreach_ratio"))
    undercoverage_ratio = _float_or_none(table.get("undercoverage_ratio"))
    hard_reasons = [str(item) for item in bg_report.get("hard_blocking_reasons", [])]
    partial_reasons = [str(item) for item in bg_report.get("partial_blocking_reasons", [])]
    if not hard_reasons and not partial_reasons:
        raw_reasons = [str(item) for item in bg_report.get("blocking_reasons", [])]
        if bg_report.get("status") == "partial":
            partial_reasons = raw_reasons
        else:
            hard_reasons = raw_reasons
    reasons = list(hard_reasons)
    if bg_report.get("status") not in {"passed", "partial"}:
        reasons.append("bg_table_qa_not_passed")
    if not table:
        reasons.append("phone_table_collision_metrics_missing")
    camera_intrinsics_source = "explicit" if camera.get("intrinsics_source") or camera.get("K") or all(key in camera for key in ("fx", "fy", "cx", "cy")) else "missing"
    camera_extrinsics_source = "explicit" if camera.get("extrinsics_source") or camera.get("T_world_to_camera") or camera.get("T_camera_to_world") else "missing"
    visual_qa_path = _first_existing(run, [
        "qa/bg_table_projection_overlay.png",
        "qa/bg3dgs_table_overlay_000000.png",
        "qa/table_collision_overlay.png",
    ])
    base_reasons = _dedupe(reasons)
    base_report = {
        "version": 1,
        "status": "passed" if not base_reasons else "blocked",
        "blocking_reasons": base_reasons,
        "partial_blocking_reasons": _dedupe(partial_reasons),
        "source_backend": table.get("source_backend") or support.get("table_collision_source_backend"),
        "geometry_type": support.get("table_collision_geometry_type") or "polygon_slab",
        "derived_from_tabletop_mask": True,
        "projection_iou": projection_iou,
        "projection_iou_with_tabletop_mask": tabletop_iou,
        "visible_projection_iou": visible_iou,
        "overreach_ratio": overreach_ratio,
        "undercoverage_ratio": undercoverage_ratio,
        "projection_surface": "phone_semantic_depth_tabletop",
        "visual_qa_path": visual_qa_path,
        "overlay_path": visual_qa_path,
        "camera_intrinsics_source": camera_intrinsics_source,
        "camera_extrinsics_source": camera_extrinsics_source,
        "bg_table_report_path": "qa/bg_table_report.json",
    }
    report_path = run / "qa" / "table_collision_report.json"
    report_path.write_text(json.dumps(base_report, indent=2), encoding="utf-8")
    if strict_claims:
        strict_reasons = list(base_reasons)
        if camera_intrinsics_source != "explicit":
            strict_reasons.append("camera_intrinsics_not_explicit")
        if camera_extrinsics_source != "explicit":
            strict_reasons.append("camera_extrinsics_not_explicit")
        if tabletop_iou is None or tabletop_iou < 0.65:
            strict_reasons.append("below_export_grade_tabletop_iou")
        if visible_iou is None or visible_iou < 0.70:
            strict_reasons.append("below_export_grade_visible_iou")
        if overreach_ratio is not None and overreach_ratio > 0.15:
            strict_reasons.append("table_collision_overreaches_visible_table")
        if undercoverage_ratio is not None and undercoverage_ratio > 0.20:
            strict_reasons.append("table_collision_under_covers_visible_table")
        strict_reasons = _dedupe(strict_reasons)
        strict_report = {
            **base_report,
            "version": 2,
            "status": "passed" if not strict_reasons else "blocked",
            "weak_status": base_report["status"],
            "blocking_reasons": strict_reasons,
            "export_grade_tabletop_iou_threshold": 0.65,
            "export_grade_visible_iou_threshold": 0.70,
            "max_overreach_ratio": 0.15,
            "max_undercoverage_ratio": 0.20,
            "legacy_report_path": "qa/table_collision_report.json",
        }
        (run / "qa" / "table_collision_report_v2.json").write_text(json.dumps(strict_report, indent=2), encoding="utf-8")
    return True


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
    if visual.get("source_kind") == "external_3dgs_renderer" and visual.get("simulator_native") is not True:
        reasons.append("background_external_render_only")
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


def _table_collision_projection_section(run: Path, manifest: dict[str, Any], *, strict_claims: bool = False) -> dict[str, Any]:
    support = manifest.get("support_surface", {})
    strict_report_path = run / "qa" / "table_collision_report_v2.json"
    report_path = strict_report_path if strict_claims and strict_report_path.is_file() else run / "qa" / "table_collision_report.json"
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
        if geometry_type not in {"polygon_slab", "convex_hull_slab", "support_plane_polygon"}:
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
        "report_path": str(report_path.relative_to(run)) if report_path.is_file() else None,
        "overlay_path": report.get("visual_qa_path") or support.get("table_collision_visual_qa_path"),
        "projection_iou": report.get("projection_iou"),
        "projection_iou_with_tabletop_mask": report.get("projection_iou_with_tabletop_mask"),
        "weak_status": report.get("weak_status"),
        "source_backend": report.get("source_backend") or support.get("table_collision_source_backend"),
        "geometry_type": report.get("geometry_type") or support.get("table_collision_geometry_type"),
        "derived_from_tabletop_mask": report.get("derived_from_tabletop_mask"),
    }


def _physics_settle_section(run: Path) -> dict[str, Any]:
    phone_runtime_path = run / "qa" / "table_physics_mvp_runtime_report.json"
    if phone_runtime_path.is_file():
        report = json.loads(phone_runtime_path.read_text(encoding="utf-8"))
        checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
        reasons = []
        if report.get("status") != "completed":
            reasons.append("genesis_settle_not_completed")
        if report.get("stability_status") != "passed":
            reasons.append("genesis_settle_failed")
        if checks.get("nan_detected"):
            reasons.append("genesis_settle_nan_detected")
        if checks.get("fall_below_table"):
            reasons.append("genesis_settle_fall_below_support")
        if checks.get("contact_ok") is False:
            reasons.append("genesis_settle_contact_gap_too_large")
        if checks.get("horizontal_drift_ok") is False:
            reasons.append("genesis_settle_excessive_displacement")
        test_object = report.get("test_object") if isinstance(report.get("test_object"), dict) else {}
        return {
            "status": "passed" if not reasons else "blocked",
            "blocking_reasons": _dedupe(reasons),
            "report_path": "qa/table_physics_mvp_runtime_report.json",
            "stability_status": report.get("stability_status"),
            "settle_steps": report.get("settle_steps"),
            "contact_gap_m": test_object.get("contact_gap_m"),
            "horizontal_drift_m": test_object.get("horizontal_drift_m"),
            "fall_below_support_detected": checks.get("fall_below_table"),
            "excessive_displacement_detected": checks.get("horizontal_drift_ok") is False,
        }
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


def _isaac_export_section(run: Path, *, strict_claims: bool = False) -> dict[str, Any]:
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
    repeatability_status = _isaac_repeatability_status(report)
    status = "passed" if not reasons else "blocked"
    if strict_claims and status == "passed" and repeatability_status == "partial_preserved_report":
        status = "partial"
        reasons.append("isaac_validation_reused")
    return {
        "status": status,
        "blocking_reasons": _dedupe(reasons),
        "path": "exports/isaac_scene.py" if path.is_file() else None,
        "report_path": "qa/isaac_load_report.json" if report else None,
        "runtime_status": report.get("status"),
        "report_source": report.get("report_source"),
        "validation_reused": bool(report.get("validation_reused", False)),
        "direct_runtime_validation": bool(report.get("direct_runtime_validation", False)),
        "repeatability_status": repeatability_status,
        "repeatability": repeatability,
        "native_3dgs_status": report.get("native_3dgs_status"),
        "background_runtime_verified": bool(report.get("background_runtime_verified", False)),
    }


def _honest_claim_gate(
    run: Path,
    manifest: dict[str, Any],
    sections: dict[str, dict[str, Any]],
    blocking_reasons: list[str],
    *,
    strict_claims: bool,
) -> dict[str, Any]:
    del strict_claims
    remaining_gaps = []
    milestone_blockers = []
    background = manifest.get("background", {})
    visual = background.get("visual_asset", {})
    registration = background.get("registration", {})
    if visual.get("source_kind") == "external_3dgs_renderer" and visual.get("simulator_native") is not True:
        remaining_gaps.append("external_3dgs_render_sidecar")
    if visual.get("has_3dgs_sidecar") and visual.get("simulator_native") is not True:
        remaining_gaps.append("simulator_native_3dgs_runtime")
    bg_table = _load_json(run / "qa" / "bg_table_report.json")
    bg_table_partial = [str(item) for item in bg_table.get("partial_blocking_reasons", [])]
    if not bg_table_partial and bg_table.get("status") == "partial":
        bg_table_partial = [str(item) for item in bg_table.get("blocking_reasons", [])]
    if "registered_3dgs_overlay_not_rendered" in bg_table_partial:
        remaining_gaps.append("registered_3dgs_overlay_not_rendered")
    if registration.get("status") != "registered" or registration.get("T_3dgs_world_to_sim_world") is None:
        milestone_blockers.append("background_unregistered")
    if sections.get("table_collision_projection", {}).get("status") != "passed":
        milestone_blockers.append("table_collision_export_grade_blocked")
    isaac = sections.get("isaac_export", {})
    if isaac.get("repeatability_status") == "partial_preserved_report":
        milestone_blockers.append("isaac_validation_reused")
        remaining_gaps.append("fresh_isaac_validation")
    collision_provenance = _load_json(run / "qa" / "collision_rebuild_report.json")
    if collision_provenance:
        collision_paper_status = str(collision_provenance.get("paper_equivalence_status") or collision_provenance.get("reproduction_status") or "")
        if collision_provenance.get("paper_equivalent") is False or collision_paper_status in {"partial", "blocked"}:
            milestone_blockers.append("coacd_collision_decomposition")
            remaining_gaps.append("coacd_collision_decomposition")
    physics_provenance = _load_json(run / "qa" / "physics_property_report.json")
    if physics_provenance:
        physics_paper_status = str(physics_provenance.get("paper_equivalence_status") or physics_provenance.get("reproduction_status") or "")
        if physics_provenance.get("paper_equivalent") is False or physics_paper_status in {"partial", "blocked"}:
            milestone_blockers.append("physics_property_inference")
            remaining_gaps.append("physics_property_inference")
    for obj in manifest.get("objects", []):
        physics = obj.get("physics", {})
        if physics.get("reproduction_status") == "partial" or physics.get("paper_equivalent") is False:
            remaining_gaps.append("physics_property_inference")
        collision = obj.get("collision_asset", {})
        if collision.get("reproduction_status") == "partial" or collision.get("paper_equivalent") is False:
            remaining_gaps.append("coacd_collision_decomposition")
    critical_sections = [
        "pose_alignment",
        "background_registration",
        "collision_assets",
        "table_collision_projection",
        "physics_settle",
        "usd_export",
        "genesis_export",
    ]
    engineering_blocked = any(sections.get(name, {}).get("status") != "passed" for name in critical_sections)
    engineering_status = "blocked" if engineering_blocked else "passed"
    simfoundry_status = "blocked" if "background_unregistered" in milestone_blockers else "partial"
    return {
        "version": 1,
        "milestone": "SF-upper-MVP-1",
        "claim": "Interactive reconstructed scene with explicit partial SimFoundry upper-pipeline reproduction",
        "engineering_interactive_scene_status": engineering_status,
        "simfoundry_upper_reproduction_status": simfoundry_status,
        "simfoundry_full_pipeline_status": "out_of_scope",
        "blocking_reasons": _dedupe(milestone_blockers),
        "report_blocking_reasons": blocking_reasons,
        "remaining_gaps": _dedupe(remaining_gaps),
        "not_claimed": [
            "digital_cousins",
            "policy_training",
            "policy_evaluation",
            "paper_level_full_visual_reconstruction",
        ],
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


def _write_isaac_fresh_validation_report(run: Path, section: dict[str, Any]) -> dict[str, Any]:
    repeatability_status = str(section.get("repeatability_status") or "missing_report")
    if repeatability_status == "fresh_direct_report":
        blocking_reasons: list[str] = []
        status = "fresh_direct_report"
    elif repeatability_status == "partial_preserved_report":
        blocking_reasons = ["isaac_validation_reused"]
        status = "partial_preserved_report"
    elif repeatability_status == "runtime_unavailable":
        blocking_reasons = ["isaac_runtime_unavailable"]
        status = "runtime_unavailable"
    elif repeatability_status == "missing_report":
        blocking_reasons = ["isaac_load_report_missing"]
        status = "missing_report"
    else:
        blocking_reasons = ["isaac_unknown_report_provenance"]
        status = repeatability_status
    report = {
        "version": 1,
        "status": status,
        "repeatability_status": repeatability_status,
        "fresh_direct_validation": repeatability_status == "fresh_direct_report",
        "validation_reused": bool(section.get("validation_reused", False)),
        "direct_runtime_validation": bool(section.get("direct_runtime_validation", False)),
        "report_source": section.get("report_source"),
        "runtime_status": section.get("runtime_status"),
        "isaac_load_report": section.get("report_path"),
        "blocking_reasons": _dedupe(blocking_reasons),
    }
    path = run / "qa" / "isaac_fresh_validation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


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


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


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



def run_export_qa(run_dir: str | Path, *, strict_claims: bool = False) -> Path:
    return write_sim_export_report(run_dir, strict_claims=strict_claims).report_path
