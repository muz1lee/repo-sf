"""Create separated collision, debug, and physics assets for scene objects."""

from __future__ import annotations

import json
import importlib.util
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


PHYSICS_SOURCE_CATEGORIES = {
    "vlm_inference",
    "material_table",
    "user_provided",
    "heuristic",
    "scene_manifest_physics_fields",
}


def ensure_collision_assets(
    run_dir: str | Path,
    *,
    backend: str = "convex-hull",
    strict_provenance: bool = False,
) -> dict[str, Any]:
    """Ensure every manifest object has visual/collision/debug/physics roles.

    Existing legacy ``mesh_path`` values are preserved for backward compatibility,
    but Genesis-facing collision geometry is written to ``collision.glb`` and
    provenance is recorded in both the scene manifest and ``physics.json``.
    """
    run = Path(run_dir)
    backend_requested = _normalise_collision_backend(backend)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    object_reports = []
    for item in manifest.get("objects", []):
        object_reports.append(
            _ensure_object_assets(
                run,
                item,
                backend_requested=backend_requested,
                strict_provenance=strict_provenance,
            )
        )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    collision_report = _collision_rebuild_report(
        object_reports,
        backend_requested=backend_requested,
        strict_provenance=strict_provenance,
    )
    collision_report["report_path"] = "qa/collision_rebuild_report.json"
    (qa_dir / "collision_rebuild_report.json").write_text(json.dumps(collision_report, indent=2), encoding="utf-8")
    physics_report = _physics_property_report(object_reports)
    (qa_dir / "physics_property_report.json").write_text(json.dumps(physics_report, indent=2), encoding="utf-8")
    return collision_report


def qa_physics(run_dir: str | Path) -> dict[str, Any]:
    """Write a standalone physics provenance report from object ``physics.json`` files."""
    run = Path(run_dir)
    manifest = json.loads((run / "scene_manifest.json").read_text(encoding="utf-8"))
    object_reports = []
    for item in manifest.get("objects", []):
        object_id = str(item["object_id"])
        object_dir = run / "objects" / object_id
        physics_path = _physics_path_from_manifest(run, object_dir, item)
        physics = _read_physics_record(physics_path, object_id, item)
        collision_report_path = object_dir / "collision_report.json"
        if collision_report_path.is_file():
            collision_report = json.loads(collision_report_path.read_text(encoding="utf-8"))
        else:
            collision_report = {
                "paper_equivalent": False,
                "reproduction_status": "blocked",
                "decomposition_backend": "missing_collision_report",
            }
        object_reports.append(
            {
                "object_id": object_id,
                "physics": physics,
                "paper_equivalent": bool(collision_report.get("paper_equivalent")),
                "collision_report": collision_report,
            }
        )
    report = _physics_property_report(object_reports)
    report["report_path"] = "qa/physics_property_report.json"
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "physics_property_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _ensure_object_assets(
    run: Path,
    item: dict[str, Any],
    *,
    backend_requested: str,
    strict_provenance: bool,
) -> dict[str, Any]:
    object_id = str(item["object_id"])
    object_dir = run / "objects" / object_id
    object_dir.mkdir(parents=True, exist_ok=True)
    visual = _resolve_visual_asset(run, object_dir, item)
    item["visual_asset"] = visual
    debug_proxy = _write_debug_proxy(run, object_dir, item, visual)
    item["debug_proxy"] = debug_proxy
    collision = _write_collision_asset(
        run,
        object_dir,
        item,
        visual,
        debug_proxy,
        backend_requested=backend_requested,
        strict_provenance=strict_provenance,
    )
    item["collision_asset"] = collision
    collision_report = _write_collision_report(
        run,
        object_dir,
        object_id,
        visual,
        collision,
        debug_proxy,
        backend_requested=backend_requested,
        strict_provenance=strict_provenance,
    )
    physics = _write_physics_json(run, object_dir, item, visual, collision, debug_proxy)
    item["physics"] = physics
    return {
        "object_id": object_id,
        "visual_asset": visual,
        "collision_asset": collision,
        "collision_report": collision_report,
        "debug_proxy": debug_proxy,
        "physics": physics,
        "collision_source": collision["source"],
        "decomposition_backend": collision_report["decomposition_backend"],
        "paper_equivalent": collision_report["paper_equivalent"],
        "reproduction_status": collision_report["reproduction_status"],
    }


def _resolve_visual_asset(run: Path, object_dir: Path, item: dict[str, Any]) -> dict[str, Any]:
    existing = item.get("visual_asset")
    if isinstance(existing, dict) and existing.get("path"):
        visual = dict(existing)
        visual.setdefault("role", "visual_asset")
        visual.setdefault("source", "manifest_visual_asset")
        visual.setdefault("status", "ready" if (run / str(visual["path"])).is_file() else "missing")
        visual.setdefault("final_visual", visual.get("status") == "ready")
        visual.setdefault("provenance", {})
        return visual

    visual_glb = object_dir / "visual.glb"
    if visual_glb.is_file():
        return {
            "role": "visual_asset",
            "path": str(visual_glb.relative_to(run)),
            "source": "visual_glb_file",
            "status": "ready",
            "final_visual": True,
            "provenance": {"selection": "objects/<id>/visual.glb"},
        }

    visual_cloud = object_dir / "visual_point_cloud.ply"
    if visual_cloud.is_file():
        return {
            "role": "visual_asset",
            "path": str(visual_cloud.relative_to(run)),
            "source": "visual_point_cloud",
            "status": "visual_point_cloud_only",
            "final_visual": False,
            "provenance": {"selection": "objects/<id>/visual_point_cloud.ply"},
        }

    legacy_path = str(item.get("mesh_path", ""))
    if not legacy_path:
        return {
            "role": "visual_asset",
            "path": None,
            "source": "missing",
            "status": "missing_visual_asset",
            "final_visual": False,
            "provenance": {},
        }
    mesh_path = run / legacy_path
    stats = _mesh_stats(mesh_path)
    is_bbox = _is_bbox_proxy(stats)
    return {
        "role": "visual_asset",
        "path": legacy_path,
        "source": "legacy_mesh_path_bbox_proxy" if is_bbox else "legacy_mesh_path",
        "status": "blocked_proxy_visual_asset" if is_bbox else "ready_legacy_visual_mesh",
        "final_visual": not is_bbox,
        "vertex_count": stats["vertex_count"],
        "face_count": stats["face_count"],
        "provenance": {"legacy_mesh_path": legacy_path},
    }


def _write_debug_proxy(run: Path, object_dir: Path, item: dict[str, Any], visual: dict[str, Any]) -> dict[str, Any]:
    debug_path = object_dir / "debug_bbox.glb"
    source_path = _asset_path(run, visual.get("path")) or _asset_path(run, item.get("mesh_path"))
    if source_path is None:
        debug_path.write_bytes(b"")
        status = "missing_source"
        source = "missing_visual_bounds"
        vertex_count = 0
        face_count = 0
    else:
        vertices = _load_vertices(source_path)
        _export_bbox(vertices, debug_path)
        status = "ready"
        source = "bbox_from_visual_bounds"
        stats = _mesh_stats(debug_path)
        vertex_count = stats["vertex_count"]
        face_count = stats["face_count"]
    return {
        "role": "debug_proxy",
        "path": str(debug_path.relative_to(run)),
        "source": source,
        "status": status,
        "vertex_count": vertex_count,
        "face_count": face_count,
    }


def _write_collision_asset(
    run: Path,
    object_dir: Path,
    item: dict[str, Any],
    visual: dict[str, Any],
    debug_proxy: dict[str, Any],
    *,
    backend_requested: str,
    strict_provenance: bool,
) -> dict[str, Any]:
    existing = item.get("collision_asset")
    if isinstance(existing, dict) and existing.get("path") and existing.get("path") != visual.get("path"):
        existing_path = run / str(existing["path"])
        existing_provenance = existing.get("provenance") if isinstance(existing.get("provenance"), dict) else {}
        existing_visual_path = existing_provenance.get("visual_asset_path")
        current_visual_path = visual.get("path")
        stale_visual_source = bool(existing_visual_path and current_visual_path and existing_visual_path != current_visual_path)
        stale_legacy_bbox = existing.get("source") == "bbox_from_legacy_mesh_proxy" and visual.get("final_visual") is True
        if existing_path.is_file() and not stale_visual_source and not stale_legacy_bbox:
            collision = dict(existing)
            collision.setdefault("role", "collision_asset")
            collision.setdefault("source", "manifest_collision_asset")
            collision.setdefault("status", "ready")
            collision.setdefault("provenance", {})
            collision.setdefault("backend_requested", backend_requested)
            collision.setdefault("strict_provenance", strict_provenance)
            return collision

    collision_path = object_dir / "collision.glb"
    visual_path = _asset_path(run, visual.get("path"))
    visual_is_bbox = visual.get("status") == "blocked_proxy_visual_asset" or visual.get("source") == "legacy_mesh_path_bbox_proxy"
    if visual_path is not None and not visual_is_bbox and visual_path.suffix.lower() == ".glb":
        vertices, faces = _load_mesh_vertices_faces(visual_path)
        if len(faces) > 0:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).convex_hull
            mesh.export(collision_path)
            source = "convex_hull_from_visual_mesh"
            method = "trimesh_convex_hull"
        else:
            _copy_or_bbox(run, debug_proxy, collision_path)
            source = "bbox_from_visual_point_cloud"
            method = "bbox_proxy"
    else:
        _copy_or_bbox(run, debug_proxy, collision_path)
        source = "bbox_from_legacy_mesh_proxy" if visual_is_bbox else "bbox_from_visual_bounds"
        method = "bbox_proxy"
    stats = _mesh_stats(collision_path)
    decomposition_backend = _collision_decomposition_backend(source, method)
    paper_equivalent = _collision_paper_equivalent(source, decomposition_backend)
    return {
        "role": "collision_asset",
        "path": str(collision_path.relative_to(run)),
        "source": source,
        "status": "ready" if collision_path.stat().st_size > 0 else "empty_collision_asset",
        "vertex_count": stats["vertex_count"],
        "face_count": stats["face_count"],
        "decomposition_backend": decomposition_backend,
        "paper_equivalent": paper_equivalent,
        "reproduction_status": "reproduced" if paper_equivalent else "partial",
        "backend_requested": backend_requested,
        "strict_provenance": strict_provenance,
        "provenance": {
            "method": method,
            "visual_asset_path": visual.get("path"),
            "debug_proxy_path": debug_proxy.get("path"),
            "backend_requested": backend_requested,
            "strict_provenance": strict_provenance,
        },
    }


def _write_physics_json(
    run: Path,
    object_dir: Path,
    item: dict[str, Any],
    visual: dict[str, Any],
    collision: dict[str, Any],
    debug_proxy: dict[str, Any],
) -> dict[str, Any]:
    physics_path = object_dir / "physics.json"
    mass = float(item.get("mass_kg", 0.2))
    friction = float(item.get("friction", 0.8))
    restitution = float(item.get("restitution", 0.0))
    field_sources = {
        "mass_kg": _physics_property_source(item, "mass_kg"),
        "friction": _physics_property_source(item, "friction"),
        "restitution": _physics_property_source(item, "restitution"),
    }
    source_category = _physics_record_source_category(item, field_sources)
    paper_equivalent = all(source["category"] == "vlm_inference" for source in field_sources.values())
    physics = {
        "version": 1,
        "object_id": str(item["object_id"]),
        "label": str(item.get("label", item["object_id"])),
        "mass_kg": mass,
        "friction": friction,
        "restitution": restitution,
        "density_basis": "manifest_mass_kg",
        "source": source_category,
        "source_category": source_category,
        "property_sources": field_sources,
        "status": "ready",
        "paper_equivalent": paper_equivalent,
        "reproduction_status": "reproduced" if paper_equivalent else "partial",
        "interactive_status": "usable",
        "paper_equivalence_status": "reproduced" if paper_equivalent else "partial",
        "blocking_for_paper": [] if paper_equivalent else _physics_not_paper_equivalent_reasons(field_sources),
        "not_paper_equivalent_reasons": [] if paper_equivalent else _physics_not_paper_equivalent_reasons(field_sources),
        "visual_asset": visual,
        "collision_asset": collision,
        "debug_proxy": debug_proxy,
        "collision_source": collision["source"],
        "collision_decomposition_backend": collision.get("decomposition_backend"),
    }
    physics_path.write_text(json.dumps(physics, indent=2), encoding="utf-8")
    record = dict(physics)
    record["path"] = str(physics_path.relative_to(run))
    return record


def _write_collision_report(
    run: Path,
    object_dir: Path,
    object_id: str,
    visual: dict[str, Any],
    collision: dict[str, Any],
    debug_proxy: dict[str, Any],
    *,
    backend_requested: str,
    strict_provenance: bool,
) -> dict[str, Any]:
    availability = _collision_backend_availability()
    source = str(collision.get("source", "unknown"))
    method = str(collision.get("provenance", {}).get("method") or collision.get("decomposition_backend") or "unknown")
    decomposition_backend = _collision_decomposition_backend(source, method)
    paper_equivalent = _collision_paper_equivalent(source, decomposition_backend)
    asset_ready = collision.get("status") == "ready"
    blocking_for_paper = _collision_not_paper_equivalent_reasons(
        source,
        decomposition_backend,
        paper_equivalent=paper_equivalent,
        availability=availability,
    )
    if strict_provenance and backend_requested == "coacd" and decomposition_backend != "coacd":
        blocking_for_paper = _dedupe(blocking_for_paper + _coacd_strict_blocking_reasons(availability))
    interactive_status = "usable" if asset_ready else "blocked"
    if not asset_ready:
        status = "blocked"
        reproduction_status = "blocked"
        paper_equivalence_status = "blocked"
    elif strict_provenance and backend_requested == "coacd" and decomposition_backend != "coacd":
        status = "blocked"
        reproduction_status = "blocked"
        paper_equivalence_status = "blocked"
    elif paper_equivalent:
        status = "reproduced"
        reproduction_status = "reproduced"
        paper_equivalence_status = "reproduced"
    else:
        status = "partial"
        reproduction_status = "partial"
        paper_equivalence_status = "partial"
    report = {
        "version": 1,
        "object_id": object_id,
        "status": status,
        "asset_status": collision.get("status"),
        "interactive_status": interactive_status,
        "paper_equivalence_status": paper_equivalence_status,
        "reproduction_status": reproduction_status,
        "paper_equivalent": paper_equivalent,
        "backend_requested": backend_requested,
        "strict_provenance": strict_provenance,
        "simfoundry_target_backend": "coacd",
        "collision_source": source,
        "decomposition_backend": decomposition_backend,
        "method": method,
        "collision_asset_path": collision.get("path"),
        "visual_asset_path": visual.get("path"),
        "debug_proxy_path": debug_proxy.get("path"),
        "availability": availability,
        "blocking_for_paper": blocking_for_paper,
        "not_paper_equivalent_reasons": blocking_for_paper,
    }
    path = object_dir / "collision_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    record = dict(report)
    record["path"] = str(path.relative_to(run))
    return record


def _collision_rebuild_report(
    object_reports: list[dict[str, Any]],
    *,
    backend_requested: str,
    strict_provenance: bool,
) -> dict[str, Any]:
    reports = [dict(item["collision_report"]) for item in object_reports]
    if any(item["reproduction_status"] == "blocked" for item in reports):
        status = "blocked"
    elif reports and all(item["paper_equivalent"] for item in reports):
        status = "reproduced"
    else:
        status = "partial"
    if any(item.get("interactive_status") != "usable" for item in reports):
        interactive_status = "blocked"
    else:
        interactive_status = "usable" if reports else "blocked"
    if any(item.get("paper_equivalence_status") == "blocked" for item in reports):
        paper_equivalence_status = "blocked"
    elif reports and all(item.get("paper_equivalence_status") == "reproduced" for item in reports):
        paper_equivalence_status = "reproduced"
    else:
        paper_equivalence_status = "partial"
    blocking_for_paper = _dedupe(
        [
            reason
            for item in reports
            for reason in item.get("blocking_for_paper", item.get("not_paper_equivalent_reasons", []))
        ]
    )
    return {
        "version": 1,
        "status": status,
        "interactive_status": interactive_status,
        "paper_equivalence_status": paper_equivalence_status,
        "reproduction_status": status,
        "paper_equivalent": bool(reports) and all(item["paper_equivalent"] for item in reports),
        "backend_requested": backend_requested,
        "strict_provenance": strict_provenance,
        "simfoundry_target_backend": "coacd",
        "availability": _collision_backend_availability(),
        "objects": object_reports,
        "collision_reports": reports,
        "collision_decomposition_backends": sorted({str(item["decomposition_backend"]) for item in reports}),
        "report_path": "qa/collision_rebuild_report.json",
        "blocking_for_paper": blocking_for_paper,
        "not_paper_equivalent_reasons": blocking_for_paper,
    }


def _physics_property_report(object_reports: list[dict[str, Any]]) -> dict[str, Any]:
    objects = []
    source_summary: dict[str, dict[str, int]] = {"mass_kg": {}, "friction": {}, "restitution": {}}
    for item in object_reports:
        physics = item["physics"]
        property_sources = physics.get("property_sources", {})
        object_record = {
            "object_id": item["object_id"],
            "label": physics.get("label"),
            "status": physics.get("status"),
            "interactive_status": "usable" if physics.get("status") == "ready" else "blocked",
            "mass_kg": physics.get("mass_kg"),
            "friction": physics.get("friction"),
            "restitution": physics.get("restitution"),
            "mass_source_category": _property_source_category(property_sources, "mass_kg"),
            "friction_source_category": _property_source_category(property_sources, "friction"),
            "restitution_source_category": _property_source_category(property_sources, "restitution"),
            "source_category": physics.get("source_category"),
            "paper_equivalent": physics.get("paper_equivalent"),
            "paper_equivalence_status": "reproduced" if physics.get("paper_equivalent") is True else "partial",
            "reproduction_status": physics.get("reproduction_status"),
            "blocking_for_paper": physics.get("blocking_for_paper", physics.get("not_paper_equivalent_reasons", [])),
            "collision_source": physics.get("collision_source"),
            "collision_decomposition_backend": physics.get("collision_decomposition_backend"),
            "collision_paper_equivalent": item.get("paper_equivalent"),
        }
        objects.append(object_record)
        for field in source_summary:
            category = str(object_record[f"{field.removesuffix('_kg')}_source_category"] if field == "mass_kg" else object_record[f"{field}_source_category"])
            source_summary[field][category] = source_summary[field].get(category, 0) + 1
    if any(item.get("status") != "ready" for item in objects):
        status = "blocked"
    elif objects and all(item.get("paper_equivalent") is True for item in objects):
        status = "reproduced"
    else:
        status = "partial"
    interactive_status = "usable" if objects and all(item.get("interactive_status") == "usable" for item in objects) else "blocked"
    if status == "blocked":
        paper_equivalence_status = "blocked"
    elif objects and all(item.get("paper_equivalent") is True for item in objects):
        paper_equivalence_status = "reproduced"
    else:
        paper_equivalence_status = "partial"
    blocking_for_paper = _dedupe(
        [reason for item in objects for reason in item.get("blocking_for_paper", [])]
    )
    return {
        "version": 1,
        "status": status,
        "interactive_status": interactive_status,
        "paper_equivalence_status": paper_equivalence_status,
        "reproduction_status": status,
        "paper_equivalent": bool(objects) and all(item.get("paper_equivalent") is True for item in objects),
        "allowed_source_categories": sorted(PHYSICS_SOURCE_CATEGORIES),
        "source_summary": source_summary,
        "objects": objects,
        "blocking_for_paper": blocking_for_paper,
        "not_paper_equivalent_reasons": blocking_for_paper,
    }


def _property_source_category(property_sources: dict[str, Any], field: str) -> str:
    source = property_sources.get(field)
    if isinstance(source, dict):
        return str(source.get("category", "heuristic"))
    return "heuristic"


def _physics_path_from_manifest(run: Path, object_dir: Path, item: dict[str, Any]) -> Path:
    physics = item.get("physics") if isinstance(item.get("physics"), dict) else {}
    if physics.get("path"):
        path = run / str(physics["path"])
        if path.is_file():
            return path
    return object_dir / "physics.json"


def _read_physics_record(path: Path, object_id: str, item: dict[str, Any]) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "version": 1,
        "object_id": object_id,
        "label": str(item.get("label", object_id)),
        "status": "missing",
        "source_category": "heuristic",
        "property_sources": {},
        "paper_equivalent": False,
        "reproduction_status": "blocked",
        "interactive_status": "blocked",
        "paper_equivalence_status": "blocked",
        "blocking_for_paper": ["missing_physics_json"],
        "not_paper_equivalent_reasons": ["missing_physics_json"],
    }


def _physics_property_source(item: dict[str, Any], field: str) -> dict[str, str]:
    explicit = _explicit_physics_property_source(item, field)
    if explicit:
        category = explicit
        source = explicit
    elif item.get("physics_source_category") or item.get("physics_source"):
        category = _normalise_physics_source_category(item.get("physics_source_category") or item.get("physics_source"))
        source = category
    elif field in item and item.get(field) is not None:
        category = "scene_manifest_physics_fields"
        source = "scene_manifest_physics_fields"
    else:
        category = "heuristic"
        source = "code_default"
    return {"category": category, "source": source}


def _explicit_physics_property_source(item: dict[str, Any], field: str) -> str | None:
    existing_physics = item.get("physics") if isinstance(item.get("physics"), dict) else {}
    for container in [
        item.get("physics_property_sources"),
        item.get("property_sources"),
        existing_physics.get("property_sources") if isinstance(existing_physics, dict) else None,
    ]:
        if not isinstance(container, dict) or field not in container:
            continue
        value = container[field]
        if isinstance(value, dict):
            value = value.get("category") or value.get("source")
        return _normalise_physics_source_category(value)
    return None


def _physics_record_source_category(item: dict[str, Any], field_sources: dict[str, dict[str, str]]) -> str:
    explicit = item.get("physics_source_category") or item.get("physics_source")
    if explicit:
        return _normalise_physics_source_category(explicit)
    categories = [source["category"] for source in field_sources.values()]
    if "vlm_inference" in categories and all(category == "vlm_inference" for category in categories):
        return "vlm_inference"
    if "user_provided" in categories and all(category == "user_provided" for category in categories):
        return "user_provided"
    if "material_table" in categories and all(category == "material_table" for category in categories):
        return "material_table"
    if "scene_manifest_physics_fields" in categories:
        return "scene_manifest_physics_fields"
    return "heuristic"


def _normalise_physics_source_category(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in PHYSICS_SOURCE_CATEGORIES:
        return raw
    if raw in {"class_default", "default", "code_default", "conservative_default"}:
        return "heuristic"
    if "scene_manifest" in raw or raw == "manifest":
        return "scene_manifest_physics_fields"
    if "material" in raw:
        return "material_table"
    if "user" in raw or "manual" in raw:
        return "user_provided"
    if "vlm" in raw or "qwen" in raw or "gemini" in raw:
        return "vlm_inference"
    return "heuristic"


def _physics_not_paper_equivalent_reasons(field_sources: dict[str, dict[str, str]]) -> list[str]:
    reasons = []
    for field, source in field_sources.items():
        category = source["category"]
        if category != "vlm_inference":
            reasons.append(f"{field}_source_not_vlm_inference:{category}")
    return reasons


def _normalise_collision_backend(value: object) -> str:
    raw = str(value or "convex-hull").strip().lower().replace("_", "-")
    if raw in {"convex", "convex-hull", "trimesh-convex-hull"}:
        return "convex-hull"
    if raw == "coacd":
        return "coacd"
    return raw


def _collision_backend_availability() -> dict[str, dict[str, Any]]:
    coacd_executable = shutil.which("coacd")
    vhacd_executable = shutil.which("vhacd")
    test_vhacd_executable = shutil.which("testVHACD")
    coacd_module = importlib.util.find_spec("coacd") is not None
    vhacd_module = importlib.util.find_spec("vhacd") is not None
    return {
        "coacd": {
            "available": bool(coacd_module or coacd_executable),
            "python_module": coacd_module,
            "executable": coacd_executable,
        },
        "vhacd": {
            "available": bool(vhacd_module or vhacd_executable or test_vhacd_executable),
            "python_module": vhacd_module,
            "executable": vhacd_executable,
            "testVHACD_executable": test_vhacd_executable,
        },
    }


def _coacd_strict_blocking_reasons(availability: dict[str, dict[str, Any]]) -> list[str]:
    if availability.get("coacd", {}).get("available"):
        return ["coacd_backend_not_integrated"]
    return ["coacd_runtime_unavailable"]


def _collision_decomposition_backend(source: str, method: str) -> str:
    source_lower = source.lower()
    method_lower = method.lower()
    if "coacd" in source_lower or "coacd" in method_lower:
        return "coacd"
    if "vhacd" in source_lower or "vhacd" in method_lower:
        return "vhacd"
    if "runtime_convex" in source_lower or "genesis_runtime_convex" in source_lower:
        return "runtime_convexification"
    if "convex_hull" in source_lower or "convex_hull" in method_lower:
        return "trimesh_convex_hull"
    if "bbox" in source_lower or "bbox" in method_lower:
        return "bbox_proxy"
    return method or "unknown"


def _collision_paper_equivalent(source: str, decomposition_backend: str) -> bool:
    return "coacd" in source.lower() or decomposition_backend == "coacd"


def _collision_not_paper_equivalent_reasons(
    source: str,
    decomposition_backend: str,
    *,
    paper_equivalent: bool,
    availability: dict[str, dict[str, Any]],
) -> list[str]:
    if paper_equivalent:
        return []
    reasons = []
    if not availability["coacd"]["available"]:
        reasons.append("coacd_unavailable")
    if not availability["vhacd"]["available"]:
        reasons.append("vhacd_unavailable")
    if decomposition_backend == "trimesh_convex_hull":
        reasons.append("convex_hull_substitute_not_coacd")
    elif decomposition_backend == "runtime_convexification":
        reasons.append("runtime_convexification_not_coacd")
    elif decomposition_backend == "bbox_proxy":
        reasons.append("bbox_proxy_not_coacd")
    else:
        reasons.append(f"{decomposition_backend}_not_coacd")
    if source not in {"convex_hull_from_visual_mesh", "runtime_convexification"} and "coacd" not in source.lower():
        reasons.append(f"collision_source_not_coacd:{source}")
    return _dedupe(reasons)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _asset_path(run: Path, value: object) -> Path | None:
    if not value:
        return None
    path = run / str(value)
    return path if path.is_file() else None


def _copy_or_bbox(run: Path, debug_proxy: dict[str, Any], destination: Path) -> None:
    debug_path = _asset_path(run, debug_proxy.get("path"))
    if debug_path is None:
        destination.write_bytes(b"")
        return
    if debug_path != destination:
        shutil.copyfile(debug_path, destination)


def _export_bbox(vertices: np.ndarray, path: Path) -> None:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.size == 0:
        path.write_bytes(b"")
        return
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    extents = np.maximum(hi - lo, 1e-4)
    center = (lo + hi) * 0.5
    mesh = trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(center))
    mesh.export(path)


def _mesh_stats(path: Path) -> dict[str, int]:
    try:
        vertices, faces = _load_mesh_vertices_faces(path)
    except Exception:  # noqa: BLE001
        return {"vertex_count": 0, "face_count": 0}
    return {"vertex_count": int(len(vertices)), "face_count": int(len(faces))}


def _is_bbox_proxy(stats: dict[str, int]) -> bool:
    return bool(stats["vertex_count"] <= 8 and 0 < stats["face_count"] <= 12)


def _load_vertices(path: Path) -> np.ndarray:
    vertices, _faces = _load_mesh_vertices_faces(path)
    return vertices


def _load_mesh_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="scene")
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    if isinstance(loaded, trimesh.Trimesh):
        verts = np.asarray(loaded.vertices, dtype=np.float64)
        vertices.append(verts)
        faces.append(np.asarray(loaded.faces, dtype=np.int64))
    elif isinstance(loaded, trimesh.Scene):
        for geom in loaded.geometry.values():
            if not hasattr(geom, "vertices"):
                continue
            verts = np.asarray(geom.vertices, dtype=np.float64)
            vertices.append(verts)
            if hasattr(geom, "faces"):
                faces.append(np.asarray(geom.faces, dtype=np.int64) + offset)
            offset += len(verts)
    elif hasattr(loaded, "vertices"):
        vertices.append(np.asarray(loaded.vertices, dtype=np.float64))
    if not vertices:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    all_vertices = np.concatenate(vertices, axis=0)
    all_faces = np.concatenate(faces, axis=0) if faces else np.zeros((0, 3), dtype=np.int64)
    return all_vertices, all_faces
