"""Isaac export helpers for simulator export bundles."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sim_export_manifest import build_sim_export_manifest, write_sim_export_manifest
from .usd_export import build_runtime_asset_report


@dataclass(frozen=True)
class IsaacExportResult:
    script_path: Path
    report_path: Path
    report: dict[str, Any]


def export_isaac_scene(
    run_dir: str | Path,
    *,
    python_executable: str | Path | None = None,
    timeout_s: int = 60,
) -> IsaacExportResult:
    run = Path(run_dir)
    export_dir = run / "exports"
    qa_dir = run / "qa"
    export_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    script_path = export_dir / "isaac_scene.py"
    script_path.write_text(_isaac_script(), encoding="utf-8")
    report_path = qa_dir / "isaac_load_report.json"
    manifest = build_sim_export_manifest(run).to_dict()

    if python_executable is None:
        report = _existing_loaded_report(report_path)
        if report is not None:
            _mark_preserved_existing_worker_report(report)
            report.setdefault("script_path", "exports/isaac_scene.py")
            report.setdefault("scene_path", "exports/scene.usda")
            _merge_isaac_asset_report(report, manifest)
        else:
            report = {
                "status": "runtime_unavailable",
                "reason": "no_isaac_python_executable_provided",
                "script_path": "exports/isaac_scene.py",
                "scene_path": "exports/scene.usda",
            }
            _mark_runtime_unavailable_report(report)
            _merge_isaac_asset_report(report, manifest)
    else:
        executable = Path(python_executable)
        if not executable.is_file():
            report = {
                "status": "runtime_unavailable",
                "reason": "isaac_python_executable_not_found",
                "python_executable": str(executable),
                "script_path": "exports/isaac_scene.py",
                "scene_path": "exports/scene.usda",
            }
            _mark_runtime_unavailable_report(report)
            _merge_isaac_asset_report(report, manifest)
        else:
            report = _run_isaac_loader(executable, script_path, run, manifest, timeout_s=timeout_s)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_sim_export_manifest(run)
    return IsaacExportResult(script_path=script_path, report_path=report_path, report=report)


def _existing_loaded_report(report_path: Path) -> dict[str, Any] | None:
    report = _read_json_report_file(report_path)
    if report is None or report.get("status") not in {"loaded", "passed"}:
        return None
    return report


def _mark_preserved_existing_worker_report(report: dict[str, Any]) -> None:
    report["report_source"] = "preserved_existing_worker_report"
    report["validation_reused"] = True
    report["direct_runtime_validation"] = False
    report["repeatability"] = {
        "status": "preserved_existing_worker_report",
        "fresh_direct_validation": False,
        "direct_worker_access_available": False,
        "caveat": (
            "Loaded status was preserved from an existing worker report; "
            "this is not a fresh direct Isaac validation from this export-sim invocation."
        ),
    }


def _mark_runtime_unavailable_report(report: dict[str, Any]) -> None:
    report["report_source"] = "direct_runtime_unavailable"
    report["validation_reused"] = False
    report["direct_runtime_validation"] = False
    report["repeatability"] = {
        "status": "direct_runtime_unavailable",
        "fresh_direct_validation": False,
        "direct_worker_access_available": False,
        "caveat": "Direct Isaac runtime validation was not run from this export-sim invocation.",
    }


def _mark_direct_runtime_report(report: dict[str, Any]) -> None:
    report.setdefault("report_source", "direct_isaac_runtime")
    report["validation_reused"] = False
    report["direct_runtime_validation"] = True
    report["repeatability"] = {
        "status": "fresh_direct_runtime_report",
        "fresh_direct_validation": True,
        "direct_worker_access_available": True,
        "caveat": None,
    }


def _run_isaac_loader(executable: Path, script_path: Path, run: Path, manifest: dict[str, Any], *, timeout_s: int) -> dict[str, Any]:
    command = [str(executable), str(script_path), "--run-dir", str(run)]
    report_path = run / "qa" / "isaac_load_report.json"
    if report_path.exists():
        report_path.unlink()
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001
        report = {
            "status": "runtime_unavailable",
            "reason": "isaac_loader_invocation_failed",
            "error": str(exc),
            "command": command,
        }
        _mark_runtime_unavailable_report(report)
        _merge_isaac_asset_report(report, manifest)
        return report
    parsed = _parse_json_report_from_stdout(completed.stdout)
    file_report = _read_json_report_file(report_path)
    report = parsed or file_report
    if report is not None:
        report.setdefault("command", command)
        report["returncode"] = completed.returncode
        report["stdout_tail"] = completed.stdout[-4000:]
        report["stderr_tail"] = completed.stderr[-4000:]
        _mark_direct_runtime_report(report)
        _merge_isaac_asset_report(report, manifest)
        return report
    if completed.returncode != 0:
        report = {
            "status": "failed",
            "reason": "isaac_loader_failed_without_report",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "command": command,
        }
        _mark_direct_runtime_report(report)
        _merge_isaac_asset_report(report, manifest)
        return report
    report = {
        "status": "failed",
        "reason": "isaac_loader_report_missing",
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "command": command,
    }
    _mark_direct_runtime_report(report)
    _merge_isaac_asset_report(report, manifest)
    return report


def _merge_isaac_asset_report(report: dict[str, Any], manifest: dict[str, Any]) -> None:
    _merge_native_3dgs_report(report, manifest)
    loaded = report.get("status") == "loaded"
    sections = build_runtime_asset_report(
        manifest,
        backend="isaac",
        visual_assets_referenced=loaded,
        collision_assets_referenced=loaded,
        background_asset_referenced=loaded,
        debug_proxy_visible=bool(report.get("debug_proxy_visible", False)),
        evidence=(
            "Isaac loader validates USD stage, prims, schemas, and asset metadata only; "
            "no native 3DGS/splat renderer is integrated or verified by this loader."
        ),
    )
    for key, value in sections.items():
        if key not in report:
            report[key] = value
        elif isinstance(value, dict) and isinstance(report[key], dict):
            merged = value.copy()
            merged.update(report[key])
            report[key] = merged
    _sync_background_native_3dgs_fields(report)


def _merge_native_3dgs_report(report: dict[str, Any], manifest: dict[str, Any]) -> None:
    required = _native_3dgs_required(manifest)
    report.setdefault("native_3dgs_required", required)
    report.setdefault("native_3dgs_supported", False)
    if "native_3dgs_status" not in report:
        if not required:
            report["native_3dgs_status"] = "not_requested"
        elif report.get("status") == "runtime_unavailable":
            report["native_3dgs_status"] = "runtime_unavailable"
        else:
            report["native_3dgs_status"] = "unsupported_unverified"
    evidence = report.get("native_3dgs_evidence")
    if not isinstance(evidence, dict):
        report["native_3dgs_evidence"] = {
            "probe_phase": "not_run" if report.get("status") == "runtime_unavailable" else "host_merge",
            "reason": report.get("reason", "isaac_runtime_probe_not_reported"),
        }
    report.setdefault(
        "background_runtime_verified",
        report.get("native_3dgs_status") in {"native_3dgs_loaded", "registered_3dgs_loaded", "verified_native_3dgs"},
    )


def _sync_background_native_3dgs_fields(report: dict[str, Any]) -> None:
    background_runtime = report.setdefault("background_runtime", {})
    for key in [
        "native_3dgs_required",
        "native_3dgs_supported",
        "native_3dgs_status",
        "native_3dgs_evidence",
    ]:
        background_runtime[key] = report.get(key)
    background_runtime["background_runtime_verified"] = bool(report.get("background_runtime_verified", False))


def _native_3dgs_required(manifest: dict[str, Any]) -> bool:
    background = manifest.get("background", {})
    visual = background.get("visual_asset", {}) or {}
    source_kind = str(visual.get("source_kind", ""))
    return bool(
        background.get("gaussian_splat_config_path")
        or background.get("gaussian_splat_checkpoint_path")
        or visual.get("has_3dgs_sidecar")
        or source_kind in {"registered_3dgs", "registered_3dgs_sidecar", "3dgs_native_asset_candidate"}
    )


def _read_json_report_file(report_path: Path) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(value, dict) and value.get("status"):
        return value
    return None



def _parse_json_report_from_stdout(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{") or not candidate.endswith("}"):
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("status"):
            return value
    return None

def _isaac_script() -> str:
    return '''#!/usr/bin/env python3
"""Verify that a real2sim USD export can be opened in Isaac/Omniverse."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def emit_report(run_dir: Path, report: dict) -> None:
    qa_dir = run_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "isaac_load_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


def _api_schema_names(prim) -> str:
    return str(prim.GetMetadata("apiSchemas") or "")


def _asset_ref(obj, key, scene_text):
    asset = obj.get(key, {}) or {}
    path = asset.get("path")
    return {
        "object_id": obj.get("object_id"),
        "path": path,
        "status": asset.get("status", "unknown"),
        "source": asset.get("source", "unknown"),
        "diagnostic_only": bool(asset.get("diagnostic_only", False)),
        "referenced_by_runtime": bool(path and path in scene_text),
    }


def _native_3dgs_status(background):
    if background.get("gaussian_splat_config_path") or background.get("gaussian_splat_checkpoint_path"):
        return "unsupported_unverified"
    return "not_requested"


def _native_3dgs_required(background):
    visual = background.get("visual_asset", {}) or {}
    source_kind = str(visual.get("source_kind", ""))
    return bool(
        background.get("gaussian_splat_config_path")
        or background.get("gaussian_splat_checkpoint_path")
        or visual.get("has_3dgs_sidecar")
        or source_kind in {"registered_3dgs", "registered_3dgs_sidecar", "3dgs_native_asset_candidate"}
    )


def _probe_native_3dgs_runtime(run_dir, manifest):
    background = manifest.get("background", {})
    required = _native_3dgs_required(background)
    native_asset = run_dir / "background" / "3dgs_native" / "splat_rgb.ply"
    evidence = {
        "probe_phase": "after_simulation_app",
        "required": required,
        "extension_manager_available": False,
        "extension_matches": [],
        "extension_errors": [],
        "module_imports": {},
        "usd_plugin_matches": [],
        "usd_plugin_errors": [],
        "sdf_file_formats": {},
        "native_asset_path": "background/3dgs_native/splat_rgb.ply",
        "native_asset_exists": native_asset.is_file(),
    }

    _probe_extension_manager(evidence)
    _probe_module_imports(evidence)
    _probe_usd_plugins(evidence)
    evidence["native_asset_validation"] = _validate_native_3dgs_asset(native_asset)

    module_supported = any(item.get("ok") for item in evidence["module_imports"].values())
    format_supported = any(ext in evidence["sdf_file_formats"] for ext in ["splat", "ksplat", "spz"])
    supported = bool(evidence["extension_matches"] or module_supported or evidence["usd_plugin_matches"] or format_supported)
    if not required:
        status = "not_requested"
    elif not supported:
        status = "unsupported_missing_native_3dgs_runtime"
    elif not native_asset.is_file():
        status = "native_3dgs_supported_missing_asset"
    elif not evidence["native_asset_validation"].get("valid_for_native_3dgs"):
        status = "native_3dgs_asset_schema_incomplete"
    else:
        status = "native_3dgs_loaded"
    return {
        "native_3dgs_required": required,
        "native_3dgs_supported": supported,
        "native_3dgs_status": status,
        "native_3dgs_evidence": evidence,
        "background_runtime_verified": status == "native_3dgs_loaded",
    }


def _probe_extension_manager(evidence):
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        evidence["extension_manager_available"] = True
        try:
            extensions = manager.get_extensions()
        except Exception as exc:  # noqa: BLE001
            evidence["extension_errors"].append("get_extensions_failed: " + str(exc))
            extensions = []
        for ext in extensions:
            text = _safe_json(ext)
            lower = text.lower()
            if "splat" in lower or "gaussian" in lower:
                evidence["extension_matches"].append(_summarize_extension(ext, text))
    except Exception as exc:  # noqa: BLE001
        evidence["extension_errors"].append(type(exc).__name__ + ": " + str(exc))


def _probe_module_imports(evidence):
    candidates = [
        "omni.splat",
        "omni.kit.splat",
        "omni.usd.splat",
        "omni.gaussian_splatting",
        "omni.kit.gaussian_splatting",
        "omni.kit.viewport.gaussian_splatting",
    ]
    for name in candidates:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            evidence["module_imports"][name] = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:300]}
        else:
            evidence["module_imports"][name] = {"ok": True}


def _probe_usd_plugins(evidence):
    try:
        from pxr import Plug, Sdf

        for plugin in Plug.Registry().GetAllPlugins():
            text = " ".join(
                [
                    str(getattr(plugin, "name", "")),
                    str(getattr(plugin, "path", "")),
                    _safe_json(getattr(plugin, "metadata", {})),
                ]
            )
            lower = text.lower()
            if "splat" in lower or "gaussian" in lower:
                evidence["usd_plugin_matches"].append(
                    {"name": str(getattr(plugin, "name", "")), "path": str(getattr(plugin, "path", ""))}
                )
        for ext in ["ply", "splat", "ksplat", "spz"]:
            file_format = Sdf.FileFormat.FindByExtension(ext)
            if file_format:
                evidence["sdf_file_formats"][ext] = {
                    "format_id": str(getattr(file_format, "formatId", "")),
                    "target": str(getattr(file_format, "target", "")),
                }
    except Exception as exc:  # noqa: BLE001
        evidence["usd_plugin_errors"].append(type(exc).__name__ + ": " + str(exc))


def _validate_native_3dgs_asset(native_asset):
    if not native_asset.is_file():
        return {"valid_for_native_3dgs": False, "reason": "splat_rgb_ply_missing"}
    try:
        header = []
        with native_asset.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(256):
                line = handle.readline()
                if not line:
                    break
                header.append(line.strip())
                if line.strip() == "end_header":
                    break
    except OSError as exc:
        return {"valid_for_native_3dgs": False, "reason": "splat_rgb_ply_read_failed", "error": str(exc)}
    properties = []
    vertex_count = None
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
            try:
                vertex_count = int(parts[2])
            except ValueError:
                vertex_count = None
        if len(parts) >= 3 and parts[0] == "property":
            properties.append(parts[-1])
    prop_set = set(properties)
    has_xyz = {"x", "y", "z"}.issubset(prop_set)
    has_dc = {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(prop_set)
    has_shape = {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}.issubset(prop_set)
    return {
        "valid_for_native_3dgs": bool(vertex_count and vertex_count > 0 and has_xyz and has_dc and has_shape),
        "vertex_count": vertex_count,
        "properties": properties,
        "has_xyz": has_xyz,
        "has_gaussian_dc": has_dc,
        "has_gaussian_shape": has_shape,
    }


def _safe_json(value):
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except Exception:
        return str(value)


def _summarize_extension(ext, text):
    if isinstance(ext, dict):
        return {
            "id": str(ext.get("id") or ext.get("name") or ext.get("package_id") or "")[:200],
            "name": str(ext.get("name") or ext.get("title") or "")[:200],
            "enabled": bool(ext.get("enabled", False)),
        }
    return {"id": text[:200], "name": text[:200], "enabled": None}


def _asset_report(manifest, scene_path, native_3dgs_report):
    try:
        scene_text = scene_path.read_text(encoding="utf-8")
    except OSError:
        scene_text = ""
    background = manifest.get("background", {})
    visual_background = background.get("visual_asset", {})
    registration = background.get("registration", {})
    bg_path = visual_background.get("path")
    debug_proxy_refs = [_asset_ref(obj, "debug_proxy", scene_text) for obj in manifest.get("objects", [])]
    debug_proxy_visible = any(item.get("referenced_by_runtime") for item in debug_proxy_refs)
    background_referenced = bool(bg_path and bg_path in scene_text)
    return {
        "object_visual_refs": [_asset_ref(obj, "visual_asset", scene_text) for obj in manifest.get("objects", [])],
        "object_collision_refs": [_asset_ref(obj, "collision_asset", scene_text) for obj in manifest.get("objects", [])],
        "debug_proxy_refs": debug_proxy_refs,
        "debug_proxy_visible": bool(debug_proxy_visible),
        "background_asset": {
            "role": visual_background.get("role", "background_visual"),
            "path": bg_path,
            "status": visual_background.get("status", "unknown"),
            "source_kind": visual_background.get("source_kind", "unknown"),
            "source_backend": background.get("source_backend", "unknown"),
            "diagnostic_only": bool(visual_background.get("diagnostic_only", False)),
            "registration_path": registration.get("path"),
            "registration_status": registration.get("status", "unknown"),
            "gaussian_splat_config_path": background.get("gaussian_splat_config_path"),
            "gaussian_splat_checkpoint_path": background.get("gaussian_splat_checkpoint_path"),
            "referenced_by_runtime": background_referenced,
        },
        "background_runtime": {
            "backend": "isaac",
            "native_3dgs_required": native_3dgs_report["native_3dgs_required"],
            "native_3dgs_supported": native_3dgs_report["native_3dgs_supported"],
            "native_3dgs_status": native_3dgs_report["native_3dgs_status"],
            "native_3dgs_evidence": native_3dgs_report["native_3dgs_evidence"],
            "background_asset_referenced": background_referenced,
            "background_runtime_verified": native_3dgs_report["background_runtime_verified"],
            "evidence": "Isaac validation opened USD after SimulationApp and probed native 3DGS/splat runtime capability.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    scene_path = run_dir / "exports" / "scene.usda"
    try:
        manifest = json.loads((run_dir / "sim_export_manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        emit_report(run_dir, {"status": "failed", "reason": "sim_export_manifest_read_failed", "error": str(exc)})
        return 2

    try:
        from isaacsim import SimulationApp
    except Exception:
        try:
            from omni.isaac.kit import SimulationApp
        except Exception as exc:
            emit_report(run_dir, {"status": "runtime_unavailable", "reason": "isaac_simulation_app_import_failed", "error": str(exc)})
            return 3

    try:
        simulation_app = SimulationApp({"headless": True})
    except Exception as exc:  # noqa: BLE001
        emit_report(run_dir, {"status": "runtime_unavailable", "reason": "isaac_simulation_app_start_failed", "error": str(exc)})
        return 3

    native_3dgs_report = None
    try:
        native_3dgs_report = _probe_native_3dgs_runtime(run_dir, manifest)
        from pxr import Usd

        stage = Usd.Stage.Open(str(scene_path))
        if stage is None:
            emit_report(run_dir, {"status": "failed", "reason": "usd_stage_open_returned_none", "scene_path": str(scene_path)})
            return 2
        missing_prims = []
        missing_api_schemas = []
        support = stage.GetPrimAtPath("/World/SupportSurface")
        if not support:
            missing_prims.append("/World/SupportSurface")
        elif "PhysicsCollisionAPI" not in _api_schema_names(support):
            missing_api_schemas.append({"prim": "/World/SupportSurface", "schema": "PhysicsCollisionAPI"})
        camera = stage.GetPrimAtPath("/World/ReferenceCamera")
        if not camera:
            missing_prims.append("/World/ReferenceCamera")
        for obj in manifest.get("objects", []):
            prim_path = "/World/Objects/" + str(obj["object_id"]).replace("-", "_").replace(" ", "_")
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                missing_prims.append(prim_path)
                continue
            schemas = _api_schema_names(prim)
            for schema in ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]:
                if schema not in schemas:
                    missing_api_schemas.append({"prim": prim_path, "schema": schema})
            collision_path = prim_path + "/Collision"
            collision = stage.GetPrimAtPath(collision_path)
            if not collision:
                missing_prims.append(collision_path)
            elif "PhysicsCollisionAPI" not in _api_schema_names(collision):
                missing_api_schemas.append({"prim": collision_path, "schema": "PhysicsCollisionAPI"})
        status = "loaded" if not missing_prims and not missing_api_schemas else "failed"
        report = {
            "status": status,
            "scene_path": "exports/scene.usda",
            "object_count": len(manifest.get("objects", [])),
            "missing_prims": missing_prims,
            "missing_api_schemas": missing_api_schemas,
            "checked_rigid_collision_metadata": True,
            "checked_support_surface": True,
            "checked_reference_camera": True,
        }
        report.update(native_3dgs_report)
        report.update(_asset_report(manifest, scene_path, native_3dgs_report))
        emit_report(run_dir, report)
        return 0 if status == "loaded" else 2
    except Exception as exc:  # noqa: BLE001
        report = {"status": "failed", "reason": "isaac_stage_validation_failed", "error": str(exc), "scene_path": "exports/scene.usda"}
        if native_3dgs_report:
            report.update(native_3dgs_report)
        emit_report(run_dir, report)
        return 2
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
'''
