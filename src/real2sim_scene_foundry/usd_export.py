"""USD scene export helpers for simulator export bundles."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .sim_export_manifest import build_sim_export_manifest, write_sim_export_manifest


@dataclass(frozen=True)
class USDExportResult:
    scene_path: Path
    report_path: Path
    report: dict[str, Any]


def build_runtime_asset_report(
    manifest: dict[str, Any],
    *,
    backend: str,
    visual_assets_referenced: bool,
    collision_assets_referenced: bool,
    background_asset_referenced: bool,
    debug_proxy_visible: bool,
    evidence: str,
) -> dict[str, Any]:
    background = manifest.get("background", {})
    visual_background = background.get("visual_asset", {})
    registration = background.get("registration", {})
    native_3dgs_status = _native_3dgs_status(background)
    background_referenced = bool(background_asset_referenced and visual_background.get("path"))
    return {
        "object_visual_refs": [
            _object_asset_ref(obj, "visual_asset", referenced_by_runtime=visual_assets_referenced)
            for obj in manifest.get("objects", [])
        ],
        "object_collision_refs": [
            _object_asset_ref(obj, "collision_asset", referenced_by_runtime=collision_assets_referenced)
            for obj in manifest.get("objects", [])
        ],
        "debug_proxy_refs": [
            _object_asset_ref(obj, "debug_proxy", referenced_by_runtime=debug_proxy_visible)
            for obj in manifest.get("objects", [])
        ],
        "debug_proxy_visible": bool(debug_proxy_visible),
        "background_asset": {
            "role": visual_background.get("role", "background_visual"),
            "path": visual_background.get("path"),
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
            "backend": backend,
            "native_3dgs_supported": False,
            "native_3dgs_status": native_3dgs_status,
            "background_asset_referenced": background_referenced,
            "evidence": evidence,
        },
        "background_completion": _background_completion_report(
            background,
            native_3dgs_status=native_3dgs_status,
            background_asset_referenced=background_referenced,
        ),
    }


def _object_asset_ref(obj: dict[str, Any], key: str, *, referenced_by_runtime: bool) -> dict[str, Any]:
    asset = obj.get(key, {}) or {}
    return {
        "object_id": obj.get("object_id"),
        "path": asset.get("path"),
        "status": asset.get("status", "unknown"),
        "source": asset.get("source", "unknown"),
        "diagnostic_only": bool(asset.get("diagnostic_only", False)),
        "referenced_by_runtime": bool(referenced_by_runtime and asset.get("path")),
    }


def _native_3dgs_status(background: dict[str, Any]) -> str:
    if background.get("gaussian_splat_config_path") or background.get("gaussian_splat_checkpoint_path"):
        return "unsupported_unverified"
    return "not_requested"


def _background_completion_report(
    background: dict[str, Any],
    *,
    native_3dgs_status: str,
    background_asset_referenced: bool,
) -> dict[str, Any]:
    visual = background.get("visual_asset", {}) or {}
    registration = background.get("registration", {}) or {}
    external_render = background.get("external_render", {}) or {}
    reasons: list[str] = []
    caveats: list[str] = []

    visual_status = str(visual.get("status") or "missing_background_visual")
    if visual_status != "ready":
        reasons.append(visual_status)
    if registration.get("status") != "registered":
        reasons.append("background_unregistered")
    reasons.extend(str(item) for item in registration.get("blocking_reasons", []))

    external_sidecar = (
        visual.get("source_kind") == "external_3dgs_renderer"
        or external_render.get("backend") == "external_3dgs_renderer"
    )
    simulator_native = bool(visual.get("simulator_native") and visual.get("native_runtime_verified"))
    if external_sidecar:
        caveats.append("external_3dgs_render_sidecar_not_native_simulator_background")
    if visual.get("has_3dgs_sidecar") and not simulator_native:
        caveats.append("native_3dgs_runtime_not_verified")
    if native_3dgs_status not in {"not_requested", "native_3dgs_loaded", "registered_3dgs_loaded", "verified_native_3dgs"}:
        caveats.append(f"native_3dgs_status:{native_3dgs_status}")

    blocking_reasons = _dedupe(reasons)
    caveats = _dedupe(caveats)
    status = "blocked" if blocking_reasons else "partial" if caveats else "passed"
    return {
        "status": status,
        "complete": status == "passed",
        "blocking_reasons": blocking_reasons,
        "caveats": caveats,
        "background_asset_referenced": bool(background_asset_referenced),
        "external_sidecar": bool(external_sidecar),
        "external_render_verified": bool(visual.get("external_render_verified")),
        "simulator_native": simulator_native,
        "registration_status": registration.get("status", "unknown"),
        "native_3dgs_status": native_3dgs_status,
    }


def export_usd_scene(run_dir: str | Path) -> USDExportResult:
    run = Path(run_dir)
    export_dir = run / "exports"
    qa_dir = run / "qa"
    export_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_sim_export_manifest(run).to_dict()
    scene_path = export_dir / "scene.usda"
    scene_path.write_text(_scene_usda(run, scene_path.parent, manifest), encoding="utf-8")
    report = {
        "status": "exported",
        "scene_path": "exports/scene.usda",
        "object_count": len(manifest["objects"]),
        "support_surface_status": manifest["support_surface"]["status"],
        "contains_visual_references": all(obj["visual_asset"].get("path") for obj in manifest["objects"]),
        "contains_collision_references": all(obj["collision_asset"].get("path") for obj in manifest["objects"]),
        "contains_physics_api": True,
        "contains_reference_camera": True,
    }
    report.update(
        build_runtime_asset_report(
            manifest,
            backend="usd",
            visual_assets_referenced=True,
            collision_assets_referenced=True,
            background_asset_referenced=bool(manifest.get("background", {}).get("visual_asset", {}).get("path")),
            debug_proxy_visible=False,
            evidence=(
                "USD export writes visual and collision mesh references plus background asset metadata; "
                "native 3DGS rendering is not implemented or verified by this exporter."
            ),
        )
    )
    report_path = qa_dir / "usd_export_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_sim_export_manifest(run)
    return USDExportResult(scene_path=scene_path, report_path=report_path, report=report)


def _scene_usda(run: Path, export_dir: Path, manifest: dict[str, Any]) -> str:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    upAxis = "Z"',
        "    metersPerUnit = 1",
        ")",
        "",
        'def Xform "World"',
        "{",
        '    custom string rsf:schema = "real2sim_scene_foundry.sim_export_manifest"',
        '    custom string rsf:coordinate_frame = "z_up_ground_plane_meters"',
    ]
    lines.extend(_support_surface_usda(run, export_dir, manifest.get("support_surface", {})))
    lines.extend(_background_usda(manifest.get("background", {})))
    lines.extend(_objects_usda(run, export_dir, manifest.get("objects", [])))
    lines.extend(_camera_usda(run))
    lines.append("}")
    return "\n".join(lines) + "\n"


def _support_surface_usda(run: Path, export_dir: Path, support: dict[str, Any]) -> list[str]:
    mesh_path = support.get("mesh_path")
    if not mesh_path:
        return [
            '    def Xform "SupportSurface"',
            "    {",
            '        custom string rsf:status = "missing"',
            "    }",
        ]
    pos = support.get("table_collision_pos_world") or [0.0, 0.0, 0.0]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(pos, dtype=np.float64)[:3]
    return [
        '    def Xform "SupportSurface" (',
        '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
        f"        prepend references = @{_ref(export_dir, run / mesh_path)}@",
        "    )",
        "    {",
        '        custom string rsf:role = "support_surface"',
        f'        custom string rsf:source_backend = "{support.get("source_backend", "unknown")}"',
        "        bool physics:collisionEnabled = true",
        f"        matrix4d xformOp:transform = {_matrix4d(transform)}",
        '        uniform token[] xformOpOrder = ["xformOp:transform"]',
        "    }",
    ]


def _background_usda(background: dict[str, Any]) -> list[str]:
    visual = background.get("visual_asset", {})
    registration = background.get("registration", {})
    lines = [
        '    def Xform "Background"',
        "    {",
        f'        custom string rsf:visual_status = "{visual.get("status", "unknown")}"',
        f'        custom string rsf:source_kind = "{visual.get("source_kind", "unknown")}"',
        f'        custom string rsf:registration_status = "{registration.get("status", "unknown")}"',
        f"        custom bool rsf:simulator_native = {_bool_token(visual.get('simulator_native'))}",
        f"        custom bool rsf:external_render_verified = {_bool_token(visual.get('external_render_verified'))}",
    ]
    if visual.get("path"):
        lines.append(f'        custom asset rsf:backgroundVisual = @{visual["path"]}@')
    if registration.get("path"):
        lines.append(f'        custom asset rsf:registration = @{registration["path"]}@')
    lines.append("    }")
    return lines


def _objects_usda(run: Path, export_dir: Path, objects: list[dict[str, Any]]) -> list[str]:
    lines = ['    def Scope "Objects"', "    {"]
    for obj in objects:
        visual = obj["visual_asset"]
        collision = obj["collision_asset"]
        physics = obj["physics"]
        transform = np.asarray(obj["pose"]["T_object_to_world"], dtype=np.float64)
        asset_scale = _asset_scale(obj)
        scaled_transform = transform.copy()
        scaled_transform[:3, :3] *= asset_scale
        object_id = _usd_identifier(str(obj["object_id"]))
        visual_ref = _ref(export_dir, run / visual["path"]) if visual.get("path") else None
        collision_ref = _ref(export_dir, run / collision["path"]) if collision.get("path") else None
        lines.append(f'        def Xform "{object_id}" (')
        lines.append('            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]')
        if visual_ref:
            lines.append(f"            prepend references = @{visual_ref}@")
        lines.extend(
            [
                "        )",
                "        {",
                f'            custom string rsf:object_id = "{obj["object_id"]}"',
                f'            custom string rsf:label = "{obj.get("label", obj["object_id"])}"',
                f'            custom string rsf:visual_asset_status = "{visual.get("status", "unknown")}"',
                f'            custom string rsf:collision_asset_status = "{collision.get("status", "unknown")}"',
                f"            custom double rsf:asset_scale = {asset_scale:.8g}",
                "            bool physics:rigidBodyEnabled = true",
                f"            float physics:mass = {float(physics['mass_kg']):.8g}",
                f"            custom double rsf:friction = {float(physics['friction']):.8g}",
                f"            custom double rsf:restitution = {float(physics['restitution']):.8g}",
                f"            matrix4d xformOp:transform = {_matrix4d(scaled_transform)}",
                '            uniform token[] xformOpOrder = ["xformOp:transform"]',
            ]
        )
        if collision_ref:
            lines.extend(
                [
                    '            def Xform "Collision" (',
                    '                prepend apiSchemas = ["PhysicsCollisionAPI"]',
                    f"                prepend references = @{collision_ref}@",
                    "            )",
                    "            {",
                    "                bool physics:collisionEnabled = true",
                    '                custom string rsf:role = "collision_mesh"',
                    "            }",
                ]
            )
        lines.append("        }")
    lines.append("    }")
    return lines


def _camera_usda(run: Path) -> list[str]:
    camera_path = run / "camera.json"
    camera = json.loads(camera_path.read_text(encoding="utf-8")) if camera_path.is_file() else {}
    fx = float(camera.get("fx", 0.0))
    fy = float(camera.get("fy", 0.0))
    cx = float(camera.get("cx", 0.0))
    cy = float(camera.get("cy", 0.0))
    width = float(camera.get("width", 0.0))
    height = float(camera.get("height", 0.0))
    return [
        '    def Camera "ReferenceCamera"',
        "    {",
        '        custom string rsf:camera_frame = "opencv_x_right_y_down_z_forward_meters"',
        f"        custom double rsf:fx = {fx:.8g}",
        f"        custom double rsf:fy = {fy:.8g}",
        f"        custom double rsf:cx = {cx:.8g}",
        f"        custom double rsf:cy = {cy:.8g}",
        f"        custom double rsf:image_width = {width:.8g}",
        f"        custom double rsf:image_height = {height:.8g}",
        "    }",
    ]


def _matrix4d(matrix: np.ndarray) -> str:
    rows = []
    for row in np.asarray(matrix, dtype=np.float64).reshape(4, 4):
        rows.append("(" + ", ".join(f"{float(v):.10g}" for v in row) + ")")
    return "(" + ", ".join(rows) + ")"


def _asset_scale(obj: dict[str, Any]) -> float:
    pose = obj.get("pose", {}) or {}
    value = float(pose.get("asset_scale", obj.get("asset_scale", 1.0)) or 1.0)
    return value if np.isfinite(value) and value > 0.0 else 1.0


def _ref(from_dir: Path, path: Path) -> str:
    return Path(os.path.relpath(path, from_dir)).as_posix()


def _usd_identifier(value: str) -> str:
    out = []
    for ch in value:
        out.append(ch if ch.isalnum() or ch == "_" else "_")
    ident = "".join(out).strip("_") or "object"
    if ident[0].isdigit():
        ident = f"object_{ident}"
    return ident


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
