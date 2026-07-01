"""Asset-level checks for native Gaussian splat PLY files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REQUIRED_GAUSSIAN_FIELDS = (
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)

RGB_FIELD_GROUPS = (
    ("red", "green", "blue"),
    ("r", "g", "b"),
    ("rgb",),
    ("f_dc_0", "f_dc_1", "f_dc_2"),
)


@dataclass(frozen=True)
class GaussianSplatPlyHeader:
    asset_path: Path
    gaussian_count: int | None
    format: str | None
    vertex_fields: tuple[str, ...]
    rgb_fields: tuple[str, ...]
    required_fields_present: dict[str, bool]
    missing_required_fields: list[str]
    comments: tuple[str, ...]


def parse_gaussian_splat_ply_header(asset_path: str | Path) -> GaussianSplatPlyHeader:
    path = Path(asset_path)
    lines = _read_ply_header_lines(path)
    ply_format: str | None = None
    gaussian_count: int | None = None
    vertex_fields: list[str] = []
    comments: list[str] = []
    current_element: str | None = None

    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        keyword = parts[0]
        if keyword == "format":
            ply_format = " ".join(parts[1:])
        elif keyword == "comment":
            comments.append(line[len("comment") :].strip())
        elif keyword == "element":
            if len(parts) < 3:
                raise ValueError(f"invalid PLY element line: {line}")
            current_element = parts[1]
            if current_element == "vertex":
                try:
                    gaussian_count = int(parts[2])
                except ValueError as exc:
                    raise ValueError(f"invalid vertex count in PLY header: {parts[2]}") from exc
        elif keyword == "property" and current_element == "vertex":
            if len(parts) < 3:
                raise ValueError(f"invalid PLY property line: {line}")
            vertex_fields.append(parts[-1])

    field_set = set(vertex_fields)
    rgb_fields = _matching_rgb_fields(field_set)
    required_fields_present = {field: field in field_set for field in REQUIRED_GAUSSIAN_FIELDS}
    required_fields_present["rgb"] = bool(rgb_fields)
    missing_required_fields = [
        field for field in REQUIRED_GAUSSIAN_FIELDS if not required_fields_present[field]
    ]
    if not required_fields_present["rgb"]:
        missing_required_fields.append("rgb")

    return GaussianSplatPlyHeader(
        asset_path=path,
        gaussian_count=gaussian_count,
        format=ply_format,
        vertex_fields=tuple(vertex_fields),
        rgb_fields=rgb_fields,
        required_fields_present=required_fields_present,
        missing_required_fields=missing_required_fields,
        comments=tuple(comments),
    )


def build_gaussian_splat_asset_report(asset_path: str | Path) -> dict[str, object]:
    path = Path(asset_path)
    required_fields_present = {field: False for field in REQUIRED_GAUSSIAN_FIELDS}
    required_fields_present["rgb"] = False

    try:
        header = parse_gaussian_splat_ply_header(path)
    except FileNotFoundError:
        return _base_report(
            asset_path=path,
            status="blocked_missing_asset",
            gaussian_count=None,
            ply_format=None,
            required_fields_present=required_fields_present,
            missing_required_fields=list(required_fields_present),
            vertex_fields=[],
            rgb_fields=[],
            blocked_reason=f"asset_missing: {path}",
        )
    except ValueError as exc:
        return _base_report(
            asset_path=path,
            status="blocked_invalid_ply_header",
            gaussian_count=None,
            ply_format=None,
            required_fields_present=required_fields_present,
            missing_required_fields=list(required_fields_present),
            vertex_fields=[],
            rgb_fields=[],
            blocked_reason=f"invalid_ply_header: {exc}",
        )

    blocked_reason = None
    status = "asset_format_complete"
    if header.missing_required_fields:
        status = "blocked_missing_required_fields"
        blocked_reason = "missing_required_fields: " + ", ".join(header.missing_required_fields)

    return _base_report(
        asset_path=path,
        status=status,
        gaussian_count=header.gaussian_count,
        ply_format=header.format,
        required_fields_present=header.required_fields_present,
        missing_required_fields=header.missing_required_fields,
        vertex_fields=list(header.vertex_fields),
        rgb_fields=list(header.rgb_fields),
        blocked_reason=blocked_reason,
    )


def write_gaussian_splat_asset_report(
    asset_path: str | Path,
    report_path: str | Path | None = None,
) -> Path:
    path = Path(asset_path)
    output = Path(report_path) if report_path is not None else path.parent / "asset_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_gaussian_splat_asset_report(path), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def build_background_3dgs_runtime_report(run_dir: str | Path) -> dict[str, object]:
    run = Path(run_dir)
    asset_path = run / "background" / "3dgs_native" / "splat_rgb.ply"
    asset_schema = build_gaussian_splat_asset_report(asset_path)
    external_render = _external_render_status(run)
    native_viewer_runtime = _native_runtime_status(
        run,
        [
            "qa/runtime_viewer_report.json",
            "exports/composite_viewer/viewer_config.json",
        ],
    )
    native_simulator_runtime = _native_runtime_status(
        run,
        [
            "qa/background_runtime_report.json",
            "qa/usd_export_report.json",
            "qa/genesis_export_report.json",
            "qa/isaac_load_report.json",
        ],
    )
    native_runtime_supported = bool(
        native_viewer_runtime["supported"] or native_simulator_runtime["supported"]
    )

    blocked_reason = None
    if native_runtime_supported:
        status = "native_3dgs_runtime_verified"
    elif asset_schema["status"] == "asset_format_complete" and external_render["verified"]:
        status = "partial_external_render_only"
        blocked_reason = (
            "External reference-view 3DGS render exists, but native viewer/simulator runtime "
            "loading is not verified."
        )
    elif asset_schema["status"] == "asset_format_complete":
        status = "blocked_native_3dgs_runtime"
        blocked_reason = (
            "3DGS asset schema is complete, but native viewer/simulator runtime loading is not verified."
        )
    elif asset_schema["status"] == "blocked_missing_asset":
        status = "blocked_missing_native_3dgs_asset"
        blocked_reason = str(asset_schema.get("blocked_reason") or "missing native 3DGS asset")
    else:
        status = "blocked_invalid_native_3dgs_asset"
        blocked_reason = str(asset_schema.get("blocked_reason") or asset_schema["status"])

    return {
        "version": 1,
        "status": status,
        "blocked_reason": blocked_reason,
        "asset_path": _rel(run, asset_path),
        "asset_schema": asset_schema,
        "external_render": external_render,
        "native_runtime_supported": native_runtime_supported,
        "native_viewer_runtime": native_viewer_runtime,
        "native_simulator_runtime": native_simulator_runtime,
        "scope_note": (
            "This report distinguishes native 3DGS asset/schema readiness from live viewer or "
            "simulator splat loading. External Nerfstudio renders are reference-view sidecars, "
            "not live reconstructed-scene runtime support."
        ),
    }


def write_background_3dgs_runtime_report(
    run_dir: str | Path,
    report_path: str | Path | None = None,
) -> Path:
    run = Path(run_dir)
    output = Path(report_path) if report_path is not None else run / "qa" / "background_3dgs_runtime_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_background_3dgs_runtime_report(run), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an asset-only report for a 3DGS PLY file.")
    parser.add_argument("asset_path", type=Path)
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args(argv)

    report_path = write_gaussian_splat_asset_report(args.asset_path, args.report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(report_path)
    return 0 if report["status"] == "asset_format_complete" else 2


def _read_ply_header_lines(path: Path, *, max_header_bytes: int = 1_048_576) -> list[str]:
    lines: list[str] = []
    bytes_read = 0
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError("missing end_header")
            bytes_read += len(raw)
            if bytes_read > max_header_bytes:
                raise ValueError(f"header exceeds {max_header_bytes} bytes")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError("PLY header is not ASCII-decodable") from exc
            lines.append(line)
            if len(lines) == 1 and line != "ply":
                raise ValueError("missing ply magic")
            if line == "end_header":
                return lines


def _matching_rgb_fields(field_set: set[str]) -> tuple[str, ...]:
    for group in RGB_FIELD_GROUPS:
        if all(field in field_set for field in group):
            return group
    return ()


def _base_report(
    *,
    asset_path: Path,
    status: str,
    gaussian_count: int | None,
    ply_format: str | None,
    required_fields_present: dict[str, bool],
    missing_required_fields: list[str],
    vertex_fields: list[str],
    rgb_fields: list[str],
    blocked_reason: str | None,
) -> dict[str, object]:
    return {
        "status": status,
        "asset_path": str(asset_path),
        "gaussian_count": gaussian_count,
        "format": ply_format,
        "required_fields_present": required_fields_present,
        "missing_required_fields": missing_required_fields,
        "vertex_fields": vertex_fields,
        "rgb_fields": rgb_fields,
        "blocked_reason": blocked_reason,
        "scope_note": (
            "This report only proves the 3DGS asset exists and its PLY header has the expected "
            "Gaussian fields. It does not prove simulator native rendering."
        ),
        "simulator_native_rendering": {
            "status": "not_verified",
            "native_rendering_proven": False,
            "blocked_reason": "asset_report_does_not_exercise_simulator_runtime",
        },
        "proxy_conversion": {
            "generated": False,
            "caveat": (
                "Do not promote this asset report, a mesh surrogate, or a point-cloud surrogate "
                "to final background completion. Any conversion remains a blocked/proxy path "
                "until a simulator native 3DGS render bridge is verified."
            ),
        },
    }


def _external_render_status(run: Path) -> dict[str, object]:
    report_path = run / "qa" / "background_3dgs_render_report.json"
    if not report_path.is_file():
        return {
            "status": "missing",
            "path": None,
            "render_path": None,
            "verified": False,
            "reference_view_only": False,
            "simulator_native": False,
        }
    report = _load_json(report_path)
    render_rel = report.get("render_path")
    verified = (
        report.get("status") == "rendered"
        and report.get("backend") == "external_3dgs_renderer"
        and report.get("registered_3dgs_rendered") is True
        and bool(render_rel)
        and (run / str(render_rel)).is_file()
    )
    simulator_native = report.get("simulator_native") is True
    return {
        "status": str(report.get("status") or "unknown"),
        "path": _rel(run, report_path),
        "backend": report.get("backend"),
        "render_path": render_rel,
        "verified": verified,
        "reference_view_only": bool(verified and not simulator_native),
        "simulator_native": simulator_native,
        "native_rendering_proven": report.get("native_rendering_proven") is True,
        "blocked_reason": report.get("blocked_reason"),
    }


def _native_runtime_status(run: Path, rel_paths: list[str]) -> dict[str, object]:
    checked: list[str] = []
    for rel_path in rel_paths:
        path = run / rel_path
        if not path.is_file():
            continue
        checked.append(rel_path)
        report = _load_json(path)
        if _report_verifies_native_3dgs(report):
            return {
                "status": "verified",
                "supported": True,
                "checked_reports": checked,
                "evidence_path": rel_path,
            }
    return {
        "status": "unsupported",
        "supported": False,
        "checked_reports": checked,
        "evidence_path": None,
        "blocked_reason": "no report proves live/native 3DGS splat loading",
    }


def _report_verifies_native_3dgs(report: dict[str, Any]) -> bool:
    if report.get("background_runtime_verified") is True:
        return True
    for key in [
        "background_visual_status",
        "visual_background_status",
        "background_status",
        "status",
    ]:
        value = str(report.get(key, "")).lower()
        if value in {"native_3dgs_loaded", "registered_3dgs_loaded", "verified_native_3dgs"}:
            return True
    return False


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _rel(run: Path, path: Path) -> str:
    try:
        return path.relative_to(run).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
