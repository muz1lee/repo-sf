"""Operator bridge helpers for running Isaac validation on an external worker."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .defaults import ISAAC_WORKER_PYTHON, ISAAC_WORKER_SSH
from .sim_export_manifest import write_sim_export_manifest
from .usd_export import build_runtime_asset_report


DEFAULT_PROJECT_DIR = "/mnt/workspace/wenqian/real2sim_scene_foundry"


@dataclass(frozen=True)
class IsaacWorkerBundle:
    manifest_path: Path
    file_list_path: Path
    files: list[str]
    commands: dict[str, str]
    missing_files: list[str]


@dataclass(frozen=True)
class IsaacWorkerReportIngestResult:
    report_path: Path
    report: dict[str, Any]


def build_isaac_worker_bundle(
    run_dir: str | Path,
    *,
    worker_run_dir: str | None = None,
    project_dir: str = DEFAULT_PROJECT_DIR,
    worker_ssh: str = ISAAC_WORKER_SSH,
    worker_python: str = ISAAC_WORKER_PYTHON,
) -> IsaacWorkerBundle:
    """Write a minimal H200-to-worker bundle manifest without attempting SSH."""

    run = Path(run_dir)
    export_dir = run / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    sim_manifest_path = write_sim_export_manifest(run)
    manifest = json.loads(sim_manifest_path.read_text(encoding="utf-8"))
    files, missing_files = _bundle_files(run, manifest)
    file_list_path = export_dir / "isaac_worker_bundle_files.txt"
    file_list_path.write_text("\n".join(files) + ("\n" if files else ""), encoding="utf-8")

    resolved_worker_run_dir = worker_run_dir or f"/root/real2sim_scene_foundry_runs/{run.name}"
    commands = _operator_commands(
        run,
        file_list_path,
        worker_run_dir=resolved_worker_run_dir,
        project_dir=project_dir,
        worker_ssh=worker_ssh,
        worker_python=worker_python,
    )
    bundle_path = export_dir / "isaac_worker_bundle.json"
    bundle = {
        "version": 1,
        "run_dir": str(run),
        "worker_run_dir": resolved_worker_run_dir,
        "files": files,
        "missing_files": missing_files,
        "commands": commands,
        "notes": [
            "Run these commands from the local operator machine, not from wenqian_h200.",
            "This bridge does not assume wenqian_h200 can SSH directly to the Isaac worker.",
        ],
    }
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return IsaacWorkerBundle(
        manifest_path=bundle_path,
        file_list_path=file_list_path,
        files=files,
        commands=commands,
        missing_files=missing_files,
    )


def ingest_isaac_worker_report(run_dir: str | Path, report_path: str | Path) -> IsaacWorkerReportIngestResult:
    """Copy an operator-returned Isaac report into the canonical run and enrich it."""

    run = Path(run_dir)
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    source_report = Path(report_path)
    out_path = qa_dir / "isaac_load_report.json"
    sim_manifest_path = write_sim_export_manifest(run)
    manifest = json.loads(sim_manifest_path.read_text(encoding="utf-8"))

    if not source_report.is_file():
        report = _runtime_unavailable_report(
            reason="isaac_worker_report_missing",
            worker_report_path=str(source_report),
            bridge_status="missing_report",
        )
    else:
        report = _read_worker_report(source_report)
        report.setdefault("status", "failed")
        report.setdefault("reason", "worker_report_missing_status")
        report["worker_bridge"] = {
            "status": "ingested",
            "ingested_from": str(source_report),
            "canonical_report_path": "qa/isaac_load_report.json",
            "h200_direct_worker_ssh_assumed": False,
        }
    _merge_runtime_asset_report(report, manifest)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_sim_export_manifest(run)
    return IsaacWorkerReportIngestResult(report_path=out_path, report=report)


def _bundle_files(run: Path, manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    requested = [
        "scene_manifest.json",
        "sim_export_manifest.json",
        "exports/scene.usda",
        "exports/isaac_scene.py",
    ]
    for obj in manifest.get("objects", []):
        for key in ("visual_asset", "collision_asset"):
            _append_path(requested, obj.get(key, {}).get("path"))
    background = manifest.get("background", {})
    _append_path(requested, background.get("visual_asset", {}).get("path"))
    _append_path(requested, background.get("registration", {}).get("path"))
    _append_path(requested, manifest.get("support_surface", {}).get("mesh_path"))

    files: list[str] = []
    missing: list[str] = []
    for rel in _dedupe(requested):
        safe_rel = _safe_relative_path(run, rel)
        if safe_rel is None:
            missing.append(str(rel))
            continue
        if (run / safe_rel).is_file():
            files.append(safe_rel)
        else:
            missing.append(safe_rel)
    return files, missing


def _operator_commands(
    run: Path,
    file_list_path: Path,
    *,
    worker_run_dir: str,
    project_dir: str,
    worker_ssh: str,
    worker_python: str,
) -> dict[str, str]:
    run_q = shlex.quote(str(run))
    file_list_q = shlex.quote(str(file_list_path))
    worker_run_q = shlex.quote(worker_run_dir)
    worker_report_q = shlex.quote(f"{worker_run_dir}/qa/isaac_load_report.json")
    project_q = shlex.quote(project_dir)
    qa_dir_q = shlex.quote(str(run / "qa"))
    returned_report_q = shlex.quote(str(run / "qa" / "isaac_worker_returned_report.json"))

    transfer = (
        f"ssh wenqian_h200 {shlex.quote(f'tar -czf - -C {run_q} -T {file_list_q}')}"
        f" | {worker_ssh} {shlex.quote(f'mkdir -p {worker_run_q} && tar -xzf - -C {worker_run_q}')}"
    )
    run_worker = (
        f"{worker_ssh} "
        f"{shlex.quote(f'cd {worker_run_q} && {shlex.quote(worker_python)} exports/isaac_scene.py --run-dir {worker_run_q}')}"
    )
    ingest = (
        f"{worker_ssh} {shlex.quote(f'cat {worker_report_q}')}"
        f" | ssh wenqian_h200 "
        f"{shlex.quote(f'cd {project_q} && source .venv/bin/activate && mkdir -p {qa_dir_q} && cat > {returned_report_q} && rsf isaac-worker-report --run-dir {run_q} --report {returned_report_q}')}"
    )
    return {
        "transfer_to_worker": transfer,
        "run_on_worker": run_worker,
        "ingest_report_to_h200": ingest,
    }


def _runtime_unavailable_report(
    *,
    reason: str,
    worker_report_path: str,
    bridge_status: str,
) -> dict[str, Any]:
    return {
        "status": "runtime_unavailable",
        "reason": reason,
        "worker_report_path": worker_report_path,
        "blocking_reasons": [reason],
        "worker_bridge": {
            "status": bridge_status,
            "ingested_from": worker_report_path,
            "canonical_report_path": "qa/isaac_load_report.json",
            "h200_direct_worker_ssh_assumed": False,
        },
    }


def _read_worker_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "reason": "isaac_worker_report_invalid_json",
            "error": str(exc),
            "worker_report_path": str(path),
        }
    if not isinstance(value, dict):
        return {
            "status": "failed",
            "reason": "isaac_worker_report_not_object",
            "worker_report_path": str(path),
        }
    return value


def _merge_runtime_asset_report(report: dict[str, Any], manifest: dict[str, Any]) -> None:
    loaded = report.get("status") in {"loaded", "passed"}
    sections = build_runtime_asset_report(
        manifest,
        backend="isaac",
        visual_assets_referenced=loaded,
        collision_assets_referenced=loaded,
        background_asset_referenced=loaded,
        debug_proxy_visible=bool(report.get("debug_proxy_visible", False)),
        evidence=(
            "Isaac report was returned through the operator bridge; "
            "this does not require or imply direct wenqian_h200-to-worker SSH access."
        ),
    )
    for key, value in sections.items():
        if key not in report:
            report[key] = value
        elif isinstance(value, dict) and isinstance(report[key], dict):
            merged = value.copy()
            merged.update(report[key])
            report[key] = merged


def _append_path(paths: list[str], value: Any) -> None:
    if value:
        paths.append(str(value))


def _safe_relative_path(run: Path, rel: str) -> str | None:
    candidate = Path(rel)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(run.resolve()).as_posix()
        except ValueError:
            return None
    if any(part == ".." for part in candidate.parts):
        return None
    return candidate.as_posix()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
