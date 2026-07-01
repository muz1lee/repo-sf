"""Nerfstudio-native viewer handoff for inspecting trained 3DGS assets."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NerfstudioViewerResult:
    report_path: Path
    launch_script_path: Path
    command: list[str]
    url: str
    report: dict[str, Any]


def export_nerfstudio_viewer(
    run_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 7014,
    ns_viewer_bin: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> NerfstudioViewerResult:
    """Write a fail-closed report and launcher for Nerfstudio's own viewer.

    This is intentionally not a simulator/native 3DGS integration. It replaces
    the browser PLY debug viewer as the quality-inspection entrypoint only.
    """

    run = Path(run_dir)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    viewer_dir = run / "exports" / "nerfstudio_viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)

    resolved_config, config_source = _resolve_config_path(run, root, config_path)
    training_report = _read_json(run / "qa" / "phone_3dgs_training_report.json")
    training = training_report.get("training", {}) if isinstance(training_report.get("training"), dict) else {}
    metrics = training_report.get("metrics", {}) if isinstance(training_report.get("metrics"), dict) else {}
    viewer_bin = _viewer_bin(root, ns_viewer_bin)
    url = f"http://{host}:{int(port)}/"
    command = [
        str(viewer_bin),
        "--load-config",
        str(resolved_config) if resolved_config is not None else "<missing-config.yml>",
        "--viewer.websocket-host",
        str(host),
        "--viewer.websocket-port",
        str(int(port)),
    ]

    status = "ready" if resolved_config is not None and resolved_config.is_file() else "blocked_missing_nerfstudio_config"
    report: dict[str, Any] = {
        "version": 1,
        "status": status,
        "viewer": "nerfstudio_native",
        "purpose": "3dgs_quality_inspection",
        "simulator_native": False,
        "replaces_ply_browser_quality_viewer": True,
        "ply_browser_viewer_status": "deprecated_debug_not_quality_reference",
        "run_dir": _display_path(run, root),
        "config_source": config_source,
        "load_config_path": _display_path(resolved_config, root) if resolved_config is not None else None,
        "checkpoint_path": training.get("checkpoint_path"),
        "training_iterations": training.get("iterations"),
        "metrics": {
            "train_psnr_last": metrics.get("train_psnr_last"),
            "eval_psnr_last": metrics.get("eval_psnr_last"),
            "eval_ssim_last": metrics.get("eval_ssim_last"),
            "eval_lpips_last": metrics.get("eval_lpips_last"),
        },
        "host": host,
        "port": int(port),
        "url": url,
        "command": command,
        "launcher_path": "exports/nerfstudio_viewer/launch.sh",
        "blocking_reason": None,
        "claim_boundary": {
            "3dgs_quality_inspection": "passed_when_viewer_loads_and_render_is_visually_checked",
            "background_registration": "not_claimed_by_nerfstudio_viewer",
            "simulator_native_3dgs": "not_claimed_by_nerfstudio_viewer",
        },
        "note": "Use this native Nerfstudio viewer to inspect trained 3DGS quality. Use composite/runtime viewers for collision, object pose, and simulator-state QA.",
    }
    if status != "ready":
        report["blocking_reason"] = "No Nerfstudio config.yml could be resolved for this run."

    report_path = qa_dir / "nerfstudio_viewer_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    launch_script_path = viewer_dir / "launch.sh"
    launch_script_path.write_text(_launch_script(root, command, status), encoding="utf-8")
    launch_script_path.chmod(0o755)
    return NerfstudioViewerResult(
        report_path=report_path,
        launch_script_path=launch_script_path,
        command=command,
        url=url,
        report=report,
    )


def _resolve_config_path(run: Path, repo_root: Path, explicit: str | Path | None) -> tuple[Path | None, str]:
    if explicit is not None:
        path = _resolve_candidate(run, repo_root, explicit)
        return (path if path.is_file() else path, "explicit")

    training_report = _read_json(run / "qa" / "phone_3dgs_training_report.json")
    training = training_report.get("training", {}) if isinstance(training_report.get("training"), dict) else {}
    if training.get("config_path"):
        path = _resolve_candidate(run, repo_root, training["config_path"])
        if path.is_file():
            return path, "phone_3dgs_training_report"

    status = _read_json(run / "video" / "3dgs_status.json")
    outputs = status.get("outputs", {}) if isinstance(status.get("outputs"), dict) else {}
    if outputs.get("latest_config"):
        path = _resolve_candidate(run, repo_root, outputs["latest_config"])
        if path.is_file():
            return path, "video_3dgs_status"

    configs = sorted((run / "video" / "3dgs").glob("**/config.yml"), key=lambda item: item.stat().st_mtime)
    if configs:
        return configs[-1], "video_3dgs_glob"
    return None, "missing"


def _resolve_candidate(run: Path, repo_root: Path, value: str | Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    repo_candidate = repo_root / raw
    if repo_candidate.is_file():
        return repo_candidate
    run_candidate = run / raw
    if run_candidate.is_file():
        return run_candidate
    return repo_candidate


def _viewer_bin(repo_root: Path, explicit: str | Path | None) -> Path | str:
    if explicit is not None:
        return Path(explicit)
    bundled = repo_root / ".venv_3dgs" / "bin" / "ns-viewer"
    return bundled if bundled.is_file() else "ns-viewer"


def _launch_script(repo_root: Path, command: list[str], status: str) -> str:
    if status != "ready":
        return (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "echo 'No Nerfstudio config.yml could be resolved for this run.' >&2\n"
            "exit 4\n"
        )
    quoted = " ".join(shlex.quote(part) for part in command)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(repo_root))}\n"
        f"exec {quoted}\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)
