"""External 3DGS background rendering through Nerfstudio."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .gaussian_splat_assets import write_background_3dgs_runtime_report


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class Background3DGSRenderResult:
    report_path: Path
    report: dict[str, Any]


def render_external_3dgs_background(
    run_dir: str | Path,
    *,
    renderer_executable: str | Path | None = None,
    split: str = "test",
    camera_idx: int = 0,
    runner: Runner | None = None,
) -> Background3DGSRenderResult:
    run = Path(run_dir)
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    report_path = qa_dir / "background_3dgs_render_report.json"
    final_render = qa_dir / "background_3dgs_render.png"
    output_dir = qa_dir / "background_3dgs_render_output"

    status = _load_3dgs_status(run)
    config_rel = _latest_config_path(status)
    checkpoint_rel = _latest_checkpoint_path(status)
    if not config_rel:
        report = _blocked_report(run, "missing_3dgs_config", None, None, None, split, camera_idx)
        return _write_result(report_path, report, run=run)
    config_path = run / config_rel
    if not config_path.is_file():
        report = _blocked_report(run, "missing_3dgs_config_file", config_rel, checkpoint_rel, None, split, camera_idx)
        return _write_result(report_path, report, run=run)

    executable = Path(renderer_executable) if renderer_executable is not None else _default_renderer_executable(run, status)
    command = [
        str(executable),
        "dataset",
        "--load-config",
        str(config_path),
        "--output-path",
        str(output_dir),
        "--split",
        split,
        "--camera-idx",
        str(int(camera_idx)),
        "--rendered-output-names",
        "rgb",
        "--image-format",
        "png",
    ]
    run_process = runner or subprocess.run
    completed = run_process(
        command,
        cwd=_find_project_root(run),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if int(completed.returncode) != 0:
        report = _blocked_report(
            run,
            "renderer_command_failed",
            config_rel,
            checkpoint_rel,
            command,
            split,
            camera_idx,
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        return _write_result(report_path, report, run=run)

    selected = _select_rendered_image(output_dir, split)
    if selected is None:
        report = _blocked_report(
            run,
            "renderer_produced_no_rgb_image",
            config_rel,
            checkpoint_rel,
            command,
            split,
            camera_idx,
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        return _write_result(report_path, report, run=run)

    shutil.copyfile(selected, final_render)
    width, height = Image.open(final_render).size
    report = {
        "version": 1,
        "status": "rendered",
        "backend": "external_3dgs_renderer",
        "source_kind": "external_3dgs_renderer",
        "registered_3dgs_rendered": True,
        "simulator_native": False,
        "native_rendering_proven": False,
        "config_path": config_rel,
        "checkpoint_path": checkpoint_rel,
        "render_path": _rel(run, final_render),
        "selected_render_source": _rel(run, selected),
        "output_dir": _rel(run, output_dir),
        "split": split,
        "camera_idx": int(camera_idx),
        "image_size": [width, height],
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
        "note": "Background is rendered by external Nerfstudio 3DGS; Genesis/Isaac still verify foreground physics and USD loading, not native splat rendering.",
    }
    return _write_result(report_path, report, run=run)


def _blocked_report(
    run: Path,
    reason: str,
    config_rel: str | None,
    checkpoint_rel: str | None,
    command: list[str] | None,
    split: str,
    camera_idx: int,
    *,
    returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "version": 1,
        "status": "blocked",
        "backend": "external_3dgs_renderer",
        "source_kind": "external_3dgs_renderer",
        "registered_3dgs_rendered": False,
        "simulator_native": False,
        "native_rendering_proven": False,
        "blocked_reason": reason,
        "config_path": config_rel,
        "checkpoint_path": checkpoint_rel,
        "render_path": None,
        "output_dir": _rel(run, run / "qa" / "background_3dgs_render_output"),
        "split": split,
        "camera_idx": int(camera_idx),
        "command": command,
        "returncode": returncode,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
    }


def _write_result(
    report_path: Path,
    report: dict[str, Any],
    *,
    run: Path | None = None,
) -> Background3DGSRenderResult:
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if run is not None:
        write_background_3dgs_runtime_report(run)
    return Background3DGSRenderResult(report_path=report_path, report=report)


def _load_3dgs_status(run: Path) -> dict[str, Any]:
    path = run / "video" / "3dgs_status.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _latest_config_path(status: dict[str, Any]) -> str | None:
    outputs = status.get("outputs") if isinstance(status.get("outputs"), dict) else {}
    value = outputs.get("latest_config") if outputs else None
    return str(value) if value else None


def _latest_checkpoint_path(status: dict[str, Any]) -> str | None:
    outputs = status.get("outputs") if isinstance(status.get("outputs"), dict) else {}
    value = outputs.get("latest_checkpoint") if outputs else None
    return str(value) if value else None


def _default_renderer_executable(run: Path, status: dict[str, Any]) -> Path:
    runtime = status.get("runtime") if isinstance(status.get("runtime"), dict) else {}
    ns_train = runtime.get("ns_train") if runtime else None
    if ns_train:
        return Path(str(ns_train)).with_name("ns-render")
    return _find_project_root(run) / ".venv_3dgs" / "bin" / "ns-render"


def _find_project_root(run: Path) -> Path:
    for candidate in [run, *run.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def _select_rendered_image(output_dir: Path, split: str) -> Path | None:
    preferred = [
        output_dir / split / "rgb" / "frame_000000.png",
        output_dir / "test" / "rgb" / "frame_000000.png",
        output_dir / "train" / "rgb" / "frame_000000.png",
    ]
    for path in preferred:
        if path.is_file():
            return path
    images = sorted(
        [
            *output_dir.rglob("*.png"),
            *output_dir.rglob("*.jpg"),
            *output_dir.rglob("*.jpeg"),
        ]
    )
    return images[0] if images else None


def _rel(run: Path, path: Path) -> str:
    try:
        return path.relative_to(run).as_posix()
    except ValueError:
        return path.as_posix()


def _tail(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]
