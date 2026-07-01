import json
import subprocess
from pathlib import Path

from PIL import Image

from real2sim_scene_foundry.background_3dgs_render import render_external_3dgs_background


def _write_3dgs_run(run_dir: Path) -> None:
    config = run_dir / "video" / "3dgs" / "run" / "config.yml"
    checkpoint = run_dir / "video" / "3dgs" / "run" / "nerfstudio_models" / "step.ckpt"
    config.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config.write_text("method_name: splatfacto\n", encoding="utf-8")
    checkpoint.write_text("checkpoint", encoding="utf-8")
    (run_dir / "video" / "3dgs_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "runtime": {"ns_train": "/opt/nerfstudio/bin/ns-train"},
                "outputs": {
                    "latest_config": "video/3dgs/run/config.yml",
                    "latest_checkpoint": "video/3dgs/run/nerfstudio_models/step.ckpt",
                },
            }
        ),
        encoding="utf-8",
    )


def test_external_3dgs_renderer_writes_blocked_report_when_command_fails(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_3dgs_run(run_dir)

    def failing_runner(command, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(command, 7, stdout="partial stdout", stderr="renderer exploded")

    result = render_external_3dgs_background(
        run_dir,
        renderer_executable=Path("/opt/nerfstudio/bin/ns-render"),
        runner=failing_runner,
    )

    assert result.report_path == run_dir / "qa" / "background_3dgs_render_report.json"
    assert result.report["status"] == "blocked"
    assert result.report["blocked_reason"] == "renderer_command_failed"
    assert result.report["returncode"] == 7
    assert result.report["registered_3dgs_rendered"] is False
    assert result.report["simulator_native"] is False
    assert "renderer exploded" in result.report["stderr_tail"]
    assert result.report["render_path"] is None
    assert not (run_dir / "qa" / "background_3dgs_render.png").exists()


def test_external_3dgs_renderer_selects_rendered_image_and_records_provenance(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_3dgs_run(run_dir)
    _write_valid_splat(run_dir / "background" / "3dgs_native" / "splat_rgb.ply")

    def rendering_runner(command, **kwargs):  # noqa: ANN001
        output_path = Path(command[command.index("--output-path") + 1])
        image_path = output_path / "test" / "rgb" / "frame_000000.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image_path)
        return subprocess.CompletedProcess(command, 0, stdout="rendered", stderr="")

    result = render_external_3dgs_background(
        run_dir,
        renderer_executable=Path("/opt/nerfstudio/bin/ns-render"),
        split="test",
        runner=rendering_runner,
    )

    assert result.report["status"] == "rendered"
    assert result.report["backend"] == "external_3dgs_renderer"
    assert result.report["source_kind"] == "external_3dgs_renderer"
    assert result.report["registered_3dgs_rendered"] is True
    assert result.report["simulator_native"] is False
    assert result.report["native_rendering_proven"] is False
    assert result.report["config_path"] == "video/3dgs/run/config.yml"
    assert result.report["checkpoint_path"] == "video/3dgs/run/nerfstudio_models/step.ckpt"
    assert result.report["render_path"] == "qa/background_3dgs_render.png"
    assert result.report["selected_render_source"] == "qa/background_3dgs_render_output/test/rgb/frame_000000.png"
    assert Image.open(run_dir / "qa" / "background_3dgs_render.png").size == (8, 6)
    runtime_report = json.loads(
        (run_dir / "qa" / "background_3dgs_runtime_report.json").read_text(encoding="utf-8")
    )
    assert runtime_report["status"] == "partial_external_render_only"
    assert runtime_report["status"] != "passed"
    assert runtime_report["external_render"]["reference_view_only"] is True
    assert runtime_report["native_runtime_supported"] is False


def _write_valid_splat(asset: Path) -> None:
    asset.parent.mkdir(parents=True)
    asset.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property float opacity",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "property float rot_0",
                "property float rot_1",
                "property float rot_2",
                "property float rot_3",
                "end_header",
                "0 0 0 255 255 255 1 0 0 0 1 0 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
