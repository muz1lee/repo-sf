import json

from real2sim_scene_foundry.cli import main
from real2sim_scene_foundry.nerfstudio_viewer import export_nerfstudio_viewer


def test_export_nerfstudio_viewer_uses_training_report_config_and_marks_ply_debug(tmp_path):
    repo_root = tmp_path
    run = repo_root / "runs" / "phone_run"
    run.mkdir(parents=True)
    config = repo_root / "runs" / "train_run" / "3dgs" / "record3d" / "splatfacto" / "train_30000" / "config.yml"
    config.parent.mkdir(parents=True)
    config.write_text("method: splatfacto\\n", encoding="utf-8")
    qa = run / "qa"
    qa.mkdir()
    (qa / "phone_3dgs_training_report.json").write_text(
        json.dumps(
            {
                "training": {
                    "iterations": 30000,
                    "config_path": str(config.relative_to(repo_root)),
                    "checkpoint_path": "runs/train_run/3dgs/record3d/splatfacto/train_30000/nerfstudio_models/step-000029999.ckpt",
                },
                "metrics": {"eval_psnr_last": 33.1, "eval_ssim_last": 0.937},
            }
        ),
        encoding="utf-8",
    )

    result = export_nerfstudio_viewer(run, repo_root=repo_root, host="127.0.0.1", port=7014)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["viewer"] == "nerfstudio_native"
    assert report["purpose"] == "3dgs_quality_inspection"
    assert report["simulator_native"] is False
    assert report["replaces_ply_browser_quality_viewer"] is True
    assert report["ply_browser_viewer_status"] == "deprecated_debug_not_quality_reference"
    assert report["training_iterations"] == 30000
    assert report["config_source"] == "phone_3dgs_training_report"
    assert report["load_config_path"] == str(config.relative_to(repo_root))
    assert report["url"] == "http://127.0.0.1:7014/"
    assert "ns-viewer" in " ".join(report["command"])
    assert result.launch_script_path.is_file()


def test_cli_nerfstudio_viewer_export_only_writes_report(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = tmp_path / "config.yml"
    config.write_text("method: splatfacto\\n", encoding="utf-8")

    code = main(["nerfstudio-viewer", "--run-dir", str(run), "--config", str(config), "--port", "7016", "--export-only"])

    assert code == 0
    report = json.loads((run / "qa" / "nerfstudio_viewer_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["viewer"] == "nerfstudio_native"
    assert report["url"] == "http://127.0.0.1:7016/"


def test_export_nerfstudio_viewer_blocks_without_config(tmp_path):
    run = tmp_path / "run"
    run.mkdir()

    result = export_nerfstudio_viewer(run, repo_root=tmp_path)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked_missing_nerfstudio_config"
    assert report["blocking_reason"] == "No Nerfstudio config.yml could be resolved for this run."
    assert report["simulator_native"] is False
