import json
from pathlib import Path

import numpy as np
from PIL import Image

from real2sim_scene_foundry.proposals import ObjectProposal
from real2sim_scene_foundry.video import prepare_rgb_video
from real2sim_scene_foundry.video_bg_only import create_video_bg_only_artifacts


class FakeProposalClient:
    def __init__(self):
        self.target_labels_seen = []

    def propose(self, image_path: Path, *, target_labels=None):
        self.target_labels_seen.append(target_labels)
        return [
            ObjectProposal(
                label="cup",
                object_id="cup",
                bbox_xyxy=(1, 1, 4, 4),
                point_xy=(2, 2),
                source="fake",
            )
        ]


class FakeMaskClient:
    def segment_box(self, image_path: Path, box_xyxy):
        mask = np.zeros((6, 8), dtype=bool)
        x0, y0, x1, y1 = box_xyxy
        mask[y0:y1, x0:x1] = True
        return mask

    def segment_text(self, image_path: Path, text_prompt: str):
        raise AssertionError("box prompt should be used when bbox exists")


class FakeInpaintClient:
    backend_name = "fake_inpaint"

    def inpaint(self, rgb, remove_mask):
        result = np.array(rgb, copy=True)
        result[remove_mask] = np.array([9, 8, 7], dtype=np.uint8)
        return result


def _write_frames(run_dir: Path) -> None:
    video_dir = run_dir / "video" / "frames"
    video_dir.mkdir(parents=True)
    for idx in (0, 10):
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        frame[:, :, 0] = idx
        Image.fromarray(frame).save(video_dir / f"frame_{idx:06d}.png")
    Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(run_dir / "video" / "reference.png")


def test_create_video_bg_only_artifacts_writes_masks_frames_and_status(tmp_path):
    run = tmp_path / "run"
    _write_frames(run)

    result = create_video_bg_only_artifacts(
        run_dir=run,
        proposal_client=FakeProposalClient(),
        mask_client=FakeMaskClient(),
        inpaint_client=FakeInpaintClient(),
        dilation_px=0,
    )

    assert result.status_path == run / "video" / "bg_only_status.json"
    assert (run / "video" / "bg_only" / "frames" / "frame_000000.png").is_file()
    assert (run / "video" / "bg_only" / "masks" / "frame_000010_mask.png").is_file()
    assert (run / "video" / "bg_only" / "proposals" / "frame_000010.json").is_file()
    bg = np.asarray(Image.open(run / "video" / "bg_only" / "frames" / "frame_000000.png").convert("RGB"))
    assert bg[2, 2].tolist() == [9, 8, 7]

    status = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["frame_count"] == 2
    assert status["target_labels"] == []
    assert status["observed_labels"] == ["cup"]
    assert status["outputs"]["frames_dir"] == "video/bg_only/frames"


def test_video_bg_only_does_not_lock_default_labels_after_first_frame(tmp_path):
    run = tmp_path / "run"
    _write_frames(run)
    proposal_client = FakeProposalClient()

    create_video_bg_only_artifacts(
        run_dir=run,
        proposal_client=proposal_client,
        mask_client=FakeMaskClient(),
        inpaint_client=FakeInpaintClient(),
        dilation_px=0,
    )

    assert proposal_client.target_labels_seen == [None, None]


def test_video_bg_only_clears_stale_outputs_before_rerun(tmp_path):
    run = tmp_path / "run"
    _write_frames(run)
    stale_frame = run / "video" / "bg_only" / "frames" / "stale.png"
    stale_mask = run / "video" / "bg_only" / "masks" / "stale_mask.png"
    stale_proposal = run / "video" / "bg_only" / "proposals" / "stale.json"
    stale_frame.parent.mkdir(parents=True)
    stale_mask.parent.mkdir(parents=True)
    stale_proposal.parent.mkdir(parents=True)
    stale_frame.write_bytes(b"old")
    stale_mask.write_bytes(b"old")
    stale_proposal.write_text("{}", encoding="utf-8")

    create_video_bg_only_artifacts(
        run_dir=run,
        proposal_client=FakeProposalClient(),
        mask_client=FakeMaskClient(),
        inpaint_client=FakeInpaintClient(),
        max_frames=1,
        dilation_px=0,
    )

    assert not stale_frame.exists()
    assert not stale_mask.exists()
    assert not stale_proposal.exists()
    assert len(list((run / "video" / "bg_only" / "frames").glob("*.png"))) == 1


def test_video_manifest_records_bg_only_status_after_artifact_creation(tmp_path):
    video = tmp_path / "input.avi"
    writer = __import__("cv2").VideoWriter(str(video), __import__("cv2").VideoWriter_fourcc(*"MJPG"), 5.0, (8, 6))
    assert writer.isOpened()
    writer.write(np.zeros((6, 8, 3), dtype=np.uint8))
    writer.release()
    run = tmp_path / "run"
    prepare_rgb_video(video_path=video, out_dir=run, frame_stride=1)

    create_video_bg_only_artifacts(
        run_dir=run,
        proposal_client=FakeProposalClient(),
        mask_client=FakeMaskClient(),
        inpaint_client=FakeInpaintClient(),
        dilation_px=0,
    )

    manifest = json.loads((run / "video" / "video_manifest.json").read_text(encoding="utf-8"))
    assert manifest["foreground_removal"]["status"] == "completed"
    assert manifest["foreground_removal"]["bg_only_status"] == "video/bg_only_status.json"
    assert manifest["background_3dgs"]["status"] == "ready_for_training"
