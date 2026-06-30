"""Command line interface for real2sim_scene_foundry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .camera import CameraIntrinsics
from .defaults import SAM3D_PROCESS_URL, SAM3_SEGMENT_URL
from .pipeline import run_extract, run_reconstruct_align, run_smoke_reconstruction
from .proposals import ObjectProposal, QwenProposalClient, load_object_proposals


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "smoke":
        camera, baseline_m = CameraIntrinsics.from_calibration_json(args.calib, camera_name=args.camera_name)
        depth_client = None
        if args.s2m2_url:
            from .clients import S2M2Client

            depth_client = S2M2Client(args.s2m2_url)
        result = run_smoke_reconstruction(
            left_image=args.left,
            right_image=args.right,
            camera=camera,
            baseline_m=baseline_m,
            out_dir=args.out,
            label=args.label,
            mask_path=args.mask,
            depth_client=depth_client,
            mock_depth_m=1.0 if args.mock_depth else None,
        )
        print(f"wrote {result.manifest_path}")
        print(f"wrote {result.usd_path}")
        print(f"wrote {result.qa_report_path}")
        return 0
    if args.command == "extract":
        camera, baseline_m = CameraIntrinsics.from_calibration_json(args.calib, camera_name=args.camera_name)
        proposals = _load_proposals_from_args(args, args.left)
        depth_client = _depth_client_from_args(args)
        sam3_client = BoxMaskClient(camera.width, camera.height) if args.mock_mask_from_bbox else _sam3_client_from_args(args)
        result = run_extract(
            left_image=args.left,
            right_image=args.right,
            camera=camera,
            baseline_m=baseline_m,
            out_dir=args.out,
            proposals=proposals,
            depth_client=depth_client,
            sam3_client=sam3_client,
            mock_depth_m=1.0 if args.mock_depth else None,
        )
        print(f"wrote {result.manifest_path}")
        return 0
    if args.command == "run":
        camera, baseline_m = CameraIntrinsics.from_calibration_json(args.calib, camera_name=args.camera_name)
        proposals = _load_proposals_from_args(args, args.left)
        extract = run_extract(
            left_image=args.left,
            right_image=args.right,
            camera=camera,
            baseline_m=baseline_m,
            out_dir=args.out,
            proposals=proposals,
            depth_client=_depth_client_from_args(args),
            sam3_client=BoxMaskClient(camera.width, camera.height) if args.mock_mask_from_bbox else _sam3_client_from_args(args),
            mock_depth_m=1.0 if args.mock_depth else None,
        )
        result = run_reconstruct_align(
            extract_dir=extract.out_dir,
            camera=camera,
            sam3d_client=_sam3d_client_from_args(args),
        )
        print(f"wrote {extract.manifest_path}")
        print(f"wrote {result.manifest_path}")
        print(f"wrote {result.usd_path}")
        print(f"wrote {result.qa_report_path}")
        return 0
    if args.command in {"reconstruct", "align"}:
        camera, _baseline_m = CameraIntrinsics.from_calibration_json(args.calib, camera_name=args.camera_name)
        result = run_reconstruct_align(
            extract_dir=args.extract_dir,
            camera=camera,
            sam3d_client=_sam3d_client_from_args(args),
        )
        print(f"wrote {result.manifest_path}")
        print(f"wrote {result.usd_path}")
        print(f"wrote {result.qa_report_path}")
        return 0
    parser.error(f"{args.command} is scaffolded but not implemented in V1")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsf")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_smoke_parser(subparsers, "smoke")
    _add_extract_parser(subparsers, "extract")
    _add_extract_parser(subparsers, "run")
    _add_reconstruct_parser(subparsers, "reconstruct")
    _add_reconstruct_parser(subparsers, "align")
    for name in ("export", "render"):
        subparsers.add_parser(name)
    return parser


def _add_smoke_parser(subparsers: argparse._SubParsersAction, name: str) -> None:
    parser = subparsers.add_parser(name)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--camera-name", default=None)
    parser.add_argument("--mask", default=None, type=Path)
    parser.add_argument("--label", default="object")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mock-depth", action="store_true")
    parser.add_argument("--s2m2-url", action="append", default=None)
    parser.set_defaults(s2m2_url=None)


def _add_extract_parser(subparsers: argparse._SubParsersAction, name: str) -> None:
    parser = subparsers.add_parser(name)
    _add_stereo_args(parser)
    _add_proposal_args(parser)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mock-depth", action="store_true")
    parser.add_argument("--mock-mask-from-bbox", action="store_true")
    parser.add_argument("--s2m2-url", action="append", default=None)
    parser.add_argument("--sam3-url", default=SAM3_SEGMENT_URL)
    parser.add_argument("--sam3d-url", default=SAM3D_PROCESS_URL)
    parser.set_defaults(s2m2_url=None)


def _add_reconstruct_parser(subparsers: argparse._SubParsersAction, name: str) -> None:
    parser = subparsers.add_parser(name)
    parser.add_argument("--extract-dir", required=True, type=Path)
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--camera-name", default=None)
    parser.add_argument("--sam3d-url", default=SAM3D_PROCESS_URL)


def _add_stereo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--camera-name", default=None)


def _add_proposal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--objects-yaml", type=Path, default=None)
    parser.add_argument("--label", default="object")
    parser.add_argument("--bbox", default=None, help="Fallback bbox as x0,y0,x1,y1")
    parser.add_argument("--point", default=None, help="Fallback point as x,y")
    parser.add_argument("--no-qwen", action="store_true", help="Disable Qwen proposals; requires --bbox or --objects-yaml")
    parser.add_argument("--target-label", action="append", default=None)
    parser.add_argument("--qwen-base-url", default=None)
    parser.add_argument("--qwen-api-key", default=None)
    parser.add_argument("--qwen-model", default=None)


def _load_proposals_from_args(args: argparse.Namespace, image_path: Path) -> list[ObjectProposal]:
    image_size = Image.open(image_path).size
    if args.objects_yaml is not None:
        return load_object_proposals(args.objects_yaml, image_size=image_size)
    if args.bbox or args.point:
        return [
            ObjectProposal(
                label=args.label,
                object_id=args.label,
                bbox_xyxy=_parse_int_tuple(args.bbox, 4) if args.bbox else None,
                point_xy=_parse_int_tuple(args.point, 2) if args.point else None,
                source="cli",
            )
        ]
    if not args.no_qwen:
        target_labels = args.target_label or ([] if args.label == "object" else [args.label])
        client = QwenProposalClient(
            base_url=args.qwen_base_url,
            api_key=args.qwen_api_key,
            model=args.qwen_model,
        )
        proposals = client.propose(image_path, target_labels=target_labels)
        if proposals:
            return proposals
        fallback_labels = _proposal_label_fallbacks(target_labels)
        for labels in fallback_labels:
            proposals = client.propose(image_path, target_labels=labels)
            if proposals:
                return proposals
        raise ValueError(f"Qwen returned no object proposals for labels {target_labels!r}")
    raise ValueError("--no-qwen requires --objects-yaml or --bbox/--point manual proposals")


def _parse_int_tuple(value: str, expected: int):
    parts = [int(round(float(part.strip()))) for part in value.split(",")]
    if len(parts) != expected:
        raise ValueError(f"expected {expected} comma-separated numbers, got {value}")
    return tuple(parts)


def _proposal_label_fallbacks(target_labels: list[str]) -> list[list[str]]:
    fallbacks: list[list[str]] = []
    for label in target_labels:
        words = [word for word in label.strip().split() if word]
        if len(words) > 1:
            fallbacks.append([words[-1]])
    if target_labels:
        fallbacks.append([])
    deduped: list[list[str]] = []
    for item in fallbacks:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _depth_client_from_args(args: argparse.Namespace):
    if args.s2m2_url:
        from .clients import S2M2Client

        return S2M2Client(args.s2m2_url)
    return None


def _sam3_client_from_args(args: argparse.Namespace):
    from .clients import SAM3Client

    return SAM3Client(args.sam3_url)


def _sam3d_client_from_args(args: argparse.Namespace):
    from .clients import SAM3DClient

    return SAM3DClient(args.sam3d_url)


class BoxMaskClient:
    def __init__(self, width: int, height: int) -> None:
        self._width = int(width)
        self._height = int(height)

    def segment_text(self, image_path: str | Path, text_prompt: str) -> np.ndarray:
        raise ValueError("--mock-mask-from-bbox requires each proposal to include bbox_xyxy")

    def segment_box(self, image_path: str | Path, box_xyxy: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = box_xyxy
        mask = np.zeros((self._height, self._width), dtype=bool)
        mask[max(0, y0) : min(self._height, y1), max(0, x0) : min(self._width, x1)] = True
        return mask


if __name__ == "__main__":
    raise SystemExit(main())
