"""Command line interface for real2sim_scene_foundry."""

from __future__ import annotations

import argparse
from pathlib import Path

from .camera import CameraIntrinsics
from .defaults import S2M2_URLS
from .pipeline import run_smoke_reconstruction


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"smoke", "run"}:
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
    parser.error(f"{args.command} is scaffolded but not implemented in V1; use 'smoke' or 'run'")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsf")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "run"):
        _add_smoke_parser(subparsers, name)
    for name in ("extract", "reconstruct", "align", "export", "render"):
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


if __name__ == "__main__":
    raise SystemExit(main())
