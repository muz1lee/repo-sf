"""Command line interface for real2sim_scene_foundry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .background import HTTPInpaintClient
from .camera import CameraIntrinsics
from .bg_table import build_table_collision as build_bg_table_collision
from .bg_table import fit_tabletop_multiframe_from_semantic_masks
from .bg_table import fit_tabletop_from_semantic_mask
from .bg_table import qa_bg_table
from .bg_table import segment_tabletop_semantic_mask
from .bg_table_sim import export_bg_table_physics_mvp, export_bg_table_physics_smoke, export_phone_bg_table_mvp
from .composite_viewer import export_composite_viewer, qa_viewer, serve_composite_viewer
from .defaults import SAM3D_PROCESS_URL, SAM3_SEGMENT_URL
from .interactive import export_interactive_scene
from .nerfstudio_viewer import export_nerfstudio_viewer
from .pipeline import run_extract, run_reconstruct_align, run_smoke_reconstruction
from .phone_capture import export_nerfstudio_from_phone_capture, import_phone_capture, validate_phone_capture
from .phone_sim_alignment import align_phone_sim_world
from .pose_refinement import (
    apply_visual_orientation_overrides,
    qa_object_alignment,
    refine_pose_rgbd,
    refine_visual_pose_to_masks,
    snap_object_poses_to_support,
)
from .proposals import ObjectProposal, QwenProposalClient, load_object_proposals
from .record3d_capture import convert_record3d_export
from .runtime_viewer import export_runtime_viewer
from .support_plane import estimate_and_apply_support_plane
from .video import prepare_rgb_video
from .video_scene import run_video_reference_scene


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
            background_inpaint_client=_background_inpaint_client_from_args(args),
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
            background_inpaint_client=_background_inpaint_client_from_args(args),
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
    if args.command == "interactive":
        result = export_interactive_scene(args.run_dir, settle_steps=args.settle_steps)
        print(f"wrote {result.script_path}")
        print(f"wrote {result.report_path}")
        return 0
    if args.command == "support-plane":
        report = estimate_and_apply_support_plane(args.run_dir, force=args.force)
        print(f"support_plane status={report['status']} source={report['source_backend']}")
        print(f"wrote {Path(args.run_dir) / 'scene_manifest.json'}")
        return 0
    if args.command == "refine-pose":
        if args.orientation_override:
            result = apply_visual_orientation_overrides(
                args.run_dir,
                _parse_orientation_overrides(args.orientation_override),
                tolerance_m=args.tolerance_m,
            )
        elif args.visual_bbox:
            result = refine_visual_pose_to_masks(args.run_dir, tolerance_m=args.tolerance_m)
        else:
            result = snap_object_poses_to_support(args.run_dir, tolerance_m=args.tolerance_m)
        print(f"pose_refinement status={result.report['status']}")
        print(f"wrote {result.report_path}")
        return 0
    if args.command == "refine-pose-rgbd":
        result = refine_pose_rgbd(args.run_dir, frames=args.frames)
        print(f"pose_rgbd status={result.report['status']}")
        print(f"wrote {result.report_path}")
        return 0
    if args.command == "qa-object-alignment":
        report = qa_object_alignment(args.run_dir)
        print(f"wrote {Path(args.run_dir) / str(report.get('report_path', 'qa/object_pose_alignment_report.json'))}")
        print(f"status={report.get('status')}")
        return 0
    if args.command == "composite-viewer":
        if args.mode == "scene":
            result = export_composite_viewer(args.run_dir, backend=args.backend, show_settled=args.show_settled)
        else:
            result = export_composite_viewer(args.run_dir, backend=args.backend, show_settled=args.show_settled, mode=args.mode)
        print(f"wrote {result.index_path}")
        print(f"wrote {result.config_path}")
        url = f"http://{args.host}:{int(args.port)}{result.url_path}"
        print(f"open {url}")
        if args.export_only:
            return 0
        if args.mode == "scene":
            serve_composite_viewer(args.run_dir, host=args.host, port=args.port, backend=args.backend, show_settled=args.show_settled)
        else:
            serve_composite_viewer(
                args.run_dir,
                host=args.host,
                port=args.port,
                backend=args.backend,
                show_settled=args.show_settled,
                mode=args.mode,
            )
        return 0
    if args.command == "qa-viewer":
        report = qa_viewer(args.run_dir)
        print(f"wrote {Path(args.run_dir) / str(report.get('report_path', 'qa/viewer_audit.json'))}")
        print(f"status={report.get('status')}")
        return 0
    if args.command == "runtime-viewer":
        result = export_runtime_viewer(args.run_dir, backend=args.backend)
        print(f"runtime_viewer status={result.report['status']} backend={result.report['backend']}")
        print(f"wrote {result.index_path}")
        print(f"wrote {result.runtime_manifest_path}")
        if result.script_path:
            print(f"wrote {result.script_path}")
            print(f"run: /mnt/workspace/wenqian/knowin-world/.venv/bin/python {result.script_path} --run-dir {args.run_dir} --host {args.host} --port {int(args.port)}")
        if args.serve:
            if result.script_path is None:
                raise RuntimeError(f"runtime viewer backend {args.backend!r} is unavailable: {result.report.get('blocked_reason')}")
            import subprocess

            return subprocess.call(
                [
                    "/mnt/workspace/wenqian/knowin-world/.venv/bin/python",
                    str(result.script_path),
                    "--run-dir",
                    str(args.run_dir),
                    "--host",
                    str(args.host),
                    "--port",
                    str(int(args.port)),
                ]
            )
        return 0
    if args.command == "genesis-settle":
        report = apply_genesis_settle_writeback(args.run_dir, writeback=args.writeback)
        print(f"wrote {Path(args.run_dir) / str(report.get('report_path', 'qa/genesis_settle_writeback_report.json'))}")
        print(f"status={report.get('status')}")
        return 0
    if args.command == "build-collision":
        report = build_collision_assets(
            args.run_dir,
            backend=args.backend,
            strict_provenance=args.strict_provenance,
        )
        print(f"wrote {Path(args.run_dir) / str(report.get('report_path', 'qa/collision_rebuild_report.json'))}")
        print(f"status={report.get('status')}")
        return 0
    if args.command == "qa-physics":
        report = qa_physics(args.run_dir)
        print(f"wrote {Path(args.run_dir) / str(report.get('report_path', 'qa/physics_property_report.json'))}")
        print(f"status={report.get('status')}")
        return 0
    if args.command == "video-prep":
        result = prepare_rgb_video(
            video_path=args.video,
            out_dir=args.out,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            reference_frame_index=args.reference_frame_index,
        )
        print(f"wrote {result.manifest_path}")
        print(f"wrote {result.reference_path}")
        return 0
    if args.command == "video-scene":
        result = run_video_reference_scene(
            run_dir=args.run_dir,
            target_labels=args.target_label,
            max_objects=args.max_objects,
            proposal_client=QwenProposalClient(
                base_url=args.qwen_base_url,
                api_key=args.qwen_api_key,
                model=args.qwen_model,
            ),
            sam3_client=_sam3_client_from_args(args),
            sam3d_client=_sam3d_client_from_args(args),
            settle_steps=args.settle_steps,
        )
        print(f"wrote {result.manifest_path}")
        print(f"wrote {result.usd_path}")
        print(f"wrote {result.qa_report_path}")
        print(f"wrote {result.interactive_script_path}")
        print(f"wrote {result.interaction_report_path}")
        return 0
    if args.command == "validate-phone-capture":
        report = validate_phone_capture(args.capture_dir)
        print(f"wrote {Path(args.capture_dir) / 'capture_contract.json'}")
        print(f"status={report.get('status')}")
        return 0
    if args.command == "import-phone-capture":
        report = import_phone_capture(args.capture_dir, args.run_dir)
        print(f"wrote {Path(args.run_dir) / 'capture_contract.json'}")
        print(f"status={report.get('status')}")
        return 0
    if args.command == "align-phone-sim-world":
        report = align_phone_sim_world(args.run_dir, force=args.force)
        print(f"wrote {Path(args.run_dir) / 'background' / 'phone_sim_alignment.json'}")
        print(f"status={report.get('status')}")
        return 0
    if args.command == "import-record3d":
        report = convert_record3d_export(
            args.record3d_dir,
            args.out,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            depth_channel=args.depth_channel,
        )
        print(f"wrote {Path(args.out) / 'record3d_import_report.json'}")
        print(f"wrote {Path(args.out) / 'capture_contract.json'}")
        print(f"status={report.get('status')}")
        return 0
    if args.command == "export-nerfstudio-from-phone-capture":
        report = export_nerfstudio_from_phone_capture(args.run_dir, pose_world=args.pose_world)
        output_path = report.get("transforms_path") or "video/nerfstudio_phone/export_report.json"
        print(f"wrote {Path(args.run_dir) / str(output_path)}")
        print(f"status={report.get('status')}")
        return 0
    if args.command == "nerfstudio-viewer":
        result = export_nerfstudio_viewer(
            args.run_dir,
            config_path=args.config,
            host=args.host,
            port=args.port,
            ns_viewer_bin=args.ns_viewer_bin,
        )
        print(f"wrote {result.report_path}")
        print(f"wrote {result.launch_script_path}")
        print(f"status={result.report.get('status')}")
        print(f"open {result.url}")
        if args.export_only:
            return 0
        if result.report.get("status") != "ready":
            return 4
        import subprocess

        return subprocess.call(result.command)
    if args.command == "export-manifest":
        manifest_path = write_sim_export_manifest(args.run_dir)
        print(f"wrote {manifest_path}")
        return 0
    if args.command == "isaac-worker-bundle":
        bundle = build_isaac_worker_bundle(
            args.run_dir,
            worker_run_dir=args.worker_run_dir,
            project_dir=args.project_dir,
            worker_ssh=args.worker_ssh,
            worker_python=args.worker_python,
        )
        print(f"wrote {bundle.manifest_path}")
        print(f"wrote {bundle.file_list_path}")
        for name, command in bundle.commands.items():
            print(f"{name}: {command}")
        return 0
    if args.command == "isaac-worker-report":
        result = ingest_isaac_worker_report(args.run_dir, args.report)
        print(f"wrote {result.report_path}")
        print(f"status={result.report.get('status')}")
        return 0
    if args.command == "render-3dgs-background":
        kwargs = {"split": args.split, "camera_idx": args.camera_idx}
        if args.renderer_executable:
            kwargs["renderer_executable"] = args.renderer_executable
        result = render_external_3dgs_background(args.run_dir, **kwargs)
        print(f"wrote {result.report_path}")
        print(f"status={result.report.get('status')}")
        return 0
    if args.command == "register-3dgs":
        result = register_3dgs_background(args.run_dir, method=args.method, write=args.write)
        print(f"wrote {result.path}")
        print(f"status={result.data.get('status')}")
        return 0
    if args.command == "qa-background-registration":
        report = qa_background_registration(args.run_dir, heldout_frames=args.heldout_frames)
        print(f"wrote {Path(args.run_dir) / 'qa' / 'background_registration_report.json'}")
        print(f"status={report.get('status')}")
        return 0
    if args.command == "segment-tabletop-mask":
        result = segment_tabletop_semantic_mask(
            args.run_dir,
            sam3_client=_sam3_client_from_args(args),
            prompt=args.prompt,
            bbox_xyxy=_parse_bbox(args.bbox) if args.bbox else None,
            frame_index=args.frame_index,
        )
        print(f"wrote {result.report_path}")
        if result.report.get("status") == "passed" and result.mask_path.is_file():
            print(f"wrote {result.mask_path}")
        print(f"status={result.report.get('status')}")
        return 0
    if args.command == "fit-tabletop-from-mask":
        result = fit_tabletop_from_semantic_mask(
            args.run_dir,
            mask=args.mask,
            frame_index=args.frame_index,
            hull=args.hull,
            plane_distance_threshold_m=args.plane_distance_threshold_m,
            write=args.write,
        )
        print(f"wrote {result.report_path}")
        if args.write and result.report.get("status") == "passed":
            for artifact_path in (result.semantic_mask_path, result.refined_mask_path, result.polygon_path):
                if artifact_path.is_file():
                    print(f"wrote {artifact_path}")
        print(f"status={result.report.get('status')}")
        return 0
    if args.command == "fit-tabletop-multiframe":
        result = fit_tabletop_multiframe_from_semantic_masks(
            args.run_dir,
            masks=args.masks,
            frames=args.frames,
            hull=args.hull,
            plane_distance_threshold_m=args.plane_distance_threshold_m,
            confidence_coverage_threshold=args.confidence_coverage_threshold,
            min_frame_count=args.min_frame_count,
            min_baseline_m=args.min_baseline_m,
            max_points_per_frame=args.max_points_per_frame,
            boundary_mode=args.boundary_mode,
            boundary_min_frame_support=args.boundary_min_frame_support,
            write=args.write,
        )
        print(f"wrote {result.report_path}")
        if args.write and result.report.get("status") == "passed":
            for artifact_path in (
                result.semantic_mask_path,
                result.refined_mask_path,
                result.plane_path,
                result.fused_points_path,
                result.polygon_path,
            ):
                if artifact_path.is_file():
                    print(f"wrote {artifact_path}")
        print(f"status={result.report.get('status')}")
        return 0
    if args.command == "build-table-collision":
        result = build_bg_table_collision(args.run_dir, source=args.source, write=args.write, thickness_m=args.height)
        print(f"wrote {result.mesh_path}")
        print(f"wrote {result.usd_path}")
        print(f"wrote {result.report_path}")
        print(f"status={result.report.get('status')}")
        return 0
    if args.command == "qa-bg-table":
        report = qa_bg_table(args.run_dir, frames=args.frames, write_overlays=args.write_overlays)
        print(f"wrote {Path(args.run_dir) / 'qa' / 'bg_table_report.json'}")
        print(f"status={report.get('status')}")
        return 0
    if args.command == "export-bg-table-mvp":
        result = export_bg_table_physics_mvp(
            args.run_dir,
            cube_size_m=args.cube_size_m,
            drop_height_m=args.drop_height_m,
            settle_steps=args.settle_steps,
        )
        print(f"wrote {result.config_path}")
        print(f"wrote {result.genesis_script_path}")
        print(f"wrote {result.usd_path}")
        print(f"wrote {result.report_path}")
        print(f"status={result.report.get('status')}")
        return 0
    if args.command == "export-bg-table-sim-smoke":
        result = export_bg_table_physics_smoke(
            args.run_dir,
            cube_size_m=args.cube_size_m,
            drop_height_m=args.drop_height_m,
            settle_steps=args.settle_steps,
        )
        print(f"wrote {result.config_path}")
        print(f"wrote {result.genesis_script_path}")
        print(f"wrote {result.usd_path}")
        print(f"wrote {result.report_path}")
        print(f"status={result.report.get('status')}")
        return 0
    if args.command == "export-phone-bg-table-mvp":
        result = export_phone_bg_table_mvp(
            args.run_dir,
            cube_size_m=args.cube_size_m,
            drop_height_m=args.drop_height_m,
            settle_steps=args.settle_steps,
        )
        print(f"wrote {result.scene_manifest_path}")
        print(f"wrote {result.sim_export_manifest_path}")
        print(f"wrote {result.usd_path}")
        print(f"wrote {result.genesis_script_path}")
        print(f"wrote {result.settle_report_path}")
        print(f"wrote {result.report_path}")
        print(f"status={result.report.get('status')}")
        return 0
    if args.command == "export-sim":
        artifacts = export_sim(args.run_dir, backends=args.backend)
        for name, path in artifacts.items():
            print(f"wrote {name}: {path}")
        return 0
    if args.command == "qa-sim":
        report_path = run_export_qa(args.run_dir, strict_claims=args.strict_claims)
        print(f"wrote {report_path}")
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
    _add_interactive_parser(subparsers, "interactive")
    _add_support_plane_parser(subparsers)
    _add_refine_pose_parser(subparsers)
    _add_refine_pose_rgbd_parser(subparsers)
    _add_qa_object_alignment_parser(subparsers)
    _add_composite_viewer_parser(subparsers)
    _add_qa_viewer_parser(subparsers)
    _add_runtime_viewer_parser(subparsers)
    _add_genesis_settle_parser(subparsers)
    _add_build_collision_parser(subparsers)
    _add_qa_physics_parser(subparsers)
    _add_video_prep_parser(subparsers, "video-prep")
    _add_video_scene_parser(subparsers, "video-scene")
    _add_validate_phone_capture_parser(subparsers)
    _add_import_phone_capture_parser(subparsers)
    _add_align_phone_sim_world_parser(subparsers)
    _add_import_record3d_parser(subparsers)
    _add_export_nerfstudio_phone_parser(subparsers)
    _add_nerfstudio_viewer_parser(subparsers)
    _add_export_manifest_parser(subparsers)
    _add_export_sim_parser(subparsers)
    _add_qa_sim_parser(subparsers)
    _add_isaac_worker_bundle_parser(subparsers)
    _add_isaac_worker_report_parser(subparsers)
    _add_render_3dgs_background_parser(subparsers)
    _add_register_3dgs_parser(subparsers)
    _add_qa_background_registration_parser(subparsers)
    _add_segment_tabletop_mask_parser(subparsers)
    _add_fit_tabletop_from_mask_parser(subparsers)
    _add_fit_tabletop_multiframe_parser(subparsers)
    _add_build_table_collision_parser(subparsers)
    _add_qa_bg_table_parser(subparsers)
    _add_export_bg_table_mvp_parser(subparsers)
    _add_export_bg_table_sim_smoke_parser(subparsers)
    _add_export_phone_bg_table_mvp_parser(subparsers)
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
    parser.add_argument("--sam3d-api", choices=("process", "mesh-job"), default="process")
    parser.add_argument("--inpaint-url", default=None)
    parser.set_defaults(s2m2_url=None)


def _add_reconstruct_parser(subparsers: argparse._SubParsersAction, name: str) -> None:
    parser = subparsers.add_parser(name)
    parser.add_argument("--extract-dir", required=True, type=Path)
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--camera-name", default=None)
    parser.add_argument("--sam3d-url", default=SAM3D_PROCESS_URL)
    parser.add_argument("--sam3d-api", choices=("process", "mesh-job"), default="process")


def _add_interactive_parser(subparsers: argparse._SubParsersAction, name: str) -> None:
    parser = subparsers.add_parser(name)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--settle-steps", type=int, default=100)


def _add_support_plane_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("support-plane")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Recompute and reapply even if support_plane already exists")


def _add_refine_pose_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("refine-pose")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--tolerance-m", type=float, default=0.005)
    parser.add_argument("--visual-bbox", action="store_true", help="Refine visual asset scale from reference-camera mask bboxes")
    parser.add_argument(
        "--orientation-override",
        action="append",
        default=None,
        help="Manual orientation override as object_id:flip_local_x_180, object_id:flip_local_y_180, or object_id:flip_local_z_180",
    )


def _add_refine_pose_rgbd_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("refine-pose-rgbd")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--frames", default="reference")


def _add_qa_object_alignment_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("qa-object-alignment")
    parser.add_argument("--run-dir", required=True, type=Path)


def _add_composite_viewer_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("composite-viewer")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7010)
    parser.add_argument("--backend", choices=("external-sidecar", "browser-3dgs"), default="external-sidecar")
    parser.add_argument("--mode", choices=("scene", "bg-table"), default="scene")
    parser.add_argument("--show-settled", action="store_true")
    parser.add_argument("--export-only", action="store_true")


def _add_qa_viewer_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("qa-viewer")
    parser.add_argument("--run-dir", required=True, type=Path)


def _add_runtime_viewer_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("runtime-viewer")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--backend", choices=("genesis", "isaac"), default="genesis")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7030)
    parser.add_argument("--serve", action="store_true", help="Start the runtime server after exporting")


def _add_genesis_settle_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("genesis-settle")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--writeback", action="store_true")


def _add_build_collision_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("build-collision")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--backend", choices=("convex-hull", "coacd"), default="convex-hull")
    parser.add_argument("--strict-provenance", action="store_true")


def _add_qa_physics_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("qa-physics")
    parser.add_argument("--run-dir", required=True, type=Path)


def _add_video_prep_parser(subparsers: argparse._SubParsersAction, name: str) -> None:
    parser = subparsers.add_parser(name)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--reference-frame-index", type=int, default=0)


def _add_video_scene_parser(subparsers: argparse._SubParsersAction, name: str) -> None:
    parser = subparsers.add_parser(name)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--target-label", action="append", default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--sam3-url", default=SAM3_SEGMENT_URL)
    parser.add_argument("--sam3d-url", default=SAM3D_PROCESS_URL)
    parser.add_argument("--sam3d-api", choices=("process", "mesh-job"), default="process")
    parser.add_argument("--qwen-base-url", default=None)
    parser.add_argument("--qwen-api-key", default=None)
    parser.add_argument("--qwen-model", default=None)


def _add_validate_phone_capture_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("validate-phone-capture")
    parser.add_argument("--capture-dir", required=True, type=Path)


def _add_import_phone_capture_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("import-phone-capture")
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)


def _add_align_phone_sim_world_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("align-phone-sim-world")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")


def _add_import_record3d_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("import-record3d")
    parser.add_argument("--record3d-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--depth-channel", type=int, default=2)


def _add_export_nerfstudio_phone_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("export-nerfstudio-from-phone-capture")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pose-world", choices=("arkit", "sim"), default="arkit")


def _add_nerfstudio_viewer_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("nerfstudio-viewer")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", default=None, type=Path, help="Nerfstudio config.yml to load; resolved from run QA reports when omitted")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7014)
    parser.add_argument("--ns-viewer-bin", default=None, type=Path)
    parser.add_argument("--export-only", action="store_true", help="Write report/launcher without starting ns-viewer")


def _add_export_manifest_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("export-manifest")
    parser.add_argument("--run-dir", required=True, type=Path)


def _add_export_sim_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("export-sim")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--backend", action="append", choices=("genesis", "usd", "isaac"), required=True)


def _add_qa_sim_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("qa-sim")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--strict-claims", dest="strict_claims", action="store_true", default=True)
    parser.add_argument("--no-strict-claims", dest="strict_claims", action="store_false")


def _add_isaac_worker_bundle_parser(subparsers: argparse._SubParsersAction) -> None:
    from .defaults import ISAAC_WORKER_PYTHON, ISAAC_WORKER_SSH

    parser = subparsers.add_parser("isaac-worker-bundle")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--worker-run-dir", default=None)
    parser.add_argument("--project-dir", default="/mnt/workspace/wenqian/real2sim_scene_foundry")
    parser.add_argument("--worker-ssh", default=ISAAC_WORKER_SSH)
    parser.add_argument("--worker-python", default=ISAAC_WORKER_PYTHON)


def _add_isaac_worker_report_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("isaac-worker-report")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)


def _add_render_3dgs_background_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("render-3dgs-background")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--renderer-executable", default=None, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--camera-idx", type=int, default=0)


def _add_register_3dgs_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("register-3dgs")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--method", choices=("camera-sim3", "known-phone-sim-world"), default="camera-sim3")
    parser.add_argument("--write", action="store_true")


def _add_qa_background_registration_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("qa-background-registration")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--heldout-frames", type=int, default=16)


def _add_segment_tabletop_mask_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("segment-tabletop-mask")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--prompt", default="tabletop / coffee table top")
    parser.add_argument("--bbox", default=None, help="Optional bbox prompt as x0,y0,x1,y1")
    parser.add_argument("--sam3-url", default=SAM3_SEGMENT_URL)


def _add_fit_tabletop_from_mask_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("fit-tabletop-from-mask")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--mask", default="background/tabletop_semantic_mask.png", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--hull", choices=("convex-hull", "clipped-convex-hull", "rotated-rectangle"), default="convex-hull")
    parser.add_argument("--plane-distance-threshold-m", type=float, default=0.02)
    parser.add_argument("--write", action="store_true")


def _add_fit_tabletop_multiframe_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("fit-tabletop-multiframe")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--masks", default="background/tabletop_semantic_masks")
    parser.add_argument("--frames", default="0,20,40,60")
    parser.add_argument("--hull", choices=("convex-hull", "clipped-convex-hull", "rotated-rectangle"), default="convex-hull")
    parser.add_argument("--plane-distance-threshold-m", type=float, default=0.02)
    parser.add_argument("--confidence-coverage-threshold", type=float, default=0.30)
    parser.add_argument("--min-frame-count", type=int, default=2)
    parser.add_argument("--min-baseline-m", type=float, default=0.02)
    parser.add_argument("--max-points-per-frame", type=int, default=50000)
    parser.add_argument("--boundary-mode", choices=("consensus-hull", "union-hull"), default="consensus-hull")
    parser.add_argument("--boundary-min-frame-support", type=int, default=2)
    parser.add_argument("--write", action="store_true")


def _add_build_table_collision_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("build-table-collision")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source", default="background/table_polygon_world.json")
    parser.add_argument("--height", type=float, default=0.03)
    parser.add_argument("--write", action="store_true")


def _add_qa_bg_table_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("qa-bg-table")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--frames", default="0,20,40,60")
    parser.add_argument("--write-overlays", action="store_true")


def _add_export_bg_table_mvp_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("export-bg-table-mvp")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cube-size-m", type=float, default=0.08)
    parser.add_argument("--drop-height-m", type=float, default=0.12)
    parser.add_argument("--settle-steps", type=int, default=120)


def _add_export_bg_table_sim_smoke_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("export-bg-table-sim-smoke")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cube-size-m", type=float, default=0.08)
    parser.add_argument("--drop-height-m", type=float, default=0.12)
    parser.add_argument("--settle-steps", type=int, default=120)


def _add_export_phone_bg_table_mvp_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("export-phone-bg-table-mvp")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cube-size-m", type=float, default=0.08)
    parser.add_argument("--drop-height-m", type=float, default=0.12)
    parser.add_argument("--settle-steps", type=int, default=120)


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


def _parse_bbox(value: str) -> tuple[int, int, int, int]:
    parts = [int(round(float(part.strip()))) for part in str(value).split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("bbox must be x0,y0,x1,y1")
    return (parts[0], parts[1], parts[2], parts[3])


def _parse_orientation_overrides(values: list[str]) -> dict[str, str]:
    overrides = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"orientation override must be object_id:operation, got {value!r}")
        object_id, operation = value.split(":", 1)
        object_id = object_id.strip()
        operation = operation.strip()
        if not object_id or not operation:
            raise ValueError(f"orientation override must be object_id:operation, got {value!r}")
        overrides[object_id] = operation
    return overrides


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
    from .clients import SAM3DClient, SAM3DMeshJobClient

    sam3d_url = str(args.sam3d_url)
    api_mode = getattr(args, "sam3d_api", "process")
    if api_mode == "mesh-job" or sam3d_url.rstrip("/").endswith("/api/jobs"):
        return SAM3DMeshJobClient(sam3d_url)
    return SAM3DClient(sam3d_url)


def _background_inpaint_client_from_args(args: argparse.Namespace):
    if getattr(args, "inpaint_url", None):
        return HTTPInpaintClient(args.inpaint_url)
    return None


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


def build_sim_export_manifest(run_dir: str | Path) -> Path:
    from .sim_export_manifest import write_sim_export_manifest as impl

    return impl(run_dir)


def write_sim_export_manifest(run_dir: str | Path) -> Path:
    from .sim_export_manifest import write_sim_export_manifest as impl

    return impl(run_dir)


def export_sim(run_dir: str | Path, *, backends: list[str]) -> dict[str, Path]:
    from .sim_export_manifest import export_sim as impl

    return impl(run_dir, backends=backends)


def run_export_qa(run_dir: str | Path, *, strict_claims: bool = False) -> Path:
    from .export_qa import run_export_qa as impl

    return impl(run_dir, strict_claims=strict_claims)


def build_isaac_worker_bundle(run_dir: str | Path, **kwargs):
    from .isaac_worker_bridge import build_isaac_worker_bundle as impl

    return impl(run_dir, **kwargs)


def ingest_isaac_worker_report(run_dir: str | Path, report_path: str | Path):
    from .isaac_worker_bridge import ingest_isaac_worker_report as impl

    return impl(run_dir, report_path)


def render_external_3dgs_background(run_dir: str | Path, **kwargs):
    from .background_3dgs_render import render_external_3dgs_background as impl

    return impl(run_dir, **kwargs)


def register_3dgs_background(run_dir: str | Path, *, method: str = "camera-sim3", write: bool = True):
    from .background_registration import register_3dgs_background as impl

    return impl(run_dir, method=method, write=write)


def qa_background_registration(run_dir: str | Path, *, heldout_frames: int = 16) -> dict:
    from .background_registration import qa_background_registration as impl

    return impl(run_dir, heldout_frames=heldout_frames)


def apply_genesis_settle_writeback(run_dir: str | Path, *, writeback: bool = False) -> dict:
    from .genesis_export import apply_genesis_settle_writeback as impl

    return impl(run_dir, writeback=writeback)


def build_collision_assets(
    run_dir: str | Path,
    *,
    backend: str = "convex-hull",
    strict_provenance: bool = False,
) -> dict:
    from .collision_assets import ensure_collision_assets as impl

    return impl(run_dir, backend=backend, strict_provenance=strict_provenance)


def qa_physics(run_dir: str | Path) -> dict:
    from .collision_assets import qa_physics as impl

    return impl(run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
