# Phone Capture Contract

This document defines the `phone_capture_bundle` input route for importing iPhone 17 Pro ARKit RGB-D captures into the real-to-sim pipeline.

## Required Layout

```text
capture_bundle/
  rgb/
    frame_000000.jpg
    frame_000001.jpg
  depth/
    frame_000000.npy
    frame_000001.npy
  confidence/
    frame_000000.png
    frame_000001.png
  camera/
    intrinsics.json
    poses.json
  metadata.json
  clean_background/        # optional
    rgb/
      frame_000000.jpg
```

The same frame stem must exist in `rgb`, `depth`, `confidence`, and `camera/poses.json`.

## `metadata.json`

Required fields:

```json
{
  "capture_kind": "phone_capture_bundle",
  "device": "iPhone 17 Pro",
  "depth_unit": "meter"
}
```

`depth_unit` must be `meter`, `meters`, or `m`. A normal RGB video file or metadata marked as `source_type=rgb_video` is not a valid RGB-D capture bundle.

## `camera/intrinsics.json`

Required fields:

```json
{
  "width": 1280,
  "height": 720,
  "fx": 980.0,
  "fy": 980.0,
  "cx": 640.0,
  "cy": 360.0
}
```

These values are treated as ARKit-explicit intrinsics. Fallback intrinsics are not allowed in strict QA.

## `camera/poses.json`

Required fields:

```json
{
  "coordinate_frame": "arkit_world",
  "poses": [
    {
      "frame_id": "frame_000000",
      "rgb_path": "rgb/frame_000000.jpg",
      "depth_path": "depth/frame_000000.npy",
      "confidence_path": "confidence/frame_000000.png",
      "T_camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "tracking_state": "normal"
    }
  ]
}
```

`tracking_state` must be usable (`normal`, `tracking`, or `mapped`) for every imported frame.

## Validation Output

`rsf validate-phone-capture --capture-dir <capture_bundle>` writes:

```text
capture_bundle/capture_contract.json
```

The report checks:

- RGB, depth, confidence, and pose frame counts match.
- Intrinsics are explicit.
- Poses are explicit.
- Depth unit is meters.
- Confidence coverage is high enough.
- Tracking state is usable.
- Trajectory baseline is large enough for reconstruction.

## Import Output

`rsf import-phone-capture --capture-dir <capture_bundle> --run-dir <run>` writes:

```text
run/
  camera.json
  trajectory.json
  frames/
  depth/
  confidence/
  capture_contract.json
```

`camera.json` must include:

```json
{
  "intrinsics_source": "arkit_explicit",
  "extrinsics_source": "arkit_explicit",
  "scale_source": "arkit_sceneDepth_meters"
}
```

Strict QA must reject imported runs that fall back to estimated camera intrinsics, fallback camera extrinsics, or non-meter depth.

## Nerfstudio Export

`rsf export-nerfstudio-from-phone-capture --run-dir <run> --pose-world arkit|sim` exports:

```text
run/video/nerfstudio_phone/transforms.json
run/video/nerfstudio_phone/images/
```

If `clean_background/rgb` exists in the capture bundle, those images are used for the Nerfstudio export. The transform file records whether camera poses are in `arkit` or `sim` world. A downstream 3DGS registration may use identity only when 3DGS was trained with `pose_world=sim` and the registration report records `identity_allowed=true` with evidence.

## Prohibited Claims

- A plain `.mp4` cannot be imported as a phone RGB-D capture.
- Fallback intrinsics or extrinsics cannot pass strict QA.
- Identity `T_3dgs_world_to_sim_world` is forbidden without explicit phone/sim-world training evidence.
- PNG sidecar renders cannot be promoted to live 3DGS.
- Bounding-box overlap metrics must not be named silhouette mask IoU.
