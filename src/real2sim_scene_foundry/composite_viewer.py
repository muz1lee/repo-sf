"""Browser composite viewer export for reconstructed assets and background geometry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CompositeViewerResult:
    index_path: Path
    config_path: Path
    url_path: str


def export_composite_viewer(run_dir: str | Path) -> CompositeViewerResult:
    run = Path(run_dir)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qa_path = run / "qa" / "qa_report.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.is_file() else {}

    viewer_dir = run / "exports" / "composite_viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    config = _viewer_config(run, viewer_dir, manifest, _qa_with_genesis_fallback(run, qa))
    config_path = viewer_dir / "viewer_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    index_path = viewer_dir / "index.html"
    index_path.write_text(_index_html(), encoding="utf-8")
    return CompositeViewerResult(
        index_path=index_path,
        config_path=config_path,
        url_path="/exports/composite_viewer/index.html",
    )


def serve_composite_viewer(run_dir: str | Path, *, host: str = "127.0.0.1", port: int = 7010) -> None:
    run = Path(run_dir)
    export_composite_viewer(run)
    handler = partial(SimpleHTTPRequestHandler, directory=str(run))
    server = ThreadingHTTPServer((host, int(port)), handler)
    server.serve_forever()


def _viewer_config(run: Path, viewer_dir: Path, manifest: dict[str, Any], qa: dict[str, Any]) -> dict[str, Any]:
    support = manifest.get("support_plane", {})
    background = manifest.get("background", {})
    physics = qa.get("physics_settle", {})
    point_cloud = run / "scene_cloud.ply"
    config = {
        "version": 1,
        "run_name": run.name,
        "coordinate_frame": manifest.get("coordinate_frames", {}).get("world", "z_up_ground_plane_meters"),
        "background": {
            "source_backend": _background_source(background, point_cloud),
            "point_cloud_path": _rel(viewer_dir, point_cloud) if point_cloud.is_file() else None,
            "point_cloud_transform": {
                "source_frame": "opencv_x_right_y_down_z_forward_meters",
                "target_frame": "z_up_ground_plane_meters",
                "support_height_m": float(support.get("original_height_world_m", 0.0)),
            },
            "gaussian_splat_config_path": background.get("gaussian_splat_config_path"),
            "note": "Browser viewer uses dense scene_cloud.ply as the background proxy unless an exported browser splat is added.",
        },
        "support_plane": {
            "status": support.get("status", "absent"),
            "table_collision_mesh_path": _rel_existing(run, viewer_dir, support.get("table_collision_mesh_path")),
            "table_collision_pos_world": support.get("table_collision_pos_world"),
            "table_collision_size_xyz": support.get("table_collision_size_xyz"),
            "table_bounds_world_xy": support.get("table_bounds_world_xy"),
        },
        "objects": [_object_config(run, viewer_dir, obj) for obj in manifest.get("objects", [])],
        "qa": {
            "stability_status": physics.get("stability_status"),
            "settle_steps": physics.get("settle_steps"),
            "max_penetration_depth_m": physics.get("max_penetration_depth_m"),
            "max_displacement_m": physics.get("max_displacement_m"),
            "fall_out_detected": physics.get("fall_out_detected"),
            "nan_detected": physics.get("nan_detected"),
        },
    }
    return config


def _qa_with_genesis_fallback(run: Path, qa: dict[str, Any]) -> dict[str, Any]:
    physics = qa.get("physics_settle", {})
    if physics.get("stability_status"):
        return qa
    genesis_path = run / "qa" / "genesis_settle_report.json"
    if not genesis_path.is_file():
        return qa
    merged = dict(qa)
    merged["physics_settle"] = json.loads(genesis_path.read_text(encoding="utf-8"))
    return merged


def _background_source(background: dict[str, Any], point_cloud: Path) -> str:
    source = str(background.get("source_backend", "unknown"))
    if source == "video_3dgs_splatfacto" and point_cloud.is_file():
        return "video_3dgs_splatfacto_proxy_point_cloud"
    return source


def _object_config(run: Path, viewer_dir: Path, obj: dict[str, Any]) -> dict[str, Any]:
    transform = np.asarray(obj["T_object_to_world"], dtype=np.float64)
    return {
        "object_id": obj["object_id"],
        "label": obj.get("label", obj["object_id"]),
        "mesh_path": _rel(viewer_dir, run / str(obj["mesh_path"])),
        "world_position": [float(v) for v in transform[:3, 3]],
        "world_quat_wxyz": _matrix_to_quat_wxyz(transform[:3, :3]),
        "mass_kg": obj.get("mass_kg"),
        "friction": obj.get("friction"),
        "needs_manual_refine": obj.get("needs_manual_refine", False),
    }


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> list[float]:
    rot = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rot[2, 1] - rot[1, 2]) / scale,
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[1, 0] - rot[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(rot)))
        if axis == 0:
            scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quat = np.array(
                [(rot[2, 1] - rot[1, 2]) / scale, 0.25 * scale, (rot[0, 1] + rot[1, 0]) / scale, (rot[0, 2] + rot[2, 0]) / scale],
                dtype=np.float64,
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quat = np.array(
                [(rot[0, 2] - rot[2, 0]) / scale, (rot[0, 1] + rot[1, 0]) / scale, 0.25 * scale, (rot[1, 2] + rot[2, 1]) / scale],
                dtype=np.float64,
            )
        else:
            scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quat = np.array(
                [(rot[1, 0] - rot[0, 1]) / scale, (rot[0, 2] + rot[2, 0]) / scale, (rot[1, 2] + rot[2, 1]) / scale, 0.25 * scale],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0 or not np.isfinite(norm):
        return [1.0, 0.0, 0.0, 0.0]
    return [float(v) for v in quat / norm]


def _rel_existing(run: Path, viewer_dir: Path, value: object) -> str | None:
    if not value:
        return None
    path = run / str(value)
    return _rel(viewer_dir, path) if path.is_file() else None


def _rel(from_dir: Path, path: Path) -> str:
    import os

    return Path(os.path.relpath(path, from_dir)).as_posix()


def _index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Real2Sim Composite Viewer</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #111; color: #eee; }
    #viewport { position: fixed; inset: 0; }
    #panel { position: fixed; left: 16px; top: 16px; width: 320px; max-height: calc(100vh - 32px); overflow: auto; background: rgba(17, 19, 23, 0.86); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 14px; backdrop-filter: blur(10px); }
    h1 { font-size: 15px; margin: 0 0 10px; font-weight: 650; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 8px 0; font-size: 12px; }
    .label { color: #aab0bb; }
    .value { color: #fff; text-align: right; overflow-wrap: anywhere; }
    .ok { color: #7ee787; }
    .warn { color: #ffcf5a; }
    button { border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.08); color: #fff; border-radius: 6px; padding: 7px 9px; cursor: pointer; font-size: 12px; }
    button[aria-pressed="true"] { background: rgba(96, 165, 250, 0.35); border-color: rgba(147,197,253,0.8); }
    #toggles { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 12px 0; }
    #status { position: fixed; right: 16px; bottom: 16px; padding: 8px 10px; border-radius: 6px; background: rgba(0,0,0,0.62); font-size: 12px; color: #d8dee9; max-width: 520px; }
  </style>
  <script type="importmap">
    {"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js"}}
  </script>
</head>
<body>
  <div id="viewport"></div>
  <aside id="panel">
    <h1>Real2Sim Composite Viewer</h1>
    <div class="row"><span class="label">Run</span><span id="runName" class="value">loading</span></div>
    <div class="row"><span class="label">Background</span><span id="bgSource" class="value">loading</span></div>
    <div class="row"><span class="label">Physics</span><span id="stability" class="value">loading</span></div>
    <div class="row"><span class="label">Objects</span><span id="objectCount" class="value">0</span></div>
    <div id="toggles">
      <button id="toggleCloud" aria-pressed="true">Point Cloud</button>
      <button id="toggleObjects" aria-pressed="true">Objects</button>
      <button id="toggleTable" aria-pressed="true">Table</button>
      <button id="toggleAxes" aria-pressed="true">Axes</button>
    </div>
    <div id="objectList"></div>
  </aside>
  <div id="status">Loading viewer_config.json...</div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
    import { GLTFLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';
    import { PLYLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/PLYLoader.js';

    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    const root = document.getElementById('viewport');
    const status = document.getElementById('status');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111318);
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 200);
    camera.up.set(0, 0, 1);
    camera.position.set(1.6, -2.2, 1.2);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    root.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.2, 0.15);
    controls.update();

    const hemi = new THREE.HemisphereLight(0xffffff, 0x222233, 1.5);
    scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.6);
    dir.position.set(1.5, -2.0, 3.0);
    scene.add(dir);
    const axes = new THREE.AxesHelper(0.5);
    scene.add(axes);
    const cloudGroup = new THREE.Group();
    const objectGroup = new THREE.Group();
    const tableGroup = new THREE.Group();
    scene.add(cloudGroup, tableGroup, objectGroup);

    function setStatus(text) { status.textContent = text; }
    function quatWxyzToThree(q) { return new THREE.Quaternion(q[1], q[2], q[3], q[0]); }
    function transformPointCloudToWorld(geometry, supportHeight) {
      const pos = geometry.getAttribute('position');
      for (let i = 0; i < pos.count; i++) {
        const cx = pos.getX(i), cy = pos.getY(i), cz = pos.getZ(i);
        pos.setXYZ(i, cx, cz, -cy - supportHeight);
      }
      pos.needsUpdate = true;
      geometry.computeBoundingSphere();
      geometry.computeBoundingBox();
    }
    function addToggle(id, target) {
      const button = document.getElementById(id);
      button.addEventListener('click', () => {
        target.visible = !target.visible;
        button.setAttribute('aria-pressed', String(target.visible));
      });
    }
    addToggle('toggleCloud', cloudGroup);
    addToggle('toggleObjects', objectGroup);
    addToggle('toggleTable', tableGroup);
    addToggle('toggleAxes', axes);

    async function main() {
      const config = await fetch('viewer_config.json').then(r => r.json());
      document.getElementById('runName').textContent = config.run_name;
      document.getElementById('bgSource').textContent = config.background.source_backend;
      document.getElementById('stability').textContent = config.qa.stability_status || 'unknown';
      document.getElementById('stability').className = config.qa.stability_status === 'passed' ? 'value ok' : 'value warn';
      document.getElementById('objectCount').textContent = String(config.objects.length);
      const objectList = document.getElementById('objectList');
      objectList.replaceChildren();
      for (const object of config.objects) {
        const row = document.createElement('div');
        const id = document.createElement('span');
        const label = document.createElement('span');
        row.className = 'row';
        id.className = 'label';
        label.className = 'value';
        id.textContent = object.object_id;
        label.textContent = object.label;
        row.append(id, label);
        objectList.append(row);
      }

      const plyLoader = new PLYLoader();
      if (config.background.point_cloud_path) {
        plyLoader.load(config.background.point_cloud_path, geometry => {
          transformPointCloudToWorld(geometry, config.background.point_cloud_transform.support_height_m || 0);
          const hasColor = Boolean(geometry.getAttribute('color'));
          const mat = new THREE.PointsMaterial({ size: 0.008, vertexColors: hasColor, color: hasColor ? 0xffffff : 0x8ab4f8, opacity: 0.82, transparent: true });
          cloudGroup.add(new THREE.Points(geometry, mat));
          setStatus('Loaded point-cloud background, object meshes, and table collision proxy.');
        }, undefined, err => setStatus(`Point cloud load failed: ${err.message || err}`));
      }

      const gltfLoader = new GLTFLoader();
      if (config.support_plane.table_collision_mesh_path) {
        gltfLoader.load(config.support_plane.table_collision_mesh_path, gltf => {
          const table = gltf.scene;
          const p = config.support_plane.table_collision_pos_world || [0, 0, 0];
          table.position.set(p[0], p[1], p[2]);
          table.traverse(node => {
            if (node.isMesh) {
              node.material = new THREE.MeshStandardMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.28, roughness: 0.8 });
              node.renderOrder = 1;
            }
          });
          tableGroup.add(table);
        });
      }

      for (const item of config.objects) {
        gltfLoader.load(item.mesh_path, gltf => {
          const object = gltf.scene;
          object.name = item.object_id;
          object.position.set(...item.world_position);
          object.quaternion.copy(quatWxyzToThree(item.world_quat_wxyz));
          object.traverse(node => {
            if (node.isMesh) {
              node.material = node.material || new THREE.MeshStandardMaterial();
              node.castShadow = true;
              node.receiveShadow = true;
            }
          });
          objectGroup.add(object);
        }, undefined, err => setStatus(`${item.object_id} mesh load failed: ${err.message || err}`));
      }
    }
    main().catch(err => setStatus(`Viewer failed: ${err.stack || err}`));
    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>
"""
