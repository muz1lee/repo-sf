"""Runtime-backed browser viewer exporters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sim_export_manifest import write_sim_export_manifest


@dataclass(frozen=True)
class RuntimeViewerExportResult:
    index_path: Path
    runtime_manifest_path: Path
    script_path: Path | None
    report: dict[str, Any]


def export_runtime_viewer(run_dir: str | Path, *, backend: str = "genesis") -> RuntimeViewerExportResult:
    run = Path(run_dir)
    write_sim_export_manifest(run)
    sim_manifest = json.loads((run / "sim_export_manifest.json").read_text(encoding="utf-8"))
    background = _runtime_background_summary(sim_manifest)
    viewer_dir = run / "exports" / "runtime_viewer" / backend
    viewer_dir.mkdir(parents=True, exist_ok=True)
    if backend == "genesis":
        script_path: Path | None = run / "exports" / "genesis_runtime_server.py"
        script_path.write_text(_genesis_runtime_server_script(), encoding="utf-8")
        status = "runtime_script_written"
        blocked_reason = None
    elif backend == "isaac":
        script_path = None
        status = "runtime_unavailable"
        blocked_reason = "isaac_runtime_unavailable:no_local_runtime_bridge_implemented_for_click_pick_apply_force"
    else:
        raise ValueError(f"unsupported runtime viewer backend: {backend}")

    runtime_manifest = {
        "version": 1,
        "backend": backend,
        "status": status,
        "blocked_reason": blocked_reason,
        "background": background,
        "script_path": str(script_path.relative_to(run)) if script_path else None,
        "index_path": f"exports/runtime_viewer/{backend}/index.html",
            "api": {
            "state": "/api/state",
            "apply_force": "/api/apply-force",
            "reset": "/api/reset",
            "replay": "/api/replay",
        },
        "run_command": (
            "/mnt/workspace/wenqian/knowin-world/.venv/bin/python "
            f"{script_path} --run-dir {run} --host 127.0.0.1 --port 7030"
        )
        if script_path
        else None,
    }
    runtime_manifest_path = viewer_dir / "runtime_manifest.json"
    runtime_manifest_path.write_text(json.dumps(runtime_manifest, indent=2), encoding="utf-8")
    index_path = viewer_dir / "index.html"
    index_path.write_text(_runtime_viewer_html(runtime_manifest), encoding="utf-8")
    report = {
        "status": status,
        "backend": backend,
        "runtime_manifest_path": str(runtime_manifest_path.relative_to(run)),
        "script_path": str(script_path.relative_to(run)) if script_path else None,
        "blocked_reason": blocked_reason,
        "background": background,
    }
    (run / "qa").mkdir(parents=True, exist_ok=True)
    (run / "qa" / "runtime_viewer_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return RuntimeViewerExportResult(index_path=index_path, runtime_manifest_path=runtime_manifest_path, script_path=script_path, report=report)


def _runtime_background_summary(sim_manifest: dict[str, Any]) -> dict[str, Any]:
    background = sim_manifest.get("background", {}) if isinstance(sim_manifest.get("background"), dict) else {}
    visual = background.get("visual_asset", {}) if isinstance(background.get("visual_asset"), dict) else {}
    external = background.get("external_render", {}) if isinstance(background.get("external_render"), dict) else {}
    registration = background.get("registration", {}) if isinstance(background.get("registration"), dict) else {}
    source_kind = str(visual.get("source_kind") or external.get("backend") or "unknown")
    simulator_native = bool(visual.get("simulator_native") or external.get("simulator_native"))
    native_runtime = bool(visual.get("native_runtime_verified") or simulator_native)
    external_verified = bool(visual.get("external_render_verified") or external.get("status") == "rendered")
    if source_kind == "external_3dgs_renderer" and external_verified and not native_runtime:
        render_mode = "external_3dgs_png_sidecar"
        viewer_status = "partial_external_render_only"
        warning = "Native/live 3DGS background is unavailable in this runtime viewer."
    elif native_runtime:
        render_mode = "live_3dgs_runtime"
        viewer_status = "live_3dgs_runtime"
        warning = None
    else:
        render_mode = source_kind
        viewer_status = registration.get("status") or visual.get("status") or "background_runtime_unavailable"
        warning = "Native/live 3DGS background is unavailable in this runtime viewer."
    return {
        "viewer_status": viewer_status,
        "render_mode": render_mode,
        "source_kind": source_kind,
        "live_3dgs_runtime": native_runtime,
        "simulator_native": simulator_native,
        "external_render_path": external.get("render_path") or visual.get("path"),
        "registration_status": registration.get("status"),
        "warning": warning,
    }


def _runtime_viewer_html(runtime_manifest: dict[str, Any]) -> str:
    backend = runtime_manifest["backend"]
    status = runtime_manifest["status"]
    background = runtime_manifest.get("background", {})
    background_status = background.get("viewer_status", "unknown")
    background_mode = background.get("render_mode", "unknown")
    background_warning = background.get("warning") or ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Real2Sim Runtime Viewer</title>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #090a0c; color: #f2f4f8; }}
    #viewport {{ position: fixed; inset: 0; }}
    #panel {{ position: fixed; left: 16px; top: 16px; width: 344px; max-height: calc(100vh - 32px); overflow: auto; background: rgba(12, 14, 18, 0.9); border: 1px solid rgba(255,255,255,0.16); border-radius: 8px; padding: 14px; }}
    h1 {{ font-size: 15px; margin: 0 0 10px; }}
    .row {{ display: flex; justify-content: space-between; gap: 8px; margin: 8px 0; font-size: 12px; }}
    .label {{ color: #aab0bb; }}
    .value {{ color: #fff; text-align: right; overflow-wrap: anywhere; }}
    .warn {{ color: #ffcf5a; }}
    .ok {{ color: #7ee787; }}
    button {{ border: 1px solid rgba(255,255,255,0.22); background: rgba(255,255,255,0.08); color: #fff; border-radius: 6px; padding: 8px 9px; cursor: pointer; font-size: 12px; }}
    button:disabled {{ opacity: 0.45; cursor: not-allowed; }}
    #controls {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 12px 0; }}
    #objectList button {{ width: 100%; display: flex; justify-content: space-between; margin: 6px 0; }}
    #status {{ position: fixed; right: 16px; bottom: 16px; padding: 8px 10px; border-radius: 6px; background: rgba(0,0,0,0.68); font-size: 12px; }}
  </style>
  <script type="importmap">
    {{"imports":{{"three":"https://unpkg.com/three@0.160.0/build/three.module.js"}}}}
  </script>
</head>
<body>
  <div id="viewport"></div>
  <aside id="panel">
    <h1>Real2Sim Runtime Viewer</h1>
    <div class="row"><span class="label">runtime backend</span><span class="value">runtime backend: {backend}</span></div>
    <div class="row"><span class="label">status</span><span id="runtimeStatus" class="value {'warn' if status != 'runtime_script_written' else 'ok'}">{status}</span></div>
    <div class="row"><span class="label">background</span><span class="value warn">{background_status}</span></div>
    <div class="row"><span class="label">background mode</span><span class="value">{background_mode}</span></div>
    <div class="row warn"><span>{background_warning}</span></div>
    <div class="row"><span class="label">selected</span><span id="selectedObject" class="value">none</span></div>
    <div id="controls">
      <button id="pushX">Apply +X force</button>
      <button id="pushY">Apply +Y force</button>
      <button id="reset">Reset</button>
      <button id="replay">Replay last</button>
    </div>
    <div id="objectList"></div>
  </aside>
  <div id="status">Loading runtime...</div>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
    import {{ GLTFLoader }} from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    const root = document.getElementById('viewport');
    const status = document.getElementById('status');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111318);
    const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 200);
    camera.up.set(0, 0, 1);
    camera.position.set(0.7, -2.0, 0.9);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    root.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.2, 0.1);
    controls.update();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x222233, 1.4));
    const dir = new THREE.DirectionalLight(0xffffff, 1.3);
    dir.position.set(1.5, -2.0, 3.0);
    scene.add(dir);
    const loader = new GLTFLoader();
    const meshes = new Map();
    let selected = null;
    let replayFrames = [];

    function setStatus(text) {{ status.textContent = text; }}
    function quatWxyzToThree(q) {{ return new THREE.Quaternion(q[1], q[2], q[3], q[0]); }}
    async function api(path, body) {{
      const opts = body ? {{ method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(body) }} : undefined;
      const res = await fetch(path, opts);
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }}
    function selectObject(id) {{
      selected = id;
      document.getElementById('selectedObject').textContent = id || 'none';
    }}
    function updateObjects(state) {{
      for (const item of state.objects || []) {{
        const existing = meshes.get(item.object_id);
        if (!existing && item.visual_mesh_path) {{
          loader.load(item.visual_mesh_path, gltf => {{
            const object = gltf.scene;
            object.userData.object_id = item.object_id;
            object.traverse(node => {{ if (node.isMesh) node.userData.object_id = item.object_id; }});
            meshes.set(item.object_id, object);
            scene.add(object);
            updateObjects(state);
          }});
          continue;
        }}
        if (existing) {{
          existing.position.set(...item.position);
          existing.quaternion.copy(quatWxyzToThree(item.quat_wxyz));
          existing.scale.setScalar(item.asset_scale || 1);
        }}
      }}
    }}
    async function refresh() {{
      const state = await api('/api/state');
      updateObjects(state);
      replayFrames = state.replay || replayFrames;
      const list = document.getElementById('objectList');
      list.replaceChildren();
      for (const item of state.objects || []) {{
        const btn = document.createElement('button');
        btn.innerHTML = `<span>${{item.object_id}}</span><span>${{item.movable ? 'movable' : 'fixed'}}</span>`;
        btn.addEventListener('click', () => selectObject(item.object_id));
        list.append(btn);
      }}
      setStatus(`runtime=${{state.backend}} step=${{state.step}} objects=${{(state.objects || []).length}}`);
    }}
    async function applyForce(vector) {{
      if (!selected) return setStatus('Select an object first.');
      const state = await api('/api/apply-force', {{ object_id: selected, force: vector, steps: 45 }});
      updateObjects(state);
      replayFrames = state.replay || [];
      setStatus(`applied force to ${{selected}}`);
    }}
    document.getElementById('pushX').addEventListener('click', () => applyForce([8, 0, 0]));
    document.getElementById('pushY').addEventListener('click', () => applyForce([0, 8, 0]));
    document.getElementById('reset').addEventListener('click', async () => updateObjects(await api('/api/reset', {{}})));
    document.getElementById('replay').addEventListener('click', async () => {{
      const data = await api('/api/replay');
      replayFrames = data.replay || [];
      for (const frame of replayFrames) {{
        updateObjects({{objects: frame.objects}});
        await new Promise(resolve => setTimeout(resolve, 40));
      }}
    }});
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    renderer.domElement.addEventListener('click', event => {{
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects([...meshes.values()], true);
      if (!hits.length) return;
      let node = hits[0].object;
      while (node && !node.userData.object_id) node = node.parent;
      selectObject(node?.userData.object_id || null);
    }});
    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});
    function animate() {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }}
    refresh().catch(err => setStatus(`runtime_unavailable: ${{err.message || err}}`));
    setInterval(() => refresh().catch(() => {{}}), 500);
    animate();
  </script>
</body>
</html>
"""


def _genesis_runtime_server_script() -> str:
    return r'''#!/usr/bin/env python3
"""Genesis-backed runtime server for click-pick/apply-force/reset/replay."""

from __future__ import annotations

import argparse
import json
import math
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import genesis as gs
import numpy as np


def matrix_to_quat_wxyz(matrix):
    rot = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * scale, (rot[2, 1] - rot[1, 2]) / scale, (rot[0, 2] - rot[2, 0]) / scale, (rot[1, 0] - rot[0, 1]) / scale], dtype=np.float64)
    else:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    return tuple(float(v) for v in quat / norm) if norm > 0.0 and np.isfinite(norm) else (1.0, 0.0, 0.0, 0.0)


def tensor_to_vec(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]


class GenesisRuntime:
    def __init__(self, run_dir: Path, *, backend: str = "cpu") -> None:
        self.run_dir = run_dir
        self.manifest = json.loads((run_dir / "sim_export_manifest.json").read_text(encoding="utf-8"))
        gs.init(backend=gs.cpu if backend == "cpu" else gs.gpu)
        self.scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01, substeps=4))
        self.entities = {}
        self.object_meta = {}
        self.initial = {}
        self.step_count = 0
        self.replay = []
        self._build_scene()

    def _build_scene(self):
        support = self.manifest.get("support_surface", {})
        if support.get("table_collision_pos_world") and support.get("table_collision_size_xyz"):
            self.scene.add_entity(
                morph=gs.morphs.Box(
                    pos=tuple(float(v) for v in support["table_collision_pos_world"]),
                    size=tuple(float(v) for v in support["table_collision_size_xyz"]),
                    fixed=True,
                    collision=True,
                ),
                material=gs.materials.Rigid(friction=1.0),
            )
        for obj in self.manifest.get("objects", []):
            transform = np.asarray(obj["pose"]["T_object_to_world"], dtype=np.float64)
            asset_scale = float(obj.get("pose", {}).get("asset_scale", 1.0) or 1.0)
            entity = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=str(self.run_dir / obj["collision_asset"]["path"]),
                    pos=tuple(float(v) for v in transform[:3, 3]),
                    quat=matrix_to_quat_wxyz(transform),
                    scale=asset_scale,
                    fixed=False,
                    collision=True,
                    convexify=True,
                ),
                material=gs.materials.Rigid(friction=float(obj["physics"].get("friction", 0.8))),
            )
            object_id = obj["object_id"]
            self.entities[object_id] = entity
            self.object_meta[object_id] = {
                "object_id": object_id,
                "label": obj.get("label", object_id),
                "asset_scale": asset_scale,
                "visual_mesh_path": "/" + obj["visual_asset"]["path"],
                "movable": True,
                "mass_kg": float(obj["physics"].get("mass_kg", 0.25)),
            }
            self.initial[object_id] = {"pos": [float(v) for v in transform[:3, 3]], "quat_wxyz": list(matrix_to_quat_wxyz(transform))}
        self.scene.build()
        for object_id, entity in self.entities.items():
            if hasattr(entity, "set_mass"):
                entity.set_mass(float(self.object_meta[object_id]["mass_kg"]))
        for _ in range(20):
            self.scene.step()
            self.step_count += 1

    def object_state(self):
        objects = []
        for object_id, entity in self.entities.items():
            meta = dict(self.object_meta[object_id])
            meta["position"] = tensor_to_vec(entity.get_pos())[:3] if hasattr(entity, "get_pos") else self.initial[object_id]["pos"]
            meta["quat_wxyz"] = tensor_to_vec(entity.get_quat())[:4] if hasattr(entity, "get_quat") else self.initial[object_id]["quat_wxyz"]
            objects.append(meta)
        return objects

    def state(self):
        return {"backend": "genesis", "status": "running", "step": self.step_count, "objects": self.object_state(), "replay": self.replay[-80:]}

    def apply_force(self, object_id, force, steps):
        if object_id not in self.entities:
            raise KeyError(f"unknown object_id: {object_id}")
        entity = self.entities[object_id]
        force_vec = np.asarray(force, dtype=np.float64).reshape(3)
        self.replay = []
        for _ in range(int(steps)):
            if hasattr(entity, "control_dofs_force"):
                try:
                    entity.control_dofs_force(force_vec)
                except Exception:
                    pass
            if hasattr(entity, "set_pos") and not hasattr(entity, "control_dofs_force"):
                current = np.asarray(tensor_to_vec(entity.get_pos())[:3], dtype=np.float64)
                entity.set_pos(tuple(float(v) for v in current + force_vec * 0.0005))
            self.scene.step()
            self.step_count += 1
            if self.step_count % 2 == 0:
                self.replay.append({"step": self.step_count, "objects": self.object_state()})
        return self.state()

    def reset(self):
        for object_id, entity in self.entities.items():
            if hasattr(entity, "set_pos"):
                entity.set_pos(tuple(self.initial[object_id]["pos"]))
            if hasattr(entity, "set_quat"):
                entity.set_quat(tuple(self.initial[object_id]["quat_wxyz"]))
            if hasattr(entity, "zero_all_dofs_velocity"):
                entity.zero_all_dofs_velocity()
        for _ in range(5):
            self.scene.step()
            self.step_count += 1
        self.replay = [{"step": self.step_count, "objects": self.object_state()}]
        return self.state()


class RuntimeHandler(SimpleHTTPRequestHandler):
    runtime: GenesisRuntime

    def _send_json(self, data, status=200):
        encoded = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(self.runtime.state())
            return
        if path == "/api/replay":
            self._send_json({"replay": self.runtime.replay[-80:]})
            return
        if path == "/":
            self.path = "/exports/runtime_viewer/index.html"
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/api/apply-force":
                self._send_json(self.runtime.apply_force(body.get("object_id"), body.get("force", [8.0, 0.0, 0.0]), int(body.get("steps", 45))))
                return
            if path == "/api/reset":
                self._send_json(self.runtime.reset())
                return
        except Exception as exc:
            self._send_json({"status": "error", "error": str(exc)}, status=500)
            return
        self._send_json({"status": "not_found", "path": path}, status=404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7030)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    RuntimeHandler.runtime = GenesisRuntime(args.run_dir, backend=args.backend)
    server = ThreadingHTTPServer((args.host, int(args.port)), lambda *a, **kw: RuntimeHandler(*a, directory=str(args.run_dir), **kw))
    print(f"runtime viewer: http://{args.host}:{args.port}/exports/runtime_viewer/index.html", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
'''
