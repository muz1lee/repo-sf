# Isaac Native 3DGS Runtime Probe

Date: 2026-07-01 Asia/Shanghai

Worker:

```text
ssh -p 1024 root@101.132.143.105
/isaac-sim/python.sh
```

Probe command:

```bash
scp -P 1024 /tmp/rsf_isaac_e_patch/isaac_3dgs_probe.py root@101.132.143.105:/tmp/rsf_isaac_3dgs_probe.py
ssh -p 1024 root@101.132.143.105 "/isaac-sim/python.sh /tmp/rsf_isaac_3dgs_probe.py > /tmp/rsf_isaac_3dgs_probe.log 2>&1"
ssh -p 1024 root@101.132.143.105 "cat /tmp/rsf_isaac_3dgs_probe_result.json"
```

Result summary:

```json
{
  "simulation_app_started": true,
  "simulation_app_source": "isaacsim",
  "extension_manager_available": true,
  "extension_matches": [],
  "module_imports": {
    "omni.splat": "ModuleNotFoundError",
    "omni.kit.splat": "ModuleNotFoundError",
    "omni.usd.splat": "ModuleNotFoundError",
    "omni.gaussian_splatting": "ModuleNotFoundError",
    "omni.kit.gaussian_splatting": "ModuleNotFoundError",
    "omni.kit.viewport.gaussian_splatting": "ModuleNotFoundError"
  },
  "usd_plugin_matches": [],
  "sdf_file_formats": {},
  "native_3dgs_supported": false,
  "native_3dgs_status": "unsupported_missing_native_3dgs_runtime"
}
```

Log evidence:

- `Simulation App Startup Complete` appears in `/tmp/rsf_isaac_3dgs_probe.log`, so the probe ran after `SimulationApp` startup.
- Isaac emitted unrelated extension import errors, including missing `/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py`; these did not prevent the 3DGS capability probe from completing.

Conclusion:

The current Isaac worker does not expose native Gaussian splat / splat PLY rendering capability through the extension manager, candidate Python modules, or USD plugin/file-format registry. `isaac_load_report.json` should therefore use:

```json
{
  "native_3dgs_required": true,
  "native_3dgs_supported": false,
  "native_3dgs_status": "unsupported_missing_native_3dgs_runtime"
}
```

Native validation of `background/3dgs_native/splat_rgb.ply` must remain blocked until a splat-capable Isaac/Omniverse extension or USD plugin is installed and detected after `SimulationApp` startup.
