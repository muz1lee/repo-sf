# Isaac Native 3DGS Runtime 探测报告

日期：2026-07-01 Asia/Shanghai

Worker：

```text
ssh -p 1024 root@101.132.143.105
/isaac-sim/python.sh
```

探测命令：

```bash
scp -P 1024 /tmp/rsf_isaac_e_patch/isaac_3dgs_probe.py root@101.132.143.105:/tmp/rsf_isaac_3dgs_probe.py
ssh -p 1024 root@101.132.143.105 "/isaac-sim/python.sh /tmp/rsf_isaac_3dgs_probe.py > /tmp/rsf_isaac_3dgs_probe.log 2>&1"
ssh -p 1024 root@101.132.143.105 "cat /tmp/rsf_isaac_3dgs_probe_result.json"
```

结果摘要：

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

日志证据：

- `/tmp/rsf_isaac_3dgs_probe.log` 中出现 `Simulation App Startup Complete`，说明探测发生在 `SimulationApp` 启动之后。
- Isaac 同时输出了一些无关 extension import error，包括缺少 `/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py`；这些错误没有阻止 3DGS capability probe 完成。

结论：

当前 Isaac worker 没有通过 extension manager、候选 Python module 或 USD plugin/file-format registry 暴露 native Gaussian splat / splat PLY 渲染能力。因此 `isaac_load_report.json` 应写成：

```json
{
  "native_3dgs_required": true,
  "native_3dgs_supported": false,
  "native_3dgs_status": "unsupported_missing_native_3dgs_runtime"
}
```

在安装并探测到支持 splat 的 Isaac/Omniverse extension 或 USD plugin 之前，`background/3dgs_native/splat_rgb.ply` 的 native validation 必须保持 blocked。
