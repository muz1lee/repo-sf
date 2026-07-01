# 3DGS 资产检查报告

日期：2026-07-01

## 范围

本报告只检查下面这个 3DGS 候选资产：

```text
runs/my_table_m7_20260630_185332/background/3dgs_native/splat_rgb.ply
```

检查内容仅包括读取 PLY header，并验证它是否像一个包含必要字段的 Gaussian splat 资产。本检查不会在 Genesis、Isaac、USD、Nerfstudio 或任何 simulator runtime 中加载该资产。

## 结果

生成的报告文件：

```text
runs/my_table_m7_20260630_185332/background/3dgs_native/asset_report.json
```

观测到的 header 摘要：

- `status`: `asset_format_complete`
- `format`: `binary_little_endian 1.0`
- `gaussian_count`: `179536`
- RGB 字段：`red`、`green`、`blue`
- 必要 Gaussian 字段存在：`x`、`y`、`z`、`opacity`、`scale_0`、`scale_1`、`scale_2`、`rot_0`、`rot_1`、`rot_2`、`rot_3`、`rgb`

## 注意事项

这不等于 native simulator completion。这个报告只证明 3DGS 风格的 PLY 资产存在，并且 header 字段完整；它不证明 simulator 能原生渲染该 splat，也不证明 splat 已经注册到 simulator world 坐标系。

当前没有生成 mesh surrogate 或 point-cloud surrogate。任何转换路线在真正实现并验证 simulator-native 3DGS render bridge 之前，都必须继续标记为 blocked/proxy。
