# render_custom_view_masks.py 使用说明

作用：给定一份训练好的 3DGS（包含 3D 目标掩码）和任意相机视角，渲染两张目标掩码并计算可见率：
- `mask_occ.png`：全场景渲染，颜色=目标标签，前景会遮挡目标。
- `mask_no_occ.png`：过滤掉非目标高斯，只渲染目标，不受其他物体遮挡。
- `visibility_ratio.txt`：`A_visible / A_full`（遮挡版面积 / 去遮挡版面积）。

## 运行示例
```bash
conda run -n sags_py310 python render_custom_view_masks.py \
  -m /raid/home/than/zhiyuan/video_data_process/results_corridor/3dgs_output \
  --iteration 30000 \
  --precomputed_mask /raid/home/than/zhiyuan/jetauto_rl_navigation/data/blue_bin_mask_from_2d.pt \
  --camera_json /path/to/cam.json \
  --output_dir /raid/home/than/zhiyuan/jetauto_rl_navigation/data/mask_renders_custom
```

`cam.json` 内容示例（需自行填写数值）：
```json
{
  "width": 1280,
  "height": 720,
  "fx": 1000.0,
  "fy": 1000.0,
  "cx": 640.0,
  "cy": 360.0,
  "w2c": [
    [r11, r12, r13, t1],
    [r21, r22, r23, t2],
    [r31, r32, r33, t3],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "znear": 0.01,   // 可选
  "zfar": 100.0    // 可选
}
```
- `w2c` 是世界到相机的 4x4 变换矩阵。
- 如果没有 `cx/cy`，可用图像中心；没有 `znear/zfar` 则用默认。

## 输出
- `mask_occ.png`：遮挡版掩码（全场景，目标可能被遮挡）。
- `mask_no_occ.png`：去遮挡版掩码（仅目标）。
- `visibility_ratio.txt`：记录 `A_full`、`A_visible`、`visible_ratio` 以及掩码路径。

## 关键输入
- 3DGS 模型目录：`-m /raid/home/than/zhiyuan/video_data_process/results_corridor/3dgs_output`
- 3D 目标掩码（bool tensor, shape N）：`--precomputed_mask /raid/home/than/zhiyuan/jetauto_rl_navigation/data/blue_bin_mask_from_2d.pt`
- 相机参数 JSON：`--camera_json /path/to/cam.json`

默认输出目录：`./mask_renders`（可用 `--output_dir` 覆盖）。
