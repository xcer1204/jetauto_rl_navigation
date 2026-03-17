"""
Given a single camera view (intrinsics + world-to-camera pose), render two masks:
- occluded mask: all Gaussians present, color = is_target label (0/1), so foreground can hide target.
- unoccluded mask: filter out non-target Gaussians, so only target remains (no occlusion by others).

Also compute visible ratio = sum(occluded_mask) / sum(unoccluded_mask) per view.
"""

import json
import math
import os
import sys
from argparse import ArgumentParser
from typing import Any, Dict

import torch
import torchvision

# Make SAGA code importable
SAGA_ROOT = "/home/ubuntu/xc_isaac/SegAnyGAussians"
if SAGA_ROOT not in sys.path:
    sys.path.append(SAGA_ROOT)

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render_with_depth
from scene import Scene, GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


class SimpleCam:
    """Lightweight camera container matching render_with_depth needs."""

    def __init__(self, width: int, height: int, fx: float, fy: float, cx: float, cy: float, w2c: torch.Tensor, znear: float = 0.01, zfar: float = 100.0, device: str = "cuda"):
        self.image_width = width
        self.image_height = height
        self.FoVx = float(2 * math.atan(width / (2 * fx)))
        self.FoVy = float(2 * math.atan(height / (2 * fy)))
        self.znear = znear
        self.zfar = zfar

        R = w2c[:3, :3].cpu().numpy()
        t = w2c[:3, 3].cpu().numpy()
        wv = torch.tensor(getWorld2View2(R, t), dtype=torch.float32, device=device).t()
        proj = getProjectionMatrix(
            znear=znear,
            zfar=zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
            w=width,
            h=height,
            cx=cx,
            cy=cy,
            allow_principle_point_shift=True,
        ).t().to(device=device)

        self.world_view_transform = wv
        self.full_proj_transform = wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        self.camera_center = torch.inverse(wv)[3, :3]


def load_cam_from_json(path: str, device: str = "cuda") -> SimpleCam:
    with open(path, "r") as f:
        data: Dict[str, Any] = json.load(f)
    w = int(data["width"])
    h = int(data["height"])
    fx = float(data.get("fx", data["focal_x"] if "focal_x" in data else None))
    fy = float(data.get("fy", data["focal_y"] if "focal_y" in data else None))
    cx = float(data.get("cx", w / 2.0))
    cy = float(data.get("cy", h / 2.0))
    znear = float(data.get("znear", 0.01))
    zfar = float(data.get("zfar", 100.0))

    w2c_list = data["w2c"]
    w2c = torch.tensor(w2c_list, dtype=torch.float32, device=device)
    if w2c.shape == (4, 4):
        pass
    elif w2c.numel() == 16:
        w2c = w2c.view(4, 4)
    else:
        raise ValueError("w2c must be 4x4 matrix")
    return SimpleCam(w, h, fx, fy, cx, cy, w2c, znear=znear, zfar=zfar, device=device)


@torch.no_grad()
def main():
    parser = ArgumentParser(description="Render occluded/unoccluded target masks for a custom camera view.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument(
        "--precomputed_mask",
        type=str,
        default="/home/ubuntu/xc_isaac/jetauto_rl_navigation/data/blue_bin_mask_from_2d.pt",
        help="Bool tensor (N,) marking target Gaussians.",
    )
    parser.add_argument(
        "--camera_json",
        type=str,
        required=True,
        help="JSON with keys: width,height,fx,fy,cx,cy,w2c (4x4 list), optional znear/zfar.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ubuntu/xc_isaac/jetauto_rl_navigation/data/mask_renders",
    )
    parser.set_defaults(
        model_path="/home/ubuntu/xc_isaac/video_data_process/results_corridor/3dgs_output",
        source_path="/home/ubuntu/xc_isaac/video_data_process/results_corridor/colmap_data_undistorted",
    )
    args = get_combined_args(parser)

    device = "cuda"

    dataset = model.extract(args)
    dataset.need_features = False
    dataset.need_masks = False
    dataset.allow_principle_point_shift = True
    pipe = pipeline.extract(args)

    print(f"Loading 3DGS from {dataset.model_path}, iter={args.iteration}")
    gaussians = GaussianModel(dataset.sh_degree)
    _ = Scene(
        dataset,
        gaussians=gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        mode="eval",
        target="scene",
        sample_rate=1.0,
    )  # loads Gaussians + cameras (cameras unused here)

    is_target = torch.load(args.precomputed_mask, map_location=device)
    if is_target.dtype != torch.bool:
        is_target = is_target.bool()
    mask_color = is_target.float().unsqueeze(1).repeat(1, 3).to(device)

    cam = load_cam_from_json(args.camera_json, device=device)
    bg = torch.zeros(3, device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    occ_path = os.path.join(args.output_dir, "mask_occ.png")
    unocc_path = os.path.join(args.output_dir, "mask_no_occ.png")
    txt_path = os.path.join(args.output_dir, "visibility_ratio.txt")

    # Occluded: all Gaussians present
    occ_res = render_with_depth(
        cam,
        gaussians,
        pipe,
        bg,
        override_mask=mask_color,
        filtered_mask=None,
    )
    occ_mask = occ_res["mask"]
    if occ_mask.dim() == 3:
        occ_mask = occ_mask[0]
    occ_mask = (occ_mask > 0.5).float()
    torchvision.utils.save_image(occ_mask, occ_path)

    # Unoccluded: filter out non-target
    unocc_res = render_with_depth(
        cam,
        gaussians,
        pipe,
        bg,
        override_mask=mask_color,
        filtered_mask=~is_target,
    )
    unocc_mask = unocc_res["mask"]
    if unocc_mask.dim() == 3:
        unocc_mask = unocc_mask[0]
    unocc_mask = (unocc_mask > 0.5).float()
    torchvision.utils.save_image(unocc_mask, unocc_path)

    A_full = float(unocc_mask.sum().item())
    A_visible = float(occ_mask.sum().item())
    visible_ratio = 0.0 if A_full == 0 else A_visible / A_full
    with open(txt_path, "w") as f:
        f.write(
            f"A_full={A_full:.0f} A_visible={A_visible:.0f} visible_ratio={visible_ratio:.4f}\n"
            f"occ_mask: {occ_path}\n"
            f"unocc_mask: {unocc_path}\n"
        )

    print(f"Saved occluded   -> {occ_path}")
    print(f"Saved unoccluded -> {unocc_path}")
    print(f"Visible ratio {visible_ratio:.4f} written to {txt_path}")


if __name__ == "__main__":
    main()
