# gs_ratio_test.py
import argparse
import math
import os
from typing import Union

import numpy as np
import torch
import torchvision
import sys
# Make SAGA code importable
SAGA_ROOT = "/home/ubuntu/SegAnyGAussians"
if SAGA_ROOT not in sys.path:
    sys.path.append(SAGA_ROOT)
    
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render_with_depth
from scene import Scene, GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

import einops
from einops import einsum
from e3nn import o3
from pytorch3d.transforms import matrix_to_quaternion


# Same alignment as Isaac env: real/3DGS -> Isaac
ALIGN_SCALE = 0.3664808278887077
ALIGN_ROT_XYZW = (0.04309, 0.03407, -0.02809, 0.99810)  # (x,y,z,w)
ALIGN_TRANSLATION = (0.089711, 0.740089, 0.192696)      # (tx,ty,tz)


@torch.jit.script
def rotation_matrix_from_quaternion_wxyz(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz -> (N,3,3)"""
    q = quaternion_wxyz
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack(
        [
            torch.stack([1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q3 * q0, 2 * q1 * q3 + 2 * q2 * q0], dim=1),
            torch.stack([2 * q1 * q2 + 2 * q3 * q0, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q1 * q0], dim=1),
            torch.stack([2 * q1 * q3 - 2 * q2 * q0, 2 * q2 * q3 + 2 * q1 * q0, 1 - 2 * q1 * q1 - 2 * q2 * q2], dim=1),
        ],
        dim=1,
    )
    return R


def to_so3(R: torch.Tensor) -> torch.Tensor:
    """Project to nearest proper rotation matrix."""
    U, _, Vt = torch.linalg.svd(R)
    R_orth = U @ Vt
    if torch.det(R_orth) < 0:
        U[..., -1] *= -1
        R_orth = U @ Vt
    return R_orth


def transform_shs(shs_feat: torch.Tensor, rotation_matrix_np: np.ndarray) -> torch.Tensor:
    """Rotate SH features up to order 3. shs_feat: (N, 15, 3) on CPU double."""
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix_np @ P

    rotation_matrix_fix = to_so3(torch.from_numpy(permuted_rotation_matrix))
    rot_angles = o3._rotation.matrix_to_angles(rotation_matrix_fix)

    D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2])

    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, "n sh rgb -> n rgb sh")
    one_degree_shs = einsum(D_1, one_degree_shs, "... i j, ... j -> ... i")
    one_degree_shs = einops.rearrange(one_degree_shs, "n rgb sh -> n sh rgb")
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, "n sh rgb -> n rgb sh")
    two_degree_shs = einsum(D_2, two_degree_shs, "... i j, ... j -> ... i")
    two_degree_shs = einops.rearrange(two_degree_shs, "n rgb sh -> n sh rgb")
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, "n sh rgb -> n rgb sh")
    three_degree_shs = einsum(D_3, three_degree_shs, "... i j, ... j -> ... i")
    three_degree_shs = einops.rearrange(three_degree_shs, "n rgb sh -> n sh rgb")
    shs_feat[:, 8:15] = three_degree_shs

    return shs_feat


def transform_gaussians_similarity_inplace(gaussians: GaussianModel, T: np.ndarray, scale: float) -> None:
    """In-place similarity transform: centers, scale, rotation, SH."""
    device = gaussians._xyz.device
    with torch.no_grad():
        ones = torch.ones((gaussians._xyz.shape[0], 1), device=device)
        xyz_h = torch.cat([gaussians._xyz, ones], dim=1)  # (N,4)
        Th = torch.tensor(T, device=device, dtype=torch.float32)
        xyz_h2 = (Th @ xyz_h.t()).t()
        gaussians._xyz = xyz_h2[:, :3]

        gaussians._scaling = gaussians._scaling + float(np.log(scale))

        R_norm = (T[:3, :3] / scale).astype(np.float32)
        R_norm_t = torch.tensor(R_norm, device=device, dtype=torch.float32)

        Rg = rotation_matrix_from_quaternion_wxyz(gaussians.get_rotation)
        Rnew = R_norm_t.unsqueeze(0) @ Rg
        gaussians._rotation = matrix_to_quaternion(Rnew)

        shs = gaussians._features_rest.detach().cpu().double()
        shs = transform_shs(shs, R_norm)
        gaussians._features_rest = shs.float().to(device)


class SimpleCam:
    """Camera container compatible with render_with_depth."""
    def __init__(self, width: int, height: int, fx: float, fy: float, cx: float, cy: float,
                 w2c: torch.Tensor, znear: float = 0.01, zfar: float = 100.0, device: str = "cuda"):
        self.image_width = int(width)
        self.image_height = int(height)
        self.FoVx = float(2 * math.atan(width / (2 * fx)))
        self.FoVy = float(2 * math.atan(height / (2 * fy)))
        self.znear = float(znear)
        self.zfar = float(zfar)

        R = w2c[:3, :3].detach().cpu().numpy()
        t = w2c[:3, 3].detach().cpu().numpy()

        wv = torch.tensor(getWorld2View2(R, t), dtype=torch.float32, device=device).t()
        proj = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
            w=self.image_width,
            h=self.image_height,
            cx=float(cx),
            cy=float(cy),
            allow_principle_point_shift=True,
        ).t().to(device=device)

        self.world_view_transform = wv
        self.full_proj_transform = wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        self.camera_center = torch.inverse(wv)[3, :3]


def w2c_from_pos_quat(pos_w: Union[list, np.ndarray], quat_wxyz: Union[list, np.ndarray], device: str) -> torch.Tensor:
    """Build w2c from camera pose in world (pos + quat wxyz)."""
    pos = torch.tensor(pos_w, dtype=torch.float32, device=device)
    quat = torch.tensor(quat_wxyz, dtype=torch.float32, device=device)
    quat = quat / torch.linalg.norm(quat)
    R_c2w = rotation_matrix_from_quaternion_wxyz(quat.view(1, 4))[0]

    c2w = torch.eye(4, dtype=torch.float32, device=device)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = pos
    w2c = torch.inverse(c2w)
    return w2c


def _fill_defaults(args, device: str) -> None:
    if args.sh_degree is None:
        args.sh_degree = 3
    if args.feature_dim is None:
        args.feature_dim = 32
    if args.init_from_3dgs_pcd is None:
        args.init_from_3dgs_pcd = False
    if args.images is None:
        args.images = "images"
    if args.resolution is None:
        args.resolution = -1
    if args.white_background is None:
        args.white_background = False
    if args.data_device is None:
        args.data_device = device
    if args.eval is None:
        args.eval = False
    if args.need_features is None:
        args.need_features = False
    if args.need_masks is None:
        args.need_masks = False
    if args.allow_principle_point_shift is None:
        args.allow_principle_point_shift = True


@torch.no_grad()
def main():
    intr = {
        "fx": 366.4996337890625,
        "fy": 366.4996337890625,
        "cx": 160.0,
        "cy": 160.0,
        "width": 320,
        "height": 320,
    }
    pos_w = [-0.38128018379211426, 2.0024003982543945, 0.20709329843521118]
    quat_wxyz = [-0.06844878941774368, -0.06373614817857742, 0.6790152788162231, 0.7281900644302368]

    model_path = "/home/ubuntu/xc_isaac/video_data_process/results_corridor/3dgs_output"
    source_path = "/home/ubuntu/xc_isaac/video_data_process/results_corridor/colmap_data_undistorted"
    iteration = 30000
    ply_path = "/home/ubuntu/xc_isaac/video_data_process/results_corridor/3dgs_output/point_cloud/iteration_30000/point_cloud.ply"
    precomputed_mask_path = "/home/ubuntu/xc_isaac/jetauto_rl_navigation/data/blue_bin_mask_from_2d.pt"
    device = "cuda"

    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.set_defaults(model_path=model_path, source_path=source_path)
    args = parser.parse_args([])
    _fill_defaults(args, device)

    dataset = model.extract(args)
    dataset.need_features = False
    dataset.need_masks = False
    dataset.allow_principle_point_shift = True
    pipe = pipeline.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(ply_path)

    is_target = torch.load(precomputed_mask_path, map_location=device)
    is_target = is_target.bool().to(device)
    mask_color = is_target.float().unsqueeze(1).repeat(1, 3)

    s = float(ALIGN_SCALE)
    tx, ty, tz = [float(v) for v in ALIGN_TRANSLATION]
    qx, qy, qz, qw = [float(v) for v in ALIGN_ROT_XYZW]  # xyzw
    q_wxyz = torch.tensor([[qw, qx, qy, qz]], dtype=torch.float32)
    R = rotation_matrix_from_quaternion_wxyz(q_wxyz)[0].cpu().numpy()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = (R * s).astype(np.float32)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    transform_gaussians_similarity_inplace(gaussians, T, s)

    w2c = w2c_from_pos_quat(pos_w, quat_wxyz, device=device)
    cam = SimpleCam(
        width=intr["width"],
        height=intr["height"],
        fx=intr["fx"],
        fy=intr["fy"],
        cx=intr["cx"],
        cy=intr["cy"],
        w2c=w2c,
        device=device,
    )

    bg = torch.zeros(3, device=device)

    occ = render_with_depth(
        cam,
        gaussians,
        pipe,
        bg,
        override_mask=mask_color,
        filtered_mask=None,
    )["mask"]
    occ_rgb = render_with_depth(
        cam,
        gaussians,
        pipe,
        bg,
        override_mask=None,
        filtered_mask=None,
    )["render"]
    if occ.dim() == 3:
        occ = occ[0]
    occ_mask = (occ > 0.5).float()

    unocc = render_with_depth(
        cam,
        gaussians,
        pipe,
        bg,
        override_mask=mask_color,
        filtered_mask=~is_target,
    )["mask"]
    unocc_rgb = render_with_depth(
        cam,
        gaussians,
        pipe,
        bg,
        override_mask=None,
        filtered_mask=~is_target,
    )["render"]
    if unocc.dim() == 3:
        unocc = unocc[0]
    unocc_mask = (unocc > 0.5).float()

    A_full = float(unocc_mask.sum().item())
    A_vis = float(occ_mask.sum().item())
    ratio = 0.0 if A_full <= 0 else (A_vis / (A_full + 1e-6))
    print(f"visible ratio = {ratio:.6f}")

    out_dir = os.path.join(os.path.dirname(__file__), "gs_ratio_test_outputs")
    os.makedirs(out_dir, exist_ok=True)
    occ_path = os.path.join(out_dir, "mask_occ.png")
    unocc_path = os.path.join(out_dir, "mask_no_occ.png")
    occ_rgb_path = os.path.join(out_dir, "rgb_occ.png")
    unocc_rgb_path = os.path.join(out_dir, "rgb_no_occ.png")
    torchvision.utils.save_image(occ_mask, occ_path)
    torchvision.utils.save_image(unocc_mask, unocc_path)
    if occ_rgb.dim() == 3:
        occ_rgb = occ_rgb[0]
    if unocc_rgb.dim() == 3:
        unocc_rgb = unocc_rgb[0]
    torchvision.utils.save_image(occ_rgb, occ_rgb_path)
    torchvision.utils.save_image(unocc_rgb, unocc_rgb_path)
    print(f"saved occ mask   -> {occ_path}")
    print(f"saved unocc mask -> {unocc_path}")
    print(f"saved occ rgb    -> {occ_rgb_path}")
    print(f"saved unocc rgb  -> {unocc_rgb_path}")


if __name__ == "__main__":
    main()
