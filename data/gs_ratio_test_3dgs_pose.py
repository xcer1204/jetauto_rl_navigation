# gs_ratio_test.py
import argparse
import math
import os
from typing import Tuple, Union

import numpy as np
import torch
import torchvision
import sys
# Make SAGA code importable
SAGA_ROOT = "/home/zgao/SegAnyGAussians"
if SAGA_ROOT not in sys.path:
    sys.path.append(SAGA_ROOT)
    
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render_mask, render_with_depth
from scene import Scene, GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

import einops
from einops import einsum
from e3nn import o3







# ===== 从isaac lab到3dgs .ply的固定变换 =====
SCALE = 0.3664808278887077
# 由于转换后的usd在isaac sim里和3dgs的坐标轴方向不一样，旋转矩阵里的所有元素变负，并且交换yz轴(第23列互换)
R = np.array(
    [
        [-0.996101, -0.060011, -0.064665],
        [ 0.058997,  0.091831, -0.994025],
        [ 0.065591, -0.993965, -0.087933]
    ],
    dtype=np.float64,
)
T = np.array([0.089711, 0.740089, 0.192696], dtype=np.float64)

# 摄像机对于机器人的3d位移(假定默认朝向与机器人一致, x向前),需要转成z向前。
# 这里需要一个旋转矩阵，相当于绕自己的坐标轴旋转
EXTRA_ROT = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


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


def quaternion_wxyz_from_euler_xyz(euler_xyz: torch.Tensor) -> torch.Tensor:
    """(N,3) xyz (rad), R = Rz(z) @ Ry(y) @ Rx(x) -> (N,4) wxyz."""
    x = euler_xyz[:, 0]
    y = euler_xyz[:, 1]
    z = euler_xyz[:, 2]
    cr = torch.cos(x * 0.5)
    sr = torch.sin(x * 0.5)
    cp = torch.cos(y * 0.5)
    sp = torch.sin(y * 0.5)
    cy = torch.cos(z * 0.5)
    sy = torch.sin(z * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return torch.stack([qw, qx, qy, qz], dim=1)


def rotation_matrix_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """(N,3,3) rotation -> (N,3) xyz Euler (rad), R = Rz(z) @ Ry(y) @ Rx(x)."""
    r31 = R[:, 2, 0]
    sy = (-r31).clamp(-1.0, 1.0)
    y = torch.asin(sy)
    cy = torch.cos(y)
    near_gimbal = torch.abs(cy) < 1e-8

    x = torch.atan2(R[:, 2, 1] / cy, R[:, 2, 2] / cy)
    z = torch.atan2(R[:, 1, 0] / cy, R[:, 0, 0] / cy)

    x = torch.where(near_gimbal, torch.zeros_like(x), x)
    z = torch.where(near_gimbal, torch.atan2(-R[:, 0, 1], R[:, 1, 1]), z)
    return torch.stack([x, y, z], dim=1)


def euler_xyz_from_quaternion_wxyz(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz -> (N,3) xyz Euler (rad), R = Rz(z) @ Ry(y) @ Rx(x)."""
    R = rotation_matrix_from_quaternion_wxyz(quaternion_wxyz)
    return rotation_matrix_to_euler_xyz(R)


def to_so3(R: torch.Tensor) -> torch.Tensor:
    """Project to nearest proper rotation matrix."""
    U, _, Vt = torch.linalg.svd(R)
    R_orth = U @ Vt
    if torch.det(R_orth) < 0:
        U[..., -1] *= -1
        R_orth = U @ Vt
    return R_orth


def rotation_matrix_to_quaternion_wxyz(Rm: torch.Tensor) -> torch.Tensor:
    """(3,3) rotation -> (4,) wxyz. Matches the old numpy implementation."""
    m = Rm
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    trace_val = float(trace.item())
    if trace_val > 0.0:
        s = torch.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0].item() > m[1, 1].item() and m[0, 0].item() > m[2, 2].item():
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1].item() > m[2, 2].item():
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    q = torch.stack([qw, qx, qy, qz], dim=0)
    q = q / torch.linalg.norm(q)
    if q[0].item() < 0.0:
        q = -q
    return q


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


def w2c_from_pos_rot(pos_w: Union[list, np.ndarray, torch.Tensor],
                     rot_c2w: Union[list, np.ndarray, torch.Tensor],
                     device: str) -> torch.Tensor:
    """Build w2c from camera pose in world (pos + rotation matrix)."""
    pos = torch.as_tensor(pos_w, dtype=torch.float32, device=device)
    rot = torch.as_tensor(rot_c2w, dtype=torch.float32, device=device)

    c2w = torch.eye(4, dtype=torch.float32, device=device)
    c2w[:3, :3] = rot
    c2w[:3, 3] = pos

    # w2c = c2w
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


def isaac_pose_to_world_pos_rot(pos_isaac: Union[list, np.ndarray, torch.Tensor],
                                quat_isaac_wxyz: Union[list, np.ndarray, torch.Tensor],
                                device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Isaac pose to 3DGS world pos + c2w rotation matrix."""
    pos = torch.as_tensor(pos_isaac, dtype=torch.float32, device=device)
    quat = torch.as_tensor(quat_isaac_wxyz, dtype=torch.float32, device=device)
    quat = quat / torch.linalg.norm(quat)

    R_isaac = rotation_matrix_from_quaternion_wxyz(quat.view(1, 4))[0]
    R_t = torch.as_tensor(R, dtype=torch.float32, device=device)
    T_t = torch.as_tensor(T, dtype=torch.float32, device=device)
    extra_rot_t = torch.as_tensor(EXTRA_ROT, dtype=torch.float32, device=device)

    pos_w = (R_t.T @ (pos - T_t)) / float(SCALE)
    R_w = R_t.T @ R_isaac @ extra_rot_t

    # quat_w = rotation_matrix_to_quaternion_wxyz(R_w)
    # R_w = rotation_matrix_from_quaternion_wxyz(quat_w.view(1, 4))[0]
    # print("R_c2w:\n", R_w.cpu().numpy())
    return pos_w, R_w


@torch.no_grad()
def main():

    model_path = "/home/zgao/video_data_process/results_corridor/3dgs_output"
    source_path = "/home/zgao/video_data_process/results_corridor/colmap_data_undistorted"
    iteration = 30000
    ply_path = "/home/zgao/video_data_process/results_corridor/3dgs_output/point_cloud/iteration_30000/scene_point_cloud.ply"
    precomputed_mask_path = "/home/zgao/jetauto_rl_navigation/data/blue_bin_mask_from_2d.pt"
    device = "cuda"
    # in isaaclab actually used intrinsics
    intr = {
        "fx": 366.4996337890625,   
        "fy": 366.4996337890625,
        "cx": 160.0,
        "cy": 160.0,
        "width": 320,
        "height": 320,
    }

    # in Isaac Sim
    # intr = {
    #     "fx": 731.78788,   
    #     "fy": 731.78788,
    #     "cx": 970.94244,
    #     "cy": 600.37482,
    #     "width": 1936,
    #     "height": 1216,
    # }

    dst_pts = np.array([[0.30, 0.76, 0.37],
                        [0.30, 0.60, 0.37],
                        [0.0, 0.76, 0.37],
                        [0.0, 0.60, 0.37],
                        [0.30, 0.76, 0.0],
                        [0.30, 0.60, 0.0],
                        [0.0, 0.60, 0.0],
                        [0.0, 0.76, 0.0]], dtype=np.float64)


    # poc_isaac=[1.0910989046096802, -0.5693130493164062, 0.20709329843521118] 
    # qoc_isaac_wxyz=[0.23803524672985077, 0.008312370628118515, 0.033895134925842285, 0.9706293940544128]

    # poc_isaac=[0.19285228848457336, 2.000744342803955, 0.20709329843521118] 
    # qoc_isaac_wxyz=[0.6719053983688354, 0.023463435471057892, -0.02583491802215576, -0.739814281463623]

    # poc_isaac=[-0.38128015398979187, 2.0024003982543945, 0.20709329843521118] 
    # qoc_isaac_wxyz=[0.7699242830276489, 0.026886362582445145, -0.022250831127166748, -0.6371803283691406]

    # poc_isaac=[0.18570639193058014, -1.201640248298645, 0.20709329843521118] 
    # qoc_isaac_wxyz=[0.650500476360321, 0.02271595038473606, 0.026494503021240234, 0.7587037086486816]

    poc_isaac=[-7.963180541992188e-05, 1.3711419105529785, 0.20709329843521118] 
    qoc_isaac_wxyz=[0.7066760659217834, 0.02467767521739006, -0.024677634239196777, -0.7066760659217834]

    pos_w, R_c2w = isaac_pose_to_world_pos_rot(poc_isaac, qoc_isaac_wxyz, device=device)


    # quat_wxyz_test = (-0.5169855943050016, -0.48200393793554497, 0.4824559007372416, 0.5173339375486095)
    # euler_test = euler_xyz_from_quaternion_wxyz(torch.tensor(quat_wxyz_test).view(1,4))
    # print("test euler (deg) =", euler_test.cpu().numpy() * 180.0 / math.pi)

    # euler_test_angles = [ -4.,0.0, 0]  # in degrees
    # euler_test_angles = [  0. ,-4.0, 0]  # in degrees
    # euler_test_rad = torch.tensor(euler_test_angles, dtype=torch.float32).view(1, 3) * (math.pi / 180.0)
    # quat_wxyz_test = quaternion_wxyz_from_euler_xyz(euler_test_rad)[0].cpu().numpy().tolist()
    # print("Using quat_wxyz_test =", quat_wxyz_test) 



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
    mask_color = is_target.float()

    w2c = w2c_from_pos_rot(pos_w, R_c2w, device=device)

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

    occ = render_mask(
        cam,
        gaussians,
        pipe,
        bg,
        precomputed_mask=mask_color,
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
    occ_mask = (occ > 0.7).float()

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
    unocc_mask = (unocc > 0.7).float()

    A_full = float(unocc_mask.sum().item())
    A_vis = float(occ_mask.sum().item())
    ratio = 0.0 if A_full <= 0 else (A_vis / (A_full + 1e-6))
    ratio = float(max(0.0, min(1.0, ratio)))
    print(f"visible ratio = {ratio:.6f}")

    out_dir = os.path.join(os.path.dirname(__file__), "gs_ratio_test_outputs")
    os.makedirs(out_dir, exist_ok=True)
    occ_path = os.path.join(out_dir, "mask_occ.png")
    unocc_path = os.path.join(out_dir, "mask_no_occ.png")
    occ_rgb_path = os.path.join(out_dir, "rgb_occ.png")
    unocc_rgb_path = os.path.join(out_dir, "rgb_no_occ.png")
    if occ_rgb.dim() == 3:
        occ_rgb = occ_rgb[0]
    if unocc_rgb.dim() == 3:
        unocc_rgb = unocc_rgb[0]
    occ_mask = torch.flip(occ_mask, dims=[-1])
    unocc_mask = torch.flip(unocc_mask, dims=[-1])
    occ_rgb = torch.flip(occ_rgb, dims=[-1])
    unocc_rgb = torch.flip(unocc_rgb, dims=[-1])
    torchvision.utils.save_image(occ_mask, occ_path)
    torchvision.utils.save_image(unocc_mask, unocc_path)
    torchvision.utils.save_image(occ_rgb, occ_rgb_path)
    torchvision.utils.save_image(unocc_rgb, unocc_rgb_path)
    print(f"saved occ mask   -> {occ_path}")
    print(f"saved unocc mask -> {unocc_path}")
    print(f"saved occ rgb    -> {occ_rgb_path}")
    print(f"saved unocc rgb  -> {unocc_rgb_path}")


if __name__ == "__main__":
    main()
