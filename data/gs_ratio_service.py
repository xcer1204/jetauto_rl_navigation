# gs_ratio_service.py
import math
import os
import threading
from typing import Any, Dict, Optional, Union, Tuple

import numpy as np
import torch
import rpyc
from rpyc.utils.server import ThreadedServer
import sys
# Make SAGA code importable
SAGA_ROOT = "/home/zgao/SegAnyGAussians"
if SAGA_ROOT not in sys.path:
    sys.path.append(SAGA_ROOT)
# ---- SAGA / 3DGS code imports (same as your first script)
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render_mask, render_with_depth
from scene import Scene, GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# ---- For rotating per-Gaussian rotations + SHs (full similarity transform)
 

# Fixed camera intrinsics; edit if needed.
FIXED_FX = 366.4996337890625
FIXED_FY = 366.4996337890625
FIXED_CX = 160.0
FIXED_CY = 160.0
FIXED_WIDTH = 320
FIXED_HEIGHT = 320

# ===== Isaac -> 3DGS conversion (same as gs_ratio_test_3dgs_pose.py)
SCALE = 0.3664808278887077
R = np.array(
    [
        [-0.996101, -0.060011, -0.064665],
        [ 0.058997,  0.091831, -0.994025],
        [ 0.065591, -0.993965, -0.087933]
    ],
    dtype=np.float64,
)
T = np.array([0.089711, 0.740089, 0.192696], dtype=np.float64)
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

    Rm = torch.stack(
        [
            torch.stack([1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q3 * q0, 2 * q1 * q3 + 2 * q2 * q0], dim=1),
            torch.stack([2 * q1 * q2 + 2 * q3 * q0, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q1 * q0], dim=1),
            torch.stack([2 * q1 * q3 - 2 * q2 * q0, 2 * q2 * q3 + 2 * q1 * q0, 1 - 2 * q1 * q1 - 2 * q2 * q2], dim=1),
        ],
        dim=1,
    )
    return Rm

def w2c_from_pos_rot(pos_w: Union[list, np.ndarray, torch.Tensor],
                     rot_c2w: Union[list, np.ndarray, torch.Tensor],
                     device: str) -> torch.Tensor:
    """Build w2c from camera pose in world (pos + rotation matrix)."""
    pos = torch.as_tensor(pos_w, dtype=torch.float32, device=device)
    rot = torch.as_tensor(rot_c2w, dtype=torch.float32, device=device)

    c2w = torch.eye(4, dtype=torch.float32, device=device)
    c2w[:3, :3] = rot
    c2w[:3, 3] = pos

    w2c = torch.inverse(c2w)
    return w2c

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
    return pos_w, R_w


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


# =========================
# 3) Engine: load once + ALIGN once + per-call ratio
# =========================
class GSVisibilityEngine:
    def __init__(self, model_path: str, source_path: str, iteration: int, precomputed_mask_path: str, device: str = "cuda"):
        self.device = device
        self.bg = torch.zeros(3, device=device)
        self._lock = threading.Lock()  # avoid concurrent CUDA rendering issues

        # Build args-like dataset/pipe
        import argparse
        parser = argparse.ArgumentParser()
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        parser.set_defaults(model_path=model_path, source_path=source_path)
        args = parser.parse_args([])
        # ModelParams with sentinel=True yields None defaults when no CLI/config is used.
        # Fill the minimal defaults needed by Scene/camera loading.
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

        dataset = model.extract(args)
        dataset.need_features = False
        dataset.need_masks = False
        dataset.allow_principle_point_shift = True
        self.pipe = pipeline.extract(args)

        # Load Gaussians (G world)
        self.gaussians = GaussianModel(dataset.sh_degree)
        has_colmap = os.path.exists(os.path.join(dataset.source_path, "sparse"))
        has_blender = os.path.exists(os.path.join(dataset.source_path, "transforms_train.json"))
        if has_colmap or has_blender:
            _ = Scene(
                dataset,
                gaussians=self.gaussians,
                load_iteration=iteration,
                shuffle=False,
                mode="eval",
                target="scene",
                sample_rate=1.0,
            )
        else:
            ply_path = os.path.join(
                model_path,
                "point_cloud",
                "iteration_" + str(iteration),
                "scene_point_cloud.ply",
            )
            if not os.path.exists(ply_path):
                raise FileNotFoundError(
                    "Missing scene data and point cloud.\n"
                    f"- source_path: {dataset.source_path} (no 'sparse' or 'transforms_train.json')\n"
                    f"- expected ply: {ply_path}\n"
                    "Please set source_path to a COLMAP/Blender dataset root or update model_path/iteration."
                )
            print(
                "Warning: source_path has no 'sparse' or 'transforms_train.json'; "
                "loading Gaussians directly from point cloud.",
                flush=True,
            )
            self.gaussians.load_ply(ply_path)

        # Load is_target (N,)
        is_target = torch.load(precomputed_mask_path, map_location=device)
        self.is_target = is_target.bool().to(device)
        self.mask_color = self.is_target.float()  # (Ng,)

    def _resolve_intrinsics(self, intr: Optional[Dict[str, Union[int, float]]]) -> Dict[str, float]:
        return {
            "width": float(FIXED_WIDTH),
            "height": float(FIXED_HEIGHT),
            "fx": float(FIXED_FX),
            "fy": float(FIXED_FY),
            "cx": float(FIXED_CX),
            "cy": float(FIXED_CY),
        }

    @torch.no_grad()
    def visible_ratio(
        self,
        poses_isaac: Any,
        znear: float = 0.01,
        zfar: float = 100.0,
        thr: float = 0.7,
    ) -> np.ndarray:
        """
        poses_isaac: list/np array of shape (N,7) = [px,py,pz,qw,qx,qy,qz] in Isaac world coordinates
        return: np.ndarray (N,) ratios
        """
        with self._lock:
            pose_np = np.asarray(poses_isaac, dtype=np.float32)
            if pose_np.ndim == 1 and pose_np.shape[0] == 7:
                pose_np = pose_np.reshape(1, 7)
            if pose_np.ndim == 0:
                raise ValueError(f"poses_isaac must be (N,7) or (7,), got scalar {pose_np}")
            if pose_np.ndim != 2 or pose_np.shape[1] != 7:
                raise ValueError(f"poses_isaac must be (N,7) or (7,), got shape {pose_np.shape}")
            pose = torch.as_tensor(pose_np, dtype=torch.float32, device=self.device)
            N = pose.shape[0]

            intr_vals = self._resolve_intrinsics(None)
            cam_w = int(round(intr_vals["width"]))
            cam_h = int(round(intr_vals["height"]))
            cam_fx = float(intr_vals["fx"])
            cam_fy = float(intr_vals["fy"])
            cam_cx = float(intr_vals["cx"])
            cam_cy = float(intr_vals["cy"])

            ratios = np.zeros((N,), dtype=np.float32)

            for i in range(N):
                pos_isaac = pose[i, 0:3]
                quat_wxyz = pose[i, 3:7]
                pos_w, R_c2w = isaac_pose_to_world_pos_rot(pos_isaac, quat_wxyz, device=self.device)
                w2c = w2c_from_pos_rot(pos_w, R_c2w, device=self.device)
                cam = SimpleCam(
                    cam_w, cam_h,
                    cam_fx, cam_fy,
                    cam_cx, cam_cy,
                    w2c,
                    znear=znear, zfar=zfar, device=self.device,
                )

                occ = render_mask(
                    cam, self.gaussians, self.pipe, self.bg,
                    precomputed_mask=self.mask_color,
                )["mask"]
                if occ.dim() == 3:
                    occ = occ[0]
                occ = (occ > thr).float()

                unocc = render_with_depth(
                    cam, self.gaussians, self.pipe, self.bg,
                    override_mask=self.mask_color,
                    filtered_mask=~self.is_target,
                )["mask"]
                if unocc.dim() == 3:
                    unocc = unocc[0]
                unocc = (unocc > thr).float()

                # If the full mask touches the image border, treat as out-of-view
                if (
                    (unocc[0, :].any())
                    or (unocc[-1, :].any())
                    or (unocc[:, 0].any())
                    or (unocc[:, -1].any())
                ):
                    ratios[i] = 0.0
                    continue

                A_full = float(unocc.sum().item())
                A_vis = float(occ.sum().item())
                ratio = 0.0 if A_full <= 0 else (A_vis / (A_full + 1e-6))
                ratios[i] = float(max(0.0, min(1.0, ratio)))

            return ratios


# =========================
# 4) rpyc service
# =========================
class RatioService(rpyc.Service):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._n_calls = 0

    def on_connect(self, conn):
        print("[GS-RPC] client connected", flush=True)

    def on_disconnect(self, conn):
        print("[GS-RPC] client disconnected", flush=True)

    def exposed_visible_ratio(self, poses_isaac):
        self._n_calls += 1
        ratios = self.engine.visible_ratio(poses_isaac)

        # 每次调用都打印一行（确认通信）
        n = len(ratios) if hasattr(ratios, "__len__") else 1
        r0 = float(ratios[0]) if n > 0 else float(ratios)
        mean_ratio = float(np.mean(ratios)) if n > 0 else float(ratios)
        print(
            f"[GS-RPC] call={self._n_calls} views={n} ratio0={r0:.4f} mean={mean_ratio:.4f}",
            flush=True,
        )

        return ratios

def main():
    engine = GSVisibilityEngine(
        model_path="/home/zgao/video_data_process/results_corridor/3dgs_output",
        source_path="/home/zgao/video_data_process/results_corridor/colmap_data_undistorted",
        iteration=30000,
        precomputed_mask_path="/home/zgao/jetauto_rl_navigation/data/blue_bin_mask_from_2d.pt",
        device="cuda",
    )
    server = ThreadedServer(
        RatioService(engine),
        port=18862,
        protocol_config={"allow_pickle": True, "allow_public_attrs": True},
    )
    print("GS ratio service running on port 18862 (returns ratio only).")
    server.start()


if __name__ == "__main__":
    main()
