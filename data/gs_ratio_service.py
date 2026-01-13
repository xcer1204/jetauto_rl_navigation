# gs_ratio_service.py
import math
import threading
from typing import Any, Dict, Union

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
from gaussian_renderer import render_with_depth
from scene import Scene, GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# ---- For rotating per-Gaussian rotations + SHs (full similarity transform)
import einops
from einops import einsum
from e3nn import o3
from pytorch3d.transforms import matrix_to_quaternion


# =========================
# 1) Put YOUR Isaac ALIGN here (same as env)
# =========================
ALIGN_SCALE = 0.3664808278887077
ALIGN_ROT_XYZW = (0.04309, 0.03407, -0.02809, 0.99810)  # (x,y,z,w)
ALIGN_TRANSLATION = (0.089711, 0.740089, 0.192696)      # (tx,ty,tz)


# =========================
# 2) Helpers
# =========================
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
    # switch axes: yzx -> xyz (same as your earlier code)
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix_np @ P

    rotation_matrix_fix = to_so3(torch.from_numpy(permuted_rotation_matrix))
    rot_angles = o3._rotation.matrix_to_angles(rotation_matrix_fix)

    D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2])

    # order-1
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, "n sh rgb -> n rgb sh")
    one_degree_shs = einsum(D_1, one_degree_shs, "... i j, ... j -> ... i")
    one_degree_shs = einops.rearrange(one_degree_shs, "n rgb sh -> n sh rgb")
    shs_feat[:, 0:3] = one_degree_shs

    # order-2
    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, "n sh rgb -> n rgb sh")
    two_degree_shs = einsum(D_2, two_degree_shs, "... i j, ... j -> ... i")
    two_degree_shs = einops.rearrange(two_degree_shs, "n rgb sh -> n sh rgb")
    shs_feat[:, 3:8] = two_degree_shs

    # order-3
    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, "n sh rgb -> n rgb sh")
    three_degree_shs = einsum(D_3, three_degree_shs, "... i j, ... j -> ... i")
    three_degree_shs = einops.rearrange(three_degree_shs, "n rgb sh -> n sh rgb")
    shs_feat[:, 8:15] = three_degree_shs

    return shs_feat


def transform_gaussians_similarity_inplace(gaussians: GaussianModel, T: np.ndarray, scale: float) -> None:
    """
    In-place similarity transform:
      X <- T * [X;1] , scaling += log(scale), rotation <- R * rotation, SH rotated.
    T is 4x4, with rotation*scale in top-left.
    """
    device = gaussians._xyz.device

    with torch.no_grad():
        # 1) centers
        ones = torch.ones((gaussians._xyz.shape[0], 1), device=device)
        xyz_h = torch.cat([gaussians._xyz, ones], dim=1)  # (N,4)
        Th = torch.tensor(T, device=device, dtype=torch.float32)
        xyz_h2 = (Th @ xyz_h.t()).t()
        gaussians._xyz = xyz_h2[:, :3]

        # 2) isotropic scaling is stored log-space
        gaussians._scaling = gaussians._scaling + float(np.log(scale))

        # 3) rotations
        R_norm = (T[:3, :3] / scale).astype(np.float32)   # pure rotation
        R_norm_t = torch.tensor(R_norm, device=device, dtype=torch.float32)  # (3,3)

        # gaussians.get_rotation: (N,4) wxyz -> (N,3,3)
        Rg = rotation_matrix_from_quaternion_wxyz(gaussians.get_rotation)
        Rnew = R_norm_t.unsqueeze(0) @ Rg
        gaussians._rotation = matrix_to_quaternion(Rnew)  # (N,4) wxyz

        # 4) rotate SH features (not strictly needed for mask-only override, but correct)
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


def _broadcast_intr(x: Union[float, int, list, np.ndarray], N: int) -> np.ndarray:
    if isinstance(x, (float, int)):
        return np.full((N,), float(x), dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        return np.full((N,), float(arr.item()), dtype=np.float32)
    if arr.size != N:
        raise ValueError(f"Intrinsic array length mismatch: got {arr.size}, expected {N}")
    return arr


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
        _ = Scene(
            dataset,
            gaussians=self.gaussians,
            load_iteration=iteration,
            shuffle=False,
            mode="eval",
            target="scene",
            sample_rate=1.0,
        )

        # Load is_target (N,)
        is_target = torch.load(precomputed_mask_path, map_location=device)
        self.is_target = is_target.bool().to(device)
        self.mask_color = self.is_target.float().unsqueeze(1).repeat(1, 3)  # (Ng,3)

        # --------- APPLY ALIGN ONCE: G -> I ----------
        self._apply_align_once()

    def _apply_align_once(self) -> None:
        s = float(ALIGN_SCALE)
        tx, ty, tz = [float(v) for v in ALIGN_TRANSLATION]
        qx, qy, qz, qw = [float(v) for v in ALIGN_ROT_XYZW]  # xyzw

        # quat -> R (wxyz batch of 1)
        q_wxyz = torch.tensor([[qw, qx, qy, qz]], dtype=torch.float32)
        R = rotation_matrix_from_quaternion_wxyz(q_wxyz)[0].cpu().numpy()  # (3,3)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = (R * s).astype(np.float32)
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)

        # In-place transform on CUDA
        transform_gaussians_similarity_inplace(self.gaussians, T, s)

    @torch.no_grad()
    def visible_ratio(self, w2c_list: Any, intr: Dict[str, Any], znear: float = 0.01, zfar: float = 100.0, thr: float = 0.5) -> np.ndarray:
        """
        w2c_list: list/np array of shape (N,4,4) in ISAAC world coordinates (I)
        intr: dict keys width,height,fx,fy,cx,cy (each can be scalar or length-N array)
        return: np.ndarray (N,) ratios
        """
        with self._lock:
            w2c = torch.tensor(w2c_list, dtype=torch.float32, device=self.device)
            if w2c.dim() == 2:
                w2c = w2c.unsqueeze(0)
            N = w2c.shape[0]

            width  = _broadcast_intr(intr["width"],  N)
            height = _broadcast_intr(intr["height"], N)
            fx     = _broadcast_intr(intr["fx"],     N)
            fy     = _broadcast_intr(intr["fy"],     N)
            cx     = _broadcast_intr(intr["cx"],     N)
            cy     = _broadcast_intr(intr["cy"],     N)

            ratios = np.zeros((N,), dtype=np.float32)

            for i in range(N):
                cam = SimpleCam(
                    int(width[i]), int(height[i]),
                    float(fx[i]), float(fy[i]),
                    float(cx[i]), float(cy[i]),
                    w2c[i],
                    znear=znear, zfar=zfar, device=self.device,
                )

                occ = render_with_depth(
                    cam, self.gaussians, self.pipe, self.bg,
                    override_mask=self.mask_color,
                    filtered_mask=None,
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

                A_full = float(unocc.sum().item())
                A_vis  = float(occ.sum().item())
                ratios[i] = 0.0 if A_full <= 0 else (A_vis / (A_full + 1e-6))

            return ratios


# =========================
# 4) rpyc service
# =========================
class RatioService(rpyc.Service):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._n_calls = 0

    def exposed_visible_ratio(self, w2c_list, intr):
        self._n_calls += 1
        ratios = self.engine.visible_ratio(w2c_list, intr)

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
