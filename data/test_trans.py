#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply the inverse similarity transform (dst -> src) with fixed parameters.

Edit the config section below to set transform parameters and query inputs.
"""
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from coordinate_transform import estimate_similarity_transform, rotation_matrix_to_euler_xyz


# ===== User config (edit here) =====
# Use precomputed src->dst transform, or set to False to estimate from calibration points.
USE_PRECOMPUTED = True
WITH_SCALE = True

# Interpret and print Euler angles in degrees if True, radians if False.
DEGREES = True

# Optional calibration points (used only when USE_PRECOMPUTED is False).
SRC_CALIB_PATH = None  # e.g., "src_pts.txt"
DST_CALIB_PATH = None  # e.g., "dst_pts.txt"
SRC_CALIB_PTS = None  # set to ndarray if not using files
DST_CALIB_PTS = None  # set to ndarray if not using files

# Query inputs (set any of these; leave as None if unused).
# All inputs are in the dst frame, outputs will be in the src frame.
DST_POINTS_PATH = None  # e.g., "query_dst_points.txt"
DST_POINTS = [0.30, 0.76, 0.37]

# Pose as [x, y, z, rx, ry, rz] in XYZ Euler angles (dst frame).
DST_POSE_EULER = None

# Pose matrices (4x4 or 3x4) in dst frame.
DST_POSE_MAT_PATH = None
DST_POSE_MAT = None

# Output controls.
PRINT_TRANSFORM = False
PRINT_CALIB_ERROR = False
PRINT_POINTS = True
PRINT_POSE_MATRIX = True
PRINT_POSE_EULER = True
PRINT_STATUS = False


DEFAULT_SRC_PTS = np.array(
    [
        [0.52469, 0.13751, 0.51037],
        [0.55617, -0.29431, 0.54819],
        [-0.26071, 0.08103, 0.47474],
        [-0.244, -0.36152, 0.49882],
        [0.61107, 0.04952, -0.49264],
        [0.61891, -0.39828, -0.44137],
        [-0.18946, -0.44312, -0.51008],
        [-0.21734, -0.00436, -0.55559],
    ],
    dtype=np.float64,
)

DEFAULT_DST_PTS = np.array(
    [
        [0.30, 0.76, 0.37],
        [0.30, 0.60, 0.37],
        [0.0, 0.76, 0.37],
        [0.0, 0.60, 0.37],
        [0.30, 0.76, 0.0],
        [0.30, 0.60, 0.0],
        [0.0, 0.60, 0.0],
        [0.0, 0.76, 0.0],
    ],
    dtype=np.float64,
)

PRECOMPUTED_SCALE = 0.3664808278887077
PRECOMPUTED_R = np.array(
    [
        [0.996101, 0.064665, 0.060011],
        [-0.058997, 0.994025, -0.091831],
        [-0.065591, 0.087933, 0.993965],
    ],
    dtype=np.float64,
)
PRECOMPUTED_T = np.array([0.089711, 0.740089, 0.192696], dtype=np.float64)


def load_points(path: str) -> np.ndarray:
    pts = np.loadtxt(path, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Points must be Nx3, got {pts.shape} from {path}")
    return pts


def load_pose_matrix(path: str) -> np.ndarray:
    mat = np.loadtxt(path, dtype=np.float64)
    if mat.shape == (3, 4):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        mat = np.vstack([mat, bottom])
    if mat.shape != (4, 4):
        raise ValueError(f"Pose matrix must be 4x4 or 3x4, got {mat.shape} from {path}")
    return mat


def normalize_points(points: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{name} must be Nx3, got {pts.shape}")
    return pts


def normalize_pose_matrix(mat: np.ndarray, name: str) -> np.ndarray:
    pose = np.asarray(mat, dtype=np.float64)
    if pose.shape == (3, 4):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        pose = np.vstack([pose, bottom])
    if pose.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4 or 3x4, got {pose.shape}")
    return pose


def normalize_pose_euler(vals: np.ndarray, name: str) -> np.ndarray:
    pose = np.asarray(vals, dtype=np.float64).reshape(-1)
    if pose.shape != (6,):
        raise ValueError(f"{name} must be 6 values: [x, y, z, rx, ry, rz]")
    return pose


def resolve_points(points: np.ndarray, path: str, name: str) -> np.ndarray:
    if path:
        return load_points(path)
    if points is None:
        return None
    return normalize_points(points, name)


def resolve_pose_matrix(mat: np.ndarray, path: str, name: str) -> np.ndarray:
    if path:
        return load_pose_matrix(path)
    if mat is None:
        return None
    return normalize_pose_matrix(mat, name)


def euler_xyz_to_rotation_matrix(euler_xyz: np.ndarray) -> np.ndarray:
    x, y, z = euler_xyz
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    return rz @ ry @ rx


def transform_points(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (R @ points.T)).T + t


def invert_similarity_transform(scale: float, R: np.ndarray, t: np.ndarray):
    inv_scale = 1.0 / scale
    R_inv = R.T
    t_inv = -inv_scale * (R_inv @ t)
    return inv_scale, R_inv, t_inv


def transform_pose_dst_to_src(pose_dst: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    R_dst = pose_dst[:3, :3]
    p_dst = pose_dst[:3, 3]
    R_src = R.T @ R_dst
    p_src = (1.0 / scale) * (R.T @ (p_dst - t))
    pose_src = np.eye(4, dtype=np.float64)
    pose_src[:3, :3] = R_src
    pose_src[:3, 3] = p_src
    return pose_src


def build_homogeneous(scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


def main() -> int:
    src_calib = None
    dst_calib = None
    if SRC_CALIB_PATH or DST_CALIB_PATH:
        if not (SRC_CALIB_PATH and DST_CALIB_PATH):
            raise ValueError("SRC_CALIB_PATH and DST_CALIB_PATH must be set together.")
        src_calib = load_points(SRC_CALIB_PATH)
        dst_calib = load_points(DST_CALIB_PATH)
    elif SRC_CALIB_PTS is not None or DST_CALIB_PTS is not None:
        if SRC_CALIB_PTS is None or DST_CALIB_PTS is None:
            raise ValueError("SRC_CALIB_PTS and DST_CALIB_PTS must be set together.")
        src_calib = normalize_points(SRC_CALIB_PTS, "SRC_CALIB_PTS")
        dst_calib = normalize_points(DST_CALIB_PTS, "DST_CALIB_PTS")

    if USE_PRECOMPUTED:
        scale = PRECOMPUTED_SCALE
        R = PRECOMPUTED_R.copy()
        t = PRECOMPUTED_T.copy()
        transform_source = "precomputed"
    else:
        if src_calib is None or dst_calib is None:
            src_calib = DEFAULT_SRC_PTS
            dst_calib = DEFAULT_DST_PTS
        scale, R, t, _ = estimate_similarity_transform(src_calib, dst_calib, with_scale=WITH_SCALE)
        transform_source = "estimated"

    inv_scale, inv_R, inv_t = invert_similarity_transform(scale, R, t)

    unit = "deg" if DEGREES else "rad"
    np.set_printoptions(precision=6, suppress=True)

    if PRINT_TRANSFORM:
        print(f"Transform source (src -> dst): {transform_source}")
        print("Scale (dst -> src):", inv_scale)
        print("Rotation R (dst -> src):\n", inv_R)
        print("Translation t (dst -> src):\n", inv_t)
        print("Homogeneous transform T (dst -> src):\n", build_homogeneous(inv_scale, inv_R, inv_t))
        euler_rad = rotation_matrix_to_euler_xyz(inv_R)
        euler_out = np.degrees(euler_rad) if DEGREES else euler_rad
        print(f"Euler XYZ ({unit}) (dst -> src):\n", euler_out)

    if PRINT_CALIB_ERROR and src_calib is not None and dst_calib is not None:
        recon = transform_points(dst_calib, inv_scale, inv_R, inv_t)
        err = np.linalg.norm(recon - src_calib, axis=1)
        print("Calibration per-point error:", err)
        print("Calibration mean error:", err.mean())

    did_query = False

    dst_points = resolve_points(DST_POINTS, DST_POINTS_PATH, "DST_POINTS")
    dst_pose_mat = resolve_pose_matrix(DST_POSE_MAT, DST_POSE_MAT_PATH, "DST_POSE_MAT")

    if dst_points is not None:
        points_out = transform_points(dst_points, inv_scale, inv_R, inv_t)
        if PRINT_POINTS:
            print("Transformed points (src frame):\n", points_out)
        did_query = True

    if DST_POSE_EULER is not None:
        pose_vals = normalize_pose_euler(DST_POSE_EULER, "DST_POSE_EULER")
        pos = pose_vals[:3]
        euler_in = pose_vals[3:]
        if DEGREES:
            euler_in = np.deg2rad(euler_in)
        R_dst = euler_xyz_to_rotation_matrix(euler_in)
        pose_dst = np.eye(4, dtype=np.float64)
        pose_dst[:3, :3] = R_dst
        pose_dst[:3, 3] = pos
        pose_src = transform_pose_dst_to_src(pose_dst, scale, R, t)
        if PRINT_POSE_MATRIX:
            print("Transformed pose matrix (src frame):\n", pose_src)
        if PRINT_POSE_EULER:
            euler_src = rotation_matrix_to_euler_xyz(pose_src[:3, :3])
            euler_src_out = np.degrees(euler_src) if DEGREES else euler_src
            print(f"Transformed pose Euler XYZ ({unit}):\n", euler_src_out)
        did_query = True

    if dst_pose_mat is not None:
        pose_dst = dst_pose_mat
        pose_src = transform_pose_dst_to_src(pose_dst, scale, R, t)
        if PRINT_POSE_MATRIX:
            print("Transformed pose matrix (src frame):\n", pose_src)
        if PRINT_POSE_EULER:
            euler_src = rotation_matrix_to_euler_xyz(pose_src[:3, :3])
            euler_src_out = np.degrees(euler_src) if DEGREES else euler_src
            print(f"Transformed pose Euler XYZ ({unit}):\n", euler_src_out)
        did_query = True

    if PRINT_STATUS and not did_query:
        if src_calib is not None and dst_calib is not None:
            print("No query points/pose provided; only calibration error printed.")
        else:
            print("No query points/pose provided.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
