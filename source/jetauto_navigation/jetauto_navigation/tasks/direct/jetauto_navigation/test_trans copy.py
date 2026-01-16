#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal dst -> src transform for:
  1) one 3D point (x, y, z)
  2) one pose (x, y, z, rx, ry, rz) in XYZ Euler
  3) one pose (x, y, z, qw, qx, qy, qz)

Edit the constants below and run:
  python test_trans.py
"""

import numpy as np

# ===== Fixed transform (src -> dst) =====
SCALE = 0.3664808278887077
# R = np.array(
#     [
#         [0.996101, 0.064665, 0.060011],
#         [-0.058997, 0.994025, -0.091831],
#         [-0.065591, 0.087933, 0.993965],
#     ],
#     dtype=np.float64,
# )

R = np.array(
    [
        [-0.996101, -0.060011, -0.064665],
        [ 0.058997,  0.091831, -0.994025],
        [ 0.065591, -0.993965, -0.087933]
    ],
    dtype=np.float64,
)


T = np.array([0.089711, 0.740089, 0.192696], dtype=np.float64)

# ===== Inputs (dst frame) =====
DEGREES = True
DST_POINT = [-0.38128018379211426, 2.0024003982543945, 0.20709329843521118]
# [0.30, 0.76, 0.37]  # np.array([x, y, z], dtype=np.float64)
DST_POSE_EULER = [-0.38128018379211426, 2.0024003982543945, 0.20709329843521118,0,0,-79.0]
# [0.30, 0.76, 0.37, 0, 0, 0]  # [x, y, z, rx, ry, rz]

# DST_POSE_QUAT =[-0.38128018379211426, 2.0024003982543945, 0.20709329843521118, -0.06844878941774368, -0.06373614817857742, 0.6790152788162231, 0.7281900644302368]
# DST_POSE_QUAT=[-0.38128018379211426, 2.0024003982543945, 0.20709329843521118, 1.0, 0.0, 0.0, 0.0]
DST_POSE_QUAT=[-0.38128018379211426, 2.0024003982543945, 0.20709329843521118, 0.7703935503959656, 0.0, 0.0, -0.6375687122344971]
# [x, y, z, qw, qx, qy, qz]

# Extra rotation applied in the body frame (orientation only; position unchanged).
EXTRA_ROT = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def euler_xyz_to_rotation_matrix(euler_xyz: np.ndarray) -> np.ndarray:
    x, y, z = euler_xyz
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    return rz @ ry @ rx


def rotation_matrix_to_euler_xyz(Rm: np.ndarray) -> np.ndarray:
    Rm = np.asarray(Rm, dtype=np.float64)
    r31 = Rm[2, 0]
    sy = np.clip(-r31, -1.0, 1.0)
    y = np.arcsin(sy)
    cy = np.cos(y)

    if abs(cy) < 1e-8:
        x = 0.0
        z = np.arctan2(-Rm[0, 1], Rm[1, 1])
    else:
        x = np.arctan2(Rm[2, 1] / cy, Rm[2, 2] / cy)
        z = np.arctan2(Rm[1, 0] / cy, Rm[0, 0] / cy)

    return np.array([x, y, z], dtype=np.float64)


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("Quaternion has zero norm.")
    q /= n
    qw, qx, qy, qz = q

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(Rm: np.ndarray) -> np.ndarray:
    m = np.asarray(Rm, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def inverse_transform_point(p_dst: np.ndarray) -> np.ndarray:
    p_dst = np.asarray(p_dst, dtype=np.float64).reshape(3)
    return (1.0 / SCALE) * (R.T @ (p_dst - T))


def inverse_transform_pose(pose_dst_xyzrpy: np.ndarray):
    pose_dst = np.asarray(pose_dst_xyzrpy, dtype=np.float64).reshape(6)
    p_dst = pose_dst[:3]
    euler_dst = pose_dst[3:]
    if DEGREES:
        euler_dst = np.deg2rad(euler_dst)
    R_dst = euler_xyz_to_rotation_matrix(euler_dst)

    p_src = (1.0 / SCALE) * (R.T @ (p_dst - T))
    R_src = R.T @ R_dst
    R_src = R_src @ EXTRA_ROT
    euler_src = rotation_matrix_to_euler_xyz(R_src)
    if DEGREES:
        euler_src = np.degrees(euler_src)
    return np.concatenate([p_src, euler_src])


def inverse_transform_pose_quat(pose_dst_xyzquat: np.ndarray):
    pose_dst = np.asarray(pose_dst_xyzquat, dtype=np.float64).reshape(7)
    p_dst = pose_dst[:3]
    qw, qx, qy, qz = pose_dst[3:]
    R_dst = quaternion_to_rotation_matrix(qw, qx, qy, qz)

    p_src = (1.0 / SCALE) * (R.T @ (p_dst - T))
    R_src = R.T @ R_dst
    R_src = R_src @ EXTRA_ROT
    q_src = rotation_matrix_to_quaternion(R_src)
    return np.concatenate([p_src, q_src])


def main() -> int:
    np.set_printoptions(precision=6, suppress=True)
    did_output = False

    if DST_POINT is not None:
        src_point = inverse_transform_point(DST_POINT)
        print("src_point_xyz:", src_point)
        did_output = True

    if DST_POSE_EULER is not None:
        src_pose = inverse_transform_pose(DST_POSE_EULER)
        print("src_pose_xyzrpy:", src_pose)
        did_output = True

    if DST_POSE_QUAT is not None:
        src_pose = inverse_transform_pose_quat(DST_POSE_QUAT)
        print("src_pose_xyzquat:", src_pose)
        did_output = True

    if not did_output:
        print("No input set: assign DST_POINT or DST_POSE_EULER or DST_POSE_QUAT.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
