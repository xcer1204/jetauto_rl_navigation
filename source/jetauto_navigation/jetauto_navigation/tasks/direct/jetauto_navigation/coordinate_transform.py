#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
box_align.py

给定同一组 3D 点在两个坐标系中的坐标（至少 3 个点，支持 8 个长方体顶点），
利用 Umeyama 算法求相似变换:

    x_dst ≈ s * R @ x_src + t

返回：
    - scale:     float，统一缩放因子 s
    - R:         (3,3) 旋转矩阵
    - t:         (3,)  平移向量
    - T:         (4,4) 齐次变换矩阵，满足 [x_dst;1] ≈ T @ [x_src;1]

使用方式示例见 main()。
"""

import numpy as np


def estimate_similarity_transform(src_pts: np.ndarray,
                                  dst_pts: np.ndarray,
                                  with_scale: bool = True):
    """
    使用 Umeyama 方法估计 3D 相似变换。

    Args:
        src_pts: (N,3) 源坐标系下的点，float
        dst_pts: (N,3) 目标坐标系下的点，float
        with_scale: 是否估计统一缩放（True=相似变换，False=刚体变换）

    Returns:
        scale: float，缩放因子 s
        R: (3,3) ndarray，旋转矩阵
        t: (3,) ndarray，平移向量
        T: (4,4) ndarray，齐次变换矩阵
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)

    assert src_pts.shape == dst_pts.shape, "src_pts 和 dst_pts 形状必须一致"
    assert src_pts.ndim == 2 and src_pts.shape[1] == 3, "点必须是 (N,3) 形式"
    n = src_pts.shape[0]
    assert n >= 3, "至少需要 3 个非共线点"

    # 1. 计算质心
    mu_src = src_pts.mean(axis=0)
    mu_dst = dst_pts.mean(axis=0)

    # 2. 去中心化
    X = src_pts - mu_src
    Y = dst_pts - mu_dst

    # 3. 协方差矩阵
    #    注意这里除以 n，是 Umeyama 原文中的定义
    Sigma = (Y.T @ X) / n  # (3,3)

    # 4. SVD 分解
    U, D, Vt = np.linalg.svd(Sigma)

    # 5. 处理可能的反射，保证 R 为旋转矩阵（det(R) = +1）
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt  # (3,3)

    # 6. 计算缩放
    if with_scale:
        var_src = (X ** 2).sum() / n  # 源点的方差
        # trace(Diag(D) @ S) = sum(D_i * S_ii)
        scale = (D * np.diag(S)).sum() / var_src
    else:
        scale = 1.0

    # 7. 计算平移 t
    t = mu_dst - scale * (R @ mu_src)

    # 8. 组装 4x4 齐次矩阵
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    return scale, R, t, T


# ===== 新增函数：从旋转矩阵求 XYZ 欧拉角（弧度） =====
def rotation_matrix_to_euler_xyz(R: np.ndarray):
    """
    从 3x3 旋转矩阵提取 XYZ 欧拉角（单位：弧度）

    假设旋转顺序为：
        R = Rz(z) @ Ry(y) @ Rx(x)

    返回:
        (x, y, z)  对应 Rotate X, Rotate Y, Rotate Z
    """
    R = np.asarray(R, dtype=np.float64)
    assert R.shape == (3, 3)

    # 对应推导：R = Rz * Ry * Rx
    r31 = R[2, 0]
    sy = -r31
    sy = np.clip(sy, -1.0, 1.0)
    y = np.arcsin(sy)
    cy = np.cos(y)

    # 处理接近 gimbal lock 的情况
    if abs(cy) < 1e-8:
        # 退化情况：强行设 x = 0，从其余元素解 z
        x = 0.0
        z = np.arctan2(-R[0, 1], R[1, 1])
    else:
        x = np.arctan2(R[2, 1] / cy, R[2, 2] / cy)
        z = np.arctan2(R[1, 0] / cy, R[0, 0] / cy)

    return np.array([x, y, z], dtype=np.float64)


def build_box_example():
    """
    构造一个简单例子：单位立方体在两个坐标系下的 8 个顶点。

    你可以把这里替换成自己的 8 个点（src_pts / dst_pts）。
    """
    # 源坐标系：单位立方体 [0,1]^3 的 8 个顶点（顺序可以任意，但两边要一一对应）
    src_pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)

    # 假设真实变换：先绕 Z 轴旋转 30°，再缩放 2.0，最后平移 (1,2,3)
    theta = np.deg2rad(30.0)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])
    s_true = 2.0
    t_true = np.array([1.0, 2.0, 3.0])

    dst_pts = (s_true * (Rz @ src_pts.T)).T + t_true  # (8,3)

    return src_pts, dst_pts


def main():
    # ===== 这里换成你自己的 8 个长方体顶点坐标 =====
    # 例子：构造一个单位立方体 + 已知变换

    dst_pts = np.array([[0.25, -0.32, 0.40],
                        [0.25, -0.45, 0.40],
                        [-0.05, -0.45, 0.40],
                        [-0.05, -0.32, 0.40],
                        [-0.05, -0.32, 0.0],
                        [0.25, -0.32, 0.0],
                        [0.25, -0.45, 0.0],
                        [-0.05, -0.45, 0.0]], dtype=np.float64)

    src_pts = np.array([[1.89034, -2.47104, -1.82542],
                        [1.89848, -3.18433, -1.90559],
                        [0.63805, -3.18798, -1.88988],
                        [0.62825, -2.48796, -1.82629],
                        [0.63687, -2.32447, -3.37908],
                        [1.89034, -2.33318, -3.38254],
                        [1.85233, -2.97721, -3.56401],
                        [0.62921, -3.00655, -3.48061]], dtype=np.float64)

    # 如果是你自己的数据，就改成：
    # src_pts = np.loadtxt("box_src.txt")   # 形状 (8,3)
    # dst_pts = np.loadtxt("box_dst.txt")   # 形状 (8,3)

    scale, R, t, T = estimate_similarity_transform(src_pts, dst_pts, with_scale=True)

    np.set_printoptions(precision=6, suppress=True)
    print("Estimated scale s:\n", scale)
    print("Estimated rotation R:\n", R)
    print("Estimated translation t:\n", t)
    print("Homogeneous transform T:\n", T)

    # ===== 新增：输出 XYZ 欧拉角（度），对应 Isaac/Omni 的 Rotate X/Y/Z =====
    euler_rad = rotation_matrix_to_euler_xyz(R)
    euler_deg = np.degrees(euler_rad)
    print("Euler angles (XYZ, degrees):\n", euler_deg)

    # 验证误差
    recon = (scale * (R @ src_pts.T)).T + t  # (N,3)
    err = np.linalg.norm(recon - dst_pts, axis=1)
    print("Per-point error:", err)
    print("Mean error:", err.mean())


if __name__ == "__main__":
    main()
