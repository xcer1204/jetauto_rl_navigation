import numpy as np

def umeyama_alignment(src, dst, estimate_scale=True):
    """
    src: Nx3 points in source coordinate system  (GS)
    dst: Nx3 points in destination coordinate system (Isaac)
    """
    assert src.shape == dst.shape

    n = src.shape[0]

    # Means
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    # Centered vectors
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    # Covariance matrix
    Sigma = dst_centered.T @ src_centered / n

    # SVD
    U, D, Vt = np.linalg.svd(Sigma)

    # Reflection handling
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1

    # Rotation
    R = U @ S @ Vt

    # Scale
    if estimate_scale:
        var_src = (src_centered ** 2).sum() / n
        s = np.trace(D @ S) / var_src
    else:
        s = 1.0

    # Translation
    t = mu_dst - s * (R @ mu_src)

    # 4×4 homogeneous matrix
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t

    return s, R, t, T


# ---------------------------------------------------
# Your 4 points (GS points as src, Isaac points as dst)
# ---------------------------------------------------

src = np.array([
    [-0.206184, 0.213495, -0.482117],
    [-0.018345, 0.257115, -0.484142],
    [-0.143717, -0.065194, -0.482694],
    [-0.017359, 0.258182, -0.711917],
])

dst = np.array([
    [1.52, 1.28, 0.26],
    [1.3, 1.08, 0.26],
    [1.52, 1.45, 0.26],
    [1.3, 1.08, 0.0],
])

# Compute transform
s, R, t, T = umeyama_alignment(src, dst, estimate_scale=True)

print("Scale s =\n", s)
print("Rotation R =\n", R)
print("Translation t =\n", t)
print("Homogeneous Transform T =\n", T)
