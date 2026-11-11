"""常用多目标评价指标：IGD、GD、Spacing、Hypervolume(2D)。"""

from __future__ import annotations

import numpy as np


def _np(a) -> np.ndarray:
    """接受 numpy / cupy 数组，统一转成 host 端 numpy。"""
    if hasattr(a, "get"):
        return a.get()
    return np.asarray(a, float)


def _dist_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A: (m, k), B: (n, k) -> (m, n) 欧氏距离"""
    d = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.sum(d * d, axis=2))


def igd(F: np.ndarray, PF: np.ndarray) -> float:
    """Inverted Generational Distance。

    参考点集 PF 中每个点到近似前沿 F 最近点的距离均值，衡量收敛性 + 分布性。
    """
    D = _dist_matrix(_np(PF), _np(F))
    return float(np.mean(np.min(D, axis=1)))


def gd(F: np.ndarray, PF: np.ndarray) -> float:
    """Generational Distance：近似前沿到参考前沿的平均距离（只反映收敛）。"""
    D = _dist_matrix(_np(F), _np(PF))
    return float(np.mean(np.min(D, axis=1)))


def spacing(F: np.ndarray) -> float:
    """Spacing（Schott, 1995）：解分布均匀程度，越小越均匀。"""
    F = _np(F)
    n = len(F)
    if n < 2:
        return 0.0
    D = _dist_matrix(F, F)
    np.fill_diagonal(D, np.inf)
    d = np.min(D, axis=1)
    return float(np.sqrt(np.sum((d - d.mean()) ** 2) / (n - 1)))


def hypervolume_2d(F: np.ndarray, ref: np.ndarray) -> float:
    """2 目标 Hypervolume（最大化方向，非支配点集），ref 为参考点（劣于所有解）。"""
    F = _np(F)
    ref = _np(ref)
    # 只保留被 ref 支配的点，并按第一目标升序
    F = F[np.all(F <= ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 0], kind="mergesort")]
    hv, prev_f2 = 0.0, ref[1]
    for f1, f2 in F:
        if f2 < prev_f2:                      # 仅非支配点贡献面积
            hv += (ref[0] - f1) * (prev_f2 - f2)
            prev_f2 = f2
    return float(hv)


def summarize(F: np.ndarray, PF: np.ndarray, ref: np.ndarray | None = None) -> dict:
    out = {"IGD": igd(F, PF), "GD": gd(F, PF), "Spacing": spacing(F), "N": len(F)}
    if ref is not None and _np(F).shape[1] == 2:
        out["HV"] = hypervolume_2d(F, ref)
    return out
