"""标准多目标测试问题集合（ZDT / DTLZ），向量化实现，支持 numpy / cupy 双后端。

后端由 Problem.xp 决定：
    problems.py     里的类用 numpy（默认）
    problems_gpu.py 里的子类把 xp 设为 cupy，evaluate() 代码完全复用

Pareto 前沿（pareto_front）始终在 CPU 上用 numpy 生成，供 IGD 使用。
"""

from __future__ import annotations

import math

import numpy as np

from nsga2 import Problem


# --------------------------------------------------------------------------- #
# 均匀参考点（Das-Dennis 单纯形格点），用于 DTLZ 的解析前沿
# --------------------------------------------------------------------------- #
def das_dennis(n_partitions: int, n_obj: int) -> np.ndarray:
    """在 sum(w) = 1 的单纯形上生成均匀格点，返回 (C(p+M-1, M-1), M)。"""
    def gen(n, m):
        if m == 1:
            yield [n]
        else:
            for i in range(n + 1):
                for tail in gen(n - i, m - 1):
                    yield [i] + tail

    w = np.array(list(gen(n_partitions, n_obj)), dtype=float)
    return w / n_partitions


def _partitions_for(n: int, n_obj: int) -> int:
    p = 1
    while math.comb(p + n_obj - 1, n_obj - 1) < n:
        p += 1
    return p


# --------------------------------------------------------------------------- #
# ZDT 系列
# --------------------------------------------------------------------------- #
class ZDT1(Problem):
    """凸前沿：f2 = 1 - sqrt(f1)，前沿 f1 ∈ [0, 1]。"""

    def __init__(self, n_var: int = 30):
        super().__init__(n_var, 2, 0.0, 1.0, name=self.__class__.__name__)

    def evaluate(self, X):
        xp = self.xp
        f1 = X[:, 0]
        g = 1.0 + 9.0 * X[:, 1:].sum(axis=1) / (self.n_var - 1)
        f2 = g * (1.0 - xp.sqrt(f1 / g))
        return xp.stack([f1, f2], axis=1), None

    def pareto_front(self, n: int = 500):
        f1 = np.linspace(0.0, 1.0, n)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


class ZDT2(Problem):
    """凹前沿：f2 = 1 - f1^2。"""

    def __init__(self, n_var: int = 30):
        super().__init__(n_var, 2, 0.0, 1.0, name=self.__class__.__name__)

    def evaluate(self, X):
        xp = self.xp
        f1 = X[:, 0]
        g = 1.0 + 9.0 * X[:, 1:].sum(axis=1) / (self.n_var - 1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return xp.stack([f1, f2], axis=1), None

    def pareto_front(self, n: int = 500):
        f1 = np.linspace(0.0, 1.0, n)
        return np.column_stack([f1, 1.0 - f1 ** 2])


class ZDT3(Problem):
    """不连续前沿（5 段）。"""

    def __init__(self, n_var: int = 30):
        super().__init__(n_var, 2, 0.0, 1.0, name=self.__class__.__name__)

    def evaluate(self, X):
        xp = self.xp
        f1 = X[:, 0]
        g = 1.0 + 9.0 * X[:, 1:].sum(axis=1) / (self.n_var - 1)
        f2 = g * (1.0 - xp.sqrt(f1 / g) - (f1 / g) * xp.sin(10.0 * xp.pi * f1))
        return xp.stack([f1, f2], axis=1), None

    def pareto_front(self, n: int = 500):
        intervals = [(0.0, 0.0830), (0.1822, 0.2578),
                     (0.4093, 0.4539), (0.6178, 0.6525), (0.8233, 0.8518)]
        parts = []
        for lo, hi in intervals:
            f1 = np.linspace(lo, hi, max(2, n // 5))
            f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
            parts.append(np.column_stack([f1, f2]))
        return np.vstack(parts)


class ZDT4(Problem):
    """多局部前沿陷阱，x1 ∈ [0,1]，其余 ∈ [-5,5]。"""

    def __init__(self, n_var: int = 10):
        xl = np.array([0.0] + [-5.0] * (n_var - 1))
        xu = np.array([1.0] + [5.0] * (n_var - 1))
        super().__init__(n_var, 2, xl, xu, name=self.__class__.__name__)

    def evaluate(self, X):
        xp = self.xp
        f1 = X[:, 0]
        Y = X[:, 1:]
        g = 1.0 + 10.0 * Y.shape[1] + xp.sum(
            Y ** 2 - 10.0 * xp.cos(4.0 * xp.pi * Y), axis=1)
        f2 = g * (1.0 - xp.sqrt(f1 / g))
        return xp.stack([f1, f2], axis=1), None

    def pareto_front(self, n: int = 500):
        f1 = np.linspace(0.0, 1.0, n)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


class ZDT6(Problem):
    """非均匀密度前沿：f1 = 1 - exp(-4x1) sin^6(6πx1)，f2 = 1 - f1^2。"""

    def __init__(self, n_var: int = 10):
        super().__init__(n_var, 2, 0.0, 1.0, name=self.__class__.__name__)

    def evaluate(self, X):
        xp = self.xp
        f1 = 1.0 - xp.exp(-4.0 * X[:, 0]) * xp.sin(6.0 * xp.pi * X[:, 0]) ** 6
        g = 1.0 + 9.0 * (X[:, 1:].sum(axis=1) / (self.n_var - 1)) ** 0.25
        f2 = g * (1.0 - (f1 / g) ** 2)
        return xp.stack([f1, f2], axis=1), None

    def pareto_front(self, n: int = 500):
        x1 = np.linspace(0.0, 1.0, n)
        f1 = 1.0 - np.exp(-4.0 * x1) * np.sin(6.0 * np.pi * x1) ** 6
        f1 = np.sort(f1)
        return np.column_stack([f1, 1.0 - f1 ** 2])


# --------------------------------------------------------------------------- #
# DTLZ 系列（可扩展到任意 M）
# --------------------------------------------------------------------------- #
class DTLZ1(Problem):
    """线性前沿 sum(f_i) = 0.5；大量局部最优，k = 5 时 n_var = M + 4。"""

    def __init__(self, n_obj: int = 3, k: int = 5):
        self.k = int(k)
        super().__init__(self.k + n_obj - 1, n_obj, 0.0, 1.0,
                         name=self.__class__.__name__)

    def evaluate(self, X):
        xp = self.xp
        M = self.n_obj
        Y = X[:, M - 1:]
        g = 100.0 * (self.k + xp.sum(
            (Y - 0.5) ** 2 - xp.cos(20.0 * xp.pi * (Y - 0.5)), axis=1))
        base = 0.5 * (1.0 + g)
        cols = []
        for i in range(M):
            if i == 0:
                cols.append(base * xp.prod(X[:, :M - 1], axis=1))
            elif i == M - 1:
                cols.append(base * (1.0 - X[:, 0]))
            else:
                cols.append(base * xp.prod(X[:, :M - 1 - i], axis=1)
                            * (1.0 - X[:, M - 1 - i]))
        return xp.stack(cols, axis=1), None

    def pareto_front(self, n: int = 500):
        M = self.n_obj
        return 0.5 * das_dennis(_partitions_for(n, M), M)


class DTLZ2(Problem):
    """球面前沿 sum(f_i^2) = 1（非负象限），k = 10 时 n_var = M + 9。"""

    def __init__(self, n_obj: int = 3, k: int = 10):
        self.k = int(k)
        super().__init__(self.k + n_obj - 1, n_obj, 0.0, 1.0,
                         name=self.__class__.__name__)

    def evaluate(self, X):
        xp = self.xp
        M = self.n_obj
        g = xp.sum((X[:, M - 1:] - 0.5) ** 2, axis=1)
        base = 1.0 + g
        cols = []
        for i in range(M):
            idx = M - 1 - i
            val = base * xp.prod(xp.cos(X[:, :idx] * xp.pi / 2.0), axis=1)
            if i > 0:
                val = val * xp.sin(X[:, idx] * xp.pi / 2.0)
            cols.append(val)
        return xp.stack(cols, axis=1), None

    def pareto_front(self, n: int = 500):
        M = self.n_obj
        w = das_dennis(_partitions_for(n, M), M)
        return w / np.linalg.norm(w, axis=1, keepdims=True)   # 投影到单位球面


# --------------------------------------------------------------------------- #
# 带约束的算例（用于验证约束支配）
# --------------------------------------------------------------------------- #
class CONSTR(Problem):
    """Deb 经典约束测试问题 CONSTR。

    min f1 = x1, f2 = (1 + x2) / x1
    s.t. g1: 6 - x2 - 9 x1 <= 0 ; g2: 1 + x2 - 9 x1 <= 0
         x1 ∈ [0.1, 1], x2 ∈ [0, 5]
    """

    def __init__(self):
        super().__init__(2, 2, np.array([0.1, 0.0]), np.array([1.0, 5.0]),
                         n_constr=2, name=self.__class__.__name__)

    def evaluate(self, X):
        xp = self.xp
        x1, x2 = X[:, 0], X[:, 1]
        F = xp.stack([x1, (1.0 + x2) / x1], axis=1)
        G = xp.stack([6.0 - (x2 + 9.0 * x1), 1.0 + x2 - 9.0 * x1], axis=1)
        return F, G

    def pareto_front(self, n: int = 500):
        # 可行域要求 x1 >= 7/18（两条约束取交集），前沿分两段：
        #   x1 ∈ [7/18, 2/3] : g1 起作用，x2 = 6 - 9x1, f2 = 7/x1 - 9
        #   x1 ∈ [2/3, 1]    : x2 = 0,                   f2 = 1/x1
        half = n // 2
        x1a = np.linspace(7.0 / 18.0, 2.0 / 3.0, half)
        x1b = np.linspace(2.0 / 3.0, 1.0, n - half)
        return np.vstack([
            np.column_stack([x1a, 7.0 / x1a - 9.0]),
            np.column_stack([x1b, 1.0 / x1b]),
        ])


# 便于按名字取用
PROBLEMS = {
    "ZDT1": ZDT1, "ZDT2": ZDT2, "ZDT3": ZDT3, "ZDT4": ZDT4, "ZDT6": ZDT6,
    "DTLZ1": DTLZ1, "DTLZ2": DTLZ2, "CONSTR": CONSTR,
}
