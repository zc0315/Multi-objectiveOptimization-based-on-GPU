"""NSGA-II 的 NumPy 实现（含约束处理）。

参考：
    K. Deb, A. Pratap, S. Agarwal, T. Meyarivan,
    "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II",
    IEEE Trans. Evolutionary Computation, 6(2): 182-197, 2002.

模块组成：
    1. Problem            问题基类（决策变量上下界 + 向量化评价）
    2. fast_non_dominated_sort   快速非支配排序 O(M N^2)
    3. crowding_distance         拥挤距离
    4. sbx_crossover             SBX 模拟二进制交叉
    5. polynomial_mutation       多项式变异
    6. NSGA2                     主算法（二元锦标赛 + 精英保留）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# 1. 问题定义
# --------------------------------------------------------------------------- #
class Problem:
    """多目标优化问题基类。

    Parameters
    ----------
    n_var : int            决策变量维数
    n_obj : int            目标个数 M
    xl, xu : float | array 决策变量下 / 上界（标量自动广播）
    n_constr : int         约束个数（g_j(x) <= 0 视为可行）

    子类实现 evaluate(X) -> (F, G)：
        X : (N, n_var)
        F : (N, n_obj)
        G : (N, n_constr)，无约束时返回 None 或 (N, 0) 空数组

    后端切换：类属性 xp 默认 numpy；GPU 子类把它设为 cupy 即可，
    evaluate() 内部统一用 `xp = self.xp` 取算子，代码只写一份。
    """

    xp = np                      # 计算后端（numpy / cupy）

    def __init__(self, n_var: int, n_obj: int,
                 xl=0.0, xu=1.0, n_constr: int = 0, name: str = "Problem"):
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.n_constr = int(n_constr)
        self.name = name
        self.xl = np.broadcast_to(np.asarray(xl, dtype=float), (self.n_var,)).copy()
        self.xu = np.broadcast_to(np.asarray(xu, dtype=float), (self.n_var,)).copy()
        self._bounds_cache: dict = {}
        if np.any(self.xu <= self.xl):
            raise ValueError("存在 xu <= xl 的决策变量")

    def bounds(self, xp):
        """返回指定后端上的 (xl, xu)，首次调用时上传并缓存。"""
        key = getattr(xp, "__name__", str(xp))
        if key not in self._bounds_cache:
            self._bounds_cache[key] = (xp.asarray(self.xl), xp.asarray(self.xu))
        return self._bounds_cache[key]

    # ---- 由子类实现 ----
    def evaluate(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

    def __call__(self, X: np.ndarray):
        xp = self.xp
        X = xp.atleast_2d(xp.asarray(X, dtype=float))
        F, G = self.evaluate(X)
        F = xp.asarray(F, dtype=float).reshape(len(X), self.n_obj)
        if G is None or self.n_constr == 0:
            G = xp.zeros((len(X), 0))
        else:
            G = xp.asarray(G, dtype=float).reshape(len(X), self.n_constr)
        return F, G

    # ---- 可选：真实 Pareto 前沿（供 IGD 使用，始终放在 CPU 上）----
    def pareto_front(self, n: int = 500) -> np.ndarray:
        raise NotImplementedError(f"{self.name} 未提供解析 Pareto 前沿")

    def __repr__(self) -> str:
        return (f"{self.name}(n_var={self.n_var}, n_obj={self.n_obj}, "
                f"n_constr={self.n_constr})")


# --------------------------------------------------------------------------- #
# 2. 快速非支配排序
# --------------------------------------------------------------------------- #
def _constraint_violation(G: np.ndarray) -> np.ndarray:
    """Deb 约束违反度：sum(max(0, g_j))。CV = 0 表示可行。"""
    if G is None or G.size == 0:
        return np.zeros(len(G) if G is not None else 0)
    return np.sum(np.maximum(G, 0.0), axis=1)


def _pairwise_domination(F: np.ndarray, CV: np.ndarray) -> np.ndarray:
    """返回布尔矩阵 D，D[i, j] = True 表示 i 约束支配 j（Deb 支配准则）。

    1) i 可行、j 不可行            -> i 支配 j
    2) 两者都不可行，CV_i < CV_j   -> i 支配 j
    3) 两者都可行且 i Pareto 支配 j -> i 支配 j
    """
    feas = CV <= 0.0
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)   # 各目标均不劣
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)    # 至少一个更优
    pareto = le & lt

    both_feas = feas[:, None] & feas[None, :]
    both_infeas = (~feas)[:, None] & (~feas)[None, :]

    D = (feas[:, None] & ~feas[None, :]) \
        | (both_feas & pareto) \
        | (both_infeas & (CV[:, None] < CV[None, :]))

    np.fill_diagonal(D, False)
    return D


def fast_non_dominated_sort(F: np.ndarray,
                            CV: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """快速非支配排序，返回按层级排列的下标数组列表 fronts[0], fronts[1], ...

    实现采用 Deb 论文的「支配计数」思路，但用向量化矩阵运算代替双层循环，
    Python 侧只剩层数（通常 5~30 层）次循环。
    """
    F = np.asarray(F, dtype=float)
    n = len(F)
    if n == 0:
        return []
    if CV is None:
        CV = np.zeros(n)
    CV = np.asarray(CV, dtype=float)

    D = _pairwise_domination(F, CV)
    n_dominated_by = D.sum(axis=0)          # 支配个体 i 的个体数量

    fronts: List[np.ndarray] = []
    unassigned = np.ones(n, dtype=bool)
    while True:
        current = unassigned & (n_dominated_by == 0)
        if not current.any():
            break
        idx = np.flatnonzero(current)
        fronts.append(idx)
        unassigned[idx] = False
        # 从被这些个体支配者的计数中减去
        n_dominated_by = n_dominated_by - D[idx].sum(axis=0)
        n_dominated_by[~unassigned] = 1     # 已分配个体不再参与判断
    return fronts


# --------------------------------------------------------------------------- #
# 3. 拥挤距离
# --------------------------------------------------------------------------- #
def crowding_distance(F: np.ndarray, front: np.ndarray) -> np.ndarray:
    """同一非支配层内的拥挤距离；边界个体为 inf。"""
    front = np.asarray(front, dtype=int)
    l = len(front)
    dist = np.zeros(l)
    if l == 0:
        return dist
    if l <= 2:
        dist[:] = np.inf
        return dist

    Ff = F[front]
    for m in range(F.shape[1]):
        order = np.argsort(Ff[:, m], kind="mergesort")
        f_min, f_max = Ff[order[0], m], Ff[order[-1], m]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = f_max - f_min
        if span <= 1e-12:
            continue
        dist[order[1:-1]] += (Ff[order[2:], m] - Ff[order[:-2], m]) / span
    return dist


def _rank_and_crowding(F: np.ndarray, CV: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    n = len(F)
    rank = np.empty(n, dtype=int)
    crowd = np.empty(n)
    fronts = fast_non_dominated_sort(F, CV)
    for r, fr in enumerate(fronts):
        rank[fr] = r
        crowd[fr] = crowding_distance(F, fr)
    return rank, crowd, fronts


# --------------------------------------------------------------------------- #
# 4. SBX 交叉（Deb 原始逐变量形式，含边界处理）
# --------------------------------------------------------------------------- #
def _betaq(rnd: float, alpha: float, eta: float) -> float:
    if rnd <= 1.0 / alpha:
        return (rnd * alpha) ** (1.0 / (eta + 1.0))
    return (1.0 / (2.0 - rnd * alpha)) ** (1.0 / (eta + 1.0))


def sbx_crossover(rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray,
                  xl: np.ndarray, xu: np.ndarray,
                  eta_c: float = 20.0, p_cross: float = 0.9
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """模拟二进制交叉，返回两个子代。"""
    n = p1.size
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > p_cross:
        return c1, c2

    for i in range(n):
        if rng.random() > 0.5:
            continue
        y1, y2 = (p1[i], p2[i]) if p1[i] <= p2[i] else (p2[i], p1[i])
        if y2 - y1 <= 1e-14:
            continue
        rnd = rng.random()

        beta = 1.0 + 2.0 * (y1 - xl[i]) / (y2 - y1)
        alpha = 2.0 - beta ** -(eta_c + 1.0)
        betaq = _betaq(rnd, alpha, eta_c)
        ch1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

        beta = 1.0 + 2.0 * (xu[i] - y2) / (y2 - y1)
        alpha = 2.0 - beta ** -(eta_c + 1.0)
        betaq = _betaq(rnd, alpha, eta_c)
        ch2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

        ch1 = min(max(ch1, xl[i]), xu[i])
        ch2 = min(max(ch2, xl[i]), xu[i])

        if rng.random() <= 0.5:
            c1[i], c2[i] = ch2, ch1
        else:
            c1[i], c2[i] = ch1, ch2
    return c1, c2


# --------------------------------------------------------------------------- #
# 5. 多项式变异
# --------------------------------------------------------------------------- #
def polynomial_mutation(rng: np.random.Generator, x: np.ndarray,
                        xl: np.ndarray, xu: np.ndarray,
                        eta_m: float = 20.0,
                        p_mut: Optional[float] = None) -> np.ndarray:
    n = x.size
    y = x.copy()
    pm = (1.0 / n) if p_mut is None else p_mut
    mut_pow = 1.0 / (eta_m + 1.0)

    for i in range(n):
        if rng.random() > pm:
            continue
        span = xu[i] - xl[i]
        if span <= 0:
            continue
        d1 = (y[i] - xl[i]) / span
        d2 = (xu[i] - y[i]) / span
        rnd = rng.random()
        if rnd < 0.5:
            xy = 1.0 - d1
            val = 2.0 * rnd + (1.0 - 2.0 * rnd) * xy ** (eta_m + 1.0)
            dq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - d2
            val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * xy ** (eta_m + 1.0)
            dq = 1.0 - val ** mut_pow
        y[i] = min(max(y[i] + dq * span, xl[i]), xu[i])
    return y


# --------------------------------------------------------------------------- #
# 6. 主算法
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    X: np.ndarray                      # 最终种群决策变量 (N, n_var)
    F: np.ndarray                      # 最终种群目标值   (N, n_obj)
    G: np.ndarray                      # 约束值          (N, n_constr)
    front: np.ndarray                  # 第一前沿个体下标
    history: dict = field(default_factory=dict)

    @property
    def X_front(self) -> np.ndarray:
        return self.X[self.front]

    @property
    def F_front(self) -> np.ndarray:
        return self.F[self.front]


class NSGA2:
    """NSGA-II 主算法。

    Parameters
    ----------
    problem   : Problem 子类实例
    pop_size  : 种群规模 N（偶数，建议 100~200）
    max_gen   : 迭代代数
    eta_c     : SBX 分布指数（越大子代越靠近父代，常用 15~30）
    eta_m     : 多项式变异分布指数（常用 15~30）
    p_cross   : 交叉概率（个体级，常用 0.9）
    p_mut     : 每个变量的变异概率，默认 1/n_var
    seed      : 随机种子
    """

    def __init__(self, problem: Problem, pop_size: int = 100, max_gen: int = 250,
                 eta_c: float = 20.0, eta_m: float = 20.0,
                 p_cross: float = 0.9, p_mut: Optional[float] = None,
                 seed: Optional[int] = None):
        if pop_size % 2:
            pop_size += 1                     # 保证成对产生子代
        self.problem = problem
        self.pop_size = int(pop_size)
        self.max_gen = int(max_gen)
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.p_cross = float(p_cross)
        self.p_mut = (1.0 / problem.n_var) if p_mut is None else float(p_mut)
        self.rng = np.random.default_rng(seed)

    # ---------------- 内部工具 ---------------- #
    def _init_population(self) -> np.ndarray:
        p = self.problem
        r = self.rng.random((self.pop_size, p.n_var))
        return p.xl + r * (p.xu - p.xl)

    def _tournament(self, rank: np.ndarray, crowd: np.ndarray) -> int:
        """二元锦标赛：先比非支配层级，再比拥挤距离。"""
        i, j = self.rng.integers(0, self.pop_size, size=2)
        if rank[i] != rank[j]:
            return int(i if rank[i] < rank[j] else j)
        if crowd[i] != crowd[j]:
            return int(i if crowd[i] > crowd[j] else j)
        return int(i if self.rng.random() < 0.5 else j)

    def _make_offspring(self, X: np.ndarray, rank: np.ndarray,
                        crowd: np.ndarray) -> np.ndarray:
        p = self.problem
        off = np.empty_like(X)
        for k in range(0, self.pop_size, 2):
            i = self._tournament(rank, crowd)
            j = self._tournament(rank, crowd)
            c1, c2 = sbx_crossover(self.rng, X[i], X[j], p.xl, p.xu,
                                   self.eta_c, self.p_cross)
            off[k] = polynomial_mutation(self.rng, c1, p.xl, p.xu,
                                         self.eta_m, self.p_mut)
            off[k + 1] = polynomial_mutation(self.rng, c2, p.xl, p.xu,
                                             self.eta_m, self.p_mut)
        return off

    @staticmethod
    def _environmental_selection(F: np.ndarray, CV: np.ndarray,
                                 n_survive: int) -> np.ndarray:
        """精英保留：按前沿依次填充，最后一层按拥挤距离降序截断。"""
        fronts = fast_non_dominated_sort(F, CV)
        chosen: List[np.ndarray] = []
        total = 0
        for fr in fronts:
            if total + len(fr) <= n_survive:
                chosen.append(fr)
                total += len(fr)
            else:
                need = n_survive - total
                if need > 0:
                    d = crowding_distance(F, fr)
                    order = np.argsort(-d, kind="mergesort")[:need]
                    chosen.append(fr[order])
                break
        return np.concatenate(chosen) if chosen else np.empty(0, dtype=int)

    # ---------------- 主流程 ---------------- #
    def run(self, verbose: bool = False) -> Result:
        prob = self.problem
        N = self.pop_size

        X = self._init_population()
        F, G = prob(X)
        CV = _constraint_violation(G) if G.size else np.zeros(N)

        history = {"n_front1": [], "n_feasible": []}

        for gen in range(1, self.max_gen + 1):
            rank, crowd, fronts = _rank_and_crowding(F, CV)
            history["n_front1"].append(len(fronts[0]))
            history["n_feasible"].append(int(np.sum(CV <= 0)))

            offspring = self._make_offspring(X, rank, crowd)
            F_off, G_off = prob(offspring)
            CV_off = _constraint_violation(G_off) if G_off.size else np.zeros(N)

            # 父代 + 子代 合并后择优（mu + lambda 精英策略）
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, F_off])
            G_all = np.vstack([G, G_off]) if G.size else G
            CV_all = np.concatenate([CV, CV_off])

            sel = self._environmental_selection(F_all, CV_all, N)
            X, F, CV = X_all[sel], F_all[sel], CV_all[sel]
            if G.size:
                G = G_all[sel]

            if verbose and (gen % 25 == 0 or gen == 1):
                print(f"gen {gen:4d} | front1 = {len(fronts[0]):3d} | "
                      f"feasible = {int(np.sum(CV <= 0)):3d}/{N}")

        fronts = fast_non_dominated_sort(F, CV)
        # 保持解的多样性：按第一前沿的目标值排序，便于出图与 IGD 计算
        first = fronts[0]
        if first.size > 1:
            first = first[np.lexsort(F[first].T[::-1])]

        return Result(X=X, F=F, G=G, front=first, history=history)
