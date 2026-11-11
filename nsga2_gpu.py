"""NSGA-II 的 GPU 实现（CuPy）—— 全种群张量化。

设计要点
--------
朴素做法（只把 np 换成 cp）在 GPU 上反而更慢：SBX / 变异 / 锦标赛是逐个体逐变量的
Python 循环，每次只处理几十个数，kernel 启动开销（Windows 上约 30 µs）会吃掉全部收益。
本模块把每一代的计算压到尽可能少的 kernel：

1. **非支配排序**：先算 (2N, 2N) 支配矩阵（一次广播），再用「分层剥离」求 rank。
   剥离时的支配计数递减用矩阵-向量乘完成（`cur @ D`，走 cuBLAS），
   避免布尔掩码 gather（那会触发 nonzero + 拷贝，慢一个量级）。
   层数用上一代的值做上界，正常情况下整代只需 1 次主机同步。
2. **拥挤距离**：不再按前沿循环。对每个目标做一次全局 argsort（键 = rank*4 + 归一化目标值），
   前沿在排序序列里是连续块，用块边界掩码 + 回散射一次算出**所有前沿**的拥挤距离。
3. **环境选择**：把 (rank, 拥挤距离) 编码成单调键 `rank + (1 - c/(1+c))/2`，一次 argsort 取前 N，
   取代逐层填充的循环。
4. **遗传算子**：配对级 / 种群级张量运算，SBX 与多项式变异各约 30 / 14 个 kernel。

`xp` 参数可设为 numpy：同一份代码退化为「CPU 向量化版」，
用于对拍或在无显卡机器上跑基准。

随机数用 `xp.random.RandomState`（cupy / numpy 的 API 交集），保证两个后端行为一致。

典型用法
--------
    from problems_gpu import PROBLEMS_GPU
    from nsga2_gpu import NSGA2GPU

    prob = PROBLEMS_GPU["ZDT1"]()
    res = NSGA2GPU(prob, pop_size=1000, max_gen=500, seed=1).run()
    F = res.F_front_np()        # 拷回主机
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# 后端与工具
# --------------------------------------------------------------------------- #
def get_backend(name: str = "auto"):
    """返回计算后端模块（cupy 或 numpy）。"""
    if name == "numpy":
        return np
    try:
        import cupy as cp
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "未检测到 CuPy。安装：pip install cupy-cuda12x（CUDA 12.x）\n"
            "若只想跑 CPU 向量化版，可传 xp=numpy 或 --backend numpy。"
        ) from exc
    if name == "auto" and cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("CuPy 已安装但未检测到可用 GPU")
    return cp


def to_numpy(a):
    """cupy / numpy 数组统一转 numpy。"""
    if hasattr(a, "get"):
        return a.get()
    return np.asarray(a)


def device_info() -> str:
    try:
        import cupy as cp
        p = cp.cuda.runtime.getDeviceProperties(0)
        name = p["name"].decode() if isinstance(p["name"], bytes) else str(p["name"])
        return f"{name} (CuPy {cp.__version__})"
    except Exception:
        return "CPU (numpy)"


INF = float("inf")
_PEEL_KERNELS = {}


def _peel_kernels(xp):
    """返回 (pick, assign) 两个函数：cupy 下是融合 kernel，numpy 下是等价张量运算。"""
    if xp is np:
        def pick(rank, nd):
            return ((rank < 0) & (nd == 0)).astype(np.float32)

        def assign(rank, cur, level):
            return np.where(cur > 0.5, np.int32(level), rank).astype(np.int32)
        return pick, assign

    key = id(xp)
    if key not in _PEEL_KERNELS:
        pick_k = xp.ElementwiseKernel(
            "int32 r, float32 nd", "float32 cur",
            "cur = (r < 0 && nd == 0.0f) ? 1.0f : 0.0f;",
            "nsga2_pick")
        assign_k = xp.ElementwiseKernel(
            "int32 r, float32 cur, int32 lv", "int32 r2",
            "r2 = (cur > 0.5f) ? lv : r;",
            "nsga2_assign")

        def pick(rank, nd, _k=pick_k):
            return _k(rank, nd)

        def assign(rank, cur, level, _k=assign_k):
            return _k(rank, cur, level)

        _PEEL_KERNELS[key] = (pick, assign)
    return _PEEL_KERNELS[key]


# --------------------------------------------------------------------------- #
# 支配关系与排序
# --------------------------------------------------------------------------- #
def domination_matrix(F, CV, xp):
    """(n, n) 布尔矩阵，D[i, j] = True 表示 i 约束支配 j（Deb 准则）。"""
    n = len(F)
    feas = CV <= 0.0
    le = xp.all(F[:, None, :] <= F[None, :, :], axis=2)     # 各目标均不劣
    lt = xp.any(F[:, None, :] < F[None, :, :], axis=2)      # 至少一个更优
    D = ((feas[:, None] & ~feas[None, :])
         | ((feas[:, None] & feas[None, :]) & le & lt)
         | ((~feas)[:, None] & (~feas)[None, :]
            & (CV[:, None] < CV[None, :])))
    idx = xp.arange(n)
    D[idx, idx] = False                                     # 去自支配
    return D


def rank_population(F, CV, xp, hint: int = 24, stats: Optional[dict] = None,
                    n_need: Optional[int] = None):
    """返回每个个体的非支配层级 rank（0 = 第一前沿）。

    两层加速：
    1. 剥离用 `cur @ D` 递减支配计数（走 cuBLAS），替代布尔掩码 gather；
       「取当前层 / 写 rank」在 cupy 上各融合成一个 ElementwiseKernel，每层 4 次启动。
    2. 若给定 n_need（= 需要保留的个体数），一旦累计已分配数达到 n_need 就停止剥离，
       其后的层对选择结果毫无影响（会被环境选择整体淘汰），直接标记为大 rank。
       这一条在收敛后期很关键：此时前沿层数会从十几层涨到几十层。

    hint 是硬上界，防止极端情况下死循环；返回的 rank 中，未被剥离到的个体为 2^20。
    """
    n = len(F)
    big = 1 << 20
    Df = domination_matrix(F, CV, xp).astype(xp.float32)
    nd = Df.sum(axis=0)                       # 支配该个体的个体数
    rank = xp.full(n, -1, dtype=xp.int32)
    level = 0
    total = 0
    need = n if n_need is None else int(n_need)

    pick, assign = _peel_kernels(xp)

    while level < hint:
        cur = pick(rank, nd)
        nd = nd - cur @ Df
        rank = assign(rank, cur, level)
        level += 1
        total += int(cur.sum())
        if total >= need:
            break

    if total < need:                          # 层数超过 hint，继续补完
        while True:
            cur = pick(rank, nd)
            cnt = int(cur.sum())
            if cnt == 0:
                break
            nd = nd - cur @ Df
            rank = assign(rank, cur, level)
            level += 1
            total += cnt

    if total < n:                             # 未剥离的个体统一放到最后
        rank = xp.where(rank < 0, xp.int32(big), rank)

    if stats is not None:
        stats["levels"] = level
        stats["assigned"] = total
    return rank


def crowding_all(F, rank, xp):
    """一次算出**所有前沿**的拥挤距离（前沿在按 rank 排序后是连续块）。"""
    n, M = F.shape
    crowd = xp.zeros(n, dtype=float)
    if n <= 2:
        crowd[:] = INF
        return crowd

    for m in range(M):
        col = F[:, m]
        cmin, cmax = col.min(), col.max()
        norm = (col - cmin) / xp.maximum(cmax - cmin, 1e-12)
        order = xp.argsort(rank.astype(F.dtype) * 4.0 + norm)   # 先按 rank 分块，块内按目标值
        sf = col[order]
        r = rank[order]

        newblk = xp.empty(n, dtype=bool)
        newblk[0] = True
        newblk[1:] = r[1:] != r[:-1]
        starts = xp.flatnonzero(newblk)
        ends = xp.concatenate([starts[1:] - 1, xp.asarray([n - 1])])
        bid = xp.cumsum(newblk) - 1

        span = xp.maximum(sf[ends] - sf[starts], 1e-12)[bid]   # 每个位置所属前沿的跨度
        contrib = xp.zeros(n, dtype=float)
        same = (bid[2:] == bid[:-2]) & (bid[1:-1] == bid[:-2])
        contrib[1:-1] = xp.where(same, (sf[2:] - sf[:-2]) / span[1:-1], 0.0)
        contrib[starts] = INF                                 # 每层边界解
        contrib[ends] = INF

        out = xp.zeros(n, dtype=float)
        out[order] = contrib                                  # 散射回原下标
        crowd = crowd + out
    return crowd


def select_survivors(rank, crowd, n_survive: int, xp):
    """按 (rank 升, crowd 降) 取前 n_survive 个，返回下标。"""
    c = xp.where(xp.isfinite(crowd), crowd, 1e12)
    key = rank.astype(xp.float64) + (1.0 - c / (1.0 + c)) * 0.5   # 落在 [rank, rank+0.5]
    return xp.argsort(key)[:n_survive]


# --------------------------------------------------------------------------- #
# 参考实现（逐前沿，用于对拍 / 小规模调试）
# --------------------------------------------------------------------------- #
def fast_non_dominated_sort(F, CV, xp) -> List:
    """Deb 原始剥离法，返回 fronts 列表。慢但直观，作为 rank_population 的参考。"""
    D = domination_matrix(F, CV, xp)
    nd = D.sum(axis=0)
    fronts: List = []
    unassigned = xp.ones(len(F), dtype=bool)
    while True:
        cur = unassigned & (nd == 0)
        if int(cur.sum()) == 0:
            break
        idx = xp.flatnonzero(cur)
        fronts.append(idx)
        unassigned[idx] = False
        nd = nd - D[idx].sum(axis=0)
        nd[~unassigned] = 1
    return fronts


def crowding_distance(F, front, xp):
    """单个前沿的拥挤距离（参考实现）。"""
    l = len(front)
    if l == 0:
        return xp.zeros(0)
    if l <= 2:
        return xp.full(l, INF)

    Ff = F[front]
    dist = xp.zeros(l, dtype=float)
    for m in range(F.shape[1]):
        order = xp.argsort(Ff[:, m])
        denom = xp.maximum(Ff[order[-1], m] - Ff[order[0], m], 1e-12)
        dist[order[:1]] = INF
        dist[order[-1:]] = INF
        dist[order[1:-1]] += (Ff[order[2:], m] - Ff[order[:-2], m]) / denom
    return dist


# --------------------------------------------------------------------------- #
# 结果容器
# --------------------------------------------------------------------------- #
@dataclass
class ResultGPU:
    X: object
    F: object
    G: object
    front: object
    history: dict = field(default_factory=dict)
    backend: str = "cupy"

    @property
    def F_front(self):
        """第一前沿目标值（仍在显存上）。"""
        return self.F[self.front]

    @property
    def X_front(self):
        return self.X[self.front]

    def F_front_np(self) -> np.ndarray:
        return to_numpy(self.F_front)

    def host(self) -> dict:
        """全部拷回主机（numpy）。"""
        return {"X": to_numpy(self.X), "F": to_numpy(self.F),
                "G": to_numpy(self.G), "front": to_numpy(self.front)}


# --------------------------------------------------------------------------- #
# 主算法
# --------------------------------------------------------------------------- #
class NSGA2GPU:
    """矢量化的 NSGA-II。

    Parameters
    ----------
    problem   : Problem 子类（GPU 版请用 problems_gpu 中的类）
    pop_size  : 种群规模（偶数）
    max_gen   : 迭代代数
    eta_c / eta_m : SBX / 多项式变异分布指数
    p_cross   : 个体级交叉概率
    p_mut     : 每变量变异概率，默认 1/n_var
    seed      : 随机种子
    xp        : 后端，默认自动选 cupy；传 numpy 得到 CPU 向量化版
    """

    def __init__(self, problem, pop_size: int = 100, max_gen: int = 250,
                 eta_c: float = 20.0, eta_m: float = 20.0,
                 p_cross: float = 0.9, p_mut: Optional[float] = None,
                 seed: Optional[int] = None, xp=None,
                 record_history: bool = True):
        if xp is None:
            xp = get_backend("auto")
        elif isinstance(xp, str):
            xp = get_backend(xp)
        self.xp = xp
        if pop_size % 2:
            pop_size += 1
        self.problem = problem
        self.pop_size = int(pop_size)
        self.max_gen = int(max_gen)
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.p_cross = float(p_cross)
        self.p_mut = (1.0 / problem.n_var) if p_mut is None else float(p_mut)
        self.record_history = record_history
        self._rs = xp.random.RandomState(seed)
        self._level_hint = 24                 # 层数上界，逐代自适应

        if getattr(problem, "xp", np) is not xp:      # 问题后端与算法后端对齐
            problem.xp = xp

    # ---------------- 初始化 / 约束 ---------------- #
    def _init_population(self):
        xp, p = self.xp, self.problem
        xl, xu = p.bounds(xp)
        r = self._rs.random_sample((self.pop_size, p.n_var))
        return xl + r * (xu - xl)

    def _cv(self, G, n: int):
        xp = self.xp
        if G.size == 0:
            return xp.zeros(n)
        return xp.sum(xp.maximum(G, 0.0), axis=1)

    def _rank_crowd(self, F, CV):
        stats: dict = {}
        rank = rank_population(F, CV, self.xp, hint=self._level_hint, stats=stats)
        crowd = crowding_all(F, rank, self.xp)
        self._level_hint = max(4, stats.get("levels", self._level_hint) + 1)
        return rank, crowd

    # ---------------- 选择 ---------------- #
    def _tournament(self, rank, crowd, k: int):
        """批量二元锦标赛：先比层级，再比拥挤距离，全平则随机。"""
        xp, rs, N = self.xp, self._rs, self.pop_size
        draw = rs.randint(0, N, size=(2, k))
        a, b = draw[0], draw[1]
        ra, rb = rank[a], rank[b]
        ca, cb = crowd[a], crowd[b]
        a_better = (ra < rb) | ((ra == rb) & (ca > cb))
        tie = (ra == rb) & (ca == cb)
        coin = rs.random_sample(k) < 0.5
        return xp.where(a_better | (tie & coin), a, b)

    # ---------------- 遗传算子（全种群张量化）---------------- #
    def _sbx(self, p1, p2):
        xp, rs = self.xp, self._rs
        xl, xu = self.problem.bounds(xp)
        K, n = p1.shape
        eta = self.eta_c
        expo = 1.0 / (eta + 1.0)

        cross = (rs.random_sample(K) <= self.p_cross)[:, None]
        var = rs.random_sample((K, n)) <= 0.5
        mask = cross & var

        y1 = xp.minimum(p1, p2)
        y2 = xp.maximum(p1, p2)
        diff = y2 - y1
        valid = mask & (diff > 1e-14)
        safe = xp.where(diff > 1e-14, diff, 1.0)

        rnd = rs.random_sample((K, n))

        beta = 1.0 + 2.0 * (y1 - xl) / safe
        alpha = 2.0 - beta ** -(eta + 1.0)
        bq1 = xp.where(rnd <= 1.0 / alpha,
                       (rnd * alpha) ** expo,
                       (1.0 / xp.maximum(2.0 - rnd * alpha, 1e-12)) ** expo)
        ch1 = 0.5 * ((y1 + y2) - bq1 * safe)

        beta = 1.0 + 2.0 * (xu - y2) / safe
        alpha = 2.0 - beta ** -(eta + 1.0)
        bq2 = xp.where(rnd <= 1.0 / alpha,
                       (rnd * alpha) ** expo,
                       (1.0 / xp.maximum(2.0 - rnd * alpha, 1e-12)) ** expo)
        ch2 = 0.5 * ((y1 + y2) + bq2 * safe)

        ch1 = xp.clip(ch1, xl, xu)
        ch2 = xp.clip(ch2, xl, xu)

        swap = rs.random_sample((K, n)) <= 0.5          # 随机交换，避免偏向父代
        c1 = xp.where(valid, xp.where(swap, ch2, ch1), p1)
        c2 = xp.where(valid, xp.where(swap, ch1, ch2), p2)
        return c1, c2

    def _mutate(self, X):
        xp, rs = self.xp, self._rs
        xl, xu = self.problem.bounds(xp)
        span = xu - xl
        eta = self.eta_m
        expo = 1.0 / (eta + 1.0)

        d1 = xp.clip((X - xl) / span, 0.0, 1.0)
        d2 = 1.0 - d1
        rnd = rs.random_sample(X.shape)

        val1 = 2.0 * rnd + (1.0 - 2.0 * rnd) * (1.0 - d1) ** (eta + 1.0)
        dq1 = xp.maximum(val1, 0.0) ** expo - 1.0
        val2 = (2.0 * (1.0 - rnd)
                + 2.0 * (rnd - 0.5) * (1.0 - d2) ** (eta + 1.0))
        dq2 = 1.0 - xp.maximum(val2, 0.0) ** expo

        dq = xp.where(rnd < 0.5, dq1, dq2)
        mask = rs.random_sample(X.shape) <= self.p_mut
        dq = xp.where(mask, dq, 0.0)
        return xp.clip(X + dq * span, xl, xu)

    def _make_offspring(self, X, rank, crowd):
        xp = self.xp
        K = self.pop_size // 2
        i1 = self._tournament(rank, crowd, K)
        i2 = self._tournament(rank, crowd, K)
        c1, c2 = self._sbx(X[i1], X[i2])
        c1 = self._mutate(c1)
        c2 = self._mutate(c2)
        off = xp.empty_like(X)
        off[0::2] = c1
        off[1::2] = c2
        return off

    # ---------------- 主流程 ---------------- #
    def run(self, verbose: bool = False) -> ResultGPU:
        xp, prob, N = self.xp, self.problem, self.pop_size

        X = self._init_population()
        F, G = prob(X)
        CV = self._cv(G, N)
        rank, crowd = self._rank_crowd(F, CV)

        hist = {"n_front1": [], "n_feasible": []}

        for gen in range(1, self.max_gen + 1):
            if self.record_history:
                hist["n_front1"].append(int((rank == 0).sum()))
                hist["n_feasible"].append(int((CV <= 0).sum()))
            if verbose and (gen % 25 == 0 or gen == 1):
                print(f"gen {gen:4d} | front1 = {int((rank == 0).sum()):4d} | "
                      f"feasible = {int((CV <= 0).sum()):4d}/{N}", flush=True)

            off = self._make_offspring(X, rank, crowd)
            F_off, G_off = prob(off)
            CV_off = self._cv(G_off, N)

            X_all = xp.vstack([X, off])
            F_all = xp.vstack([F, F_off])
            CV_all = xp.concatenate([CV, CV_off])
            G_all = xp.vstack([G, G_off]) if G.size else G

            # 每代只做一次排序（父代 + 子代），幸存者直接继承 rank / crowd
            stats: dict = {}
            r_all = rank_population(F_all, CV_all, xp, hint=self._level_hint,
                                    stats=stats, n_need=N)
            self._level_hint = max(6, stats.get("levels", self._level_hint) + 3)
            c_all = crowding_all(F_all, r_all, xp)
            sel = select_survivors(r_all, c_all, N, xp)

            X, F, CV = X_all[sel], F_all[sel], CV_all[sel]
            rank, crowd = r_all[sel], c_all[sel]
            if G.size:
                G = G_all[sel]

        first = xp.flatnonzero(rank == 0)
        if len(first) > 1:                       # 按 f1 排序，便于出图
            Ff = to_numpy(F[first])
            order = np.lexsort(Ff.T[::-1])
            first = first[xp.asarray(order)]

        return ResultGPU(X=X, F=F, G=G, front=first, history=hist,
                         backend=getattr(xp, "__name__", "?"))
