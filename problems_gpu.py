"""GPU 版的 ZDT / DTLZ / CONSTR 测试问题。

做法：直接继承 problems.py 里的类，只把类属性 xp 换成 cupy。
evaluate() 的数学表达式一行都不用重写 —— 所以 CPU 版和 GPU 版永远不会「算得不一样」。
"""

from __future__ import annotations

import cupy as cp

from problems import (CONSTR, DTLZ1, DTLZ2, PROBLEMS, ZDT1, ZDT2, ZDT3,
                      ZDT4, ZDT6)


class ZDT1GPU(ZDT1):
    xp = cp


class ZDT2GPU(ZDT2):
    xp = cp


class ZDT3GPU(ZDT3):
    xp = cp


class ZDT4GPU(ZDT4):
    xp = cp


class ZDT6GPU(ZDT6):
    xp = cp


class DTLZ1GPU(DTLZ1):
    xp = cp


class DTLZ2GPU(DTLZ2):
    xp = cp


class CONSTRGPU(CONSTR):
    xp = cp


PROBLEMS_GPU = {
    "ZDT1": ZDT1GPU, "ZDT2": ZDT2GPU, "ZDT3": ZDT3GPU, "ZDT4": ZDT4GPU,
    "ZDT6": ZDT6GPU, "DTLZ1": DTLZ1GPU, "DTLZ2": DTLZ2GPU, "CONSTR": CONSTRGPU,
}


def make_problem(name: str, backend: str = "auto", **kwargs):
    """按名字和后端构造问题实例。backend ∈ {auto, cupy, numpy}。"""
    name = name.upper()
    if name not in PROBLEMS:
        raise KeyError(f"未知算例 {name}，可选：{sorted(PROBLEMS)}")
    table = PROBLEMS if backend == "numpy" else PROBLEMS_GPU
    return table[name](**kwargs)
