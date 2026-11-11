"""GPU 版 NSGA-II 的命令行入口 / 基准脚本。

用法：
    python demo_gpu.py -p ZDT1 -n 400 -g 500                 # 默认用 GPU
    python demo_gpu.py -p ZDT1 -n 400 -g 500 --compare       # 同时跑 CPU 循环版对比
    python demo_gpu.py -p DTLZ2 -n 300 -g 500 --n-obj 3      # 3 目标
    python demo_gpu.py -p ZDT1 --backend numpy -n 1000       # 无卡时跑 CPU 向量化版
    python demo_gpu.py -p ZDT1 -n 1000 -g 300 --sweep        # 规模扫描，看加速比趋势

输出：result_<问题>.csv、pareto_<问题>.png、终端性能与指标表。
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from metrics import summarize
from problems import PROBLEMS


def _build(name: str, backend: str, n_obj: int = 3):
    """构造问题实例（GPU 版 / CPU 版）。"""
    cls_cpu = PROBLEMS[name]
    is_dtlz = name.startswith("DTLZ")
    if backend == "numpy":
        return cls_cpu(n_obj=n_obj) if is_dtlz else cls_cpu()
    from problems_gpu import PROBLEMS_GPU
    cls_gpu = PROBLEMS_GPU[name]
    return cls_gpu(n_obj=n_obj) if is_dtlz else cls_gpu()


def _sync(backend: str) -> None:
    if backend == "cupy":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()


def _peak_gpu_mem_mb() -> float:
    try:
        import cupy as cp
        return cp.get_default_memory_pool().used_bytes() / 1024 ** 2
    except Exception:
        return float("nan")


def run_gpu(problem, args, backend: str):
    from nsga2_gpu import NSGA2GPU, device_info, get_backend

    xp = get_backend("numpy" if backend == "numpy" else "auto")
    best, res = float("inf"), None
    for k in range(args.repeat):
        _sync(backend)
        t0 = time.perf_counter()
        algo = NSGA2GPU(problem, pop_size=args.pop, max_gen=args.gen,
                        eta_c=args.eta_c, eta_m=args.eta_m, p_cross=args.pc,
                        seed=args.seed + k, xp=xp,
                        record_history=not args.no_history)
        res = algo.run(verbose=(args.verbose and k == 0))
        _sync(backend)
        best = min(best, time.perf_counter() - t0)
    return res, best, device_info(), _peak_gpu_mem_mb()


def run_cpu_loop(problem, args):
    from nsga2 import NSGA2
    best, res = float("inf"), None
    for k in range(args.repeat):
        t0 = time.perf_counter()
        res = NSGA2(problem, pop_size=args.pop, max_gen=args.gen,
                    eta_c=args.eta_c, eta_m=args.eta_m, p_cross=args.pc,
                    seed=args.seed + k).run()
        best = min(best, time.perf_counter() - t0)
    return res, best


def report(tag: str, F: np.ndarray, problem, sec: float) -> None:
    try:
        PF = problem.pareto_front(500)
        ref = PF.max(axis=0) * 1.1
        m = summarize(F, PF, ref)
        print(f"{tag:<10} {sec:8.3f}s | IGD {m['IGD']:.3e} | GD {m['GD']:.3e} "
              f"| Spacing {m['Spacing']:.3e} | |PF| = {m['N']}")
    except NotImplementedError:
        print(f"{tag:<10} {sec:8.3f}s | |PF| = {len(F)}（无解析前沿）")


def plot_front(F: np.ndarray, problem, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 5), dpi=150)
    if F.shape[1] == 2:
        ax = fig.add_subplot(111)
        try:
            PF = problem.pareto_front(500)
            ax.plot(PF[:, 0], PF[:, 1], "k-", lw=1.2, label="True PF")
        except NotImplementedError:
            pass
        ax.scatter(F[:, 0], F[:, 1], s=18, c="#c0392b", label="NSGA-II (GPU)")
        ax.set_xlabel("$f_1$")
        ax.set_ylabel("$f_2$")
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=14, c="#c0392b",
                   label="NSGA-II (GPU)")
        ax.set_xlabel("$f_1$")
        ax.set_ylabel("$f_2$")
        ax.set_zlabel("$f_3$")
    ax.set_title(f"NSGA-II on {problem.name}")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="NSGA-II GPU (CuPy) benchmark")
    ap.add_argument("-p", "--problem", default="ZDT1", choices=list(PROBLEMS))
    ap.add_argument("-n", "--pop", type=int, default=400)
    ap.add_argument("-g", "--gen", type=int, default=500)
    ap.add_argument("-s", "--seed", type=int, default=1)
    ap.add_argument("--backend", default="cupy", choices=["cupy", "numpy"])
    ap.add_argument("--n-obj", type=int, default=3)
    ap.add_argument("--pc", type=float, default=0.9)
    ap.add_argument("--eta-c", type=float, default=20.0)
    ap.add_argument("--eta-m", type=float, default=20.0)
    ap.add_argument("--repeat", type=int, default=1, help="重复运行取最短耗时")
    ap.add_argument("--compare", action="store_true", help="同时跑 CPU 循环版对比")
    ap.add_argument("--sweep", action="store_true",
                    help="规模扫描：N = 100/200/400/800/1600，输出加速比")
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--plot", default="")
    args = ap.parse_args()

    if args.sweep:
        sweep(args)
        return

    problem = _build(args.problem, args.backend, args.n_obj)
    res, sec, dev, mem = run_gpu(problem, args, args.backend)

    F = res.F_front_np()
    print(f"\n设备 : {dev}")
    print(f"配置 : {problem} | N={args.pop} gen={args.gen} "
          f"seed={args.seed} repeat={args.repeat}")
    report(f"GPU/向量", F, _build(args.problem, "numpy", args.n_obj), sec)
    if not np.isnan(mem):
        print(f"显存 : 峰值池占用 ≈ {mem:.1f} MB")

    if args.compare and args.backend == "cupy":
        cpu_prob = _build(args.problem, "numpy", args.n_obj)
        res_cpu, sec_cpu = run_cpu_loop(cpu_prob, args)
        report("CPU 循环", res_cpu.F_front, cpu_prob, sec_cpu)
        print(f"加速比 : {sec_cpu / sec:.2f}×")

    csv = f"result_{args.problem}.csv"
    np.savetxt(csv, F, delimiter=",",
               header=",".join(f"f{i+1}" for i in range(F.shape[1])), comments="")
    print(f"已保存: {os.path.abspath(csv)}")

    if not args.no_plot:
        png = args.plot or f"pareto_{args.problem}.png"
        plot_front(F, _build(args.problem, "numpy", args.n_obj), png)
        print(f"已保存: {os.path.abspath(png)}")


def sweep(args) -> None:
    """固定代数，扫描种群规模，观察 GPU 相对两种 CPU 实现的加速比。

    CPU 向量化 = 同一份 nsga2_gpu 代码 + xp=numpy；CPU 循环 = nsga2.py 的逐个体实现。
    循环版在 N >= 800 时太慢，自动跳过。
    """
    sizes = [200, 400, 800, 1600, 3200]
    gen = min(args.gen, 100)
    from nsga2_gpu import NSGA2GPU
    NSGA2GPU(_build(args.problem, "cupy", args.n_obj), pop_size=60,
             max_gen=3, seed=0).run()          # 预热：触发 kernel JIT 编译
    _sync("cupy")
    print(f"算例 {args.problem} | gen={gen}（每档取最短耗时）\n")
    print(f"{'N':>6} | {'GPU':>9} | {'CPU向量化':>10} | {'加速比':>7} | "
          f"{'CPU循环':>9} | {'加速比':>7} | {'IGD':>9}")
    print("-" * 82)
    for n in sizes:
        a = argparse.Namespace(**vars(args))
        a.pop, a.gen = n, gen
        a.repeat = args.repeat
        a.no_history = True

        res, sec_g, _, _ = run_gpu(_build(args.problem, "cupy", args.n_obj), a, "cupy")

        vec_prob = _build(args.problem, "numpy", args.n_obj)
        from nsga2_gpu import NSGA2GPU
        t0 = time.perf_counter()
        NSGA2GPU(vec_prob, pop_size=n, max_gen=gen, seed=a.seed,
                 record_history=False, xp="numpy").run()
        sec_v = time.perf_counter() - t0

        if n <= 400:
            _, sec_c = run_cpu_loop(vec_prob, a)
            c_txt = f"{sec_c:9.2f} | {sec_c / sec_g:6.2f}×"
        else:
            c_txt = f"{'—':>9} | {'—':>7}"

        try:
            igd = summarize(res.F_front_np(), vec_prob.pareto_front(500))["IGD"]
        except NotImplementedError:
            igd = float("nan")
        print(f"{n:6d} | {sec_g:9.2f} | {sec_v:10.2f} | {sec_v / sec_g:6.2f}× | "
              f"{c_txt} | {igd:9.2e}")


if __name__ == "__main__":
    main()
