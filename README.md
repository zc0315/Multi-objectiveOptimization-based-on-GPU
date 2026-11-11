# NSGA-II GPU 版（CuPy）

GPU 版不是把 `np` 换成 `cp` 就完事。第一版直接替换的结果是**比 CPU 还慢**：
SBX / 变异 / 锦标赛在 `nsga2.py` 里是逐个体逐变量的 Python 循环，每次只处理几十个数，
Windows 上单个 kernel 启动约 25–40 µs，全部收益被启动开销吃掉（N=400 时 20 ms/代）。

本模块把每一代压到尽量少的 kernel 上，同样配置降到 11 ms/代。下面是要点、实测数据和坑。

## 文件

| 文件 | 说明 |
|---|---|
| `nsga2_gpu.py` | GPU 主算法 `NSGA2GPU`；另含 `rank_population` / `crowding_all` / `select_survivors` 等可单独复用的组件 |
| `problems_gpu.py` | ZDT / DTLZ / CONSTR 的 GPU 版：继承 CPU 版，只把类属性 `xp` 改成 cupy，数学表达式零重复 |
| `demo_gpu.py` | 命令行入口；`--compare` 与 CPU 循环版对比，`--sweep` 做规模扫描 |
| `bench_gpu.py` | 批量基准：多算例 × 多 seed，输出明细/汇总 CSV 与 Excel |
| `TUTORIAL_GPU.md` | **详细教程**：安装、用法、原理、接自定义问题、批量实验、调优、报错速查 |

`problems.py` 已改成后端无关（`Problem.xp` 决定用 numpy 还是 cupy），
所以 CPU 版和 GPU 版共用同一份目标函数代码，不会出现「两个版本算得不一样」。

## 安装（有一个必须注意的坑）

```bash
pip install cupy-cuda12x==13.6.0 --no-deps   # CUDA 12.x
pip install fastrlock                         # CuPy 的必需依赖
```

**不要装最新的 `cupy-cuda12x` 14.x**：它要求 `numpy>=2.0`，pip 会把 numpy 升到 2.5，
从而打破 numba 0.60（要求 <2.1）、scipy 1.13（要求 <2.3）、gensim（要求 <2.0）。
用 13.6.0 + numpy 1.26.4 这一组合即可共存（本机 RTX 4060 Laptop / CUDA 12.4 实测通过）。

验证：

```bash
python -c "import cupy as cp; print(cp.__version__, cp.cuda.runtime.getDeviceCount())"
```

## 用法

```bash
python demo_gpu.py -p ZDT1 -n 1000 -g 500              # GPU
python demo_gpu.py -p ZDT1 -n 400 -g 400 --compare     # 与 CPU 循环版对比
python demo_gpu.py -p ZDT1 --sweep                     # 规模扫描（N=200…3200）
python demo_gpu.py -p DTLZ2 -n 300 -g 500 --n-obj 3
python demo_gpu.py -p ZDT1 -n 400 --backend numpy      # 无卡时跑 CPU 向量化版
```

代码里调用：

```python
from problems_gpu import PROBLEMS_GPU
from nsga2_gpu import NSGA2GPU

prob = PROBLEMS_GPU["ZDT1"]()
res = NSGA2GPU(prob, pop_size=1000, max_gen=500, seed=1).run()
F = res.F_front_np()     # 拷回 host；res.host() 返回 X/F/G/front 全量
```

`xp="numpy"` 时同一份代码退化为 CPU 向量化版，且比 `nsga2.py` 的循环版快 2–9 倍。

## 四个关键改动

1. **非支配排序**：`(2N, 2N)` 支配矩阵一次广播算出，然后分层剥离。
   剥离时用矩阵-向量乘递减支配计数（`cur @ D` 走 cuBLAS），**不用**布尔掩码 gather
   —— 后者会触发 `nonzero` + 拷贝，慢一个量级。取当前层 / 写 rank 各融合成一个
   `ElementwiseKernel`，每层只剩 4 次 kernel 启动。
2. **提前停止剥离**：一旦已分配的个体数达到 N（需要保留的规模），剩下的层对选择结果
   没有任何影响（会被环境选择整层淘汰），直接跳过并把它们标成大 rank。
   收敛后期前沿层数会从十几层涨到上百层，这一条省掉大量无用迭代。
3. **拥挤距离**：不再按前沿循环。对每个目标做一次全局 argsort（键 = `rank*4 + 归一化目标值`），
   前沿在排序序列里是连续块，用块边界掩码 + 回散射，一次算出**所有前沿**的拥挤距离。
   N=400 时从 26 ms 降到 2.5 ms。
4. **环境选择**：把 `(rank, 拥挤距离)` 编码成单调键 `rank + (1 - c/(1+c))/2`，
   一次 argsort 取前 N，取代「逐层填充 + 最后一层按拥挤距离截断」的循环。

## 性能实测（RTX 4060 Laptop，ZDT1，100 代，取最短耗时）

| N | GPU | CPU 向量化（`xp=numpy`） | 加速比 | CPU 循环（`nsga2.py`） | 加速比 | IGD |
|---|---|---|---|---|---|---|
| 200 | 0.96 s | 0.62 s | 0.64× | 1.56 s | 1.63× | 6.96e-03 |
| 400 | 1.19 s | 2.24 s | 1.87× | 4.23 s | 3.54× | 3.60e-03 |
| 800 | 1.87 s | 8.74 s | 4.68× | — | — | 1.73e-03 |
| 1600 | 4.61 s | 34.11 s | 7.40× | — | — | 9.93e-04 |
| 3200 | 15.29 s | 135.31 s | 8.85× | — | — | 5.35e-04 |

每代耗时：N=200 约 9.6 ms，N=400 约 12 ms，N=1600 约 46 ms，N=3200 约 153 ms。

**结论要看清**：`ZDT1` 这类目标函数极便宜的问题，每次迭代的计算量本来就小
（N=200 时一步只有 200×30 个数），CPU 上 numpy 向量化几乎无启动开销，所以
**N ≲ 250 时 GPU 反而更慢**；N ≥ 400 才开始赢，到 N=3200 约 9 倍。
真正吃 GPU 的场景是：① 大规模种群（N ≥ 1000）；② 目标函数昂贵且可向量化
（CFD/结构仿真代理模型、神经网络评估），那时评估本身在 GPU 上，加速比会远超表中数字。

## 质量对齐（N=100/105，250 代，seed=1）

| 问题 | IGD（CPU 循环） | IGD（GPU） | Spacing（CPU / GPU） |
|---|---|---|---|
| ZDT1 | 5.30e-03 | 5.10e-03 | 5.00e-03 / 7.20e-03 |
| ZDT2 | 5.09e-03 | 4.72e-03 | 6.76e-03 / 5.33e-03 |
| ZDT3 | 5.40e-03 | 5.58e-03 | 7.32e-03 / 7.92e-03 |
| ZDT4 | 6.36e-03 | 8.54e-03 | 5.62e-03 / 6.18e-03 |
| ZDT6 | 8.86e-03 | 9.17e-03 | 4.31e-03 / 3.54e-03 |
| DTLZ2 (M=3) | 6.50e-02 | 6.65e-02 | 3.81e-02 / 3.80e-02 |
| CONSTR | 2.16e-02 | 2.34e-02 | 3.57e-02 / 5.12e-02 |

差异来自随机数流不同，不是算法差异（同量级、同趋势，ZDT4 这种多局部最优问题抖动本就大）。

## 正确性验证

新写的三个组件都与「逐前沿参考实现」逐元素比对通过（numpy 与 cupy 两种后端各 5 组随机数据，
含 20% 不可行解、故意把层数上界设小以触发兜底分支）：

- `rank_population` vs `fast_non_dominated_sort`：rank 完全一致
- `crowding_all` vs 逐前沿 `crowding_distance`：拥挤距离完全一致
- `select_survivors` vs 逐层填充：选中集合完全一致
- 提前停止剥离（`n_need`）与完整排序：选择结果的 `(rank, -crowding)` 键多重集完全一致
  （差异仅出现在 key 完全相同的平局个体之间，属 NSGA-II 允许的任意选择）

SBX / 变异的向量化实现与 `nsga2.py` 的标量 Deb 实现做过统计对拍（各 20 万样本）：

| 统计量 | 向量化 | 标量 |
|---|---|---|
| `E[c1] - 中点` | -0.0826, 0.0276, -0.1380, 0.1371 | -0.0828, 0.0276, -0.1385, 0.1379 |
| `std(c1)` | 0.1255, 0.0418, 0.2091, 0.2097 | 0.1254, 0.0418, 0.2088, 0.2091 |
| 变异 `std`（x=0.5, η_m=20） | 0.063 | 0.061–0.063 |

（`E[c1]` 偏离中点属于 Deb 原始 SBX 的边界修正特性，标量版同样如此，不是 bug。）

## 显存

支配矩阵布尔 `(2N)²` + float32 副本 + 目标比较临时张量 `(2N)²·M`：
N=1000 约 60 MB，N=3200 约 700 MB，8 GB 显存可跑到 N ≈ 5000。
用 `cupy.get_default_memory_pool().free_all_blocks()` 可主动归还。
