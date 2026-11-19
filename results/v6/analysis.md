# V6 — Async Copy + Double Buffering GEMM 性能分析报告

## 1. 实验环境与优化方案

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 5090 32GB |
| NVIDIA Driver | 580.178.04 |
| CUDA / NCU | 13.0 / 2025.2.0 |
| 矩阵规模 | M=N=K=2048 |
| 输入 / 累加 / 输出 | FP16 / FP32 / FP32 |
| Tile | BM=64, BN=32, BK=32 |
| Block | (32, 8)，256 Threads |
| Kernel | `gemmV6DoubleBuffer` |

V6 在 V5 的基础上引入 `cp.async` 异步 Global → Shared 复制和双缓冲机制。

使用两组 Shared Memory 缓冲区，在计算当前 K Tile 时提前加载下一 Tile，尝试实现数据加载与计算重叠。

Shared Memory 数据缓冲量由 6 KiB 增加到 12 KiB，Register Tiling R8 计算逻辑保持不变。

## 2. Benchmark 结果

三次测试均通过正确性验证，最大误差为 `0.000366211`。

| 测试 | 时间 (ms) | TFLOPS |
|---|---:|---:|
| 第一次 | 0.657 | 26.156 |
| 第二次 | 0.653 | 26.304 |
| 第三次 | 0.657 | 26.130 |
| **平均** | **0.656** | **26.197** |

与 V5 对比：

| 指标 | V5 | V6 |
|---|---:|---:|
| Average Time | 0.661 ms | 0.656 ms |
| Performance | 25.999 TFLOPS | 26.197 TFLOPS |

**V6 吞吐性能提升约 0.76%。** 提升幅度较小，需要更多重复测试确认稳定性。

## 3. Memory Workload Analysis

| 指标 | V5 | V6 |
|---|---:|---:|
| 普通 Global Load Instructions | 1,572,864 | 0 |
| Global Load To Shared Store 请求 | 0 | 1,572,864 |
| 普通 Shared Store Instructions | 1,572,864 | 0 |
| Shared Load Instructions | 100,663,296 | 100,663,296 |
| Shared Load Bank Conflicts | 84,114 | 9,354 |

V6 的 Global → Shared 数据搬运成功切换到 `cp.async` 路径，不再使用普通 Global Load 与 Shared Store 指令组合。

异步复制请求数为 1,572,864，普通 Shared Load 数量保持不变。

这说明数据搬运机制已经改变，但计算阶段的 Shared Memory 读取工作量没有减少。

## 4. Warp State 与计算管线

### 4.1 Warp State Statistics

| 指标 | V5 | V6 |
|---|---:|---:|
| Warp Cycles / Issued Instruction | 12.81 | 11.99 |
| Barrier | ≈3.33 | ≈1.66 |
| Long Scoreboard | ≈0.46 | ≈0.02 |
| Math Pipe Throttle | ≈2.31 | ≈2.88 |
| Not Selected | ≈3.95 | ≈4.77 |

注：Stall 周期为 NCU 柱状图近似读数。

V6 的 Barrier 和 Long Scoreboard 归一化等待明显下降，与异步复制及双缓冲的设计目标相符。

但 Math Pipe Throttle 和 Not Selected 增加，计算管线与 Warp 调度仍然具有一定优化空间。

这些指标不能直接量化异步加载与计算的实际重叠时间。

### 4.2 Scheduler 与 Compute Workload

| 指标 | V5 | V6 |
|---|---:|---:|
| Active Warps / Scheduler | 10.48 | 10.06 |
| Eligible Warps / Scheduler | 4.05 | 4.84 |
| Issued Warp / Scheduler | 0.82 | 0.84 |
| No Eligible | 18.21% | 16.05% |
| Issued IPC Active | 3.27 | 3.36 |
| Issue Slots Busy | 81.81% | 83.99% |
| FMA Pipeline Utilization | 58.5% | 58.3% |

V6 的 Eligible Warps 增加，No Eligible 减少，Warp 指令发射效率略有提高。

但 FMA 管线利用率基本不变，说明双缓冲没有显著改变计算阶段的执行效率。

## 5. Occupancy 分析

| 指标 | V5 | V6 |
|---|---:|---:|
| Theoretical Occupancy | 100% | 100% |
| Achieved Occupancy | 87.38% | 83.89% |
| Active Warps / SM | 41.94 | 40.27 |
| Block Limit Registers | 6 | 6 |
| Block Limit Shared Mem | 14 | 7 |

V6 的双缓冲使 Shared Memory 需求增加，但没有降低理论最大 Warp 驻留数量。

实际 Occupancy 下降约 3.49 个百分点，但指令就绪与发射效率有所改善。

## 6. 最终结论

V6 实现了以下变化：

- 使用 `cp.async` 替代普通 Global Load + Shared Store。
- 通过双缓冲实现下一 Tile 的提前预取。
- Barrier 与 Long Scoreboard 归一化等待下降。
- Warp 指令就绪和发射效率略有改善。
- Shared Load 指令数量不变。
- Shared Memory 用量翻倍，实际 Occupancy 略降。

**V6 成功实现异步复制与双缓冲设计，但相比 V5 仅获得约 0.76% 的性能提升。**

现有数据支持部分等待得到缓解，但尚不能证明充分的加载与计算重叠。

最终成绩：

**V6 — 0.656 ms / 26.197 TFLOPS / Correctness PASS**

### 下一阶段

进一步研究 Global Memory 到 Shared Memory 的异步流水线设计，以及计算阶段的 Shared Load 和数学指令执行开销，尝试提高数据搬运与计算的重叠效率。