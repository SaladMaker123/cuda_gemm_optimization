# V5 — Vectorized Shared Store GEMM 性能分析报告

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
| Kernel | `gemmV5VectorizedSharedStore` |

V5 在 V4 的 128-bit Global Load 基础上，使用 `uint4` 实现 128-bit Shared Store。

通过直接将 16 Bytes 数据写入 Shared Memory，替代 V4 的拆分存储方式，减少 Shared Store 指令和访问冲突。Register Tiling R8 与计算逻辑保持不变。

## 2. Benchmark 结果

三次测试均通过正确性验证，最大误差为 `0.000366211`。

| 测试 | 时间 (ms) | TFLOPS |
|---|---:|---:|
| 第一次 | 0.661 | 25.991 |
| 第二次 | 0.661 | 25.998 |
| 第三次 | 0.661 | 26.007 |
| **平均** | **0.661** | **25.999** |

与 V4 对比：

| 指标 | V4 | V5 |
|---|---:|---:|
| Average Time | 0.718 ms | 0.661 ms |
| Performance | 23.924 TFLOPS | 25.999 TFLOPS |

**V5 吞吐性能提升约 8.7%，执行时间降低约 7.9%。**

## 3. Memory Workload Analysis

| 指标 | V4 | V5 |
|---|---:|---:|
| Global Load Instructions | 1,572,864 | 1,572,864 |
| Shared Load Instructions | 100,663,296 | 100,663,296 |
| Shared Store Instructions | 3,145,728 | 1,572,864 |
| Shared Store Wavefronts | 13,046,064 | 6,761,413 |
| Shared Store Bank Conflicts | 6,754,176 | 469,336 |

V5 保持 Global Load 指令数量不变，Shared Store Instructions 减少 **50%**。

Shared Store Wavefronts 减少约 **48.2%**，Bank Conflicts 减少约 **93.1%**。

这表明 128-bit Shared Store 显著降低了存储指令数量，并改善了 Shared Memory 存储访问的冲突情况。

## 4. Warp State 与计算管线

### 4.1 Warp State Statistics

| 指标 | V4 | V5 |
|---|---:|---:|
| Warp Cycles / Issued Instruction | 13.32 | 12.81 |
| Not Selected | ≈5.0 | ≈3.95 |
| Math Pipe Throttle | ≈3.2 | ≈2.31 |
| Barrier | ≈1.8 | ≈3.33 |
| MIO Throttle | ≈0.3 | ≈0.33 |

注：Stall 周期为 NCU 柱状图的近似读数。

V5 的归一化 Warp 周期数下降，Math Pipe Throttle 与 Not Selected 减少。

Barrier 等待有所增加，成为较突出的 Warp Stall 类型。由于该指标按已发射指令归一化，不能直接认定同步总耗时同比例增加。

### 4.2 Scheduler 与 Compute Workload

| 指标 | V4 | V5 |
|---|---:|---:|
| Active Warps / Scheduler | 9.87 | 10.48 |
| Issued Warp / Scheduler | 0.74 | 0.82 |
| No Eligible | 25.92% | 18.21% |
| Issued IPC Active | 2.96 | 3.27 |
| Issue Slots Busy | 74.11% | 81.81% |
| FMA Pipeline Utilization | 52.0% | 58.5% |

V5 的 Warp 发射效率明显改善，No Eligible 下降 7.71 个百分点。

FMA 管线利用率从 52.0% 提高至 58.5%，说明数据存储优化后，计算资源得到了更充分的利用。

## 5. Occupancy 分析

| 指标 | V4 | V5 |
|---|---:|---:|
| Theoretical Occupancy | 100% | 100% |
| Achieved Occupancy | 82.26% | 87.38% |
| Active Warps / SM | 39.49 | 41.94 |
| Block Limit Registers | 6 | 6 |

V5 的实际 Occupancy 提高 **5.12 个百分点**。

两个版本的理论 Occupancy 均为 100%，寄存器资源允许的最大驻留 Block 数量也相同。

因此，V5 的性能改善伴随着实际 Warp 活跃程度提高，但不是通过增加理论驻留上限实现的。

## 6. 最终结论

V5 通过 128-bit Shared Store 实现了：

- Shared Store Instructions 减少 **50%**。
- Shared Store Bank Conflicts 减少 **93.1%**。
- Shared Store Wavefronts 减少 **48.2%**。
- Warp 指令发射效率提高。
- FMA 管线利用率由 52.0% 提高至 58.5%。
- 实际 Occupancy 由 82.26% 提高至 87.38%。

**V5 的主要收益来自更高效的 Shared Memory 存储，减少存储指令和 Bank Conflict 开销，并改善整体 Warp 调度与计算资源利用情况。**

最终性能：

**V5 — 0.661 ms / 25.999 TFLOPS / Correctness PASS**

### 下一阶段

V6 将引入 Double Buffering（双缓冲），研究 Global Memory 数据加载与 GEMM 计算之间的重叠执行，进一步减少数据准备阶段对计算的阻塞。