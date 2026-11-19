# V7 — BK64 Double Buffer GEMM 性能分析报告

## 1. 实验环境与优化方案

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 5090 32GB |
| Driver | 580.178.04 |
| CUDA / NCU | 13.0 / 2025.2.0 |
| 矩阵规模 | M=N=K=2048 |
| 输入 / 累加 / 输出 | FP16 / FP32 / FP32 |
| Tile | BM=64, BN=32, BK=64 |
| Block | (32, 8)，256 Threads |
| Kernel | `gemmV7BK64DoubleBuffer` |
| 编译参数 | `-O3 -std=c++17 -arch=sm_120 -lineinfo` |

V7 在 V6 的 `cp.async` 双缓冲基础上，将 BK 从 32 增加到 64。

K 阶段数从 64 减少到 32，以减少流水线切换和同步操作，并增加单阶段计算工作量。

双缓冲 Shared Memory 数据容量从 12 KiB 增加到 24 KiB。

## 2. Benchmark 结果

三次测试均通过正确性验证，最大误差为 `0.000366211`。

| 测试 | 时间 (ms) | TFLOPS |
|---|---:|---:|
| 第一次 | 0.647 | 26.560 |
| 第二次 | 0.646 | 26.597 |
| 第三次 | 0.647 | 26.556 |
| **平均** | **0.647** | **26.571** |

| 指标 | V6 | V7 |
|---|---:|---:|
| Average Time | 0.656 ms | 0.647 ms |
| Performance | 26.197 TFLOPS | 26.571 TFLOPS |

**V7 吞吐提升约 1.43%。**

## 3. Memory Workload Analysis

| 指标 | V6 | V7 |
|---|---:|---:|
| cp.async Requests | 1,572,864 | 1,572,864 |
| Global → Shared 读取 Bytes | 805,306,368 | 805,306,368 |
| Shared Load Instructions | 100,663,296 | 100,663,296 |
| Shared 异步写入 Wavefronts | 12,582,912 | 8,388,608 |
| Shared Load Bank Conflicts | 9,354 | 19,397 |

BK64 使每阶段加载量翻倍，但阶段数量减半，因此总异步复制请求数和缓存请求数据量保持不变。

Shared 异步写入 Wavefronts 减少约 **33.3%**，说明访问处理批次有所减少，但不能直接换算成等比例的时间收益。

Shared Load 指令数量保持不变，Bank Conflicts 虽有所增加，但相对于总请求规模仍较少。

## 4. Warp State 与计算管线

| 指标 | V6 | V7 |
|---|---:|---:|
| Warp Cycles / Issued Instruction | 11.99 | 8.73 |
| Barrier Stall | ≈1.66 | ≈0.63 |
| Math Pipe Throttle | ≈2.88 | ≈2.11 |
| Not Selected | ≈4.77 | ≈3.50 |
| Long Scoreboard | ≈0.02 | ≈0.02 |

注：Stall 周期为柱状图近似读数。

V7 的 K 阶段切换次数减半，Barrier 归一化等待明显下降。

Math Pipe Throttle 和 Not Selected 同样下降，但这些指标受到指令数量及活跃 Warp 数量影响，不能直接作为总耗时变化比例。

## 5. Scheduler 与 Occupancy

| 指标 | V6 | V7 |
|---|---:|---:|
| Registers / Thread | — | 40 |
| Block Limit Shared Mem | 7 | 4 |
| Theoretical Occupancy | 100% | 66.67% |
| Achieved Occupancy | 83.89% | 60.32% |
| Active Warps / Scheduler | 10.06 | 7.24 |
| Eligible Warps / Scheduler | 4.84 | 3.74 |
| Issued Warp / Scheduler | 0.84 | 0.83 |
| No Eligible | 16.05% | 17.02% |
| Issued IPC Active | 3.36 | 3.32 |
| FMA Utilization | 58.3% | 59.0% |

V7 的 Shared Memory 占用增加，使每个 SM 最多驻留 4 个 Block，理论 Occupancy 降至 66.67%。

尽管驻留 Warp 数量明显下降，Issued Warp、Issued IPC 和 FMA 利用率均基本保持稳定，说明当前 Warp 数量仍能提供较充分的指令发射机会。

## 6. 最终结论

V7 将 BK 从 32 增大到 64，取得以下结果：

- K 阶段数量减少 **50%**。
- Barrier 归一化等待明显下降。
- Shared 异步写入 Wavefronts 减少 **33.3%**。
- 理论 Occupancy 从 100% 降至 66.67%。
- 指令发射吞吐和 FMA 利用率基本保持稳定。

**V7 在 Occupancy 明显下降的情况下，仍通过更大的 K Tile 获得约 1.43% 的性能提升。**

实验表明，适当增大 Tile、减少流水线切换开销，可能比单纯维持更高 Occupancy 更有利于当前 GEMM Kernel。

最终成绩：

**V7 — 0.647 ms / 26.571 TFLOPS / Correctness PASS**

### 下一阶段

V8 将研究二维 Register Tiling（二维寄存器分块），调整每线程负责的输出矩阵形状，进一步优化数据复用与计算阶段的访存开销。