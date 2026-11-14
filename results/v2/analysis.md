# V2 Register Tiling GEMM — Performance Analysis

## 1. 优化目标

V1 使用 Shared Memory Tiling，显著减少了 Global Load 指令，但仍然存在较高的 MIO Throttle（内存相关指令队列限流）。

V2 引入 Register Tiling，使每个线程计算两个 C 元素，通过寄存器复用 B 操作数，尝试减少 Shared Memory 加载指令和 MIO 队列压力。

## 2. Kernel 配置

| 参数 | V1 | V2-R2 |
|---|---:|---:|
| BM | 8 | 16 |
| BN | 32 | 32 |
| BK | 32 | 32 |
| Threads / Block | 256 | 256 |
| Outputs / Thread | 1 | 2 |
| C Tile | 8×32 | 16×32 |
| Grid | 64×256 | 64×128 |

V2 使用 FP16 输入、FP32 累加及 FP32 输出。

每个线程维护 `sum0` 和 `sum1` 两个累加器，并复用同一个 B 操作数完成两次乘加。

## 3. Benchmark 结果

测试环境：NVIDIA GeForce RTX 5070、CUDA 13.0、矩阵尺寸 M=N=K=2048。

| 指标 | V1 | V2-R2 |
|---|---:|---:|
| Correctness | PASS | PASS |
| Average Time | 6.719 ms | 4.125 ms |
| Performance | 2.557 TFLOPS | 4.165 TFLOPS |

V2 相比 V1：

- Kernel 执行时间减少约 **38.6%**。
- 计算吞吐提升约 **62.9%**。
- 加速比约为 **1.63×**。

两个版本均在当前 RTX 5070 实例上重新运行 Benchmark。

## 4. Memory Workload Analysis

| 指标 | V1 | V2-R2 | 变化 |
|---|---:|---:|---|
| Global Load Instructions | 41,943,040 | 25,165,824 | -40% |
| Shared Load Instructions | 335,544,320 | 201,326,592 | -40% |
| Shared Store Instructions | 41,943,040 | 25,165,824 | -40% |

V2 减少了加载指令数量，说明更大的 C Tile 和线程内部寄存器复用有效减少了重复的数据加载工作。

以上 V1 内存指令数据来自此前保存的 NCU 基线，V2 数据来自本次测试；正式进行严格的内存计数器对照时，应在相同环境重新采集 V1。

## 5. Scheduler Statistics

| 指标 | V1 | V2-R2 |
|---|---:|---:|
| Active Warps / Scheduler | 11.91 | 11.84 |
| Eligible Warps / Scheduler | 2.40 | 2.60 |
| Issued Warp / Scheduler | 0.59 | 0.67 |
| No Eligible | 41.26% | 33.40% |

分析：

- Active Warps 基本保持不变。
- Eligible Warps 增加，说明平均每周期有更多 Warp 满足指令发射条件。
- Issued Warp 增加，表明调度器的平均指令发射效率有所改善。
- No Eligible 降低约 7.86 个百分点。

V2 的性能改善并不是通过增加 Warp 驻留数量实现，而是与更好的 Warp 就绪和指令发射情况相伴随。

## 6. Warp State Statistics

| Stall Reason | V1 | V2-R2 |
|---|---:|---:|
| MIO Throttle | 8.02 | 5.58 |
| Long Scoreboard | 2.92 | 3.11 |
| Barrier | 2.61 | 1.77 |

上述指标单位为按已发射 Warp 指令归一化的平均等待周期。

主要发现：

- MIO Throttle 下降约 30.4%，表明相关指令队列限流有所缓解。
- Barrier 等待下降约 32.2%。
- Long Scoreboard 略微增加，但不能据此直接认定单次 Global Memory 加载延迟变高。

MIO Throttle 仍然是 V2 的主要等待原因之一。

## 7. Source / SASS 与 PC Sampling 分析

使用 `-lineinfo` 编译 V2，并结合 NCU Source Counters 和 SASS 进行分析。

在 Source 视图中观察到：

| CUDA 源码行 | 对应操作 | MIO Throttle 采样占比 |
|---|---|---:|
| 84 | 从 Shared Memory 读取 B 操作数 | 37.51% |
| 86 | sum0 乘加 | 8.02% |
| 87 | sum1 乘加 | 8.47% |

通过 SASS 观察到：

- `LDS.U16` 和 `LDS.64` 等 Shared Memory 加载指令。
- 两条不同的 `FFMA` 指令复用相同的寄存器操作数。
- Shared Memory 加载相关源码位置集中出现 MIO Throttle 采样。

PC Sampling 表示采样时 Warp 的指令位置，不代表该指令一定是造成队列拥堵的根本原因；采样百分比也不能直接解释为 Kernel 时间占比。

## 8. 结论与下一步

V2 Register Tiling 成功实现每线程两个输出元素，并通过寄存器复用减少了数据加载指令。

实验表明：

1. 正确性验证通过。
2. 相比 V1，Benchmark 性能明显提升。
3. Shared Load 指令数量减少约 40%。
4. MIO Throttle 和 No Eligible 指标均有所改善。
5. 计算循环中的 Shared Memory 加载仍是重要的优化候选区域。

下一步可尝试 **R4 Register Tiling（每线程计算四个输出元素）**，进一步提高 B 操作数的寄存器复用率。

实验时重点观察 Shared Load Instructions、MIO Throttle、Registers Per Thread、Occupancy 和 Benchmark，验证继续增加寄存器分块是否仍能带来性能收益。
