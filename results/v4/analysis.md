# V4 — 128-bit Vectorized Global Load GEMM 性能分析报告

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
| 编译参数 | `-O3 -std=c++17 -arch=sm_120 -lineinfo` |
| Kernel | `gemmV4Vectorized128` |

V4 在 V3 的基础上使用 `uint4` 实现 128-bit Global Load，将单线程每次加载的数据宽度从 4 Bytes 提高至 16 Bytes。

每次可加载 8 个连续 FP16 元素。计算逻辑和 Register Tiling 配置保持不变。

## 2. Benchmark 结果

三次运行均通过正确性验证，最大误差为 `0.000366211`。

| 测试 | 时间 (ms) | TFLOPS |
|---|---:|---:|
| 第一次 | 0.718 | 23.938 |
| 第二次 | 0.717 | 23.956 |
| 第三次 | 0.720 | 23.877 |
| **平均** | **0.718** | **23.924** |

相同 GPU、更新驱动后的性能对比：

| 指标 | V3 | V4 |
|---|---:|---:|
| Average Time | 0.788 ms | 0.718 ms |
| Performance | 21.789 TFLOPS | 23.924 TFLOPS |

**V4 吞吐性能提升约 9.8%，执行时间降低约 8.9%。**

## 3. Memory Workload Analysis

### 3.1 Global Memory

| 指标 | V3 | V4 |
|---|---:|---:|
| Global Load Instructions | 6,291,456 | 1,572,864 |
| Sectors/Request | 4 | 16 |
| Total Sectors | 25,165,824 | 25,165,824 |
| Load Bytes | 805,306,368 | 805,306,368 |

Global Load 指令数量减少 **75%**。

V4 每次加载 16 Bytes，是 V3 的 4 倍。虽然每个 Warp 请求涉及更多 Sector，但总 Sector 数量和 L1/TEX 统计的数据传输量保持不变。

**128-bit 向量化加载减少了加载指令及请求处理开销，而不是减少数据传输总量。**

### 3.2 Shared Memory

| 指标 | V3 | V4 |
|---|---:|---:|
| Shared Load Instructions | 100,663,296 | 100,663,296 |
| Shared Store Instructions | 6,291,456 | 3,145,728 |
| Shared Store Wavefronts | 6,495,389 | 13,046,064 |
| Shared Store Bank Conflicts | 194,946 | 6,754,176 |

Shared Load 数量不变，Shared Store 指令减少 50%。

但 V4 的 Shared Store Bank Conflicts 大幅增加，Wavefronts 总量也明显上升。

这表明 V4 的 Shared Memory 存储操作产生了更高的访问处理开销，可能抵消部分 Global Load 向量化收益。

具体冲突来源仍需要结合 SASS 和线程地址映射进一步确认。

## 4. Warp State 与计算管线

### 4.1 Warp State Statistics

| 指标 | V4 |
|---|---:|
| Warp Cycles / Issued Instruction | 13.32 |
| Not Selected | ≈5.0 cycles |
| Math Pipe Throttle | ≈3.2 cycles |
| Barrier | ≈1.8 cycles |
| Long Scoreboard | ≈0.6 cycles |
| MIO Throttle | ≈0.3 cycles |

注：除 NCU 明确给出的 Not Selected 外，其余 Stall 周期为柱状图近似读数。

V4 的 Long Scoreboard 和 MIO Throttle 较低，说明内存加载相关数据依赖等待和 MIO 队列限流并不突出。

虽然 Shared Store Bank Conflicts 显著增加，但没有观察到严重的 MIO Throttle。

Math Pipe Throttle 和 Barrier 是当前更值得关注的等待因素。

### 4.2 Compute Workload Analysis

| 指标 | V4 |
|---|---:|
| Issued IPC Active | 2.96 inst/cycle |
| Issue Slots Busy | 74.11% |
| FMA Pipeline Utilization | 52.0% |
| LSU 峰值指令吞吐利用率 | 约 67% |

FMA 是利用率较高的计算管线，但尚未表现出持续饱和。

LSU 指令吞吐利用率较高，说明访存相关指令仍然占用较多执行资源。

## 5. Scheduler Statistics 与 Occupancy

### 5.1 Warp 调度

| 指标 | V4 |
|---|---:|
| Active Warps / Scheduler | 9.87 |
| Eligible Warps / Scheduler | 4.45 |
| Issued Warp / Scheduler | 0.74 |
| No Eligible | 25.92% |

V4 平均有较多 Eligible Warps，但每个 Scheduler 每周期实际发射约 0.74 条 Warp 指令。

相比 V3，此指标略有下降，但 V4 仍然获得更高性能，说明减少总访存指令数量比单纯提高每周期指令发射数量更有价值。

### 5.2 寄存器与 Occupancy

| 指标 | V3 | V4 |
|---|---:|---:|
| Registers Per Thread | 40 | 37 |
| Theoretical Occupancy | 100% | 100% |
| Achieved Occupancy | 83.35% | 82.26% |
| Active Warps / SM | 40.01 | 39.49 |
| Block Limit Registers | 6 | 6 |

V4 每线程使用 37 个寄存器，低于 V3 的 40 个。

两个版本均可达到 100% 理论 Occupancy，实际 Occupancy 接近。

因此，V4 的性能提升并非来自更高的 Warp 驻留数量。

## 6. 最终结论

V4 使用 `uint4` 将 Global Load 宽度提高至 128-bit，实现：

- Global Load Instructions 减少 **75%**。
- Shared Store Instructions 减少 **50%**。
- Global Load 总数据传输量不变。
- Shared Store Bank Conflicts 明显增加。
- Long Scoreboard 和 MIO Throttle 保持较低水平。
- 寄存器占用略有下降，理论 Occupancy 保持 100%。

**V4 通过更宽的向量化加载显著降低了 Global Load 指令数量，并取得约 9.8% 的吞吐提升。**

但 Shared Store 的 Bank Conflicts 和访问 Wavefronts 明显增加，说明数据搬运过程仍存在优化空间。

最终成绩：

**V4 — 0.718 ms / 23.924 TFLOPS / Correctness PASS**

### 下一阶段

在保持 128-bit Global Load 的基础上，进一步研究 Shared Memory 存储向量化，减少数据拆分与 Shared Store 开销，并分析 Bank Conflict 的改善效果。