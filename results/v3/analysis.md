# V3 — Vectorized Global Load GEMM 性能分析报告

## 1. 实验环境与优化方案

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 5090 32GB |
| CUDA / NCU | 13.0 / 2025.2.0 |
| 矩阵规模 | M=N=K=2048 |
| 输入 / 累加 / 输出 | FP16 / FP32 / FP32 |
| Tile | BM=64, BN=32, BK=32 |
| Block | (32, 8)，256 Threads |
| 编译参数 | `-O3 -std=c++17 -arch=sm_120 -lineinfo` |
| Kernel | `gemmV3Vectorized` |

V3 在 V2-R8 Register Tiling 的基础上，引入 `__half2` 向量化全局内存加载，将单次 Global Load 宽度从 16-bit 提高到 32-bit。

每次读取两个连续 FP16 元素，减少 Global Load 指令数量。计算逻辑、Tile 配置及每线程 8 个输出的设计保持不变。

## 2. Benchmark 结果

所有测试均通过 cuBLAS 正确性验证，最大误差为 `0.000366211`。

| 测试 | 时间 (ms) | TFLOPS |
|---|---:|---:|
| 第一次 | 0.796 | 21.579 |
| 第二次 | 0.797 | 21.551 |
| 第三次 | 0.792 | 21.682 |
| **平均** | **0.795** | **21.604** |

与 V2-R8 对比：

| 指标 | V2-R8 | V3 |
|---|---:|---:|
| Average Time | 0.863 ms | 0.795 ms |
| Performance | 19.898 TFLOPS | 21.604 TFLOPS |

**V3 吞吐提升 8.6%，执行时间降低约 7.9%。**

## 3. Memory Workload Analysis

### 3.1 Global Memory

| 指标 | V2-R8 | V3 |
|---|---:|---:|
| Global Load Instructions | 12,582,912 | 6,291,456 |
| Sectors/Request | 2.00 | 4.00 |
| Total Sectors | 25,165,824 | 25,165,824 |
| Load Bytes | 805,306,368 | 805,306,368 |

Global Load 指令数量减少 **50%**，而总 Sector 数和 L1/TEX 统计的数据传输量保持不变。

说明 `__half2` 向量化加载成功降低了内存加载指令及请求处理数量，但没有减少需要传输的数据总量。

### 3.2 Shared Memory

| 指标 | V2-R8 | V3 |
|---|---:|---:|
| Shared Load Instructions | 100,663,296 | 100,663,296 |
| Shared Store Instructions | 12,582,912 | 6,291,456 |
| Shared Load Bank Conflicts | 6,580 | 7,616 |
| Shared Store Bank Conflicts | 154,266 | 194,946 |

Shared Load 数量不变，因为计算阶段没有修改。

Shared Store 指令减少 **50%**，与编译器将相邻 FP16 存储合并为更宽指令的解释相符。

虽然 Shared Store Bank Conflicts 增加，但额外 Wavefront 约占 Shared Store Requests 的 3.24%，目前没有证据表明它是主要性能瓶颈。

## 4. Warp 调度与计算管线

### 4.1 Warp State Statistics

| 指标 | V2-R8 | V3 |
|---|---:|---:|
| Warp Cycles / Issued Instruction | 14.61 | 13.19 |
| Long Scoreboard | ≈4.2 | ≈2.5 |
| Math Pipe Throttle | ≈2.3 | ≈2.6 |
| MIO Throttle | ≈0.24 | ≈0.2 |

注：Stall 周期为 NCU 柱状图近似读数。

V3 的 Long Scoreboard 明显下降，说明 Global Load 指令减少的同时，加载相关数据依赖等待有所缓解。

Math Pipe Throttle 略有增加，数学执行管线开始表现出更明显的处理压力，但尚不能认定为主要瓶颈。

### 4.2 Scheduler Statistics

| 指标 | V2-R8 | V3 |
|---|---:|---:|
| Active Warps / Scheduler | 10.07 | 9.99 |
| Eligible Warps / Scheduler | 2.95 | 3.88 |
| Issued Warp / Scheduler | 0.69 | 0.76 |
| No Eligible | 31.05% | 24.26% |

V3 的 Eligible Warps 提高约 31.5%，Issued Warp 提高约 10.1%，No Eligible 下降 6.79 个百分点。

说明 Warp 就绪情况改善，调度器能够更加频繁地发射指令。

### 4.3 Compute Workload

| 指标 | V3 |
|---|---:|
| Issued IPC Active | 3.03 inst/cycle |
| Issue Slots Busy | 75.69% |
| FMA Pipeline Utilization | 49.8% |

FMA 是利用率较高的计算管线，但尚未表现出持续满载的特征。

## 5. Occupancy 与资源占用

| 指标 | V2-R8 | V3 |
|---|---:|---:|
| Registers Per Thread | 40 | 40 |
| Shared Memory / Block | 6.14 KB | 6.14 KB |
| Theoretical Occupancy | 100% | 100% |
| Achieved Occupancy | ≈84.1% | 83.35% |
| Active Warps / SM | 40.38 | 40.01 |

V3 与 V2-R8 的寄存器使用量、Shared Memory 配置及理论 Occupancy 相同。

因此，V3 的加速不是通过提高 Warp 驻留上限实现的。

V3 的 Grid 为 2048 Blocks，Waves Per SM 为 2.01，存在尾部波次利用率不足的问题，但尚未通过实际实验量化其性能影响。

## 6. 最终结论

V3 通过 `__half2` 向量化加载实现了以下优化：

- Global Load Instructions 减少 **50%**。
- Shared Store Instructions 减少 **50%**。
- Global Load 数据传输总量不变。
- Long Scoreboard 等待明显降低。
- Warp 就绪和指令发射效率改善。
- 寄存器使用量及理论 Occupancy 保持不变。

**V3 的主要收益来自减少内存操作指令数量，并伴随着数据依赖等待和 Warp 调度情况改善。**

最终性能：**0.795 ms / 21.604 TFLOPS**，相比 V2-R8 提升 **8.6%**。

### 下一阶段：V4

将 Global Load 宽度进一步提高到 **128-bit（每次加载 8 个 FP16）**，重点研究 Global Load 指令数量、Shared Memory Store、Bank Conflicts 及 Warp 停顿变化。

目标是在 V3 的基础上进一步降低访存指令开销，提高 GEMM 吞吐性能。