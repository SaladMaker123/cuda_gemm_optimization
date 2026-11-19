# V1 — Shared Memory Tiling GEMM 性能分析报告

## 1. 实验环境与 Kernel 配置

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 32GB |
| NVIDIA Driver | 580.95.05 |
| CUDA Toolkit | 13.0 |
| Nsight Compute | 2025.2.0 |
| 矩阵规模 | M=N=K=2048 |
| 输入类型 | FP16 |
| 累加与输出类型 | FP32 |
| Block | (32, 8)，256 Threads |
| Grid | (64, 256) |
| BM / BN / BK | 8 / 32 / 32 |
| 编译参数 | `-O3 -std=c++17 -arch=sm_120 -lineinfo` |
| Kernel | `gemmV1` |

V1 引入 Shared Memory Tiling（共享内存分块）。

每个 Block 协作将 A、B 的 Tile 从 Global Memory 加载至 Shared Memory，随后使用 Shared Memory 中的数据执行乘加计算。

每个线程仍然负责一个输出元素。

## 2. 正确性与 Benchmark

每次程序运行包含 5 次 Warmup 和 30 次 CUDA Event 计时。

| 测试次数 | Correctness | Max Difference | Average Time | Performance |
|---|---|---:|---:|---:|
| 1 | PASS | 0.000366211 | 2.083 ms | 8.250 TFLOPS |
| 2 | PASS | 0.000366211 | 2.113 ms | 8.131 TFLOPS |
| 3 | PASS | 0.000366211 | 2.080 ms | 8.260 TFLOPS |
| **平均** | **PASS** | **0.000366211** | **2.092 ms** | **8.214 TFLOPS** |

与同一张 RTX 5090 上的 V0 比较：

| 指标 | V0 | V1 |
|---|---:|---:|
| Average Time | 2.398 ms | 2.092 ms |
| Performance | 7.166 TFLOPS | 8.214 TFLOPS |

V1 相对 V0：

- 吞吐性能提高约 **14.6%**。
- Kernel 执行时间降低约 **12.8%**。
- 加速比约为 **1.15×**。

以上性能基线来自正常 CUDA Event Benchmark，而不是 NCU Profiling 的执行时间。

## 3. NCU 关键指标分析

### 3.1 GPU Speed Of Light

位置：`Details → GPU Speed Of Light Throughput`

| 指标 | V0 | V1 |
|---|---:|---:|
| Compute (SM) Throughput | 96.36% | 93.86% |
| L1/TEX Cache Throughput | 96.97% | 94.47% |
| L2 Cache Throughput | 17.99% | 17.18% |
| DRAM Throughput | 0.29% | 0.34% |
| FMA Pipeline Utilization | 24.5% | 25.5% |

V1 的 L1/TEX 吞吐利用率略有下降，但仍然很高。

DRAM Throughput 仅为 0.34%，没有出现外部显存带宽饱和。

FMA 管线利用率约 25.5%，当前没有表现出 FP32 FMA 计算管线饱和。

### 3.2 Memory Workload Analysis

位置：`Details → Memory Workload Analysis → Memory Tables`

| 指标 | V0 | V1 |
|---|---:|---:|
| Global 读取相关指令 | 536,870,912 | 41,943,040 |
| Shared Load Instructions | 0 | 335,544,320 |
| Shared Store Instructions | 0 | 41,943,040 |
| Shared Load Bank Conflicts | 0 | 5,271 |

注意，V1 的 Global Memory 读取操作由 NCU 统计在 `Global Load To Shared Store (access)` 类别下，而不是普通的 `Global Load` 类别。

V1 的 Global Memory 读取相关指令数量相比 V0 减少约 **92.19%**。

这表明 Shared Memory Tiling 显著减少了重复 Global Memory 加载操作。

与此同时，V1 产生了约 3.36 亿条 Shared Load 指令。

Shared Load Bank Conflicts 为 5,271，当前没有观察到严重的 Shared Memory Bank Conflict 问题。

### 3.3 Cache Analysis

位置：`Details → Memory Workload Analysis`

| 指标 | V1 |
|---|---:|
| L1/TEX Hit Rate | 0.00% |
| L2 Hit Rate | 98.80% |
| Mem Pipes Busy | 93.86% |

V1 的 Global → Shared 复制采用 L1 Access 路径，但相关请求没有观察到 L1 缓存命中。

大量请求继续进入 L2，并且大部分在 L2 中命中。

L1/TEX Hit Rate 为 0% 并不表示 Shared Memory 没有复用数据，也不表示 L1/TEX 访存资源没有工作。

Mem Pipes Busy 达到 93.86%，表明内存相关指令处理管线非常繁忙。

### 3.4 Scheduler Statistics

位置：`Details → Scheduler Statistics`

| 指标 | V0 | V1 |
|---|---:|---:|
| Active Warps Per Scheduler | 11.70 | 11.71 |
| Eligible Warps Per Scheduler | 1.97 | 2.28 |
| No Eligible | 59.00% | 42.05% |

V1 的 Active Warps 基本不变，说明性能提升并不是通过增加 Warp 驻留数量实现。

Eligible Warps 增加，No Eligible 下降 16.95 个百分点，表明 Warp 调度器在更多周期中能够找到可发射指令的 Warp。

### 3.5 Warp State Statistics

位置：`Details → Warp State Statistics`

| 指标 | V0 | V1 |
|---|---|---|
| Warp Cycles Per Issued Instruction | 28.55 cycles | 20.20 cycles |
| 主要停顿类型 | LG Throttle | MIO Throttle |
| 突出停顿的平均周期 | LG：约 15.1 | MIO：约 7.7 |
| 对应归一化 Warp 周期比例 | 约 53.0% | 约 38.3% |

V1 的突出停顿类型由 LG Throttle 转变为 MIO Throttle。

MIO Throttle 表示 Warp 等待 MIO 指令队列出现可用容量。

结合 Shared Load Instructions 达到 335,544,320，以及 Mem Pipes Busy 达到 93.86%，可以判断 Shared Memory 相关指令处理压力是重要的性能限制因素。

需要注意，MIO 管线也处理部分其他类型的指令，不能将全部 MIO Throttle 停顿都直接归因于 Shared Memory Load。

## 4. 主要性能瓶颈

**V1 的突出瓶颈表现为 MIO Throttle，且与频繁的 Shared Memory 指令及较高的内存指令管线负载相伴随。**

主要证据：

1. Global Memory 读取相关指令数量相比 V0 降低约 92.19%。
2. Shared Load Instructions 达到约 3.36 亿条。
3. Mem Pipes Busy 为 93.86%。
4. MIO Throttle 约为 7.7 cycles，占归一化 Warp 周期的约 38.3%。
5. No Eligible 从 59.00% 降低到 42.05%。
6. DRAM Throughput 仅为 0.34%，外部显存带宽不是当前主要瓶颈。
7. Shared Memory Bank Conflict 很少，没有观察到严重冲突问题。

V1 减少了 Global Memory 加载压力，但仍存在大量 Shared Memory 读取指令。

## 5. 下一版本优化方向：V2 Register Tiling

下一步引入 Register Tiling（寄存器分块）。

核心思想：

让每个线程负责多个输出元素，使加载到寄存器中的部分 A、B 操作数能够参与多次乘加。

重点验证：

- Shared Load Instructions 是否下降。
- MIO Throttle 是否缓解。
- Warp 的 Eligible 和 Issued 情况是否改善。
- Registers Per Thread 是否增加。
- Occupancy 是否受到寄存器使用量影响。
- 正常 CUDA Event Benchmark 是否进一步提升。

## 6. 阶段结论

在 RTX 5090 上，V1 Shared Memory Tiling GEMM 正确性验证通过。

平均执行时间为 **2.092 ms**，平均吞吐性能为 **8.214 TFLOPS**。

相比 V0，吞吐提升约 **14.6%**。

NCU 表明，V1 显著减少了重复 Global Memory 读取指令，使 Warp 调度就绪情况有所改善。

但 Shared Memory 指令数量较大，MIO Throttle 成为当前突出的等待原因。

**下一阶段通过 V2 Register Tiling，提高线程内部的数据复用率，重点减少 Shared Memory 加载指令与相关队列压力。**