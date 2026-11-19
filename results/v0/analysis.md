# V0 — Naive GEMM 性能分析报告

## 1. 实验配置

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 32GB |
| CUDA Toolkit | 13.0 |
| Nsight Compute | 2025.2.0 |
| NVIDIA Driver | 580.95.05 |
| 矩阵规模 | M=N=K=2048 |
| 输入精度 | FP16 |
| 累加与输出精度 | FP32 |
| Block | (32, 8)，256 Threads |
| Grid | (64, 256)，16384 Blocks |
| Kernel | `gemmNaive` |
| 编译参数 | `-O3 -std=c++17 -arch=sm_120 -lineinfo` |

V0 采用 Naive GEMM 实现，每个线程计算输出矩阵 C 的一个元素。

线程在 K 方向循环中直接从 Global Memory 加载 A、B 元素，并执行 FP32 累加，不使用 Shared Memory Tiling、Register Tiling 或 Tensor Core WMMA。

## 2. Benchmark 结果

使用 CUDA Event 测量 Kernel 执行时间。每次程序运行包含 5 次 Warmup 和 30 次正式计时。

| 测试次数 | Correctness | Max Difference | Average Time | Performance |
|---|---|---:|---:|---:|
| 1 | PASS | 0.000366211 | 2.360 ms | 7.280 TFLOPS |
| 2 | PASS | 0.000366211 | 2.402 ms | 7.154 TFLOPS |
| 3 | PASS | 0.000366211 | 2.432 ms | 7.065 TFLOPS |
| **平均** | **PASS** | **0.000366211** | **2.398 ms** | **7.166 TFLOPS** |

**RTX 5090 的 V0 正式基线为约 2.398 ms、7.166 TFLOPS。**

该结果来自正常 CUDA Event Benchmark，不采用 NCU Profiling 下的 Kernel Duration 作为性能基线。

## 3. NCU 关键性能指标

### 3.1 GPU Speed Of Light

位置：`Details → GPU Speed Of Light Throughput`

| 指标 | RTX 5090 实测值 |
|---|---:|
| Compute (SM) Throughput | 96.36% |
| Memory Throughput | 96.36% |
| L1/TEX Cache Throughput | 96.97% |
| L2 Cache Throughput | 17.99% |
| DRAM Throughput | 0.29% |
| FMA Pipeline Utilization | 24.5% |

分析：

- L1/TEX Cache Throughput 达到 96.97%，是当前高度繁忙的硬件资源。
- DRAM Throughput 仅为 0.29%，外部显存带宽没有饱和。
- FMA Pipeline Utilization 约 24.5%，FP32 FMA 计算管线没有表现出同等级别的饱和。
- Compute (SM) Throughput 是综合吞吐指标，不等于 FP32 运算单元利用率。

### 3.2 Scheduler Statistics

位置：`Details → Scheduler Statistics`

| 指标 | 实测值 |
|---|---:|
| Active Warps Per Scheduler | 11.70 |
| Theoretical Warps Per Scheduler | 12 |
| Eligible Warps Per Scheduler | 1.97 |
| Issued Warp Per Scheduler | 0.41 |
| No Eligible | 59.00% |

分析：

每个 Scheduler 平均拥有 11.70 个 Active Warp，接近理论上限 12。

但是，平均仅有 1.97 个 Warp 具备指令发射条件。

59.00% 的调度周期没有 Eligible Warp，平均每周期仅发射 0.41 条 Warp 指令。

说明 V0 的主要问题不是驻留 Warp 数量不足，而是大量 Warp 因等待而无法继续发射指令。

### 3.3 Warp State Statistics

位置：`Details → Warp State Statistics → LG Throttle Stalls`

| 指标 | 实测值 |
|---|---:|
| Warp Cycles Per Issued Instruction | 28.55 cycles |
| LG Throttle Stall | 约 15.1 cycles |
| LG Throttle 占归一化 Warp 周期比例 | 约 53.0% |

分析：

LG Throttle 表示 Warp 等待 Global/Local Memory 相关的 L1 指令请求队列出现空余容量。

本次约 15.1 cycles 的等待占 Warp 指令发射间隔统计的约 53.0%。

这一结果说明 Global/Local Memory 请求队列限流是 V0 的突出停顿原因。

该百分比不是 Kernel 总执行时间占比，不能直接换算成可获得的性能提升。

### 3.4 L1/TEX Cache

位置：`Details → Memory Workload Analysis → Memory Tables → L1/TEX Cache`

| 指标 | Global Load | Global Store |
|---|---:|---:|
| Instructions | 536,870,912 | 131,072 |
| Requests | 536,870,912 | 131,072 |
| Sectors | 805,306,368 | 524,288 |
| Sectors/Req | 1.50 | 4.00 |
| Hit Rate | 89.51% | 0% |
| Bytes | 25,769,803,776 | 16,777,216 |

Global Load 指令数量可以从 Kernel 结构推导：

- 总共 131072 个 Warp。
- 每个 Warp 沿 K 方向循环 2048 次。
- 每次循环分别加载 A、B，执行两条 Warp 级 Global Load 指令。

因此：

`131072 × 2048 × 2 = 536870912`

与 NCU 实测的 Global Load Instructions 完全一致。

对于 Sectors/Req：

- A：同一 Warp 的线程访问相同 A 元素，通常涉及一个 Sector。
- B：32 个线程读取连续的 32 个 FP16 元素，通常涉及两个 Sector。

平均：

`(1 + 2) / 2 = 1.50`

这与 NCU 结果相符。

当前访存模式下，主要优化机会是减少重复 Global Load 指令，而不只是改变单次加载的地址合并方式。

### 3.5 L2 Cache

位置：`Details → Memory Workload Analysis → Memory Tables → L2 Cache → L1/TEX Load`

| 指标 | 实测值 |
|---|---:|
| Requests | 50,754,905 |
| Sectors | 84,473,920 |
| Sectors/Req | 1.66 |
| Hit Rate | 99.38% |
| Bytes | 2,703,165,440 |
| Sector Misses to Device | 524,288 |

L2 的加载命中率达到 99.38%。

继续向 GPU Device Memory 请求的读取数据约为：

`524288 × 32 = 16777216 Bytes`

即约 16.78 MB。

虽然 V0 执行了大量 Global Load，但大部分访问可以在缓存层级中获得数据，因此 DRAM 带宽利用率很低。

## 4. 主要性能瓶颈

**V0 的主要瓶颈是重复 Global Load 导致的 L1/TEX 访存指令与请求处理压力，并伴随明显的 LG Throttle 停顿。**

证据如下：

1. Global Load Instructions 高达 536,870,912。
2. L1/TEX Cache Throughput 达到 96.97%。
3. LG Throttle 约占归一化 Warp 周期的 53.0%。
4. No Eligible 达到 59.00%，Warp 指令发射能力受到明显限制。
5. DRAM Throughput 仅为 0.29%，没有出现外部显存带宽饱和。
6. FMA Pipeline Utilization 约为 24.5%，当前未表现出 FP32 FMA 管线饱和。

这些结果说明：当前应该优先减少重复访存指令，而不是简单增加 Warp 数量。

## 5. 下一版本优化方向：V1 Shared Memory Tiling

V1 引入 Shared Memory Tiling（共享内存分块）。

主要目标：

1. 让 Block 内线程协作加载 A、B Tile。
2. 将加载的数据暂存在 Shared Memory。
3. 在计算中复用 Shared Memory 中的数据。
4. 减少重复的 Global Load 指令和 L1/TEX 请求压力。

V1 需要重点验证：

- Global Load Instructions 是否明显减少。
- L1/TEX Throughput 是否下降。
- LG Throttle 是否得到缓解。
- Shared Load Instructions 是否明显增加。
- 新的主要停顿原因是否转变为 MIO Throttle。
- 正常 Benchmark 性能是否获得提升。

这些是后续需要通过 NCU 验证的优化假设。

## 6. 阶段结论

RTX 5090 上，V0 Naive GEMM 正确性验证通过。

正式 Benchmark 平均执行时间为 **2.398 ms**，平均吞吐性能为 **7.166 TFLOPS**。

NCU 表明，V0 的 Global Load 指令数量非常大，L1/TEX 资源接近峰值利用率，LG Throttle 是突出的 Warp 停顿原因。

大量加载由 L1、L2 Cache 满足，外部显存带宽并未饱和。

**下一步进入 V1 Shared Memory Tiling，验证数据复用能否减少 Global Load 指令数量并缓解 L1/TEX 请求处理压力。**