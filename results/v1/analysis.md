# V1 — Shared Memory GEMM 性能分析报告

## 1. 实验配置

| 项目             | 配置                            |
| -------------- | ----------------------------- |
| GPU            | NVIDIA GeForce RTX 5070       |
| CUDA Toolkit   | 13.0                          |
| Nsight Compute | 2025.2.0                      |
| 矩阵规模           | M=N=K=2048                    |
| 输入精度           | FP16                          |
| 累加与输出精度        | FP32                          |
| Block          | 32×8，256 Threads              |
| Tile           | BM=8，BN=32，BK=32              |
| 编译参数           | `-O3 -std=c++17 -arch=sm_120` |

V1 使用 Shared Memory Tiling（共享内存分块），让 Block 协作加载 A、B Tile，并在 Shared Memory 中复用数据。每个线程计算一个 C 元素。

## 2. Benchmark 结果

| 指标             |            V0 |           V1 |
| -------------- | ------------: | -----------: |
| Correctness    |          PASS |         PASS |
| Max Difference |    约 0.000366 |  0.000366211 |
| Average Time   |      约 8.0 ms |     6.794 ms |
| Performance    | 约 2.14 TFLOPS | 2.529 TFLOPS |

**V1 相比 V0，吞吐量提升约 18%，执行时间降低约 15%。**

正式性能数据来自 CUDA Event Benchmark，而不是 NCU Profiling 下的运行时间。

## 3. NCU 关键指标

| NCU Section / Metric                        |       V1 实测值 |
| ------------------------------------------- | -----------: |
| GPU SOL / L1/TEX Throughput                 |       95.72% |
| GPU SOL / DRAM Throughput                   |        0.32% |
| Compute / FMA Pipeline Utilization          |      约 25.8% |
| Occupancy / Achieved Occupancy              |       99.26% |
| Memory / Mem Pipes Busy                     |       95.58% |
| Memory Tables / Global Load Instructions    |   41,943,040 |
| Memory Tables / Global Load Sectors/Request |         2.00 |
| Memory Tables / Shared Load Instructions    |  335,544,320 |
| Memory Tables / Shared Load Bank Conflicts  |        6,043 |
| Scheduler / Eligible Warps Per Scheduler    |         2.40 |
| Scheduler / Issued Warps Per Scheduler      |         0.59 |
| Scheduler / No Eligible                     |       41.26% |
| Warp State / Stall MIO Throttle             | 约 8.0 cycles |

## 4. V0 → V1 优化效果

**Global Load Instructions：**

- V0：536,870,912
- V1：41,943,040
- 减少约 92.19%，指令数量降低 12.8 倍。

Shared Memory Tiling 显著减少了重复 Global Load 指令。

Global Load Sectors/Request 为 2.00，与 Warp 连续加载 32 个 FP16 元素的访问方式一致。

Shared Memory Bank Conflict 数量相对于总请求数量非常小，当前没有明显的严重 Bank Conflict 问题。

调度器发射情况也有所改善：

- V0 Issued Warps/Scheduler：约 0.41
- V1 Issued Warps/Scheduler：0.59



说明 V1 优化后，调度器能够更频繁地发射 Warp 指令。

## 5. 当前主要瓶颈

**V1 的主要性能瓶颈表现为MIO Throttle（内存相关指令队列限流）。**

证据：

1. Warp State Statistics 显示 Stall MIO Throttle 约为 8.0 cycles，占 Warp 平均指令发射间隔的约 39.6%。
2. Memory Workload Analysis 显示 Mem Pipes Busy 达到 95.58%。
3. Shared Load Instructions 达到约 3.36 亿条。
4. Shared Memory Bank Conflict 很少，说明严重 Bank Conflict 不是主要原因。
5. DRAM Throughput 只有 0.32%，没有表现出显存带宽饱和。
6. FMA Pipeline Utilization 约 25.8%，FP32 计算流水线未表现出明显饱和。
7. Achieved Occupancy 达到 99.26%，增加 Warp 驻留数量不是当前优先方向。

V0 的突出停顿原因为 LG Throttle，V1 则转变为 MIO Throttle。

**当前假设：频繁的 Shared Memory Load/Store 可能带来较大的 MIO 指令队列压力。** 由于 MIO 还处理其他类型指令，不能认定全部停顿都来自 Shared Memory。

## 6. 下一步优化方向：V2

计划研究 **Register Tiling（寄存器分块）**。

主要思路：

1. 让每个线程负责多个 C 元素。
2. 将部分从 Shared Memory 读取的操作数保存在寄存器中复用。
3. 尝试减少单位输出计算所需的 Shared Memory Load 指令。
4. 在提升数据复用的同时控制寄存器使用量。

V2 重点验证：

- Correctness 保持 PASS。
- Benchmark 执行时间是否进一步降低。
- Shared Load Instructions 是否减少。
- MIO Throttle 是否改善。
- Registers Per Thread 和 Occupancy 是否发生变化。

以上为待验证的优化假设，不预设 V2 一定加速。

## 7. 阶段结论

V1 通过 Shared Memory Tiling 成功减少重复 Global Load 指令，并取得约 18% 的初步吞吐量提升。

NCU 数据表明，当前主要限制已经从 V0 的 LG Throttle 转向 V1 的 MIO Throttle。

下一阶段将尝试通过 Register Tiling 改善寄存器级数据复用，进一步降低内存指令处理压力。
