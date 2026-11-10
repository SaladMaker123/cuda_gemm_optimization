# V0 — Naive GEMM 性能分析报告

## 1. 实验配置

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| CUDA Toolkit | 13.0 |
| Nsight Compute | 2025.2.0 |
| 矩阵规模 | M=N=K=2048 |
| 输入精度 | FP16 |
| 累加与输出精度 | FP32 |
| Block | 32×8，256 Threads |
| 计算方式 | 每个 Thread 计算一个 C 元素 |
| 编译参数 | `-O3 -std=c++17 -arch=sm_120` |

## 2. Benchmark 结果

| 指标 | 第一次 | 第二次 |
|---|---:|---:|
| Correctness | PASS | PASS |
| Max Difference | 0.00036211 | 0.00036211 |
| Average Time | 8.023 ms | 8.058 ms |
| Performance | 2.141 TFLOPS | 2.132 TFLOPS |

**V0 基线：约 8.0 ms，2.13 TFLOPS。**

正式性能基线来自 CUDA Event Benchmark，不使用 NCU Profiling 下的运行时间。

## 3. NCU 关键证据

| NCU 指标 | 实测值 | 说明 |
|---|---:|---|
| Achieved Occupancy | 99.25% | Warp 驻留数量已接近上限 |
| L1/TEX Throughput | 97.54% | L1/TEX 相关资源非常繁忙 |
| DRAM Throughput | 0.28% | 外部显存带宽没有饱和 |
| LG Throttle | 约 16.1 cycles | 内存请求队列限流是主要停顿原因 |
| No Eligible | 58.77% | 调度器经常没有可发射指令的 Warp |
| Global Load Instructions | 536,870,912 | 大量 Global Load 指令 |
| Global Load Sectors/Request | 1.50 | 与 A 广播、B 连续加载的访问模型一致 |
| L1/TEX Global Load Hit Rate | 89.48% | 大量加载由 L1 缓存满足 |
| L2 Load Hit Rate | 99.38% | 到达 L2 的加载大部分命中 |
| Global Load Data Utilization | 22 B / 32 B | 存在改善加载数据使用效率的空间 |

## 4. 主要性能瓶颈

**主要瓶颈：Global Memory Load 指令过多，导致 LSU/L1TEX 请求处理压力较大，并产生明显的 LG Throttle 停顿。**

证据如下：

1. V0 每个线程独立计算一个输出元素，在 K 循环中反复读取 A、B。
2. NCU 统计到约 5.37 亿条 Warp 级 Global Load 指令。
3. LSU 指令吞吐接近峰值，L1/TEX 吞吐利用率达到 97.54%。
4. LG Throttle 是最突出的 Warp 停顿原因。
5. 虽然 Occupancy 达到 99.25%，调度器仍有 58.77% 的周期没有 Eligible Warp。

**不是主要瓶颈的部分：**

- DRAM 带宽：利用率仅 0.28%，没有出现饱和。
- Warp 驻留数量：Occupancy 已接近 100%。
- FMA 计算流水线：本次利用率约 24.5%，没有表现出明显饱和。

以上判断基于当前 NCU 采集结果，不代表其他瓶颈完全不存在。

## 5. 下一步优化方向：V1

**V1 — Shared Memory Tiling（共享内存分块）**

计划：

1. 让一个 Block 共同计算一个 C Tile。
2. 线程协作把 A、B 的 Tile 从 Global Memory 搬运到 Shared Memory。
3. 在 K 方向循环中重复使用 Shared Memory 中的数据。
4. 减少重复的 Global Memory Load 指令及相关请求。

**V1 重点验证：**

- 正确性保持 PASS。
- Kernel 执行时间是否下降。
- Global Load 指令数量是否减少。
- LG Throttle 是否改善。
- 新增 Shared Memory 访问与同步是否造成额外瓶颈。

如果 V1 没有加速，应保留实测结果并分析原因，不预设优化一定成功。

## 6. 阶段结论

V0 已完成正确性验证、Benchmark 和 NCU 分析。

当前 GPU 缓存已发挥明显的数据复用作用，但 Kernel 仍存在大量重复 Global Load 指令，限制了指令发射效率。

**下一阶段重点不是提高 Occupancy，而是通过 Shared Memory Tiling 改善数据复用并减少 Global Memory 加载压力。**