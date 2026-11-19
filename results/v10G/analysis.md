# V10-G — Warp Specialization Double Buffered WMMA GEMM

## 1. Optimization Overview

V10-G 在双缓冲 WMMA GEMM 的基础上引入 Warp Specialization，通过 Producer-Consumer 分工，将异步数据搬运与矩阵计算分配给不同 Warp。

- GPU: NVIDIA RTX 5090
- CUDA: 13.0
- Matrix: M=N=K=2048
- Precision: FP16 Input / FP32 Accumulation
- Block Tile: 32×32×16
- WMMA Tile: 16×16×16
- Block Size: 160 Threads / 5 Warps
- Producer: 1 Warp
- Consumers: 4 Warps
- Buffer Stages: 2
- Data Transfer: `cuda::memcpy_async`
- Synchronization: `cuda::pipeline`

Producer Warp 负责 Global-to-Shared 异步数据搬运，四个 Consumer Warps 分别负责一个 16×16 输出 Tile 的 WMMA 计算。

通过双缓冲机制尝试重叠下一 K Tile 的加载与当前 Tile 的计算。

## 2. Benchmark Results

| Metric | V10-E | V10-F | V10-G |
|---|---:|---:|---:|
| Average Time | 0.252 ms | 0.257877 ms | **0.257340 ms** |
| Performance | **68.211 TFLOPS** | 66.621 TFLOPS | 66.760 TFLOPS |
| Correctness | PASS | PASS | PASS |
| Max Difference | 0 | 0 | 0 |

V10-G 三次 Benchmark 结果：

| Run | Average Time | Performance |
|---|---:|---:|
| 1 | 0.257517 ms | 66.714 TFLOPS |
| 2 | 0.256826 ms | 66.893 TFLOPS |
| 3 | 0.257677 ms | 66.672 TFLOPS |
| **Mean** | **0.257340 ms** | **66.760 TFLOPS** |

V10-G 相比 V10-F 吞吐提升约 **0.21%**，相比 V10-E 下降约 **2.13%**。

当前 Warp Specialization 实现未超过 V10-E 的最佳性能。

## 3. NCU Performance Analysis

### 3.1 Memory Workload

| Metric | V10-F | V10-G |
|---|---:|---:|
| TMA Load Instructions | 1,048,576 | 0 |
| Global Load To Shared Store (Bypass) | 0 | 2,097,152 |
| Shared Store From Global Load | 0 | 2,097,152 |
| Shared Load Matrix Instructions | 4,194,304 | 4,194,304 |
| Shared Load Matrix Wavefronts | 52.67 M | 52.61 M |
| Shared Load Bank Conflicts | 35.90 M | 35.82 M |
| L2 Hit Rate | 97.59% | 97.71% |
| Mem Busy | 84.39% | 88.49% |

V10-G 使用 `cuda::memcpy_async` 实现 Global-to-Shared 数据搬运，并由独立 Producer Warp 负责异步加载。

NCU 确认数据搬运进入异步拷贝路径。

Shared Load Matrix Wavefronts 和 Bank Conflicts 与 V10-F 基本一致，说明 Warp Specialization 未能显著改善 WMMA 的 Shared Memory 访问效率。

Shared Load Bank Conflicts 仍达到 35.82 M，是后续优化的重要方向。

### 3.2 Warp State & Compute

| Metric | V10-F | V10-G |
|---|---:|---:|
| Warp Cycles / Issued Instruction | 26.65 | 21.82 |
| Long Scoreboard | ≈8.2 | ≈8.5 |
| Issued IPC Active | 1.67 | 1.90 |
| Issue Slots Busy | 41.85% | 47.52% |
| SM Busy | 41.85% | 47.52% |
| Tensor Pipe Utilization | ≈14.7% | ≈15.7% |
| ALU Active-Cycle Utilization | ≈11.8% | 21.3% |

V10-G 的 Warp Cycles / Issued Instruction 下降约 18.1%，Issued IPC Active 提升约 13.8%。

Issue Slots Busy 提升至 47.52%，表明整体指令发射活动增加。

Long Scoreboard 仍是主要 Stall 来源，MIO Throttle 与 Branch Resolving 也占据明显比例。

Tensor Pipe Utilization 仅小幅改善，未获得与整体指令发射率相匹配的计算吞吐提升。

### 3.3 Scheduler & Occupancy

| Metric | V10-F | V10-G |
|---|---:|---:|
| Active Warps / Scheduler | 11.30 | 10.10 |
| Eligible Warps / Scheduler | 0.68 | 0.84 |
| Issued Warp / Scheduler | 0.42 | 0.46 |
| No Eligible | 57.60% | 53.73% |
| Theoretical Occupancy | 100.00% | 93.75% |
| Achieved Occupancy | 93.30% | 86.84% |
| Achieved Active Warps / SM | 44.78 | 41.68 |

V10-G 通过 Producer-Consumer 分工提高了 Eligible Warps 数量与指令发射率。

但每个 Block 的 Warp 数量由 4 增加至 5，导致 Warp 资源最多允许每个 SM 驻留 9 个 Block。

理论最大驻留 Warp 数：

**9 × 5 = 45 Warps**

对应理论 Occupancy：

**45 ÷ 48 × 100% = 93.75%**

Achieved Occupancy 下降至 86.84%，但 Warp 发射效率仍略高于 V10-F。

## 4. Conclusion

V10-G 验证了 Warp Specialization 与双缓冲异步流水线在 WMMA GEMM 中的实现效果。

- Producer Warp 与 Consumer Warps 实现独立的加载和计算分工。
- `cuda::pipeline` 支持异步数据搬运与计算之间的协作。
- Issued IPC Active 从 1.67 提升至 1.90。
- Eligible Warps / Scheduler 从 0.68 提升至 0.84。
- Achieved Occupancy 因每个 Block 的 Warp 数增加而下降至 86.84%。
- Shared Memory Matrix Load Bank Conflicts 仍然较高。
- 计算吞吐较 V10-F 小幅改善，但未超过 V10-E。

本次实验表明，Warp Specialization 能够改善指令调度效率，但最终性能仍受到线程资源分配、流水线同步及 Shared Memory 访问效率的共同影响。

**Final Performance: 66.760 TFLOPS / 0.257340 ms**

**Performance Change vs V10-F: +0.21%**

**Performance Change vs V10-E: -2.13%**

**Best Configuration So Far: V10-E — 68.211 TFLOPS**