# V10-H — Warp Specialization Triple Buffered WMMA GEMM

## 1. Optimization Overview

V10-H 基于 V10-G 的 Warp Specialization 实现，将双缓冲扩展为三缓冲，通过增加 Pipeline Stages 提升 Producer 的数据预取能力，尝试进一步重叠异步数据搬运与 WMMA 计算。

- GPU: NVIDIA RTX 5090
- CUDA: 13.0
- Matrix: M=N=K=2048
- Precision: FP16 Input / FP32 Accumulation
- Block Tile: 32×32×16
- WMMA Tile: 16×16×16
- Block Size: 160 Threads / 5 Warps
- Producer: 1 Warp
- Consumers: 4 Warps
- Buffer Stages: 3
- Shared Memory: 6 KiB (Logical Triple Buffer Storage)
- Data Transfer: `cuda::memcpy_async`
- Synchronization: `cuda::pipeline`

三个 Shared Memory Buffer 通过 `phase % 3` 循环使用。Producer Warp 负责异步加载，Consumer Warps 负责 WMMA 计算，通过 Pipeline Acquire、Commit、Wait 和 Release 管理缓冲区生命周期。

## 2. Benchmark Results

| Metric | V10-E | V10-G | V10-H |
|---|---:|---:|---:|
| Average Time | 0.252 ms | 0.257340 ms | **0.256367 ms** |
| Performance | **68.211 TFLOPS** | 66.760 TFLOPS | 67.013 TFLOPS |
| Correctness | PASS | PASS | PASS |
| Max Difference | 0 | 0 | 0 |

V10-H 三次 Benchmark 结果：

| Run | Average Time | Performance |
|---|---:|---:|
| 1 | 0.256585 ms | 66.956 TFLOPS |
| 2 | 0.256367 ms | 67.013 TFLOPS |
| 3 | 0.256150 ms | 67.069 TFLOPS |
| **Mean** | **0.256367 ms** | **67.013 TFLOPS** |

V10-H 相比 V10-G 吞吐提升约 **0.38%**，平均执行时间降低约 **0.38%**。

相比当前最佳版本 V10-E，吞吐仍低约 **1.76%**。

## 3. NCU Performance Analysis

### 3.1 Memory Workload

| Metric | V10-G | V10-H |
|---|---:|---:|
| Global Load To Shared Store (Bypass) | 2,097,152 | 2,097,152 |
| Shared Store From Global Load | 2,097,152 | 2,097,152 |
| Global Load Sectors | 33,554,432 | 33,554,432 |
| Global Load Bytes | 1 GiB | 1 GiB |
| Shared Load Matrix Instructions | 4,194,304 | 4,194,304 |
| Shared Load Matrix Wavefronts | 52.61 M | 52.77 M |
| Shared Load Bank Conflicts | 35.82 M | 36.00 M |
| L2 Hit Rate | 97.71% | 97.72% |
| Mem Busy | 88.49% | 90.34% |

V10-H 保持了与 V10-G 相同的异步 Global-to-Shared 数据搬运方式。

Global Load Instructions、Sectors 和 Bytes 均未发生明显变化，说明增加 Buffer Stages 未改变总数据搬运规模。

Shared Load Matrix Wavefronts 增加约 0.32%，Bank Conflicts 增加约 0.48%。

三缓冲未改善 WMMA Shared Memory Matrix Load 的 Bank Conflict 问题。

Mem Busy 从 88.49% 增至 90.34%，显示内存子系统保持较高活动水平。

### 3.2 Warp State & Compute

| Metric | V10-G | V10-H |
|---|---:|---:|
| Warp Cycles / Issued Instruction | 21.82 | 21.71 |
| Long Scoreboard | ≈8.5 | ≈6.5 |
| Short Scoreboard | ≈2.8 | ≈3.9 |
| Issued IPC Active | 1.90 | 1.96 |
| Issue Slots Busy | 47.52% | 49.10% |
| SM Busy | 47.52% | 49.10% |
| Tensor Pipe Utilization | ≈15.7% | ≈16.1% |
| ALU Active-Cycle Utilization | 21.3% | 20.3% |

V10-H 的 Warp Cycles / Issued Instruction 略有下降，Long Scoreboard Stall 减少，Short Scoreboard Stall 增加。

Issued IPC Active 从 1.90 提升至 1.96，Issue Slots Busy 提升至 49.10%。

Tensor Pipe Utilization 仅小幅改善，说明三缓冲未显著提高 Tensor Core 的持续计算利用率。

整体结果与三缓冲带来的有限性能收益一致。

### 3.3 Scheduler & Occupancy

| Metric | V10-G | V10-H |
|---|---:|---:|
| Active Warps / Scheduler | 10.10 | 10.39 |
| Eligible Warps / Scheduler | 0.84 | 0.91 |
| Issued Warp / Scheduler | 0.46 | 0.48 |
| No Eligible | 53.73% | 52.12% |
| Theoretical Occupancy | 93.75% | 93.75% |
| Achieved Occupancy | 86.84% | 88.17% |
| Achieved Active Warps / SM | 41.68 | 42.32 |
| Block Limit Shared Mem | 19 | 14 |
| Block Limit Warps | 9 | 9 |

V10-H 的 Eligible Warps / Scheduler 增至 0.91，No Eligible 降至 52.12%，表明 Warp 就绪状态有所改善。

三缓冲增加了 Shared Memory 资源消耗，使 Block Limit Shared Mem 从 19 降至 14。

但由于两个版本均受 Block Limit Warps 限制，每个 SM 的理论最大驻留 Block 数仍为 9。

理论最大驻留 Warp 数：

**9 × 5 = 45 Warps**

理论 Occupancy：

**45 ÷ 48 × 100% = 93.75%**

V10-H 的 Achieved Occupancy 达到 88.17%，比 V10-G 提高 1.33 个百分点。

增加 Buffer Stages 未降低理论 Occupancy，并伴随小幅调度效率改善。

## 4. Conclusion

V10-H 验证了 Triple Buffering 在 Warp Specialization WMMA GEMM 中的性能表现。

- Buffer Stages 从 2 增加至 3。
- Global-to-Shared 异步搬运方式与数据规模保持不变。
- Eligible Warps / Scheduler 从 0.84 提升至 0.91。
- Achieved Occupancy 从 86.84% 提升至 88.17%。
- Issued IPC Active 从 1.90 提升至 1.96。
- Shared Memory Matrix Load Bank Conflicts 仍然较高。
- Tensor Core 利用率仅小幅改善。
- 相比 V10-G，吞吐提升约 0.38%，但仍低于 V10-E。

本次实验表明，在当前 Warp Specialization 配置下，增加 Pipeline Stages 可以小幅改善 Warp 调度状态，但进一步提高缓冲深度的性能收益有限。


**Final Performance: 67.013 TFLOPS / 0.256367 ms**

**Performance Change vs V10-G: +0.38%**

**Performance Change vs V10-E: -1.76%**

**Best Configuration So Far: V10-E — 68.211 TFLOPS**