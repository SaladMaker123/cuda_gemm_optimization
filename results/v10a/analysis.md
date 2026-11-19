# V10-A — One-Warp WMMA Tensor Core GEMM

## 1. Optimization Overview

V10-A 引入 NVIDIA WMMA API，使用 Tensor Core 执行 FP16 矩阵乘法与 FP32 累加，建立 Tensor Core GEMM 性能基线。

- GPU: NVIDIA RTX 5090
- CUDA: 13.0
- Matrix: M=N=K=2048
- Precision: FP16 Input / FP32 Accumulation
- WMMA Tile: 16×16×16
- Block Size: 32 Threads
- Memory Access: Direct Global Memory Load

每个 Block 使用一个 Warp 计算 16×16 输出 Tile，不使用显式 Shared Memory Tiling。

## 2. Benchmark Results

| Metric | V9-A SIMT | V10-A WMMA |
|---|---:|---:|
| Average Time | 0.458 ms | 0.604 ms |
| Performance | 37.538 TFLOPS | 28.458 TFLOPS |
| Correctness | PASS | PASS |
| Max Difference | 0.000366211 | 0 |

V10-A 相比 V9-A 的吞吐下降约 **24.2%**。

## 3. NCU Performance Analysis

### 3.1 Memory Workload

| Metric | V10-A |
|---|---:|
| Global Load Instructions | 16.78 M |
| L1/TEX Global Load Bytes | 4.00 GiB |
| L1/TEX Hit Rate | 50.90% |
| L2 Load Hit Rate | 99.20% |
| Device Memory Load Bytes | 16.01 MiB |
| Device Memory Peak Utilization | 1.19% |
| Shared Load Instructions | 0 |

Global Load 请求量较大，但 L2 Cache 命中率达到 99.20%，显存带宽未达到饱和状态。

Direct Global Memory Load 缺少 Block 内数据复用，增加了缓存访问和内存指令处理开销。

### 3.2 Warp State & Compute

| Metric | V10-A |
|---|---:|
| Warp Cycles / Issued Instruction | 101.98 |
| Long Scoreboard | ≈60.5 cycles |
| MIO Throttle | ≈18.5 cycles |
| LG Throttle | ≈14.1 cycles |
| Issued IPC Active | 0.23 |
| Issue Slots Busy | 5.71% |
| Tensor Pipe Utilization | ≈6.3% |

Long Scoreboard 为主要 Stall 来源，表明数据加载依赖限制了指令发射。

Issue Slots Busy 和 Tensor Pipe Utilization 均较低，Tensor Core 未得到持续的计算指令供给。

### 3.3 Scheduler & Occupancy

| Metric | V10-A |
|---|---:|
| Active Warps / Scheduler | 5.82 |
| Eligible Warps / Scheduler | 0.08 |
| Issued Warp / Scheduler | 0.06 |
| No Eligible | 94.30% |
| Theoretical Occupancy | 50.00% |
| Achieved Occupancy | 48.40% |
| Block Limit SM | 24 |

Single-Warp Block 设计受到 SM 最大驻留 Block 数限制，理论 Occupancy 仅为 50%。

同时，94.30% 的调度周期没有 Eligible Warp，进一步限制了指令吞吐和 Tensor Core 利用率。

## 4. Conclusion

V10-A 成功建立了基于 WMMA 的 Tensor Core GEMM 基线，但暴露出明显的数据供给与执行效率问题：

- Direct Global Load 产生大量重复访存请求。
- Long Scoreboard 与访存指令队列等待显著。
- Single-Warp Block 限制理论 Occupancy。
- Tensor Core Utilization 和指令发射效率较低。

**Baseline Performance: 28.458 TFLOPS / 0.604 ms**

下一版本 V10-B 将引入 Multi-Warp Block 与 Shared Memory Tiling，改善数据复用和 Warp 并行度，降低 Global Load 开销。