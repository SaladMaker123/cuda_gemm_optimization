# V10-B — Multi-Warp Shared Memory WMMA GEMM

## 1. Optimization Overview

V10-B 在 V10-A 的基础上引入 Multi-Warp Execution 和 Shared Memory Tiling，以提升数据复用效率及 Tensor Core 指令供给。

**Configuration**

- GPU: NVIDIA RTX 5090
- CUDA: 13.0
- Matrix: M=N=K=2048
- Precision: FP16 Input / FP32 Accumulation
- Block Tile: 32×32×16
- WMMA Tile: 16×16×16
- Block Size: 128 Threads / 4 Warps
- Shared Memory: 2 KiB (Logical Tile Storage)

## 2. Benchmark Results

| Metric | V10-A | V10-B |
|---|---:|---:|
| Average Time | 0.604 ms | **0.334 ms** |
| Performance | 28.458 TFLOPS | **51.455 TFLOPS** |
| Correctness | PASS | PASS |
| Max Difference | 0 | 0 |

V10-B 实现 **1.81× Speedup**，吞吐提升 **80.81%**，执行时间降低约 **44.7%**。

## 3. NCU Performance Analysis

### 3.1 Memory Workload

| Metric | V10-A | V10-B |
|---|---:|---:|
| Global Load Instructions | 16.78 M | 16.78 M |
| Global Load Sectors | 134.22 M | 33.55 M |
| L1/TEX Global Load Bytes | 4 GiB | 1 GiB |
| L2 Hit Rate | 99.20% | 97.49% |
| Shared Load Matrix Instructions | 0 | 4.19 M |
| Shared Load Bank Conflicts | 0 | 34.09 M |

Shared Memory Tiling 将 L1/TEX Global Load 流量减少 75%，改善了 Block 内的数据复用与访存组织。

但 WMMA Shared Memory Matrix Load 产生大量 Bank Conflicts，仍存在加载效率优化空间。

### 3.2 Warp State & Compute

| Metric | V10-A | V10-B |
|---|---:|---:|
| Warp Cycles / Issued Instruction | 101.98 | 20.67 |
| Long Scoreboard | ≈60.5 | ≈11.5 |
| Issued IPC Active | 0.23 | 2.24 |
| Issue Slots Busy | 5.71% | 56.05% |
| Tensor Pipe Utilization | ≈6.3% | ≈10.5% |

Long Scoreboard 显著下降，指令发射效率明显改善。

Tensor Pipe Utilization 有所提高，但仍处于较低水平。ALU、LSU 以及同步开销仍有优化空间。

### 3.3 Scheduler & Occupancy

| Metric | V10-A | V10-B |
|---|---:|---:|
| Active Warps / Scheduler | 5.82 | 11.50 |
| Eligible Warps / Scheduler | 0.08 | 1.18 |
| Issued Warp / Scheduler | 0.06 | 0.56 |
| No Eligible | 94.30% | 44.39% |
| Theoretical Occupancy | 50.00% | 100.00% |
| Achieved Occupancy | 48.40% | 93.93% |

Multi-Warp Execution 消除了 Single-Warp Block 的驻留容量限制，显著提高了 Warp 并行度和调度效率。

但仍有 44.39% 的调度周期缺少 Eligible Warp，说明数据依赖和同步等待仍然限制执行效率。

## 4. Conclusion

V10-B 通过 Multi-Warp Execution 与 Shared Memory Tiling，显著改善了 V10-A 的内存访问与 Warp 调度效率，取得 **80.81% 的吞吐提升**。

后续优化方向：

- 优化 Shared Memory Matrix Load 的 Bank Conflicts。
- 减少 Global-to-Shared 标量搬运指令开销。
- 改进 K 方向数据流水线，降低同步与加载等待。

**Final Performance: 51.455 TFLOPS / 0.334 ms**

**Next: V10-C — WMMA K Tiling (BK=32)**