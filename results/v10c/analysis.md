# V10-C — WMMA K Tiling (BK=32)

## 1. Optimization Overview

V10-C 基于 V10-B 扩大 K Tile，由 BK=16 调整为 BK=32，在一次 Shared Memory 数据加载后连续执行两次 WMMA MMA。

- GPU: NVIDIA RTX 5090
- CUDA: 13.0
- Matrix: M=N=K=2048
- Precision: FP16 Input / FP32 Accumulation
- Block Tile: 32×32×32
- WMMA Tile: 16×16×16
- Block Size: 128 Threads / 4 Warps
- Shared Memory: 4 KiB (Logical Tile Storage)

K Phases 从 128 降至 64，每个 Block 的同步点数从 256 降至 128，WMMA MMA 总次数保持不变。

## 2. Benchmark Results

| Metric | V10-B (BK16) | V10-C (BK32) |
|---|---:|---:|
| Average Time | **0.334 ms** | 0.337 ms |
| Performance | **51.455 TFLOPS** | 50.915 TFLOPS |
| Correctness | PASS | PASS |
| Max Difference | 0 | 0 |

V10-C 相比 V10-B 的计算吞吐下降 **1.05%**。

## 3. NCU Performance Analysis

### 3.1 Memory Workload

| Metric | V10-B | V10-C |
|---|---:|---:|
| Global Load Instructions | 16.78 M | 16.78 M |
| L1/TEX Global Load Bytes | 1 GiB | 1 GiB |
| Shared Load Matrix Instructions | 4.19 M | 4.19 M |
| Shared Load Matrix Wavefronts | 50.88 M | 68.36 M |
| Shared Load Bank Conflicts | 34.09 M | 51.58 M |
| Shared Store Bank Conflicts | 0 | 0.66 M |
| Total Shared Wavefronts | 70.79 M | 90.55 M |

BK=32 改变了 Shared Memory 行跨度，Shared Load Bank Conflicts 增加约 51.3%，Matrix Load Wavefronts 增加约 34.4%。

Global Load 规模不变，但 Shared Memory 访问处理成本明显增加。

### 3.2 Warp State & Compute

| Metric | V10-B | V10-C |
|---|---:|---:|
| Warp Cycles / Issued Instruction | 20.67 | 17.60 |
| Long Scoreboard | ≈11.50 | ≈4.94 |
| MIO Throttle | ≈0.35 | ≈3.31 |
| Barrier Stall | ≈1.75 | ≈1.59 |
| Issued IPC Active | 2.24 | 1.98 |
| Tensor Pipe Utilization | ≈10.5% | ≈10.8% |

K Tiling 降低了 Long Scoreboard 和 Barrier Stall，但 MIO Throttle 明显增加，与更高的 Shared Memory Wavefronts 和 Bank Conflicts 相符。

Tensor Core 利用率基本持平，未获得明显的计算管线收益。

### 3.3 Scheduler & Occupancy

| Metric | V10-B | V10-C |
|---|---:|---:|
| Active Warps / Scheduler | 11.50 | 8.66 |
| Eligible Warps / Scheduler | 1.18 | 1.46 |
| Issued Warp / Scheduler | 0.56 | 0.49 |
| No Eligible | 44.39% | 50.77% |
| Block Limit Registers | 12 | 10 |
| Theoretical Occupancy | 100.00% | 83.33% |
| Achieved Occupancy | 93.93% | 72.40% |

V10-C 的寄存器资源将每个 SM 的最大驻留 Block 数限制为 10，理论 Active Warps 降至 40。

实际 Occupancy 下降至 72.40%，同时指令发射率降低，削弱了 Warp 级延迟隐藏能力。

## 4. Conclusion

V10-C 验证了扩大 K Tile 的性能权衡：

- 外层 K 循环和同步次数减半。
- Long Scoreboard 显著下降。
- Shared Memory Bank Conflicts 和 Wavefronts 增加。
- 寄存器限制使 Warp 驻留能力下降。
- Tensor Core 利用率未明显改善。

额外的 Shared Memory 访问开销和较低的 Warp 驻留率可能抵消 K Tiling 的同步优化收益，最终产生约 **1.05% 的性能回退**。

**Final Performance: 50.915 TFLOPS / 0.337 ms**

**Preferred Configuration: V10-B (BK=16)**

**Next: V10-D — 128-bit Vectorized Global Load and Shared Store**