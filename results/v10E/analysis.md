# V10-E — cp.async Double Buffered WMMA GEMM

## 1. Optimization Overview

V10-E 基于 V10-D 引入 `cp.async` 异步 Global-to-Shared 数据搬运与双缓冲机制，尝试重叠下一 K Tile 的数据加载和当前 Tile 的 WMMA 计算。

- GPU: NVIDIA RTX 5090
- CUDA: 13.0
- Matrix: M=N=K=2048
- Precision: FP16 Input / FP32 Accumulation
- Block Tile: 32×32×16
- WMMA Tile: 16×16×16
- Block Size: 128 Threads / 4 Warps
- Shared Memory: 4 KiB (Logical Double Buffer Storage)
- Copy Mechanism: `cp.async`, 16 Bytes per participating thread
- Synchronization: Pipeline Commit/Wait + CTA Barrier

采用两组 Shared Memory Buffer 交替执行 Prefetch 与 Compute，通过异步拷贝尝试隐藏 Global Memory 数据加载延迟。

## 2. Benchmark Results

| Metric | V10-D | V10-E |
|---|---:|---:|
| Average Time | 0.254 ms | **0.252 ms** |
| Performance | 67.697 TFLOPS | **68.211 TFLOPS** |
| Correctness | PASS | PASS |
| Max Difference | 0 | 0 |

V10-E 相比 V10-D 实现约 **0.76% 的吞吐提升**。

三次 Benchmark 分别为 68.230、68.210 和 68.193 TFLOPS。鉴于性能差异较小，仍需进一步重复测量以确认增益稳定性。

## 3. NCU Performance Analysis

### 3.1 Memory Workload

| Metric | V10-D | V10-E |
|---|---:|---:|
| Ordinary Global Load Instructions | 2,097,152 | 0 |
| Global Load To Shared Store (Bypass) | 0 | 2,097,152 |
| Ordinary Shared Store Instructions | 2,097,152 | 0 |
| Shared Store From Global Load | 0 | 4,194,304 |
| Global Load Sectors | 33.55 M | 41.87 M |
| L1/TEX Global Load Bytes | 1.00 GiB | 1.25 GiB |
| Shared Load Matrix Instructions | 4.19 M | 4.19 M |
| Shared Load Matrix Wavefronts | 50.47 M | 50.55 M |
| Shared Load Bank Conflicts | 33.69 M | 33.78 M |

NCU 确认 Global-to-Shared 数据搬运已切换至异步 Copy 路径，普通 Global Load 与 Shared Store 指令被替代。

V10-E 的 Global Load Sectors 增加约 24.8%，表明当前异步加载路径存在额外的缓存访问处理开销。

Shared Load Matrix Bank Conflicts 基本不变，说明双缓冲并未改善 WMMA Matrix Load 的 Shared Memory Bank 映射问题。

### 3.2 Warp State & Compute

| Metric | V10-D | V10-E |
|---|---:|---:|
| Warp Cycles / Issued Instruction | 96.14 | 79.28 |
| Long Scoreboard | ≈43.1 | 40.2 |
| Barrier Stall | ≈23.9 | 27.3 |
| Issued IPC Active | 0.37 | 0.58 |
| Issue Slots Busy | 9.37% | 14.54% |
| SM Busy | 15.33% | 15.25% |
| Tensor Pipe Utilization | ≈15.3% | ≈15.4% |

V10-E 的 Warp Cycles / Issued Instruction 降低，指令发射率有所提升。

但 Long Scoreboard 和 Barrier Stall 仍然占据主要等待周期，表明异步加载与计算之间仍存在明显的依赖和同步开销。

Tensor Pipe Utilization 基本持平，未观察到显著的计算管线利用率改善。

### 3.3 Scheduler & Occupancy

| Metric | V10-D | V10-E |
|---|---:|---:|
| Active Warps / Scheduler | 8.94 | 11.25 |
| Eligible Warps / Scheduler | 0.13 | 0.17 |
| Issued Warp / Scheduler | 0.09 | 0.14 |
| No Eligible | 90.71% | 85.81% |
| Theoretical Occupancy | 83.33% | 100.00% |
| Achieved Occupancy | 74.43% | 97.11% |
| Achieved Active Warps / SM | 35.73 | 46.62 |

V10-E 的理论 Occupancy 恢复至 100%，实际 Occupancy 达到 97.11%。

更高的 Warp 驻留率改善了 Scheduler 的指令发射机会，但 No Eligible 仍达到 85.81%，数据依赖和同步等待仍然显著。

## 4. Conclusion

V10-E 验证了 `cp.async` 与 Double Buffering 在 WMMA GEMM 中的实现效果。

- 异步 Global-to-Shared 数据搬运成功替代同步 Load/Store 路径。
- 双缓冲允许下一 K Tile 的 Prefetch 与当前 Tile 的计算尝试重叠。
- Achieved Occupancy 从 74.43% 提升至 97.11%。
- Warp 指令发射效率有所改善。
- Long Scoreboard 与 Barrier Stall 仍是主要等待来源。
- Global Load Sectors 增加，Shared Memory Bank Conflicts 基本不变。
- 最终计算吞吐仅小幅提升，异步流水线的收益仍然有限。

后续优化可重点研究更合理的 Pipeline Synchronization、降低异步拷贝访存开销，以及改善 WMMA Shared Memory Matrix Load 布局。

**Final Performance: 68.211 TFLOPS / 0.252 ms**

**Performance Gain over V10-D: +0.76%**

**Next: V10-F — Pipeline and Shared Memory Layout Optimization**