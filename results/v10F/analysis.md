# V10-F — TMA 2D Double Buffered WMMA GEMM

## 1. Optimization Overview

V10-F 基于 V10-E，将 `cp.async` 替换为 Tensor Memory Accelerator (TMA) 2D Bulk Copy，通过 Tensor Map Descriptor 和 Block-scoped Barriers 实现双缓冲数据流水线。

- GPU: NVIDIA RTX 5090
- CUDA: 13.0
- Matrix: M=N=K=2048
- Precision: FP16 Input / FP32 Accumulation
- Block Tile: 32×32×16
- WMMA Tile: 16×16×16
- Block Size: 128 Threads / 4 Warps
- Shared Memory: 4 KiB (Logical Double Buffer Storage)
- Data Transfer: TMA 2D Global-to-Shared
- Synchronization: Transaction Barriers + CTA Synchronization

每个 K Phase 使用两次 TMA 2D Copy 分别加载 A、B Tile，尝试将下一阶段数据搬运与当前阶段的 WMMA 计算重叠。

## 2. Benchmark Results

| Metric | V10-E | V10-F |
|---|---:|---:|
| Average Time | 0.252 ms | 0.257877 ms |
| Performance | **68.211 TFLOPS** | 66.621 TFLOPS |
| Correctness | PASS | PASS |
| Max Difference | 0 | 0 |

V10-F 三次 Benchmark 分别达到 66.634、66.630、66.598 TFLOPS。

相比 V10-E，计算吞吐下降 **2.33%**，表明当前 TMA Double Buffering 配置尚未优于 `cp.async` 实现。

## 3. NCU Performance Analysis

### 3.1 Memory Workload

| Metric | V10-E | V10-F |
|---|---:|---:|
| Global Load To Shared Store (Bypass) | 2,097,152 | 0 |
| TMA Load Instructions | 0 | 1,048,576 |
| Shared Load Matrix Instructions | 4,194,304 | 4,194,304 |
| Shared Load Matrix Wavefronts | 50.55 M | 52.67 M |
| Shared Load Bank Conflicts | 33.78 M | 35.90 M |
| L2 Hit Rate | 97.69% | 97.59% |
| Mem Busy | 83.94% | 84.39% |

NCU 确认数据搬运路径已切换为 TMA 2D。

TMA Load 指令数与计算规模、K Phases 及每阶段两次 Tensor Copy 的实现一致。

Shared Load Matrix Wavefronts 增加约 4.19%，Bank Conflicts 增加约 6.30%，表明 WMMA Shared Memory 读取开销没有获得改善。

### 3.2 Warp State & Compute

| Metric | V10-E | V10-F |
|---|---:|---:|
| Warp Cycles / Issued Instruction | 79.28 | 26.65 |
| Long Scoreboard | 40.2 | ≈8.2 |
| Barrier Stall | 27.3 | ≈2.35 |
| Issued IPC Active | 0.58 | 1.67 |
| Issue Slots Busy | 14.54% | 41.85% |
| SM Busy | 15.25% | 41.85% |
| Tensor Pipe Utilization | ≈15.4% | ≈14.7% |

V10-F 的归一化 Warp Stall 显著下降，指令发射率大幅提高。

但 Tensor Core 利用率基本持平，说明额外的指令发射活动没有带来相应的矩阵计算吞吐收益。

TMA Barrier 协作、分支处理及流水线控制仍然需要额外指令和同步操作。

### 3.3 Scheduler & Occupancy

| Metric | V10-E | V10-F |
|---|---:|---:|
| Active Warps / Scheduler | 11.25 | 11.30 |
| Eligible Warps / Scheduler | 0.17 | 0.68 |
| Issued Warp / Scheduler | 0.14 | 0.42 |
| No Eligible | 85.81% | 57.60% |
| Theoretical Occupancy | 100.00% | 100.00% |
| Achieved Occupancy | 97.11% | 93.30% |
| Achieved Active Warps / SM | 46.62 | 44.78 |

TMA 实现显著改善了 Warp 就绪与指令发射效率。

Achieved Occupancy 略有下降，但仍保持在 93% 以上，未表现出明显的驻留容量限制。

## 4. Conclusion

V10-F 验证了 TMA 2D Double Buffering 与 WMMA GEMM 的集成效果。

- TMA 2D Copy 成功替代 `cp.async` 数据搬运路径。
- Warp 调度效率与 IPC 明显提高。
- Tensor Core 利用率未出现显著改善。
- Shared Memory Matrix Load Bank Conflicts 增加约 6.30%。
- 当前较小的 TMA Tile 与频繁同步可能限制流水线收益。
- 最终吞吐较 V10-E 下降 2.33%。

本次实验表明，更高的 Warp 调度效率不一定带来更高的 GEMM 吞吐。TMA 的性能收益仍然依赖 Tile 粒度、同步组织与实际计算重叠程度。

**Final Performance: 66.621 TFLOPS / 0.257877 ms**

**Performance Change vs V10-E: -2.33%**

**Preferred Configuration: V10-E (68.211 TFLOPS)**

**Next: V10-G — WMMA Warp Specialization**