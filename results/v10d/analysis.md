# V10-D — 128-bit Vectorized Load and Shared Store WMMA GEMM

## 1. Optimization Overview

V10-D 在 V10-B 的基础上引入 128-bit Vectorized Global Load 和 Shared Memory Store，通过 `uint4` 替代标量 FP16 数据搬运，保持 BK=16 和 4-Warp WMMA 计算配置不变。

- GPU: NVIDIA RTX 5090
- CUDA: 13.0
- Matrix: M=N=K=2048
- Precision: FP16 Input / FP32 Accumulation
- Block Tile: 32×32×16
- WMMA Tile: 16×16×16
- Block Size: 128 Threads / 4 Warps
- Shared Memory: 2 KiB (Logical Tile Storage)

每次向量化加载和写入处理 8 个连续 FP16 元素（16 Bytes），减少 Global-to-Shared 数据搬运指令数量。

## 2. Benchmark Results

| Metric | V10-B | V10-C | V10-D |
|---|---:|---:|---:|
| Average Time | 0.334 ms | 0.337 ms | **0.254 ms** |
| Performance | 51.455 TFLOPS | 50.915 TFLOPS | **67.697 TFLOPS** |
| Correctness | PASS | PASS | PASS |
| Max Difference | 0 | 0 | 0 |

V10-D 相比 V10-B 实现 **1.316× 吞吐提升**，计算吞吐提高 **31.6%**，平均执行时间降低约 **24.0%**。

三次 Benchmark 吞吐分别为 67.694、67.750 和 67.646 TFLOPS。

## 3. NCU Performance Analysis

### 3.1 Memory Workload

| Metric | V10-B | V10-D |
|---|---:|---:|
| Global Load Instructions | 16,777,216 | 2,097,152 |
| Shared Store Instructions | 16,777,216 | 2,097,152 |
| Global Load Sectors | 33,554,432 | 33,554,432 |
| Global Load Sectors/Request | 2 | 16 |
| L1/TEX Global Load Bytes | 1 GiB | 1 GiB |
| Shared Load Matrix Instructions | 4,194,304 | 4,194,304 |
| Shared Load Matrix Wavefronts | 50,875,553 | 50,468,383 |
| Shared Load Bank Conflicts | 34,094,105 | 33,686,004 |
| Shared Store Wavefronts | 16,777,216 | 8,388,608 |
| Shared Store Bank Conflicts | 0 | 0 |

128-bit Vectorization 将 Global Load 和 Shared Store 指令数量均减少 **87.5%**，同时保持 Global Load 总访问量不变。

优化收益主要来自访存指令和相关地址计算开销的降低，而不是减少数据传输总量。

Shared Load Matrix Bank Conflicts 基本保持不变，表明当前 WMMA Shared Memory 布局仍有进一步优化空间。

### 3.2 Warp State & Compute

| Metric | V10-B | V10-D |
|---|---:|---:|
| Warp Cycles / Issued Instruction | 20.67 | 96.14 |
| Long Scoreboard | ≈11.5 | ≈43.1 |
| Barrier Stall | ≈1.75 | ≈23.9 |
| MIO Throttle | ≈0.35 | ≈9.7 |
| Issued IPC Active | 2.24 | 0.37 |
| Issue Slots Busy | 56.05% | 9.37% |
| SM Busy | 56.05% | 15.33% |
| Tensor Pipe Utilization | ≈10.5% | ≈15.3% |

V10-D 的指令发射速率降低，归一化 Warp Stall 指标明显增加。

由于向量化显著减少了搬运指令数量，Cycles Per Issued Instruction 的增加不能直接解释为单次内存访问延迟或 Kernel 总等待时间同比例增加。

整体 Benchmark 证明，指令工作量减少带来的收益超过了指令发射效率下降的影响。

Tensor Pipe Utilization 有所提高，但计算管线仍未达到持续高利用率。

### 3.3 Scheduler & Occupancy

| Metric | V10-B | V10-D |
|---|---:|---:|
| Active Warps / Scheduler | 11.50 | 8.94 |
| Eligible Warps / Scheduler | 1.18 | 0.13 |
| Issued Warp / Scheduler | 0.56 | 0.09 |
| No Eligible | 44.39% | 90.71% |
| Theoretical Occupancy | 100.00% | 83.33% |
| Achieved Occupancy | 93.93% | 74.43% |
| Achieved Active Warps / SM | 45.09 | 35.73 |

V10-D 的 Warp 驻留数量和 Scheduler 发射效率均低于 V10-B。

理论 Occupancy 降至 83.33%，实际 Occupancy 为 74.43%。具体资源限制仍需结合完整 Launch Statistics 验证。

尽管存在上述变化，向量化数据搬运仍实现了显著的整体性能提升。

## 4. Conclusion

V10-D 验证了 128-bit Vectorized Global-to-Shared 数据搬运对 WMMA GEMM 的优化效果：

- Global Load Instructions 和 Shared Store Instructions 均减少 **87.5%**。
- L1/TEX Global Load Bytes 保持不变，表明收益主要来自指令开销降低。
- 相比 V10-B，GEMM 吞吐提升 **31.6%**。
- Shared Memory Matrix Load Bank Conflicts 仍然较高。
- Scheduler 利用率及 Occupancy 下降，但未抵消向量化带来的整体加速。

**Final Performance: 67.697 TFLOPS / 0.254 ms**

**Next: V10-E — cp.async Double Buffering**