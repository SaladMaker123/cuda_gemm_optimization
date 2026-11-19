# V8 — 2D Register Tiling GEMM Performance Analysis

## 1. Optimization Overview

V8 引入二维 Register Tiling，通过提高 A、B 操作数在寄存器中的复用率，减少计算阶段的 Shared Memory Load 开销。

本版本实现并比较两种线程输出映射：

| Configuration | V8-A | V8-B |
|---|---|---|
| Register Tile | 4×4 | 8×2 |
| Block Dimensions | (16,16) | (32,8) |
| Block Tile | 64×64×32 | 64×64×32 |
| Outputs / Thread | 16 | 16 |
| Global Load | 128-bit | 128-bit |
| Accumulation | FP32 | FP32 |

实验环境：RTX 5090、CUDA 13.0、M=N=K=2048，FP16 输入、FP32 输出。

## 2. Benchmark Results

| Metric | V8-A 4×4 | V8-B 8×2 |
|---|---:|---:|
| Average Time | **0.466 ms** | 0.581 ms |
| Performance | **36.833 TFLOPS** | 29.568 TFLOPS |
| Max Difference | 0.000366211 | 0.000366211 |
| Correctness | PASS | PASS |

**V8-A 相比 V8-B 实现 24.6% 的吞吐提升。**

## 3. NCU Performance Analysis

### 3.1 Memory Workload

| Metric | V8-A 4×4 | V8-B 8×2 |
|---|---:|---:|
| Global Load Instructions | 1.05 M | 1.05 M |
| Shared Load Instructions | **50.33 M** | 83.89 M |
| Shared Load Wavefronts | 100.66 M | 83.89 M |
| Shared Load Bank Conflicts | 33.55 M | **0** |
| Global Store Sectors | 2.10 M | **1.05 M** |

V8-A 的 Shared Load Instructions 减少 **40%**，体现了 4×4 Register Tiling 更高的操作数复用效率。

但 V8-A 存在明显的 Shared Memory Bank Conflicts，且 Global Store Sectors 增加一倍，表明其内存访问模式仍有优化空间。

### 3.2 Warp State & Compute Workload

| Metric | V8-A 4×4 | V8-B 8×2 |
|---|---:|---:|
| Warp Cycles / Issued Instruction | 10.78 | 10.84 |
| Math Pipe Throttle | ≈2.24 | ≈2.78 |
| Issued IPC Active | 2.93 | 2.89 |
| Issue Slots Busy | 73.36% | 72.23% |
| FMA Utilization | **56.4%** | 54.0% |
| LSU Utilization | ≈55% | ≈72% |

V8-A 的 Shared Load 指令数量显著下降，LSU 指令吞吐压力有所减轻，同时维持了相近的 Warp 指令发射效率。

FMA 利用率略有提高，Math Pipe Throttle 下降，与计算阶段执行效率改善的结果一致。

### 3.3 Scheduler & Occupancy

| Metric | V8-A 4×4 | V8-B 8×2 |
|---|---:|---:|
| Active Warps / Scheduler | 7.91 | 7.83 |
| Eligible Warps / Scheduler | 3.43 | 3.58 |
| Issued Warp / Scheduler | 0.73 | 0.72 |
| No Eligible | 26.55% | 27.70% |
| Theoretical Occupancy | 100% | 100% |
| Achieved Occupancy | 65.79% | 65.29% |

两个版本的 Occupancy 与 Warp 发射效率基本一致，性能差异主要不能由 Warp 驻留数量解释。

## 4. Conclusion

V8-A 通过 4×4 Register Tiling 实现更均衡的操作数复用，在保持相近计算管线利用率和 Warp 调度效率的情况下，将 Shared Load Instructions 降低 **40%**。

尽管 Bank Conflicts 和 Global Store 访问开销增加，V8-A 仍取得 **24.6% 的吞吐提升**。

**Best Configuration: V8-A (4×4)**

- Performance: **36.833 TFLOPS**
- Latency: **0.466 ms**
- Correctness: **PASS**

后续优化将重点关注 Shared Memory Bank Conflicts、Global Store 合并访问，以及计算阶段的指令执行效率。