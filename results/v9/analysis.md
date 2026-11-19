# V9 — Shared Memory Layout Optimization

## 1. Optimization Overview

V9 基于 V8-A（4×4 Register Tiling），研究两种 Shared Memory 布局优化策略：

- **V9-A Padding**：增加 Shared Memory 行跨度，消除 A 矩阵读取时的 Bank Conflicts。
- **V9-B Swizzle**：通过 XOR 地址重排改善 Bank 映射，避免额外 Padding。

实验环境：RTX 5090、CUDA 13.0，M=N=K=2048，FP16 Input、FP32 Accumulation。

## 2. Benchmark Results

| Metric | V8-A Baseline | V9-A Padding | V9-B Swizzle |
|---|---:|---:|---:|
| Average Time | 0.466 ms | **0.458 ms** | 0.586 ms |
| Performance | 36.833 | **37.538** | 29.331 |
| Correctness | PASS | PASS | PASS |
| Max Difference | 0.000366211 | 0.000366211 | 0.000366211 |

单位：Performance 为 TFLOPS。

V9-A 相比 V8-A 提升 **1.91%**，V9-B 相比 V9-A 下降 **21.86%**。

## 3. NCU Analysis

### 3.1 Memory Workload

| Metric | V8-A | V9-A | V9-B |
|---|---:|---:|---:|
| Shared Load Instructions | 50.33 M | 50.33 M | 83.89 M |
| Shared Load Wavefronts | 100.66 M | 67.11 M | 100.66 M |
| Shared Load Bank Conflicts | 33.55 M | 443 | 911 |
| Shared Store Bank Conflicts | 0 | 2.10 M | 0 |
| Total Shared Wavefronts | 105.00 M | 73.55 M | 105.01 M |

Padding 几乎消除了 Shared Load Bank Conflicts，将 Total Shared Wavefronts 降低约 30%，但引入了额外 Shared Store Conflicts。

Swizzle 同样消除了主要 Bank Conflicts，却使 Shared Load Instructions 相比 Padding 增加 66.7%，抵消了访问重排的收益。

### 3.2 Compute & Warp Scheduling

| Metric | V9-A | V9-B |
|---|---:|---:|
| Warp Cycles / Issued Instruction | 10.52 | 9.66 |
| Issued IPC Active | 2.96 | 3.33 |
| Issue Slots Busy | 74.06% | 83.14% |
| FMA Utilization | 56.1% | 52.7% |
| LSU Utilization | ≈56% | ≈73% |
| ALU Utilization | ≈5% | ≈19% |
| No Eligible | 26.18% | 16.87% |
| Achieved Occupancy | 65.08% | 66.97% |

V9-B 的指令发射率与 Occupancy 均高于 V9-A，但 LSU、ALU 资源使用显著增加，FMA Utilization 略有下降。

较高的 IPC 未能转化为有效计算吞吐，表明额外访存与地址处理工作降低了整体执行效率。

## 4. Conclusion

本次实验验证了 Shared Memory 布局优化的性能权衡：

- Padding 有效降低 Bank Conflicts，保持原有 Shared Load 指令数量，并取得稳定的性能提升。
- Swizzle 虽然同样消除了 Bank Conflicts，但增加了 Shared Load 与地址处理开销，导致明显的性能回退。
- Bank Conflict 数量并不能单独决定 Kernel 性能，还需综合评估指令数量、访问批次及执行管线开销。

**Best Configuration: V9-A Padding**

- Performance: **37.538 TFLOPS**
- Latency: **0.458 ms**
- Correctness: **PASS**

后续优化以 V9-A 为参考，继续研究更高效的 Global-to-Shared 数据搬运和计算流水线。