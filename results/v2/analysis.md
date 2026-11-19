# V2 — Register Tiling GEMM 性能分析报告

## 1. 实验环境与优化目标

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 32GB |
| NVIDIA Driver | 580.95.05 |
| CUDA Toolkit | 13.0 |
| Nsight Compute | 2025.2.0 |
| 矩阵规模 | M=N=K=2048 |
| 输入精度 | FP16 |
| 累加、输出精度 | FP32 |
| Block Size | 256 Threads |
| 编译参数 | `-O3 -std=c++17 -arch=sm_120 -lineinfo` |

V1 Shared Memory Tiling 虽然减少了 Global Memory 读取指令，但仍存在大量 Shared Load，并出现明显的 MIO Throttle 停顿。

V2 引入 Register Tiling（寄存器分块），让每个线程计算多个输出元素，从而在寄存器中复用 Shared Memory 加载的操作数。

本次分别测试 R2、R4、R8 三个版本，观察寄存器复用、访存指令、Warp 调度、Occupancy 和整体性能之间的关系。

## 2. 三个子版本的实现区别

| 配置 | R2 | R4 | R8 |
|---|---:|---:|---:|
| BM | 16 | 32 | 64 |
| BN | 32 | 32 | 32 |
| BK | 32 | 32 | 32 |
| 每线程输出数 | 2 | 4 | 8 |
| 每 Block 输出 Tile | 16×32 | 32×32 | 64×32 |
| Grid Blocks | 8192 | 4096 | 2048 |
| Shared Memory 数据用量 | 3 KB | 4 KB | 6 KB |

三个版本均使用相同的 256 Threads/Block。

优化的主要变化是沿 M 方向扩大输出 Tile，让一个线程计算更多行的 C 元素。

对于同一个 K：

- R2：一个 B 操作数可以参与 2 次乘加。
- R4：一个 B 操作数可以参与 4 次乘加。
- R8：一个 B 操作数可以参与 8 次乘加。

因此，随着 R 增大，B 操作数的线程内部复用次数增加。

此外，BM 增大还减少了不同 Block 对同一 B Tile 的重复 Global Memory 加载。

## 3. 正确性与 Benchmark

所有版本均通过 cuBLAS 参考结果验证，最大误差为 0.000366211。

每次程序运行执行 5 次 Warmup 和 30 次 CUDA Event 计时；每个版本进行三次独立运行。

### 3.1 三次 Benchmark 结果

| 版本 | 第一次 | 第二次 | 第三次 | 平均时间 | 平均 TFLOPS |
|---|---:|---:|---:|---:|---:|
| R2 | 1.324 ms | 1.294 ms | 1.324 ms | **1.314 ms** | **13.077** |
| R4 | 0.920 ms | 0.918 ms | 0.915 ms | **0.918 ms** | **18.721** |
| R8 | 0.865 ms | 0.863 ms | 0.862 ms | **0.863 ms** | **19.898** |

### 3.2 性能提升

| 对比 | 吞吐提升 |
|---|---:|
| R2 → R4 | +43.2% |
| R4 → R8 | +6.3% |
| R2 → R8 | +52.2% |

V2-R8 是三个子版本中性能最高的实现。

但 R4 → R8 的边际收益明显降低，需要结合 NCU 分析原因。

以上性能数据来自正常 CUDA Event Benchmark，不采用 NCU Profiling 期间的执行时间作为正式性能结果。

## 4. Memory Workload Analysis

查看位置：

`Details → Memory Workload Analysis → Memory Tables`

### 4.1 Global / Shared Memory 指令数量

| 指标 | R2 | R4 | R8 |
|---|---:|---:|---:|
| Global Load Instructions | 25,165,824 | 16,777,216 | 12,582,912 |
| Shared Load Instructions | 201,326,592 | 134,217,728 | 100,663,296 |
| Shared Store Instructions | 25,165,824 | 16,777,216 | 12,582,912 |

从 R2 到 R4：

- Global Load 减少 33.33%。
- Shared Load 减少 33.33%。
- Shared Store 减少 33.33%。

从 R4 到 R8：

- Global Load 减少 25%。
- Shared Load 减少 25%。
- Shared Store 减少 25%。

**分析：**

随着 BM 从 16 增加到 32、64，单个 Block 计算更多输出行。

对于完整的输出矩阵，A 的逻辑加载总量基本保持不变，但 B Tile 在不同 Block 之间的重复加载次数减少。

与此同时，单线程计算多个 C 元素后，同一个 B 操作数可以在寄存器中复用，从而减少 Shared Load 指令。

因此，V2-R2 → R4 → R8 的 Global Load 和 Shared Load 指令数量持续下降。

### 4.2 Shared Memory Bank Conflicts

查看位置：

`Memory Tables → Shared Memory → Bank Conflicts`

| 指标 | R2 | R4 | R8 |
|---|---:|---:|---:|
| Shared Load Bank Conflicts | 59,724 | 26,424 | 6,580 |
| Shared Store Bank Conflicts | 0 | 0 | 154,266 |

R8 的 Shared Load Bank Conflicts 明显减少，但 Shared Store 新增了 154,266 次 Bank Conflict。

NCU 同时显示 R8：

- Shared Store Requests：12,582,912
- Shared Store Wavefronts：12,739,385

额外 Wavefront 约占 Requests 的 1.24%。

因此，虽然 R8 的 Shared Store 出现了新的 Bank Conflict，但当前没有充分证据将其认定为主要性能瓶颈。

## 5. Warp State Statistics

查看位置：

`Details → Warp State Statistics`

### 5.1 Warp Cycles Per Issued Instruction

| 版本 | Cycles / Issued Instruction |
|---|---:|
| R2 | 17.62 |
| R4 | 12.83 |
| R8 | 14.61 |

该指标反映每条已发射 Warp 指令平均对应的 Warp 周期数，包括 Warp 的等待状态。

它不是单条 GPU 指令的实际执行延迟。

R2 → R4：该数值明显下降。

R4 → R8：该数值反而上升。

说明 R8 虽然执行了更少的访存指令，但 Warp 等待与指令发射状态没有继续全面改善。

### 5.2 主要停顿类型

| 指标 | R2 | R4 | R8 |
|---|---:|---:|---:|
| MIO Throttle | ≈5.2 cycles | ≈3.36 cycles | ≈0.24 cycles |
| Long Scoreboard | ≈3.6 cycles | ≈1.65 cycles | ≈4.2 cycles |
| Math Pipe Throttle | 较低 | ≈0.67 cycles | ≈2.3 cycles |

注：以上 Stall 数值为 NCU Warp State 柱状图的近似读数。

**分析：**

R2 和 R4 中，MIO Throttle 是突出的停顿原因。

随着 Shared Load 指令减少，MIO Throttle 逐步下降。

到 R8 时，MIO Throttle 已经很低，但 Long Scoreboard 和 Math Pipe Throttle 成为更加突出的等待因素。

Long Scoreboard 表示 Warp 等待与 L1/TEX 相关操作的数据依赖满足。

Math Pipe Throttle 表示相关数学执行管线暂时无法接受更多指令。

这说明 R8 进一步减少 Shared Load 后，性能开始更多地受到其他指令处理与数据依赖因素影响。

不过，具体哪些指令造成 Long Scoreboard 和 Math Pipe Throttle，仍需要结合 Source/SASS 或更详细的计数器分析才能最终确定。

## 6. Scheduler Statistics

查看位置：

`Details → Scheduler Statistics`

| 指标 | R2 | R4 | R8 |
|---|---:|---:|---:|
| Active Warps / Scheduler | 11.47 | 9.31 | 10.07 |
| Eligible Warps / Scheduler | 2.39 | 2.86 | 2.95 |
| Issued Warp / Scheduler | 0.65 | 0.73 | 0.69 |
| No Eligible | 34.93% | 27.46% | 31.05% |

**分析：**

R2 → R4：

虽然 Active Warps 减少，但 Eligible 和 Issued Warp 均增加，No Eligible 下降，说明调度器的指令发射情况明显改善。

R4 → R8：

Active Warps 和 Eligible Warps 增加，但 Issued Warp 从 0.73 降至 0.69，No Eligible 从 27.46% 上升至 31.05%。

因此，R8 并没有继续提高平均 Warp 指令发射吞吐。

结合 Warp State 数据，R8 的 Long Scoreboard 和 Math Pipe Throttle 变得更加突出，这可能是 R4 → R8 性能提升趋缓的重要因素。

## 7. Launch Statistics 与 Occupancy

查看位置：

`Details → Launch Statistics → Registers Per Thread`

以及：

`Details → Occupancy`

| 指标 | R2 | R4 | R8 |
|---|---:|---:|---:|
| Registers Per Thread | 图示约 40，待直接核验 | 48 | 40 |
| Theoretical Occupancy | 100% | 83.33% | 100% |
| Achieved Occupancy | 95.49% | 77.62% | 约 84.1% |
| Active Warps / Scheduler | 11.47 | 9.31 | 10.07 |

### 7.1 R4 的寄存器资源限制

R4 每线程使用 48 个寄存器。

一个 Block 包含 256 个线程，因此单 Block 需要：

`48 × 256 = 12288 Registers`

每个 SM 有 65,536 个 32-bit 寄存器。

因此，寄存器资源最多支持 5 个 R4 Block 同时驻留：

`5 × 8 Warps = 40 Warps`

理论 Occupancy：

`40 / 48 = 83.33%`

NCU 的 `Block Limit Registers = 5` 也验证了这一结果。

### 7.2 R8 的寄存器使用量反而减少

R8 每线程使用 40 个寄存器。

虽然源代码中有 8 个 FP32 累加器，但编译器最终分配的寄存器数量低于 R4。

这说明寄存器使用量不仅取决于累加器数量，还与编译器生成的指令、临时变量和变量生命周期有关。

40 个寄存器允许一个 SM 驻留 6 个 R8 Block，从而达到 100% 理论 Occupancy。

但 R8 实测 Occupancy 只有约 84.1%，并没有长期保持满额驻留。

### 7.3 Tail Effect

R8 的 Launch Statistics 显示：

- Grid Size：2048 Blocks
- SM 数量：170
- Waves Per SM：2.01

在每个 SM 最多驻留 6 个 Block 的配置下，整张 GPU 一轮最多驻留：

`170 × 6 = 1020 Blocks`

因此整个 Grid 大约包含两轮完整 Block 波次和少量尾部 Block。

最后的不完整波次可能降低部分 SM 的平均利用率。

NCU 给出的 Tail Effect 估计收益为 33.33%，属于分析模型给出的潜在优化提示，不是已经实测得到的性能提升。

## 8. 三个版本的综合比较

### R2：首次引入寄存器复用

- 每线程计算 2 个输出。
- 相比 V1 大幅减少 Shared Load。
- MIO Throttle 仍然明显。
- 理论 Occupancy 达到 100%。
- 性能为 13.077 TFLOPS。

### R4：寄存器复用带来明显加速

- 每线程计算 4 个输出。
- Global/Shared Load 指令继续减少三分之一。
- MIO Throttle 进一步缓解。
- 每线程寄存器数量增加到 48，理论 Occupancy 降至 83.33%。
- 尽管驻留 Warp 减少，调度器发射情况仍有改善。
- 性能达到 18.721 TFLOPS。

### R8：访存减少，但收益开始递减

- 每线程计算 8 个输出。
- Global/Shared Load 指令相比 R4 再减少 25%。
- MIO Throttle 已经不再突出。
- Long Scoreboard 和 Math Pipe Throttle 成为主要关注点。
- 实际寄存器使用量为 40，理论 Occupancy 恢复 100%。
- No Eligible 与 Warp Cycles Per Issued Instruction 相比 R4 有所上升。
- 性能达到 19.898 TFLOPS，相比 R4 仅提升约 6.3%。

## 9. V2 最终结论与后续方向

**V2-R8 是当前三个 Register Tiling 子版本中性能最好的版本。**

V2 的实验验证了：

1. 增加每线程输出数量，可以提高操作数的寄存器复用率。
2. 扩大 BM 可以减少跨 Block 重复加载 B Tile。
3. Global Load 和 Shared Load 指令数量随 R 增加而持续下降。
4. MIO Throttle 明显缓解，但其他类型的等待可能变得突出。
5. Occupancy 与性能不存在简单的正比关系。
6. 寄存器使用量不一定随着源代码累加器数量单调增加。
7. R4 → R8 已经出现明显的性能收益递减。

下一阶段 V3 将研究 Vectorized Global Load（向量化全局内存加载）。

重点观察：

- Global Load 指令数量及单次访问宽度。
- Global Memory Load 的合并访问情况。
- LG Throttle、Long Scoreboard 等停顿。
- 向量化加载对总体 Benchmark 性能的影响。

**阶段最佳结果：V2-R8，0.863 ms，19.898 TFLOPS。**