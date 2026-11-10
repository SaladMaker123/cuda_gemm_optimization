# CUDA GEMM Deep Optimization

An independent, reproducible CUDA C++ kernel optimization project for **NVIDIA RTX 5090 (SM 120)**.

## Project roadmap

- `V0`–`V9`: Progressive GEMM optimization, from naive CUDA through tiling, WMMA Tensor Core, double buffering, TMA, warp specialization, and parameter tuning.
- `F0`–`F2`: Fused GEMM epilogues (ReLU, Bias + ReLU) and fair end-to-end benchmarking.
- Compare results with **cuBLAS**, keeping workloads, data types, correctness criteria, and timing methodology fixed.

These are planned experiments; benchmark improvement is **not guaranteed** for every optimization attempt.

## Repository layout

- `kernels/`: `.cu` source files for all GEMM and fusion versions.
- `benchmarks/`: correctness checks and repeatable timing programs.
- `results/`: measurements and profiling reports.
- `scripts/setup_env.sh`: provision a new Vast.ai Ubuntu VM, verify CUDA and actually test Nsight Compute counter access.

## New GPU environment

For a **fresh Vast.ai Ubuntu 22.04/24.04 x86-64 RTX 5090 VM**, clone **this repository**, then run:

```bash
cd /workspace
# Clone once (omit if already cloned)
git clone https://github.com/SaladMaker123/cuda_gemm_optimization.git
cd cuda_gemm_optimization
bash scripts/setup_env.sh
```

The script currently pins:

- `cuda-toolkit-13-0=13.0.3-1`
- `nsight-compute-2025.2.0=2025.2.0.11-1`

It requires root permissions and network access, checks NVIDIA driver >=580, does **not** replace the driver, and compiles/profiles a temporary CUDA kernel. A provider may deny GPU hardware performance counters even if Nsight Compute is installed (`ERR_NVGPUCTRPERM`).

## Measurement rules

- Main benchmark: `M=N=K=2048`, FP16 inputs, FP32 accumulation/output (subject to per-kernel shape constraints and documented adjustments).
- Keep a stable input seed, compilation flags, clocks/environment where feasible, warmup, iterations, validation and error tolerance.
- Benchmark all retained versions on **one GPU instance** for final charts. Never compare raw speedups across different VM instances without rerunning baselines.
- Store rejected experiments and explain regressions; never fabricate monotonically improving benchmark data.
