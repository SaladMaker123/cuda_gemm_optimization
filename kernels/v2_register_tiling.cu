
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <iomanip>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "../common/cuda_check.h"
#include "../common/cublas_check.h"

constexpr int BM = 16;
constexpr int BN = 32;
constexpr int BK = 32;

// V2: Register Tiling GEMM (2 outputs per thread)
__global__ void gemmV2(
    const __half* A,
    const __half* B,
    float* C,
    int M,
    int N,
    int K
)
{
    __shared__ __half sA[BM][BK];
    __shared__ __half sB[BK][BN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int row0 = blockIdx.y * BM + threadIdx.y;
    int row1 = row0 + blockDim.y;
    int col = blockIdx.x * BN + threadIdx.x;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int phase = 0; phase < K / BK; ++phase)
    {
        int kBase = phase * BK;

        // Load A Tile: Global Memory -> Shared Memory
        for (int idx = tid; idx < BM * BK; idx += blockDim.x * blockDim.y)
        {
            int aRow = idx / BK;
            int aCol = idx % BK;

            if (blockIdx.y * BM + aRow < M && kBase + aCol < K)
            {
                sA[aRow][aCol] = A[(blockIdx.y * BM + aRow) * K + kBase + aCol];
            }
            else
            {
                sA[aRow][aCol] = __float2half(0.0f);
            }
        }

        // Load B Tile: Global Memory -> Shared Memory
        for (int idx = tid; idx < BK * BN; idx += blockDim.x * blockDim.y)
        {
            int bRow = idx / BN;
            int bCol = idx % BN;

            if (kBase + bRow < K && blockIdx.x * BN + bCol < N)
            {
                sB[bRow][bCol] = B[(kBase + bRow) * N + blockIdx.x * BN + bCol];
            }
            else
            {
                sB[bRow][bCol] = __float2half(0.0f);
            }
        }

        // Wait for all threads to finish loading
        __syncthreads();

        // Compute two C elements per thread
        for (int k = 0; k < BK; ++k)
        {
            float b = __half2float(sB[k][threadIdx.x]);

            sum0 += __half2float(sA[threadIdx.y][k]) * b;
            sum1 += __half2float(sA[threadIdx.y + blockDim.y][k]) * b;
        }

        // Wait for all threads to finish using current Tile
        __syncthreads();
    }

    // Store C
    if (row0 < M && col < N)
    {
        C[row0 * N + col] = sum0;
    }

    if (row1 < M && col < N)
    {
        C[row1 * N + col] = sum1;
    }
}

int main()
{
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    const int WARMUP = 5;
    const int RUNS = 30;

    std::vector<__half> hA(M * K);
    std::vector<__half> hB(K * N);
    std::vector<float> hC(M * N);
    std::vector<float> hRef(M * N);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& x : hA)
    {
        x = __float2half(dist(rng));
    }

    for (auto& x : hB)
    {
        x = __float2half(dist(rng));
    }

    __half* dA = nullptr;
    __half* dB = nullptr;
    float* dC = nullptr;
    float* dRef = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRef, hRef.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        dA, hA.data(),
        hA.size() * sizeof(__half),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB, hB.data(),
        hB.size() * sizeof(__half),
        cudaMemcpyHostToDevice
    ));

    dim3 block(32, 8);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    // Correctness test
    gemmV2<<<grid, block>>>(dA, dB, dC, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // cuBLAS reference
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        dB, CUDA_R_16F, N,
        dA, CUDA_R_16F, K,
        &beta,
        dRef, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        hC.data(), dC,
        hC.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        hRef.data(), dRef,
        hRef.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // Correctness comparison
    float maxDiff = 0.0f;
    int errors = 0;

    for (size_t i = 0; i < hC.size(); ++i)
    {
        float diff = std::fabs(hC[i] - hRef[i]);
        float tolerance = 0.01f + 0.001f * std::fabs(hRef[i]);

        if (diff > maxDiff)
        {
            maxDiff = diff;
        }

        if (!std::isfinite(hC[i]) ||
            !std::isfinite(hRef[i]) ||
            diff > tolerance)
        {
            ++errors;

            if (errors <= 3)
            {
                std::cout << "Mismatch at " << i
                          << ": got " << hC[i]
                          << ", expected " << hRef[i]
                          << std::endl;
            }
        }
    }

    std::cout << "V2 Register Tiling GEMM (R2)" << std::endl;

    std::cout << "Matrix: M=" << M
              << " N=" << N
              << " K=" << K << std::endl;

    std::cout << "Correctness: "
              << (errors == 0 ? "PASS" : "FAIL")
              << std::endl;

    std::cout << "Max difference: " << maxDiff << std::endl;

    if (errors != 0)
    {
        CUBLAS_CHECK(cublasDestroy(handle));

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaFree(dRef));

        return 1;
    }

    // Warmup
    for (int i = 0; i < WARMUP; ++i)
    {
        gemmV2<<<grid, block>>>(dA, dB, dC, M, N, K);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA Event Benchmark
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < RUNS; ++i)
    {
        gemmV2<<<grid, block>>>(dA, dB, dC, M, N, K);
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    float avgMs = elapsedMs / RUNS;
    double tflops = 2.0 * M * N * K / (avgMs * 1e9);

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Average time: "
              << avgMs << " ms" << std::endl;

    std::cout << "Performance: "
              << tflops << " TFLOPS" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));

    return 0;
}
