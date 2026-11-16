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

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;

// V8 4x4 Register Tiling GEMM: 128-bit Global Load + Shared Store
__global__ void gemmV9SharedMemoryPadding(
    const __half* A,
    const __half* B,
    float* C,
    int M,
    int N,
    int K
)
{
    __shared__ __align__(16) __half sA[BM][BK + 8];
    __shared__ __align__(16) __half sB[BK][BN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float sum[4][4] = {};

    for (int phase = 0; phase < K / BK; ++phase)
    {
        // 1. Global Memory -> Shared Memory: A tile
        for (int idx = tid; idx < BM * (BK / 8);
             idx += blockDim.x * blockDim.y)
        {
            int aRow = idx / (BK / 8);
            int aCol = (idx % (BK / 8)) * 8;

            int globalRow = blockIdx.y * BM + aRow;
            int globalCol = phase * BK + aCol;

            if (globalRow < M && globalCol + 7 < K && K % 8 == 0)
            {
                const uint4* ptr = reinterpret_cast<const uint4*>(
                    A + globalRow * K + globalCol
                );
                uint4 value = *ptr;
                uint4* dst = reinterpret_cast<uint4*>(
                    &sA[aRow][aCol]
                );
                *dst = value;
            }
            else
            {
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    int globalCol2 = globalCol + i;
                    sA[aRow][aCol + i] =
                        (globalRow < M && globalCol2 < K)
                            ? A[globalRow * K + globalCol2]
                            : __half(0.0f);
                }
            }
        }

        // 2. Global Memory -> Shared Memory: B tile
        for (int idx = tid; idx < BK * (BN / 8);
             idx += blockDim.x * blockDim.y)
        {
            int bRow = idx / (BN / 8);
            int bCol = (idx % (BN / 8)) * 8;

            int globalRow = phase * BK + bRow;
            int globalCol = blockIdx.x * BN + bCol;

            if (globalRow < K && globalCol + 7 < N && N % 8 == 0)
            {
                const uint4* ptr = reinterpret_cast<const uint4*>(
                    B + globalRow * N + globalCol
                );
                uint4 value = *ptr;
                uint4* dst = reinterpret_cast<uint4*>(
                    &sB[bRow][bCol]
                );
                *dst = value;
            }
            else
            {
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    int globalCol2 = globalCol + i;
                    sB[bRow][bCol + i] =
                        (globalRow < K && globalCol2 < N)
                            ? B[globalRow * N + globalCol2]
                            : __half(0.0f);
                }
            }
        }

        // All threads finish loading before computation.
        __syncthreads();

        // 3. Compute: 4 consecutive rows x 4 consecutive columns
        for (int k = 0; k < BK; ++k)
        {
            float regA[4];
            float regB[4];

            #pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                regA[r] = __half2float(
                    sA[threadIdx.y * 4 + r][k]
                );
            }

            #pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                regB[c] = __half2float(
                    sB[k][threadIdx.x * 4 + c]
                );
            }

            #pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                #pragma unroll
                for (int c = 0; c < 4; ++c)
                {
                    sum[r][c] += regA[r] * regB[c];
                }
            }
        }

        // All threads finish reading this tile before it is overwritten.
        __syncthreads();
    }

    // 4. Register -> Global Memory: store 4x4 results
    #pragma unroll
    for (int r = 0; r < 4; ++r)
    {
        #pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            int row = blockIdx.y * BM
                    + threadIdx.y * 4 + r;
            int col = blockIdx.x * BN
                    + threadIdx.x * 4 + c;

            if (row < M && col < N)
            {
                C[row * N + col] = sum[r][c];
            }
        }
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

    dim3 block(16, 16);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    // Correctness test
    gemmV9SharedMemoryPadding<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major C=A*B via the column-major cuBLAS interface.
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
            !std::isfinite(hRef[i]) || diff > tolerance)
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

    std::cout << "gemmV9 SharedMemory Padding" << std::endl;
    std::cout << "Matrix: M=" << M
              << " N=" << N
              << " K=" << K << std::endl;
    std::cout << "Correctness: "
              << (errors == 0 ? "PASS" : "FAIL") << std::endl;
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
        gemmV9SharedMemoryPadding<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA Event benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < RUNS; ++i)
    {
        gemmV9SharedMemoryPadding<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    float avgMs = elapsedMs / RUNS;
    double tflops = 2.0 * M * N * K / (avgMs * 1e9);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average time: " << avgMs << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));
    return 0;
}
