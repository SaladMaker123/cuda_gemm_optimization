
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>

#include "../common/cuda_check.h"
#include "../common/cublas_check.h"

namespace wmma = nvcuda::wmma;

// WMMA Tile
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

// Block Tile
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 32;


// V10-C: Multi-Warp WMMA + Shared Memory + K Tiling
__global__ void gemmV10CWMMA(
    const __half* A,
    const __half* B,
    float* C,
    int M,
    int N,
    int K
)
{
    const int tid = threadIdx.x;

    const int warpId = tid / 32;
    const int warpRow = warpId / 2;
    const int warpCol = warpId % 2;

    const int tileRow = blockIdx.y * BM;
    const int tileCol = blockIdx.x * BN;

    // Only complete tiles
    if (tileRow + BM > M ||
        tileCol + BN > N ||
        K % BK != 0)
    {
        return;
    }

    // Shared Memory: BK = 32
    __shared__ __align__(32) __half sA[BM][BK];
    __shared__ __align__(32) __half sB[BK][BN];

    // A Fragment
    wmma::fragment<
        wmma::matrix_a,
        WM, WN, WK,
        __half,
        wmma::row_major
    > aFrag;

    // B Fragment
    wmma::fragment<
        wmma::matrix_b,
        WM, WN, WK,
        __half,
        wmma::row_major
    > bFrag;

    // C Accumulator Fragment
    wmma::fragment<
        wmma::accumulator,
        WM, WN, WK,
        float
    > cFrag;

    wmma::fill_fragment(cFrag, 0.0f);

    // Outer K Tiling: 32 elements per iteration
    for (int k = 0; k < K; k += BK)
    {
        // Global Memory -> Shared Memory (A)
        for (int idx = tid;
             idx < BM * BK;
             idx += blockDim.x)
        {
            const int aRow = idx / BK;
            const int aCol = idx % BK;

            sA[aRow][aCol] =
                A[(tileRow + aRow) * K + k + aCol];
        }

        // Global Memory -> Shared Memory (B)
        for (int idx = tid;
             idx < BK * BN;
             idx += blockDim.x)
        {
            const int bRow = idx / BN;
            const int bCol = idx % BN;

            sB[bRow][bCol] =
                B[(k + bRow) * N + tileCol + bCol];
        }

        // Ensure Global -> Shared is finished
        __syncthreads();

        // Inner K Tiling: 16 elements per WMMA
        for (int kk = 0; kk < BK; kk += WK)
        {
            // Shared Memory -> A Fragment
            wmma::load_matrix_sync(
                aFrag,
                &sA[warpRow * WM][kk],
                BK
            );

            // Shared Memory -> B Fragment
            wmma::load_matrix_sync(
                bFrag,
                &sB[kk][warpCol * WN],
                BN
            );

            // Tensor Core Matrix Multiply-Accumulate
            wmma::mma_sync(
                cFrag,
                aFrag,
                bFrag,
                cFrag
            );
        }

        // Wait before overwriting Shared Memory
        __syncthreads();
    }

    // Store C Fragment -> Global Memory
    wmma::store_matrix_sync(
        C + (tileRow + warpRow * WM) * N
          + tileCol + warpCol * WN,
        cFrag,
        N,
        wmma::mem_row_major
    );
}


int main()
{
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    const int WARMUP = 5;
    const int RUNS = 30;

    // Host Memory
    std::vector<__half> hA(M * K);
    std::vector<__half> hB(K * N);

    std::vector<float> hC(M * N);
    std::vector<float> hRef(M * N);

    // Random Input
    std::mt19937 rng(1234);

    std::uniform_real_distribution<float> dist(
        -1.0f,
        1.0f
    );

    for (auto& x : hA)
    {
        x = __float2half(dist(rng));
    }

    for (auto& x : hB)
    {
        x = __float2half(dist(rng));
    }

    // Device Pointers
    __half* dA = nullptr;
    __half* dB = nullptr;

    float* dC = nullptr;
    float* dRef = nullptr;

    // Allocate Device Memory
    CUDA_CHECK(cudaMalloc(
        &dA,
        hA.size() * sizeof(__half)
    ));

    CUDA_CHECK(cudaMalloc(
        &dB,
        hB.size() * sizeof(__half)
    ));

    CUDA_CHECK(cudaMalloc(
        &dC,
        hC.size() * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &dRef,
        hRef.size() * sizeof(float)
    ));

    // Host -> Device
    CUDA_CHECK(cudaMemcpy(
        dA,
        hA.data(),
        hA.size() * sizeof(__half),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB,
        hB.data(),
        hB.size() * sizeof(__half),
        cudaMemcpyHostToDevice
    ));

    // Kernel Configuration
    dim3 block(128);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    // V10-C Correctness Test
    gemmV10CWMMA<<<grid, block>>>(
        dA,
        dB,
        dC,
        M,
        N,
        K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // cuBLAS Reference
    cublasHandle_t handle;

    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major C = A * B using cuBLAS
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        dB,
        CUDA_R_16F,
        N,
        dA,
        CUDA_R_16F,
        K,
        &beta,
        dRef,
        CUDA_R_32F,
        N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Device -> Host
    CUDA_CHECK(cudaMemcpy(
        hC.data(),
        dC,
        hC.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        hRef.data(),
        dRef,
        hRef.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // Correctness Check
    float maxDiff = 0.0f;
    int errors = 0;

    for (size_t i = 0; i < hC.size(); ++i)
    {
        const float diff = std::fabs(
            hC[i] - hRef[i]
        );

        const float tolerance =
            0.01f + 0.001f * std::fabs(hRef[i]);

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
                std::cout
                    << "Mismatch at " << i
                    << ": got " << hC[i]
                    << ", expected " << hRef[i]
                    << std::endl;
            }
        }
    }

    std::cout
        << "V10-C Multi-Warp WMMA K Tiling GEMM"
        << std::endl;

    std::cout
        << "Matrix: M=" << M
        << " N=" << N
        << " K=" << K
        << std::endl;

    std::cout
        << "Correctness: "
        << (errors == 0 ? "PASS" : "FAIL")
        << std::endl;

    std::cout
        << "Max difference: "
        << maxDiff
        << std::endl;

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
        gemmV10CWMMA<<<grid, block>>>(
            dA,
            dB,
            dC,
            M,
            N,
            K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA Event Benchmark
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < RUNS; ++i)
    {
        gemmV10CWMMA<<<grid, block>>>(
            dA,
            dB,
            dC,
            M,
            N,
            K
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsedMs,
        start,
        stop
    ));

    const float avgMs = elapsedMs / RUNS;

    const double tflops =
        2.0 * M * N * K / (avgMs * 1e9);

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "Average time: "
        << avgMs << " ms"
        << std::endl;

    std::cout
        << "Performance: "
        << tflops << " TFLOPS"
        << std::endl;

    // Destroy Events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Destroy cuBLAS Handle
    CUBLAS_CHECK(cublasDestroy(handle));

    // Free Device Memory
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));

    return 0;
}
