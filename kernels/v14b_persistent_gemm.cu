
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
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;

// ======================================================
// V14-C: Persistent GEMM Scheduling Comparison
//
// Mode 0: Linear Grid-Stride
// Mode 1: Fixed-Column
// ======================================================

__global__ void gemmV14Persistent(
    const __half* A,
    const __half* B,
    float* C,
    int M,
    int N,
    int K,
    int scheduleMode
)
{
    const int tid = threadIdx.x;
    const int warpId = tid / 32;

    const int warpRow = warpId / 2;
    const int warpCol = warpId % 2;

    const int tilesM = M / BM;
    const int tilesN = N / BN;
    const int totalTiles = tilesM * tilesN;

    // Shared Memory
    __shared__ __align__(32) __half sA[BM][BK];
    __shared__ __align__(32) __half sB[BK][BN];

    // ==================================================
    // Step 1: Persistent Tile Scheduling
    // ==================================================

    // Fixed-column scheduling parameters
    const int blockCol = blockIdx.x % tilesN;
    const int blockGroup = blockIdx.x / tilesN;

    const int groupsPerCol =
        gridDim.x / tilesN +
        (blockCol < gridDim.x % tilesN ? 1 : 0);

    for (int iteration = 0; ; ++iteration)
    {
        int tileRowIndex = 0;
        int tileColIndex = 0;

        // ----------------------------------------------
        // Mode 0: Linear Grid-Stride
        // ----------------------------------------------

        if (scheduleMode == 0)
        {
            const int tileId =
                blockIdx.x + iteration * gridDim.x;

            if (tileId >= totalTiles)
            {
                break;
            }

            tileRowIndex = tileId / tilesN;
            tileColIndex = tileId % tilesN;
        }

        // ----------------------------------------------
        // Mode 1: Fixed-Column
        // ----------------------------------------------

        else
        {
            tileRowIndex =
                blockGroup + iteration * groupsPerCol;

            if (tileRowIndex >= tilesM)
            {
                break;
            }

            tileColIndex = blockCol;
        }

        // Current C Tile position
        const int tileRow = tileRowIndex * BM;
        const int tileCol = tileColIndex * BN;

        // ==============================================
        // Step 2: Initialize C Accumulators
        // ==============================================

        // Each Warp computes 32x32:
        // Four 16x16 accumulators.

        wmma::fragment<
            wmma::accumulator,
            WM, WN, WK, float
        > cFrag[2][2];

        #pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            #pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                wmma::fill_fragment(
                    cFrag[i][j], 0.0f
                );
            }
        }

        // ==============================================
        // Step 3: K Tiling
        // ==============================================

        for (int phase = 0; phase < K / BK; ++phase)
        {
            const int kStart = phase * BK;

            // ------------------------------------------
            // Step 3.1: Global -> Shared A
            // ------------------------------------------

            if (tid < 64)
            {
                #pragma unroll
                for (int part = 0; part < 4; ++part)
                {
                    const int idx = tid + part * 64;

                    const int aRow = idx / 4;
                    const int aCol = (idx % 4) * 8;

                    const uint4 value =
                        *reinterpret_cast<const uint4*>(
                            &A[(tileRow + aRow) * K
                               + kStart + aCol]
                        );

                    *reinterpret_cast<uint4*>(
                        &sA[aRow][aCol]
                    ) = value;
                }
            }

            // ------------------------------------------
            // Step 3.2: Global -> Shared B
            // ------------------------------------------

            else
            {
                const int vectorId = tid - 64;

                #pragma unroll
                for (int part = 0; part < 4; ++part)
                {
                    const int idx = vectorId + part * 64;

                    const int bRow = idx / 8;
                    const int bCol = (idx % 8) * 8;

                    const uint4 value =
                        *reinterpret_cast<const uint4*>(
                            &B[(kStart + bRow) * N
                               + tileCol + bCol]
                        );

                    *reinterpret_cast<uint4*>(
                        &sB[bRow][bCol]
                    ) = value;
                }
            }

            // All Global -> Shared loads finish
            __syncthreads();

            // ------------------------------------------
            // Step 3.3: Shared -> Registers -> WMMA
            // ------------------------------------------

            #pragma unroll
            for (int k = 0; k < BK; k += WK)
            {
                wmma::fragment<
                    wmma::matrix_a,
                    WM, WN, WK,
                    __half, wmma::row_major
                > aFrag[2];

                wmma::fragment<
                    wmma::matrix_b,
                    WM, WN, WK,
                    __half, wmma::row_major
                > bFrag[2];

                // Load A Fragments
                #pragma unroll
                for (int i = 0; i < 2; ++i)
                {
                    wmma::load_matrix_sync(
                        aFrag[i],
                        &sA[warpRow * 32 + i * WM][k],
                        BK
                    );
                }

                // Load B Fragments
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                {
                    wmma::load_matrix_sync(
                        bFrag[j],
                        &sB[k][warpCol * 32 + j * WN],
                        BN
                    );
                }

                // Tensor Core MMA
                #pragma unroll
                for (int i = 0; i < 2; ++i)
                {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j)
                    {
                        wmma::mma_sync(
                            cFrag[i][j],
                            aFrag[i],
                            bFrag[j],
                            cFrag[i][j]
                        );
                    }
                }
            }

            // All Warps finish current Shared Tile
            __syncthreads();
        }

        // ==============================================
        // Step 4: Registers -> Global C
        // ==============================================

        #pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            #pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                const int row =
                    tileRow +
                    warpRow * 32 +
                    i * WM;

                const int col =
                    tileCol +
                    warpCol * 32 +
                    j * WN;

                wmma::store_matrix_sync(
                    &C[row * N + col],
                    cFrag[i][j],
                    N,
                    wmma::mem_row_major
                );
            }
        }

        // Continue to next C Tile
    }
}

// ======================================================
// Main
// ======================================================

int main()
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;

    constexpr int WARMUP = 5;
    constexpr int RUNS = 30;

    static_assert(
        M % BM == 0 &&
        N % BN == 0 &&
        K % BK == 0,
        "Matrix dimensions must match tile sizes"
    );

    static_assert(
        N % 8 == 0 &&
        K % 8 == 0,
        "Row strides must support 128-bit loads"
    );

    // ==================================================
    // Step 1: Host Memory
    // ==================================================

    std::vector<__half> hA(M * K);
    std::vector<__half> hB(K * N);

    std::vector<float> hC(M * N);
    std::vector<float> hRef(M * N);

    std::mt19937 rng(1234);

    std::uniform_real_distribution<float> dist(
        -1.0f, 1.0f
    );

    for (auto& x : hA)
    {
        x = __float2half(dist(rng));
    }

    for (auto& x : hB)
    {
        x = __float2half(dist(rng));
    }

    // ==================================================
    // Step 2: Device Memory
    // ==================================================

    __half* dA = nullptr;
    __half* dB = nullptr;

    float* dC = nullptr;
    float* dRef = nullptr;

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

    // Host -> Device A
    CUDA_CHECK(cudaMemcpy(
        dA,
        hA.data(),
        hA.size() * sizeof(__half),
        cudaMemcpyHostToDevice
    ));

    // Host -> Device B
    CUDA_CHECK(cudaMemcpy(
        dB,
        hB.data(),
        hB.size() * sizeof(__half),
        cudaMemcpyHostToDevice
    ));

    // ==================================================
    // Step 3: cuBLAS Reference
    // ==================================================

    cublasHandle_t handle;

    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // A and B are row-major.
    // Compute C^T = B^T * A^T using cuBLAS.

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

    CUDA_CHECK(cudaMemcpy(
        hRef.data(),
        dRef,
        hRef.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // ==================================================
    // Step 4: GPU Configuration
    // ==================================================

    int device = 0;
    int smCount = 0;

    CUDA_CHECK(cudaGetDevice(&device));

    CUDA_CHECK(cudaDeviceGetAttribute(
        &smCount,
        cudaDevAttrMultiProcessorCount,
        device
    ));

    const int tilesN = N / BN;
    const int totalTiles = (M / BM) * tilesN;

    const int multipliers[4] = {1, 2, 4, 8};

    dim3 block(128);

    std::cout
        << "V14-C Persistent Scheduling Comparison\n"
        << "GPU SM Count: " << smCount << '\n'
        << "Total C Tiles: " << totalTiles << '\n'
        << "Matrix: M=" << M
        << " N=" << N
        << " K=" << K
        << "\n\n";

    int failedConfigs = 0;

    // ==================================================
    // Step 5: Scheduling Comparison
    // ==================================================

    // scheduleMode = 0: Linear Grid-Stride
    // scheduleMode = 1: Fixed-Column

    for (int scheduleMode = 0;
         scheduleMode < 2;
         ++scheduleMode)
    {
        std::cout
            << "==================================\n"
            << (scheduleMode == 0
                ? "Mode 0: Linear Grid-Stride"
                : "Mode 1: Fixed-Column")
            << "\n==================================\n";

        // Test different Block counts
        for (int config = 0; config < 5; ++config)
        {
            int blockCount = totalTiles;

            if (config < 4)
            {
                blockCount =
                    smCount * multipliers[config];

                if (blockCount > totalTiles)
                {
                    blockCount = totalTiles;
                }
            }

            // Fixed-column mode requires
            // at least one Block per Tile column.
            if (blockCount < tilesN)
            {
                continue;
            }

            dim3 grid(blockCount);

            // ==========================================
            // Step 5.1: Correctness
            // ==========================================

            // Clear C before each configuration
            CUDA_CHECK(cudaMemset(
                dC,
                0,
                hC.size() * sizeof(float)
            ));

            gemmV14Persistent<<<grid, block>>>(
                dA, dB, dC, M, N, K, scheduleMode
            );

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(
                hC.data(),
                dC,
                hC.size() * sizeof(float),
                cudaMemcpyDeviceToHost
            ));

            int errors = 0;
            float maxDiff = 0.0f;

            for (size_t i = 0; i < hC.size(); ++i)
            {
                const float diff =
                    std::fabs(hC[i] - hRef[i]);

                const float tolerance =
                    0.01f +
                    0.001f * std::fabs(hRef[i]);

                if (diff > maxDiff)
                {
                    maxDiff = diff;
                }

                if (!std::isfinite(hC[i]) ||
                    !std::isfinite(hRef[i]) ||
                    diff > tolerance)
                {
                    ++errors;
                }
            }

            std::cout
                << "----------------------------------\n"
                << "Persistent Blocks: "
                << blockCount << '\n'
                << "Correctness: "
                << (errors == 0 ? "PASS" : "FAIL")
                << '\n'
                << "Max difference: "
                << maxDiff << '\n';

            if (errors != 0)
            {
                ++failedConfigs;

                std::cout
                    << "Errors: "
                    << errors << "\n\n";

                continue;
            }

            // ==========================================
            // Step 5.2: Benchmark
            // ==========================================

            // Warmup
            for (int i = 0; i < WARMUP; ++i)
            {
                gemmV14Persistent<<<grid, block>>>(
                    dA, dB, dC, M, N, K, scheduleMode
                );
            }

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEvent_t start;
            cudaEvent_t stop;

            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start));

            // Benchmark
            for (int i = 0; i < RUNS; ++i)
            {
                gemmV14Persistent<<<grid, block>>>(
                    dA, dB, dC, M, N, K, scheduleMode
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
                << std::setprecision(6)
                << "Average time: "
                << avgMs << " ms\n"
                << "Performance: "
                << tflops << " TFLOPS\n\n";

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
    }

    // ==================================================
    // Step 6: Cleanup
    // ==================================================

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));

    return failedConfigs == 0 ? 0 : 1;
}
