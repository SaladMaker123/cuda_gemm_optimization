#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "../common/cuda_check.h"
#include "../common/cublas_check.h"

// MMA Instruction Tile
constexpr int WM = 16;
constexpr int WN = 8;
constexpr int WK = 16;
// Block Tile
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
static_assert(
    sizeof(uint4) == 16 && alignof(uint4) == 16,
    "uint4 must be 16-byte aligned"
);
// V13: ldmatrix + mma.sync GEMM
__global__ void gemmV13MMA
(
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
    const int laneId = tid % 32;

    const int warpRow = warpId / 2;
    const int warpCol = warpId % 2;

    const int tileRow = blockIdx.y * BM;
    const int tileCol = blockIdx.x * BN; // Only complete and aligned tiles
    if (tileRow + BM > M ||
        tileCol + BN > N ||
        K <= 0 ||
        K % BK != 0 ||
        K % 8 != 0 ||
        N % 8 != 0)
    {
        return;
    }
    // ==========================================
    // Step 1: Shared Memory
    // ==========================================
    __shared__ __align__(32) __half sA[BM][BK];
    __shared__ __align__(32) __half sB[BK][BN];
    // Each Warp computes 32x32.
    // Each mma.sync computes 16x8.
    // Therefore each Warp needs 2x4 accumulators.
    // Each accumulator holds 4 FP32 registers per lane.
    float cFrag[2][4][4];
    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            #pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                cFrag[i][j][r] = 0.0f;
            }
        }
    }
    // ==================================================
    // Step 2: K Tiling
    // ==================================================
    for(int phase = 0;phase < K / BK; ++phase)
    {
        const int kStart = phase * BK;
        // ----------------------------------------------
        // Step 2.1: Global -> Shared A
        // ----------------------------------------------
        // First 64 threads load A
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

        // ----------------------------------------------
        // Step 2.2: Global -> Shared B
        // ----------------------------------------------

        // Remaining 64 threads load B
        else
        {
            const int vectorTid = tid - 64;

            #pragma unroll
            for (int part = 0; part < 4; ++part)
            {
                const int idx = vectorTid + part * 64;

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
        // All input tiles must be ready
        __syncthreads();
        // ==================================================
        // Step 3: Shared -> Registers -> Tensor Core
        // ==================================================
        #pragma unroll
        for(int k = 0;k < BK;k += WK)
        {
            uint32_t aRegs[2][4];
            uint32_t bRegs[4][2];
            // ------------------------------------------
            // Step 3.1: ldmatrix A
            // ------------------------------------------
            #pragma unroll
            for(int i =0;i < 2;++i)
            {
                const int matrixId = laneId / 8;
                const int matrixRow = laneId % 8;
                // Shared Memory row
                const int sARow = warpRow * 32 + i * WM + (matrixId % 2) * 8 + matrixRow;
                // Shared Memory column
                const int sACol = k + (matrixId / 2) * 8;
                // Convert to Shared Memory address
                const uint32_t address =
                    static_cast<uint32_t>(
                        __cvta_generic_to_shared(
                            &sA[sARow][sACol]
                        )
                    );
                // Load four 8x8 matrices
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(aRegs[i][0]),
                      "=r"(aRegs[i][1]),
                      "=r"(aRegs[i][2]),
                      "=r"(aRegs[i][3])
                    : "r"(address)
                );
            }
            // ------------------------------------------
            // Step 3.2: ldmatrix B
            // ------------------------------------------
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                // Two 8x8 matrices:
                // Matrix 0: rows 0-7
                // Matrix 1: rows 8-15

                // Shared Memory row
                const int sBRow =
                    k + laneId % 16;

                // Shared Memory column
                const int sBCol =
                    warpCol * 32 +
                    j * WN;

                // Convert to Shared Memory address
                const uint32_t address =
                    static_cast<uint32_t>(
                        __cvta_generic_to_shared(
                            &sB[sBRow][sBCol]
                        )
                    );

                // Load two 8x8 matrices
                // with transposed register layout
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                    "{%0, %1}, [%2];\n"
                    : "=r"(bRegs[j][0]),
                      "=r"(bRegs[j][1])
                    : "r"(address)
                );
            }
            // ------------------------------------------
            // Step 3.3: mma.sync
            // ------------------------------------------
            #pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+f"(cFrag[i][j][0]),
                          "+f"(cFrag[i][j][1]),
                          "+f"(cFrag[i][j][2]),
                          "+f"(cFrag[i][j][3])
                        : "r"(aRegs[i][0]),
                          "r"(aRegs[i][1]),
                          "r"(aRegs[i][2]),
                          "r"(aRegs[i][3]),
                          "r"(bRegs[j][0]),
                          "r"(bRegs[j][1])
                    );
                }
            }
    
        }
        // All Warps finish reading current Shared tiles
        __syncthreads();
    }
     // ==================================================
    // Step 4: Registers -> Global Memory C
    // ==================================================
    // MMA output fragment mapping
    const int fragmentRow = laneId / 4;
    const int fragmentCol = (laneId % 4) * 2;
     #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            // Global C row
            const int row =
                tileRow +
                warpRow * 32 +
                i * WM +
                fragmentRow;

            // Global C column
            const int col =
                tileCol +
                warpCol * 32 +
                j * WN +
                fragmentCol;

            // Four FP32 results per Lane
            C[row * N + col] =
                cFrag[i][j][0];

            C[row * N + col + 1] =
                cFrag[i][j][1];

            C[(row + 8) * N + col] =
                cFrag[i][j][2];

            C[(row + 8) * N + col + 1] =
                cFrag[i][j][3];
        }
    }
}
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

    // ==================================================
    // Step 1: Host Memory
    // ==================================================

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

    // ==================================================
    // Step 3: Kernel Launch
    // ==================================================

    dim3 block(128);
    dim3 grid(N / BN, M / BM);

    gemmV13MMA<<<grid, block>>>(
        dA, dB, dC, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==================================================
    // Step 4: cuBLAS Reference
    // ==================================================

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // A and B are row-major.
    // cuBLAS uses column-major matrices.
    // Therefore calculate C^T = B^T * A^T.

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

    // ==================================================
    // Step 5: Device -> Host
    // ==================================================

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

    // ==================================================
    // Step 6: Correctness Check
    // ==================================================

    float maxDiff = 0.0f;
    int errors = 0;

    for (size_t i = 0; i < hC.size(); ++i)
    {
        const float diff =
            std::fabs(hC[i] - hRef[i]);

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

            if (errors <= 5)
            {
                std::cout
                    << "Mismatch at " << i
                    << ": got " << hC[i]
                    << ", expected " << hRef[i]
                    << '\n';
            }
        }
    }

    std::cout
        << "V13 ldmatrix + mma.sync GEMM\n"
        << "Matrix: M=" << M
        << " N=" << N
        << " K=" << K << '\n'
        << "Correctness: "
        << (errors == 0 ? "PASS" : "FAIL")
        << '\n'
        << "Max difference: "
        << maxDiff << '\n';

    // ==================================================
    // Step 7: Benchmark
    // ==================================================

    if (errors == 0)
    {
        // Warmup
        for (int i = 0; i < WARMUP; ++i)
        {
            gemmV13MMA<<<grid, block>>>(
                dA, dB, dC, M, N, K
            );
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start;
        cudaEvent_t stop;

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < RUNS; ++i)
        {
            gemmV13MMA<<<grid, block>>>(
                dA, dB, dC, M, N, K
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
            << tflops << " TFLOPS\n";

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // ==================================================
    // Step 8: Cleanup
    // ==================================================

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));

    return errors == 0 ? 0 : 1;
}
