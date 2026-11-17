#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda_pipeline.h>

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
constexpr int BK = 16;

static_assert(
    BM == 32 && BN == 32 && BK == 16,
    "V10-E uses fixed tile sizes"
);
// V10-E: cp.async + Double Buffering + WMMA
__global__ void gemmV10EWMMA(
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
    // Only complete, aligned tiles
    if (tileRow + BM > M ||
        tileCol + BN > N ||
        K % BK != 0 ||
        K % 8 != 0 ||
        N % 8 != 0)
    {
        return;
    }
    // Double Buffered Shared Memory
    __shared__ __align__(32) __half sA[2][BM][BK];
    __shared__ __align__(32) __half sB[2][BK][BN];
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

    const int numPhases = K / BK;
    // ==========================================
    // Step 1: Prefetch Tile 0 into Buffer 0
    // ==========================================
    if(tid < 64)
    {
        const int aRow = tid / 2;
        const int aCol = (tid % 2) * 8;
        __pipeline_memcpy_async(&sA[0][aRow][aCol],&A[(tileRow + aRow) * K + aCol],16);
    }
    else
    {
        const int tid2 = tid - 64;
        const int bRow = tid2 / 4;
        const int bCol = (tid2 % 4) * 8;
        __pipeline_memcpy_async(&sB[0][bRow][bCol],&B[bRow * N + tileCol + bCol],16);
    }
    // Commit the initial async copy
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    // ==========================================
    // Step 2: Double Buffered K Tiling
    // ==========================================
    for(int phase = 0;phase < numPhases; ++phase)
    {
        const int current = phase % 2;
        const int next = (current + 1) % 2;
        // --------------------------------------
        // Prefetch the next K Tile
        // --------------------------------------
        if(phase + 1 < numPhases)
        {
            const int nextK = (phase + 1) * BK;
            //warp0,1:load A
            if(tid < 64)
            {
                const int aRow = tid / 2;
                const int aCol = (tid % 2) * 8;
                __pipeline_memcpy_async(&sA[next][aRow][aCol],&A[(tileRow + aRow) * K + nextK + aCol],16);
            }
             // Warp 2,3: Load B
            else
            {
                const int tid2 = tid - 64;
                const int bRow = tid2 / 4;
                const int bCol = (tid2 % 4) * 8;
                __pipeline_memcpy_async(&sB[next][bRow][bCol],&B[(nextK + bRow) * N + tileCol + bCol],16);
            }
            __pipeline_commit();
        }
        // --------------------------------------
        // Compute using the current buffer
        // --------------------------------------
        wmma::load_matrix_sync(aFrag,&sA[current][warpRow * WM][0],BK);
        wmma::load_matrix_sync(bFrag,&sB[current][0][warpCol * WN],BN);
        wmma::mma_sync(cFrag,aFrag,bFrag,cFrag);
        // --------------------------------------
        // Wait for next buffer before using it
        // --------------------------------------
        if (phase + 1 < numPhases)
        {
            __pipeline_wait_prior(0);
        }
         __syncthreads();
    }
    // Store C Fragment -> Global Memory
    wmma::store_matrix_sync(C + (tileRow + warpRow * WM) * N + tileCol + warpCol * WN,cFrag,N,wmma::mem_row_major);
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

    // V10-E Correctness Test
    gemmV10EWMMA<<<grid, block>>>(
        dA, dB, dC, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // cuBLAS Reference
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major C = A * B
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
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
        << "V10-E cp.async Double Buffer WMMA GEMM"
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

    if (errors == 0)
    {
        // Warmup
        for (int i = 0; i < WARMUP; ++i)
        {
            gemmV10EWMMA<<<grid, block>>>(
                dA, dB, dC, M, N, K
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
            gemmV10EWMMA<<<grid, block>>>(
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
            << std::setprecision(3);

        std::cout
            << "Average time: "
            << avgMs << " ms"
            << std::endl;

        std::cout
            << "Performance: "
            << tflops << " TFLOPS"
            << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));

    return errors == 0 ? 0 : 1;
}