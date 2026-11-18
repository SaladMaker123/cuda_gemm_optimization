#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

#include "../common/cuda_check.h"
#include "../common/cublas_check.h"

namespace wmma = nvcuda::wmma;
namespace cg = cooperative_groups;

// ==========================================
// WMMA Tile
// ==========================================
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

// ==========================================
// Block Tile
// ==========================================
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 16;
// ==========================================
// Warp Specialization + Triple Buffer
// ==========================================
constexpr int STAGES = 3;
constexpr int PRODUCER_THREADS = 32;
constexpr int CONSUMER_WARPS = 4;
constexpr int BLOCK_THREADS = PRODUCER_THREADS + CONSUMER_WARPS * 32;
// ==========================================
// V10-H Kernel
// ==========================================
__global__ void gemmV10HWmma(
    const __half* A,
    const __half* B,
    float* C,
    int M,
    int N,
    int K
)
{
    // ======================================
    // Step 1: Thread Mapping
    // ======================================
    const int tid = threadIdx.x;

    const int warpId = tid / 32;
    const int laneId = tid % 32;

    const bool isProducer = (warpId == 0);

    const int tileRow = blockIdx.y * BM;
    const int tileCol = blockIdx.x * BN;

    // Only complete and aligned tiles
    if (tileRow + BM > M ||
        tileCol + BN > N ||
        K <= 0 ||
        K % BK != 0 ||
        K % 8 != 0 ||
        N % 8 != 0)
    {
        return;
    }
    // ======================================
    // Step 2: Triple Buffered Shared Memory
    // ======================================
    __shared__ __align__(32) __half sA[STAGES][BM][BK];
    __shared__ __align__(32) __half sB[STAGES][BK][BN];
    // ======================================
    // Step 3: Partitioned Pipeline
    // ======================================
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block,STAGES> sharedState;
    auto threadBlock = cg::this_thread_block();
    cuda::std::size_t producerCount = PRODUCER_THREADS;
    auto pipe = cuda::make_pipeline(threadBlock,&sharedState,producerCount);
    const int numPhases = K / BK;
    // ======================================
    // Step 4: Producer Warp
    // Warp 0: Thread 0~31
    // ======================================
    if(isProducer)
    {
        for (int phase = 0;phase < numPhases;++phase)
        {
            const int buffer = phase % STAGES;
            const int kStart = phase * BK;
            // Wait until this stage is writable
            pipe.producer_acquire();
            // Load A Tile and B Tile
            for (int part = 0; part < 2; ++part)
            {
                // --------------------------
                // A Tile: 32x16 FP16
                // --------------------------
                const int aIdx = laneId + part * 32;

                const int aRow = aIdx / 2;
                const int aCol = (aIdx % 2) * 8;

                cuda::memcpy_async(
                    &sA[buffer][aRow][aCol],
                    &A[(tileRow + aRow) * K
                       + kStart + aCol],
                    cuda::aligned_size_t<16>(16),
                    pipe
                );
                // --------------------------
                // B Tile: 16x32 FP16
                // --------------------------
                const int bIdx =
                    laneId + part * 32;

                const int bRow = bIdx / 4;
                const int bCol = (bIdx % 4) * 8;

                cuda::memcpy_async(
                    &sB[buffer][bRow][bCol],
                    &B[(kStart + bRow) * N
                       + tileCol + bCol],
                    cuda::aligned_size_t<16>(16),
                    pipe
                );
            }
            // Submit the current stage
            pipe.producer_commit();
        }
    }
    // ======================================
    // Step 5: Consumer Warps
    // Warp 1~4: Thread 32~159
    // ======================================
    else
    {
        const int consumerWarp = warpId - 1;

        const int warpRow = consumerWarp / 2;
        const int warpCol = consumerWarp % 2;
        // WMMA A Fragment
        wmma::fragment<wmma::matrix_a,WM, WN, WK,__half,wmma::row_major> aFrag;
        // WMMA B Fragment
        wmma::fragment<wmma::matrix_b,WM, WN, WK,__half,wmma::row_major> bFrag;
        // WMMA C Accumulator
        wmma::fragment<wmma::accumulator,WM, WN, WK,float> cFrag;
        wmma::fill_fragment(cFrag, 0.0f);
        // ==================================
        // Consumer K Tiling
        // ==================================
        for (int phase = 0;phase < numPhases;++phase)
        {
            const int buffer = phase % STAGES;
            // Wait for the oldest submitted stage
            pipe.consumer_wait();
            // Load A Fragment
            wmma::load_matrix_sync(aFrag,&sA[buffer][warpRow * WM][0],BK);
            // Load B Fragment
            wmma::load_matrix_sync(bFrag,&sB[buffer][0][warpCol * WN],BN);
            // Tensor Core GEMM
            wmma::mma_sync(cFrag,aFrag,bFrag,cFrag);
            // All consumers finish this stage
            pipe.consumer_release();
        }
        // ==================================
        // Step 6: Store Result
        // ==================================
        wmma::store_matrix_sync(
            C + (tileRow + warpRow * WM) * N+ tileCol + warpCol * WN,
            cFrag,
            N,
            wmma::mem_row_major
        );
    }
}
// ==========================================
// Main
// ==========================================
int main()
{
    // ======================================
    // Step 1: Matrix Configuration
    // ======================================
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    const int WARMUP = 5;
    const int RUNS = 30;


    // ======================================
    // Step 2: Host Memory
    // ======================================
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


    // ======================================
    // Step 3: Device Memory
    // ======================================
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


    // ======================================
    // Step 4: Host -> Device
    // ======================================
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


    // ======================================
    // Step 5: Kernel Configuration
    // ======================================
    dim3 block(BLOCK_THREADS);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );


    // ======================================
    // Step 6: Correctness Test
    // ======================================
    gemmV10HWmma<<<grid, block>>>(
        dA, dB, dC, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // ======================================
    // Step 7: cuBLAS Reference
    // ======================================
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

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


    // ======================================
    // Step 8: Device -> Host
    // ======================================
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


    // ======================================
    // Step 9: Correctness Check
    // ======================================
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
        << "V10-H Warp Specialization Triple Buffer WMMA GEMM"
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


    // ======================================
    // Step 10: Benchmark
    // ======================================
    if (errors == 0)
    {
        for (int i = 0; i < WARMUP; ++i)
        {
            gemmV10HWmma<<<grid, block>>>(
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
            gemmV10HWmma<<<grid, block>>>(
                dA, dB, dC, M, N, K
            );
        }

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsedMs = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsedMs, start, stop
        ));

        const float avgMs = elapsedMs / RUNS;

        const double tflops =
            2.0 * M * N * K / (avgMs * 1e9);

        std::cout
            << std::fixed
            << std::setprecision(6);

        std::cout
            << "Average time: "
            << avgMs << " ms"
            << std::endl;

        std::cout
            << std::setprecision(3);

        std::cout
            << "Performance: "
            << tflops << " TFLOPS"
            << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }


    // ======================================
    // Step 11: Release Resources
    // ======================================
    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));

    return errors == 0 ? 0 : 1;
}