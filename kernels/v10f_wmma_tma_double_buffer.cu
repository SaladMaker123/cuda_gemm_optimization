#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <utility>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda/barrier>

#include "../common/cuda_check.h"
#include "../common/cublas_check.h"
namespace wmma = nvcuda::wmma;
namespace cde = cuda::device::experimental;
using barrier = cuda::barrier<cuda::thread_scope_block>;
// WMMA Tile
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;
// Block Tile
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 16;
constexpr int TILE_BYTES = (BM * BK + BK * BN) * sizeof(__half);
static_assert(BM == 32 && BN == 32 && BK == 16,"V10-F uses fixed tile sizes");
// Create a 2D Tensor Map on the host
void createTensorMap(
    CUtensorMap& map,
    void* ptr,
    cuuint64_t width,
    cuuint64_t height,
    cuuint32_t tileWidth,
    cuuint32_t tileHeight
)
{
    const cuuint64_t globalDim[2] = {width,height};
    const cuuint64_t globalStrides[1] = {width * sizeof(__half)};
    const cuuint32_t boxDim[2] = {tileWidth,tileHeight};
    const cuuint32_t elementStrides[2] = {1,1};
    CUresult status = cuTensorMapEncodeTiled(
        &map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,
        ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if(status != CUDA_SUCCESS)
    {
        const char* errorName = nullptr;
        cuGetErrorName(status,&errorName);
        std::cerr<<"Tensor Map Error:"
        <<(errorName ? errorName : "Unknown")<<std::endl;
        std::exit(EXIT_FAILURE);
    }
}
// V10-F: TMA 2D + Double Buffering + WMMA
__global__ void gemmV10FWMMA(
    const __grid_constant__ CUtensorMap mapA,
    const __grid_constant__ CUtensorMap mapB,
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
    if(tileRow + BM > M || tileCol + BN > N || K % BK != 0)
    {
        return;
    }
    // Double Buffered Shared Memory
    __shared__ __align__(128) __half sA[2][BM][BK];
    __shared__ __align__(128) __half sB[2][BK][BN];
    // One barrier for each buffer
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bars[2];
    if(tid == 0)
    {
        init(&bars[0],blockDim.x);
        init(&bars[1],blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();
     // WMMA Fragments
    wmma::fragment<wmma::matrix_a,WM, WN, WK,__half,wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b,WM, WN, WK,__half,wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator,WM, WN, WK,float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);
    const int numPhases = K / BK;
    // ==========================================
    // Step 1: Load Tile 0 into Buffer 0
    // ==========================================
    barrier::arrival_token initialToken;
    if(tid == 0)
    {
       // TMA: A Tile
       cde::cp_async_bulk_tensor_2d_global_to_shared(
        &sA[0][0][0],
        &mapA,
        0,
        tileRow,
        bars[0]
       );
       // TMA: B Tile
       cde::cp_async_bulk_tensor_2d_global_to_shared(
        &sB[0][0][0],
        &mapB,
        tileCol,
        0,
        bars[0]
       );
       initialToken = cuda::device::barrier_arrive_tx(bars[0],1,TILE_BYTES);
    }
    else
    {
        initialToken = bars[0].arrive();
    }
    bars[0].wait(std::move(initialToken));
    __syncthreads();
    // ==========================================
    // Step 2: Double Buffered K Tiling
    // ==========================================
    for(int phase = 0;phase < numPhases;++phase)
    {
        const int current = phase % 2;
        const int next = 1 - current;
        const bool hasNext = phase + 1 < numPhases;
        // Prefetch next tile
        if(hasNext)
        {
            const int nextK = (phase + 1) * BK;
            barrier::arrival_token nextToken;
            if(tid == 0)
            {
                // TMA: Next A Tile
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &sA[next][0][0],
                    &mapA,
                    nextK,
                    tileRow,
                    bars[next]
                );
                // TMA: Next B Tile
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &sB[next][0][0],
                    &mapB,
                    tileCol,
                    nextK,
                    bars[next]
                );
                nextToken = cuda::device::barrier_arrive_tx(bars[next],1,TILE_BYTES);
            }
            else
            {
                nextToken = bars[next].arrive();
            }
            // Compute using current buffer
            wmma::load_matrix_sync(aFrag,&sA[current][warpRow * WM][0],BK);
            wmma::load_matrix_sync(bFrag,&sB[current][0][warpCol * WN],BN);
            wmma::mma_sync(cFrag,aFrag,bFrag,cFrag);
            // Wait for next tile
            bars[next].wait(std::move(nextToken));
        }
        else
        {
            // Last phase: no more prefetch
            wmma::load_matrix_sync(aFrag,&sA[current][warpRow * WM][0],BK);
            wmma::load_matrix_sync(bFrag,&sB[current][0][warpCol * WN],BN);
            wmma::mma_sync(cFrag,aFrag,bFrag,cFrag);
        }
        // All warps finish using the current buffer
        __syncthreads();
    }
    // Store C Fragment -> Global Memory
    wmma::store_matrix_sync(C + (tileRow + warpRow * WM) * N + tileCol + warpCol * WN,cFrag,N,wmma::mem_row_major);
    __syncthreads();
    if (tid == 0)
    {
        bars[0].~barrier();
        bars[1].~barrier();
    }
}
int main()
{
    // ==========================================
    // Step 1: Matrix Configuration
    // ==========================================
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const int WARMUP = 5;
    const int RUNS = 30;
    // ==========================================
    // Step 2: Host Memory
    // ==========================================
    std::vector<__half> hA(M * K);
    std::vector<__half> hB(K * N);
    std::vector<float> hC(M * N);
    std::vector<float> hRef(M * N);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    for(auto& x : hA)
    {
        x = __float2half(dist(rng));
    }
    for(auto& x : hB)
    {
        x = __float2half(dist(rng));
    }
    // ==========================================
    // Step 3: Device Memory
    // ==========================================
    __half* dA = nullptr;
    __half* dB = nullptr;
    float* dC = nullptr;
    float* dRef = nullptr;

    CUDA_CHECK(cudaMalloc(&dA,hA.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB,hB.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC,hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRef,hRef.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA,hA.data(),hA.size() * sizeof(__half),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB.data(),hB.size() * sizeof(__half),cudaMemcpyHostToDevice));
    // ==========================================
    // Step 4: Create Tensor Maps
    // ==========================================
    CUtensorMap mapA{};
    CUtensorMap mapB{};

    createTensorMap(mapA,dA,K,M,BK,BM);
    createTensorMap(mapB,dB,N,K,BN,BK);
    // ==========================================
    // Step 5: Kernel Configuration
    // ==========================================
    dim3 block(128);
    dim3 grid(
        (N + BN -1) / BN,
        (M + BM -1) / BM
    );
    // ==========================================
    // Step 6: Correctness Test
    // ==========================================
    gemmV10FWMMA<<<grid,block>>>(mapA,mapB,dC,M,N,K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // ==========================================
    // Step 7: cuBLAS Reference
    // ==========================================
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,M,K,
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
    // ==========================================
    // Step 8: Copy Results to Host
    // ==========================================
    CUDA_CHECK(cudaMemcpy(hC.data(),dC,hC.size() * sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hRef.data(),dRef,hRef.size() * sizeof(float),cudaMemcpyDeviceToHost));
    // ==========================================
    // Step 9: Compare Results
    // ==========================================
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
        << "V10-F: TMA 2D Double Buffer WMMA GEMM"
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
    // ==========================================
    // Step 10: Benchmark
    // ==========================================
        if (errors == 0)
    {
        // Warmup
        for (int i = 0; i < WARMUP; ++i)
        {
            gemmV10FWMMA<<<grid, block>>>(
                mapA,
                mapB,
                dC,
                M,
                N,
                K
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
            gemmV10FWMMA<<<grid, block>>>(
                mapA,
                mapB,
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
    // ==========================================
    // Step 11: Release Resources
    // ==========================================
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));
    return errors == 0 ? 0 : 1;
}
