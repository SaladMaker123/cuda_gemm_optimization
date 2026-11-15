#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>

#include "../common/cuda_check.h"
#include "../common/cublas_check.h"

constexpr int BM = 64;
constexpr int BN = 32;
constexpr int BK = 64;
// ============================================
// V6: Async Global-to-Shared Tile Loading
// ============================================
template<int ROWS,int COLS>
__device__ __forceinline__ void asyncLoadTile(
    const __half* src,
    __half(&tile)[ROWS][COLS],
    int rowBase,
    int colBase,
    int height,
    int width
)
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(int idx = tid;idx < ROWS * (COLS / 8);idx += blockDim.x * blockDim.y)
    {
        int localRow = idx / (COLS / 8);
        int localCol = (idx % (COLS / 8)) * 8;
        int globalRow = rowBase + localRow;
        int globalCol = colBase + localCol;

        if(globalRow < height && globalCol + 7 < width && width % 8 == 0)
        {
            __pipeline_memcpy_async(&tile[localRow][localCol],src + globalRow * width + globalCol,16);
        }else{
            #pragma unroll
            for(int i = 0;i < 8;++i)
            {
                tile[localRow][localCol + i] =
                (globalRow < height && globalCol + i < width) ? src[globalRow * width + globalCol + i] : __half(0.0f);
            }
        }
    }
}
// ============================================
// 2. V6 Double Buffer GEMM Kernel
// ============================================
__global__ void gemmV7BK64DoubleBuffer(
    const __half* A,
    const __half* B,
    float* C,
    int M,
    int N,
    int K
)
{
    __shared__ __align__(16) __half sA[2][BM][BK];
    __shared__ __align__(16) __half sB[2][BK][BN];
    int col = blockIdx.x * BN + threadIdx.x;
    float sum[8] = {};
    int PHASES = K / BK;

    // ========================================
    // 2.1 Preload Phase 0 into Buffer 0
    // ========================================
    asyncLoadTile(A,sA[0],blockIdx.y * BM,0,M,K);
    asyncLoadTile(B,sB[0],0,blockIdx.x * BN,K,N);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    // ========================================
    // 2.2 Double Buffer Pipeline
    // ========================================
    for(int phase = 0;phase < PHASES ;++phase)
    {
        int current = phase % 2;
        int next = (phase + 1) % 2;
        // ====================================
        // Prefetch the next phase
        // ====================================
        if(phase + 1 < PHASES)
        {
            asyncLoadTile(A,sA[next],blockIdx.y * BM,(phase + 1) * BK,M,K);
            asyncLoadTile(B,sB[next],(phase + 1) * BK,blockIdx.x * BN,K,N);
            __pipeline_commit();
        }
        // ====================================
        // Compute the current phase
        // ====================================
        for(int k = 0;k < BK;++k)
        {
            float b = __half2float(sB[current][k][threadIdx.x]);
            #pragma unroll
            for(int r = 0;r < 8;++r)
            {
                sum[r] += __half2float(sA[current][threadIdx.y + r * blockDim.y][k]) * b;
            }
        }
        // ====================================
        // Wait for async copies
        // ====================================
        if(phase + 1 < PHASES)
        {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
    }
     // ========================================
    // 2.3 Store C
    // ========================================
    #pragma unroll
    for(int r = 0;r < 8;++r)
    {
        int row = blockIdx.y * BM + threadIdx.y + r * blockDim.y;
        if(row < M && col < N)
        {
            C[row * N + col] = sum[r];
        }
    }
}
// ============================================
// 3. Host: Correctness + Benchmark
// ============================================
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
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    for(auto& x : hA)
    {
        x = __half(dist(rng));
    }
    for(auto& x : hB)
    {
        x = __half(dist(rng));
    }
    // ========================================
    // 3.1 GPU Memory Allocation
    // ========================================
    __half* dA = nullptr;
    __half* dB = nullptr;
    float* dC = nullptr;
    float* dRef = nullptr;

    CUDA_CHECK(cudaMalloc(&dA,hA.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB,hB.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC,hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRef,hRef.size() * sizeof(float)));
    // ========================================
    // 3.2 Copy Input Data
    // ========================================
    CUDA_CHECK(cudaMemcpy(dA,hA.data(),hA.size() * sizeof(__half),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB.data(),hB.size() * sizeof(__half),cudaMemcpyHostToDevice));
    // ========================================
    // 3.3 Kernel Configuration
    // ========================================
    dim3 block(32,8);
    dim3 grid(
        (N + BN -1) / BN,
        (M + BM -1) / BM
    );
    // ========================================
    // 3.4 Correctness Test
    // ========================================
    gemmV7BK64DoubleBuffer<<<grid,block>>>(dA,dB,dC,M,N,K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // ========================================
    // 3.5 cuBLAS Reference
    // ========================================
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
        dB,CUDA_R_16F,N,
        dA,CUDA_R_16F,K,
        &beta,
        dRef,CUDA_R_32F,N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));
    CUDA_CHECK(cudaDeviceSynchronize());
    // ========================================
    // 3.6 Copy Results to CPU
    // ========================================
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
    // ========================================
    // 3.7 Compare Results
    // ========================================
    float maxDiff = 0.0f;
    int errors = 0;
    for(size_t i = 0;i<hC.size();++i)
    {
        float diff = std::fabs(hRef[i] - hC[i]);
        float tolerance = 0.01f + 0.001f * std::fabs(hRef[i]);
        if(diff > maxDiff)
        {
            maxDiff = diff;
        }
        if(!std::isfinite(hC[i]) || !std::isfinite(hRef[i]) || diff > tolerance)
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
        << "gemmV7 BK64 DoubleBuffer"
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
    // ========================================
    // 3.8 Exit if Incorrect
    // ========================================
    if (errors != 0)
    {
        CUBLAS_CHECK(cublasDestroy(handle));

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaFree(dRef));

        return 1;
    }
    // ========================================
    // 3.9 Warmup
    // ========================================
    for (int i = 0; i < WARMUP; ++i)
    {
        gemmV7BK64DoubleBuffer<<<grid, block>>>(
            dA, dB, dC, M, N, K
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // ========================================
    // 3.10 CUDA Event Benchmark
    // ========================================
    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0; i < RUNS;++i)
    {
        gemmV7BK64DoubleBuffer<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs,start,stop));
    float avgMs = elapsedMs / RUNS;
    double tflops = 2.0 * M * N * K / (avgMs * 1e9);
    std::cout<<std::fixed<<std::setprecision(3);
    std::cout<< "Average time: "<< avgMs << " ms"<< std::endl;
    std::cout<< "Performance: "<< tflops << " TFLOPS"<< std::endl;
    // ========================================
    // 3.11 Cleanup
    // ========================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));
    return 0;
}