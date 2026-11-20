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
// One WMMA instruction computes a 16 x 16 x 16 tile.
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;
// V11: larger block tile, 4 warps, multiple accumulators per warp.
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
__global__ void gemmV11WMMA(
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
    if (tileRow + BM > M || tileCol + BN > N ||
        K <= 0 || K % BK != 0 || K % 8 != 0 || N % 8 != 0)
    {
        return;
    }
    __shared__ __align__(32) __half sA[BM][BK];
    __shared__ __align__(32) __half sB[BK][BN];
    // Each warp owns a 32x32 output region = four 16x16 fragments.
    wmma::fragment<wmma::accumulator,WM,WN,WK,float> cFrag[2][2];
    #pragma unroll
    for(int i = 0;i < 2;++i)
    {
        for(int j = 0;j < 2;++j)
        {
            wmma::fill_fragment(cFrag[i][j],0.0f);
        }
    }
    // K is divided into 32-element stages, with two WMMA K steps per stage.
    for(int phase = 0;phase < K / BK;++phase)
    {
        const int kStart = phase * BK;
        // Warps 0,1: load A[64][32], 4 chunks of 8 FP16 per lane.
        if(tid < 64)
        {
            #pragma unroll
            for(int part = 0;part < 4;part++)
            {
                const int idx = tid + part * 64;
                const int aRow = idx / 4;
                const int aCol = (idx % 4) * 8;
                const uint4 value = *reinterpret_cast<const uint4*>(&A[(tileRow + aRow) * K + kStart + aCol]);
                *reinterpret_cast<uint4*>(&sA[aRow][aCol]) = value;
            }
        }
        // Warps 2,3: load B[32][64], 4 chunks of 8 FP16 per lane.
        else
        {
            const int vectorId = tid - 64;
            #pragma unroll
            for(int part = 0;part < 4;part++)
            {
                const int idx = vectorId + part * 64;
                const int bRow = idx / 8;
                const int bCol = (idx % 8) * 8;
                const uint4 value = *reinterpret_cast<const uint4*>(&B[(kStart + bRow) * N + tileCol + bCol]);
                *reinterpret_cast<uint4*>(&sB[bRow][bCol]) = value;
            }
        }
        __syncthreads();
        // Load two A and two B fragments; reuse them for 4 WMMA calls.
        #pragma unroll
        for(int k = 0;k < BK;k += WK)
        {
            wmma::fragment<wmma::matrix_a, WM, WN, WK,__half, wmma::row_major> aFrag[2];
            wmma::fragment<wmma::matrix_b, WM, WN, WK,__half, wmma::row_major> bFrag[2];
            #pragma unroll
            for(int i = 0;i < 2;i++)
            {
                wmma::load_matrix_sync(aFrag[i],&sA[warpRow * 32 + i * WM][k],BK);
            }
            #pragma unroll
            for(int j = 0;j < 2;j++)
            {
                wmma::load_matrix_sync(bFrag[j],&sB[k][warpCol * 32 + j * WN],BN);
            }
            #pragma unroll 
            for(int i = 0;i < 2;++i)
            {
                #pragma unroll
                for(int j = 0;j < 2;++j)
                {
                    wmma::mma_sync(cFrag[i][j],aFrag[i],bFrag[j],cFrag[i][j]);
                }
            }
        }
        __syncthreads();
    }
    // Store four 16x16 output fragments per warp.
    #pragma unroll
    for(int i = 0; i < 2;i++)
    {
        #pragma unroll
        for(int j = 0; j < 2;j++)
        {
            const int row = tileRow + warpRow * 32 + i * WM;
            const int col = tileCol + warpCol * 32 + j * WN;
            wmma::store_matrix_sync(&C[row * N + col],cFrag[i][j],N,wmma::mem_row_major);
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

    static_assert(M % BM == 0 && N % BN == 0 && K % BK == 0,
                  "V11 baseline requires complete tiles");

    static_assert(N % 8 == 0 && K % 8 == 0,
                  "Global-memory row strides must be 16-byte aligned");

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

    dim3 block(128);
    dim3 grid(N / BN, M / BM);

    // V11 correctness test
    gemmV11WMMA<<<grid, block>>>(dA, dB, dC, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Row-major C=A*B via column-major cuBLAS: C^T=B^T*A^T.
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
        dB, CUDA_R_16F, N,
        dA, CUDA_R_16F, K,
        &beta,
        dRef, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

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

    // Correctness check
    float maxDiff = 0.0f;
    int errors = 0;

    for (size_t i = 0; i < hC.size(); ++i)
    {
        const float diff = std::fabs(hC[i] - hRef[i]);
        const float tolerance = 0.01f + 0.001f * std::fabs(hRef[i]);

        if (diff > maxDiff)
        {
            maxDiff = diff;
        }

        if (!std::isfinite(hC[i]) || !std::isfinite(hRef[i]) ||
            diff > tolerance)
        {
            ++errors;

            if (errors <= 3)
            {
                std::cout
                    << "Mismatch at " << i
                    << ": got " << hC[i]
                    << ", expected " << hRef[i] << '\n';
            }
        }
    }

    std::cout
        << "V11 WMMA Large Tile + Four Accumulators per Warp\n"
        << "Matrix: M=" << M
        << " N=" << N
        << " K=" << K << '\n'
        << "Correctness: "
        << (errors == 0 ? "PASS" : "FAIL") << '\n'
        << "Max difference: " << maxDiff << '\n';

    // Benchmark
    if (errors == 0)
    {
        for (int i = 0; i < WARMUP; ++i)
        {
            gemmV11WMMA<<<grid, block>>>(dA, dB, dC, M, N, K);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < RUNS; ++i)
        {
            gemmV11WMMA<<<grid, block>>>(dA, dB, dC, M, N, K);
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
            << std::fixed << std::setprecision(6)
            << "Average time: " << avgMs << " ms\n"
            << "Performance: " << tflops << " TFLOPS\n";

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