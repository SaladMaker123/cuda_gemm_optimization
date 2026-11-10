#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call)                          \
do{                                                 \
    cudaError_t err = (call);                      \
    if(err != cudaSuccess){                                              \
        std::cerr<<"CUDA ERROR: "                                 \
                 <<cudaGetErrorString(err)                      \
                 <<" at line "<<__LINE__<<std::endl;                      \
                 std::exit(EXIT_FAILURE);                           \
    }                                                               \
}while(0)

