#pragma once

#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>

#define CUBLAS_CHECK(call)                                  \
do {                                                        \
    cublasStatus_t status = (call);                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                  \
        std::cerr << "cuBLAS ERROR at line "                \
                  << __LINE__ << std::endl;                 \
        std::exit(EXIT_FAILURE);                            \
    }                                                       \
} while (0)

