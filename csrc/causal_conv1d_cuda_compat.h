#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

inline void causal_conv1d_cuda_check_impl(
    cudaError_t error,
    const char* expr,
    const char* file,
    int line) {
    if (error == cudaSuccess) {
        return;
    }
    throw std::runtime_error(
        std::string("CUDA runtime error at ") + file + ":" + std::to_string(line)
        + " for " + expr + ": " + cudaGetErrorString(error));
}

#define CAUSAL_CONV1D_CUDA_CHECK(expr) \
    causal_conv1d_cuda_check_impl((expr), #expr, __FILE__, __LINE__)

#define CAUSAL_CONV1D_CUDA_KERNEL_LAUNCH_CHECK() \
    CAUSAL_CONV1D_CUDA_CHECK(cudaGetLastError())
