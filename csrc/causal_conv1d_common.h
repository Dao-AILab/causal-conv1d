/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

//#include <cuda_bf16.h>
#include <hip/hip_bf16.h>
#include <cuda_fp16.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

// rocm functions
#include <algorithm>
constexpr size_t my_max(std::initializer_list<size_t> ilist)
{
    return *std::max_element(ilist.begin(), ilist.end());
}

// added adeem
template<typename T>
constexpr T constexpr_min(T a, T b) {
    return a < b ? a : b;
}

template<int BYTES> struct BytesToType {};

template<> struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
};

template<> struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};

template<> struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};

template<> struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template<> struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ inline T operator()(T const & x, T const & y) { return x + y; }
};

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        // TODO: document this
        // x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        x = op(x, __shfl_xor(x, OFFSET)); // 
        return Allreduce<OFFSET>::run(x, op);
    }
};

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ inline T run(T x, Operator &op) {
    // TODO: document this
    // x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    x = op(x, __shfl_xor(x, 1)); // 
    return x;
}
};
