#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>

#include "causal_conv1d_kernel.hip"

// Helper macro for HIP error checking
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ << " - " \
                      << hipGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constexpr max helper
template<typename T>
constexpr T const_max(T a, T b) {
    return (a > b) ? a : b;
}

// Kernel traits similar to CUDA version
template<int kNThreads_, int kWidth_, int kNElts_>
struct CausalConv1dKernelTraits {
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNElts = kNElts_;
    static constexpr int kChunkSize = kNThreads * kNElts;
    
    using vec_t = float4;  // 4 floats for vectorization
    
    // BlockLoad and BlockStore for coalesced memory access
    using BlockLoadT = hipcub::BlockLoad<float, kNThreads, kNElts, hipcub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = hipcub::BlockStore<float, kNThreads, kNElts, hipcub::BLOCK_STORE_WARP_TRANSPOSE>;
    
    static constexpr int kSmemIOSize = const_max(sizeof(typename BlockLoadT::TempStorage), 
                                                  sizeof(typename BlockStoreT::TempStorage));
    static constexpr int kSmemExchangeSize = kNThreads * sizeof(vec_t);
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

// Test case configuration
struct TestCase {
    const char* name;
    int batch;
    int dim;
    int seqlen;
    int width;
    bool use_bias;
    bool use_silu;
};

// CPU reference implementation for causal conv1d
void causal_conv1d_ref_cpu(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int batch,
    int dim,
    int seqlen,
    int width,
    bool use_silu
) {
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < dim; ++d) {
            for (int t = 0; t < seqlen; ++t) {
                float acc = 0.0f;
                
                // Causal convolution
                for (int i = 0; i < width; ++i) {
                    int input_t = t - width + 1 + i;
                    if (input_t >= 0) {
                        int input_idx = b * (dim * seqlen) + d * seqlen + input_t;
                        acc += x[input_idx] * weight[d * width + i];
                    }
                }
                
                // Add bias
                if (bias != nullptr) {
                    acc += bias[d];
                }
                
                // Apply SiLU activation
                if (use_silu) {
                    acc = acc / (1.0f + expf(-acc));
                }
                
                int out_idx = b * (dim * seqlen) + d * seqlen + t;
                out[out_idx] = acc;
            }
        }
    }
}

// Run a single test case
bool run_test_case(const TestCase& tc, int test_idx) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Case " << test_idx << ": " << tc.name << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Batch: " << tc.batch << ", Dim: " << tc.dim 
              << ", SeqLen: " << tc.seqlen << ", Width: " << tc.width << std::endl;
    std::cout << "Bias: " << (tc.use_bias ? "Yes" : "No") 
              << ", Activation: " << (tc.use_silu ? "SiLU" : "None") << std::endl;
    
    // Kernel configuration matching CUDA version
    constexpr int kNThreads = 128;  // threads per block
    constexpr int kWidth = 4;       // must match width
    constexpr int kNElts = 4;       // elements per thread
    
    using Ktraits = CausalConv1dKernelTraits<kNThreads, kWidth, kNElts>;
    
    const int total_elements = tc.batch * tc.dim * tc.seqlen;
    const int weight_size = tc.dim * tc.width;
    const int bias_size = tc.dim;
    
    std::cout << "Total elements: " << total_elements << std::endl;
    std::cout << "Kernel config: " << kNThreads << " threads, " 
              << kNElts << " elts/thread, chunk size: " << Ktraits::kChunkSize << std::endl;
    
    // Allocate host memory
    std::vector<float> h_x(total_elements);
    std::vector<float> h_weight(weight_size);
    std::vector<float> h_bias(tc.use_bias ? bias_size : 0);
    std::vector<float> h_out_gpu(total_elements);
    std::vector<float> h_out_ref(total_elements);
    
    // Initialize with random values
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < total_elements; ++i) {
        h_x[i] = dis(gen);
    }
    for (int i = 0; i < weight_size; ++i) {
        h_weight[i] = dis(gen);
    }
    if (tc.use_bias) {
        for (int i = 0; i < bias_size; ++i) {
            h_bias[i] = dis(gen);
        }
    }
    
    // CPU reference computation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    causal_conv1d_ref_cpu(
        h_x.data(),
        h_weight.data(),
        tc.use_bias ? h_bias.data() : nullptr,
        h_out_ref.data(),
        tc.batch, tc.dim, tc.seqlen, tc.width,
        tc.use_silu
    );
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU reference time: " << cpu_ms << " ms" << std::endl;
    
    // Allocate GPU memory
    float *d_x, *d_weight, *d_bias, *d_out, *d_out_naive;
    HIP_CHECK(hipMalloc(&d_x, total_elements * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(float)));
    if (tc.use_bias) {
        HIP_CHECK(hipMalloc(&d_bias, bias_size * sizeof(float)));
    }
    HIP_CHECK(hipMalloc(&d_out, total_elements * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_naive, total_elements * sizeof(float)));
    
    // Copy data to GPU
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), total_elements * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), weight_size * sizeof(float), hipMemcpyHostToDevice));
    if (tc.use_bias) {
        HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), bias_size * sizeof(float), hipMemcpyHostToDevice));
    }
    
    // Calculate strides (row-major layout: batch, dim, seqlen)
    const int x_batch_stride = tc.dim * tc.seqlen;
    const int x_c_stride = tc.seqlen;
    const int weight_c_stride = tc.width;
    const int weight_width_stride = 1;
    const int out_batch_stride = tc.dim * tc.seqlen;
    const int out_c_stride = tc.seqlen;
    
    // Setup kernel launch parameters for optimized kernel
    // Grid: (batch, dim) - each block handles one (batch, channel) pair
    dim3 grid_opt(tc.batch, tc.dim);
    dim3 block_opt(kNThreads);
    size_t smem_size = Ktraits::kSmemSize;
    
    std::cout << "Shared memory size: " << smem_size << " bytes" << std::endl;
    
    // Warmup optimized kernel
    hipLaunchKernelGGL(
        causal_conv1d_fwd_kernel<Ktraits>,
        grid_opt, block_opt, smem_size, 0,
        d_x, d_weight, tc.use_bias ? d_bias : nullptr, d_out,
        tc.batch, tc.dim, tc.seqlen, tc.width,
        x_batch_stride, x_c_stride,
        weight_c_stride, weight_width_stride,
        out_batch_stride, out_c_stride,
        tc.use_silu
    );
    HIP_CHECK(hipDeviceSynchronize());
    
    // Performance test - optimized kernel
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    const int num_iters = 100;
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < num_iters; ++i) {
        hipLaunchKernelGGL(
            causal_conv1d_fwd_kernel<Ktraits>,
            grid_opt, block_opt, smem_size, 0,
            d_x, d_weight, tc.use_bias ? d_bias : nullptr, d_out,
            tc.batch, tc.dim, tc.seqlen, tc.width,
            x_batch_stride, x_c_stride,
            weight_c_stride, weight_width_stride,
            out_batch_stride, out_c_stride,
            tc.use_silu
        );
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / num_iters;
    std::cout << "GPU optimized kernel time (" << num_iters << " iters): " << avg_ms << " ms" << std::endl;
    
    // Test naive kernel for comparison
    dim3 grid_naive((total_elements + 255) / 256);
    dim3 block_naive(256);
    
    hipLaunchKernelGGL(
        causal_conv1d_kernel_naive,
        grid_naive, block_naive, 0, 0,
        d_x, d_weight, tc.use_bias ? d_bias : nullptr, d_out_naive,
        tc.batch, tc.dim, tc.seqlen, tc.width, tc.use_silu
    );
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < num_iters; ++i) {
        hipLaunchKernelGGL(
            causal_conv1d_kernel_naive,
            grid_naive, block_naive, 0, 0,
            d_x, d_weight, tc.use_bias ? d_bias : nullptr, d_out_naive,
            tc.batch, tc.dim, tc.seqlen, tc.width, tc.use_silu
        );
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    float avg_ms_naive = ms / num_iters;
    std::cout << "GPU naive kernel time (" << num_iters << " iters): " << avg_ms_naive << " ms" << std::endl;
    std::cout << "Speedup (naive vs optimized): " << (avg_ms_naive / avg_ms) << "x" << std::endl;
    
    // Calculate bandwidth
    float bytes = total_elements * sizeof(float) * 2 +  // read x, write out
                  weight_size * sizeof(float) +          // read weight
                  (tc.use_bias ? bias_size * sizeof(float) : 0);  // read bias
    float bandwidth_gb = (bytes / 1e9) / (avg_ms / 1000.0f);
    std::cout << "Effective bandwidth: " << bandwidth_gb << " GB/s" << std::endl;
    
    // Copy results back to host
    std::vector<float> h_out_gpu_naive(total_elements);
    HIP_CHECK(hipMemcpy(h_out_gpu.data(), d_out, total_elements * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_out_gpu_naive.data(), d_out_naive, total_elements * sizeof(float), hipMemcpyDeviceToHost));
    
    // Accuracy verification - optimized kernel vs CPU
    float max_diff_opt = 0.0f;
    float avg_diff_opt = 0.0f;
    int error_count_opt = 0;
    const float threshold = 1e-4f;
    
    for (int i = 0; i < total_elements; ++i) {
        float diff = std::abs(h_out_gpu[i] - h_out_ref[i]);
        max_diff_opt = std::max(max_diff_opt, diff);
        avg_diff_opt += diff;
        if (diff > threshold) {
            error_count_opt++;
            if (error_count_opt <= 5) {
                int b = i / (tc.dim * tc.seqlen);
                int d = (i / tc.seqlen) % tc.dim;
                int t = i % tc.seqlen;
                std::cout << "Optimized kernel error at [" << b << "," << d << "," << t << "]: "
                         << "GPU=" << h_out_gpu[i] << ", CPU=" << h_out_ref[i]
                         << ", diff=" << diff << std::endl;
            }
        }
    }
    avg_diff_opt /= total_elements;
    
    // Accuracy verification - naive kernel vs CPU
    float max_diff_naive = 0.0f;
    float avg_diff_naive = 0.0f;
    int error_count_naive = 0;
    
    for (int i = 0; i < total_elements; ++i) {
        float diff = std::abs(h_out_gpu_naive[i] - h_out_ref[i]);
        max_diff_naive = std::max(max_diff_naive, diff);
        avg_diff_naive += diff;
        if (diff > threshold) {
            error_count_naive++;
        }
    }
    avg_diff_naive /= total_elements;
    
    // Compare optimized vs naive
    float max_diff_kernels = 0.0f;
    for (int i = 0; i < total_elements; ++i) {
        float diff = std::abs(h_out_gpu[i] - h_out_gpu_naive[i]);
        max_diff_kernels = std::max(max_diff_kernels, diff);
    }
    
    std::cout << "\n=== Accuracy Results ===" << std::endl;
    std::cout << "Optimized kernel vs CPU:" << std::endl;
    std::cout << "  Max absolute difference: " << max_diff_opt << std::endl;
    std::cout << "  Average absolute difference: " << avg_diff_opt << std::endl;
    std::cout << "  Elements with error > " << threshold << ": " << error_count_opt 
              << " (" << (100.0f * error_count_opt / total_elements) << "%)" << std::endl;
    std::cout << "  Status: " << (max_diff_opt < 1e-3f ? "PASS ✓" : "FAIL ✗") << std::endl;
    
    std::cout << "\nNaive kernel vs CPU:" << std::endl;
    std::cout << "  Max absolute difference: " << max_diff_naive << std::endl;
    std::cout << "  Average absolute difference: " << avg_diff_naive << std::endl;
    std::cout << "  Elements with error > " << threshold << ": " << error_count_naive 
              << " (" << (100.0f * error_count_naive / total_elements) << "%)" << std::endl;
    std::cout << "  Status: " << (max_diff_naive < 1e-3f ? "PASS ✓" : "FAIL ✗") << std::endl;
    
    std::cout << "\nOptimized vs Naive kernel:" << std::endl;
    std::cout << "  Max absolute difference: " << max_diff_kernels << std::endl;
    std::cout << "  Status: " << (max_diff_kernels < 1e-5f ? "IDENTICAL ✓" : "DIFFERENT") << std::endl;
    
    // Cleanup
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_weight));
    if (tc.use_bias) {
        HIP_CHECK(hipFree(d_bias));
    }
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_out_naive));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    bool passed = (max_diff_opt < 1e-3f && max_diff_naive < 1e-3f);
    std::cout << "\n>>> Test Case " << test_idx << ": " << (passed ? "PASSED ✓" : "FAILED ✗") << " <<<" << std::endl;
    
    return passed;
}

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Causal Conv1D Comprehensive Test Suite                 ║" << std::endl;
    std::cout << "║   CUDA-style Optimized Kernel for AMD ROCm               ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    
    // Define 5 different test cases with varying parameters
    TestCase test_cases[] = {
        // Test Case 1: Small scale - quick validation
        {
            "Small Scale (Quick Test)",
            2,      // batch
            32,     // dim (channels)
            512,    // seqlen
            4,      // width
            true,   // use_bias
            true    // use_silu
        },
        // Test Case 2: Medium scale - typical usage
        {
            "Medium Scale (Typical Workload)",
            4,      // batch
            64,     // dim
            2048,   // seqlen
            4,      // width
            true,   // use_bias
            true    // use_silu
        },
        // Test Case 3: Large scale - stress test
        {
            "Large Scale (Stress Test)",
            8,      // batch
            128,    // dim
            4096,   // seqlen
            4,      // width
            true,   // use_bias
            true    // use_silu
        },
        // Test Case 4: Long sequence - memory intensive
        {
            "Long Sequence (Memory Intensive)",
            2,      // batch
            64,     // dim
            8192,   // seqlen
            4,      // width
            true,   // use_bias
            false   // use_silu (test without activation)
        },
        // Test Case 5: Wide channels - computation intensive
        {
            "Wide Channels (Computation Intensive)",
            4,      // batch
            256,    // dim
            1024,   // seqlen
            4,      // width
            false,  // use_bias (test without bias)
            true    // use_silu
        }
    };
    
    const int num_tests = sizeof(test_cases) / sizeof(TestCase);
    
    // Track overall results
    int passed_count = 0;
    int failed_count = 0;
    
    // Run all test cases
    for (int i = 0; i < num_tests; ++i) {
        bool passed = run_test_case(test_cases[i], i + 1);
        if (passed) {
            passed_count++;
        } else {
            failed_count++;
        }
    }
    
    // Print summary
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    TEST SUMMARY                           ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "Total tests: " << num_tests << std::endl;
    std::cout << "Passed: " << passed_count << " ✓" << std::endl;
    std::cout << "Failed: " << failed_count << " ✗" << std::endl;
    std::cout << "Success rate: " << (100.0f * passed_count / num_tests) << "%" << std::endl;
    
    if (failed_count == 0) {
        std::cout << "\n🎉 All tests PASSED! 🎉" << std::endl;
    } else {
        std::cout << "\n⚠️  Some tests FAILED - please review the results above." << std::endl;
    }
    
    return (failed_count == 0) ? 0 : 1;
}