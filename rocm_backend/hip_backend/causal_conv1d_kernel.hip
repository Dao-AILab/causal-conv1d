#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>


// SiLU activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}


// Causal Conv1D kernel 
// Each block processes one (batch, channel) pair
// Threads within a block collaborate to process the entire sequence for that channel
template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int batch,
    int dim,
    int seqlen,
    int width,
    int x_batch_stride,
    int x_c_stride,
    int weight_c_stride,
    int weight_width_stride,
    int out_batch_stride,
    int out_c_stride,
    bool use_silu
) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kChunkSize = Ktraits::kChunkSize;
    using vec_t = typename Ktraits::vec_t;
    
    // Shared memory for block load/store and data exchange
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize);
    
    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    
    // Pointer to this block's input, weight, and output
    const float *x_ptr = x + batch_id * x_batch_stride + channel_id * x_c_stride;
    const float *weight_ptr = weight + channel_id * weight_c_stride;
    float *out_ptr = out + batch_id * out_batch_stride + channel_id * out_c_stride;
    
    float bias_val = (bias == nullptr) ? 0.0f : bias[channel_id];
    
    // Thread 0 initializes the boundary (previous chunk's last elements) to 0
    if (tidx == 0) {
        float zeros[kNElts] = {0.0f};
        smem_exchange[kNThreads - 1] = *reinterpret_cast<vec_t*>(zeros);
    }
    
    // Load weights into registers (each thread loads all weights for this channel)
    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) {
        weight_vals[i] = weight_ptr[i * weight_width_stride];
    }
    
    // Process sequence in chunks
    const int n_chunks = (seqlen + kChunkSize - 1) / kChunkSize;
    
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        float x_vals_load[2 * kNElts] = {0.0f};
        
        // Load current chunk using BlockLoad for coalesced access
        __syncthreads();
        int remaining = seqlen - chunk * kChunkSize;
        typename Ktraits::BlockLoadT(smem_load).Load(
            x_ptr, 
            reinterpret_cast<float(&)[kNElts]>(x_vals_load[kNElts]), 
            remaining
        );
        x_ptr += kChunkSize;
        
        __syncthreads();
        
        // Share data with neighboring threads for causal convolution
        // Thread i writes its data so thread i+1 can read it
        if (tidx < kNThreads - 1) {
            smem_exchange[tidx] = *reinterpret_cast<vec_t*>(&x_vals_load[kNElts]);
        }
        
        __syncthreads();
        
        // Thread i reads from thread i-1 (or thread kNThreads-1 if i==0)
        *reinterpret_cast<vec_t*>(x_vals_load) = smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
        
        __syncthreads();
        
        // Now thread kNThreads-1 can write its data for the next chunk
        if (tidx == kNThreads - 1) {
            smem_exchange[tidx] = *reinterpret_cast<vec_t*>(&x_vals_load[kNElts]);
        }
        
        // Perform causal convolution
        float out_vals[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            out_vals[i] = bias_val;
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                // Access pattern ensures causality: 
                // For output position i, we access inputs from positions [i-(kWidth-1), ..., i]
                out_vals[i] += weight_vals[w] * x_vals_load[kNElts + i - (kWidth - w - 1)];
            }
        }
        
        // Apply SiLU activation if requested
        if (use_silu) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                out_vals[i] = silu(out_vals[i]);
            }
        }
        
        // Store results using BlockStore for coalesced access
        __syncthreads();
        typename Ktraits::BlockStoreT(
        ).Store(
            out_ptr, 
            out_vals, 
            remaining
        );
        out_ptr += kChunkSize;
    }
}

// Simple naive kernel for comparison and fallback
__global__ void causal_conv1d_kernel_naive(
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * dim * seqlen;
    
    if (idx >= total_elements) return;
    
    int t = idx % seqlen;
    int d = (idx / seqlen) % dim;
    int b = idx / (dim * seqlen);
    
    float acc = 0.0f;
    
    for (int i = 0; i < width; ++i) {
        int input_t = t - width + 1 + i;
        if (input_t >= 0) {
            int input_idx = b * (dim * seqlen) + d * seqlen + input_t;
            acc += x[input_idx] * weight[d * width + i];
        }
    }
    
    if (bias != nullptr) {
        acc += bias[d];
    }
    
    if (use_silu) {
        acc = silu(acc);
    }
    
    out[idx] = acc;
}