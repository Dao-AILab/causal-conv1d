/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <Python.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include "causal_conv1d.h"

using torch::stable::Tensor;
using torch::headeronly::ScalarType;
using torch::headeronly::IntHeaderOnlyArrayRef;

namespace {

int64_t last_dim(const Tensor &t) { return t.dim() - 1; }

cudaStream_t current_cuda_stream(const Tensor &t) {
    auto device_index = static_cast<int32_t>(t.get_device_index());
#if defined(TORCH_VERSION_2_13_0) && TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0
    void* stream_ptr =
        torch::stable::accelerator::getCurrentStream(device_index).nativeHandle();
#else
    // Torch < 2.13 has no Stream::nativeHandle(); fall back to the AOTI shim.
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
#endif
    return static_cast<cudaStream_t>(stream_ptr);
}

void check_shape(const Tensor &t, std::initializer_list<int64_t> shape, const char* name) {
    STD_TORCH_CHECK(t.dim() == static_cast<int64_t>(shape.size()),
                    name, " must have ", shape.size(), " dimensions");
    int64_t i = 0;
    for (int64_t s : shape) {
        STD_TORCH_CHECK(t.size(i) == s, name, " has wrong size at dim ", i);
        ++i;
    }
}

}  // namespace

#define CHECK_SHAPE(x, ...) check_shape(x, {__VA_ARGS__}, #x)

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == ScalarType::Half) {                                                \
        using input_t = torch::headeronly::Half;                                    \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == ScalarType::BFloat16) {                                     \
        using input_t = torch::headeronly::BFloat16;                                \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == ScalarType::Float)  {                                       \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        STD_TORCH_CHECK(false, #NAME, " not implemented for this input type");      \
    }

#define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)                     \
    if (WTYPE == ScalarType::Half) {                                                 \
        using weight_t = torch::headeronly::Half;                                    \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == ScalarType::BFloat16) {                                      \
        using weight_t = torch::headeronly::BFloat16;                                \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == ScalarType::Float)  {                                        \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        STD_TORCH_CHECK(false, #NAME, " not implemented for this weight type");      \
    }

template<typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);
template <typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void causal_conv1d_bwd_cuda(ConvParamsBwd &params, cudaStream_t stream);
template<typename input_t, typename weight_t>
void causal_conv1d_channellast_bwd_cuda(ConvParamsBwd &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase &params, cudaStream_t stream);

void set_conv_params_fwd(ConvParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t seqlen,
                         const size_t width,
                         // device pointers
                         const Tensor &x,
                         const Tensor &weight,
                         const Tensor &out,
                         void* bias_ptr,
                         bool silu_activation) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;

    params.silu_activation = silu_activation;

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.weight_ptr = weight.data_ptr();
    params.bias_ptr = bias_ptr;
    params.out_ptr = out.data_ptr();
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.x_c_stride = x.stride(1);
    params.x_l_stride = x.stride(last_dim(x));
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
    params.out_batch_stride = out.stride(0);
    params.out_c_stride = out.stride(1);
    params.out_l_stride = out.stride(last_dim(out));
}


void set_conv_params_bwd(ConvParamsBwd &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t seqlen,
                         const size_t width,
                         // device pointers
                         const Tensor &x,
                         const Tensor &weight,
                         void* bias_ptr,
                         const Tensor &dout,
                         const Tensor &dx,
                         const Tensor &dweight,
                         void* dbias_ptr,
                         bool silu_activation) {
    // Pass in "dout" instead of "out", we're not gonna use "out" at all.
    set_conv_params_fwd(params, batch, dim, seqlen, width,
                        x, weight, dout, bias_ptr, silu_activation);

    // Set the pointers and strides.
    params.dout_ptr = dout.data_ptr();
    params.dx_ptr = dx.data_ptr();
    params.dweight_ptr = dweight.data_ptr();
    params.dbias_ptr = dbias_ptr;
    // All stride are in elements, not bytes.
    params.dout_batch_stride = dout.stride(0);
    params.dout_c_stride = dout.stride(1);
    params.dout_l_stride = dout.stride(2);
    params.dweight_c_stride = dweight.stride(0);
    params.dweight_width_stride = dweight.stride(1);
    params.dx_batch_stride = dx.stride(0);
    params.dx_c_stride = dx.stride(1);
    params.dx_l_stride = dx.stride(2);
}

void
causal_conv1d_fwd(Tensor x,
                  Tensor weight,
                  std::optional<Tensor> bias_,
                  std::optional<Tensor> seq_idx_,
                  std::optional<Tensor> initial_states_,
                  Tensor out,
                  std::optional<Tensor> final_states_out_,
                  bool silu_activation) {
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    STD_TORCH_CHECK(input_type == ScalarType::Float || input_type == ScalarType::Half || input_type == ScalarType::BFloat16);
    STD_TORCH_CHECK(weight_type == ScalarType::Float || weight_type == ScalarType::Half || weight_type == ScalarType::BFloat16);

    STD_TORCH_CHECK(x.is_cuda());
    STD_TORCH_CHECK(weight.is_cuda());

    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int width = weight.size(last_dim(weight));

    CHECK_SHAPE(x, batch_size, dim, seqlen);
    CHECK_SHAPE(weight, dim, width);

    STD_TORCH_CHECK(x.stride(2) == 1 || x.stride(1) == 1);
    const bool is_channel_last = x.stride(1) == 1 && x.stride(2) > 1;

    if (is_channel_last) {
        STD_TORCH_CHECK(dim % 8 == 0, "causal_conv1d only supports channel dimension divisible by 8 for now");
        STD_TORCH_CHECK(x.stride(2) % 8 == 0 && x.stride(0) % 8 == 0, "causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8");
    }
    STD_TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        STD_TORCH_CHECK(bias.scalar_type() == weight_type);
        STD_TORCH_CHECK(bias.is_cuda());
        STD_TORCH_CHECK(bias.stride(last_dim(bias)) == 1);
        CHECK_SHAPE(bias, dim);
    }

    if (seq_idx_.has_value()) {
        STD_TORCH_CHECK(is_channel_last, "seq_idx is only supported for channel last layout");
        auto seq_idx = seq_idx_.value();
        STD_TORCH_CHECK(seq_idx.scalar_type() == ScalarType::Int);
        STD_TORCH_CHECK(seq_idx.is_cuda());
        STD_TORCH_CHECK(seq_idx.is_contiguous());
        CHECK_SHAPE(seq_idx, batch_size, seqlen);
    }

    ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out,
                        bias_.has_value() ? bias_.value().data_ptr() : nullptr,
                        silu_activation);

    if (seq_idx_.has_value()) {
        params.seq_idx_ptr = seq_idx_.value().data_ptr();
    } else {
        params.seq_idx_ptr = nullptr;
    }

    if (initial_states_.has_value()) {
        STD_TORCH_CHECK(is_channel_last, "initial_states is only supported for channel last layout");
        auto initial_states = initial_states_.value();
        STD_TORCH_CHECK(initial_states.scalar_type() == input_type);
        STD_TORCH_CHECK(initial_states.is_cuda());
        CHECK_SHAPE(initial_states, batch_size, dim, width - 1);
        STD_TORCH_CHECK(initial_states.stride(1) == 1);
        params.initial_states_ptr = initial_states.data_ptr();
        params.initial_states_batch_stride = initial_states.stride(0);
        params.initial_states_c_stride = initial_states.stride(1);
        params.initial_states_l_stride = initial_states.stride(2);
    } else {
        params.initial_states_ptr = nullptr;
    }

    if (final_states_out_.has_value()) {
        STD_TORCH_CHECK(is_channel_last, "final_states is only supported for channel last layout");
        auto final_states = final_states_out_.value();
        STD_TORCH_CHECK(final_states.scalar_type() == input_type);
        STD_TORCH_CHECK(final_states.is_cuda());
        CHECK_SHAPE(final_states, batch_size, dim, width - 1);
        STD_TORCH_CHECK(final_states.stride(1) == 1);
        params.final_states_ptr = final_states.data_ptr();
        params.final_states_batch_stride = final_states.stride(0);
        params.final_states_c_stride = final_states.stride(1);
        params.final_states_l_stride = final_states.stride(2);
    } else {
        params.final_states_ptr = nullptr;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    torch::stable::accelerator::DeviceGuard device_guard(x.get_device_index());
    auto stream = current_cuda_stream(x);
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_fwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(weight.scalar_type(), "causal_conv1d_fwd", [&] {
            if (!is_channel_last) {
                causal_conv1d_fwd_cuda<input_t, weight_t>(params, stream);
            } else {
                causal_conv1d_channellast_fwd_cuda<input_t, weight_t>(params, stream);
            }
        });
    });
}

void
causal_conv1d_bwd(Tensor x,
                  Tensor weight,
                  std::optional<Tensor> bias_,
                  Tensor dout,
                  std::optional<Tensor> seq_idx_,
                  std::optional<Tensor> initial_states_,
                  std::optional<Tensor> dfinal_states_,
                  Tensor dx,
                  Tensor dweight,
                  std::optional<Tensor> dbias_,
                  std::optional<Tensor> dinitial_states_,
                  bool silu_activation,
                  bool deterministic) {
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    STD_TORCH_CHECK(input_type == ScalarType::Float || input_type == ScalarType::Half || input_type == ScalarType::BFloat16);
    STD_TORCH_CHECK(weight_type == ScalarType::Float || weight_type == ScalarType::Half || weight_type == ScalarType::BFloat16);

    STD_TORCH_CHECK(x.is_cuda());
    STD_TORCH_CHECK(weight.is_cuda());
    STD_TORCH_CHECK(dout.is_cuda());
    STD_TORCH_CHECK(bias_.has_value() == dbias_.has_value());

    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int width = weight.size(last_dim(weight));

    STD_TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    CHECK_SHAPE(x, batch_size, dim, seqlen);
    CHECK_SHAPE(weight, dim, width);
    CHECK_SHAPE(dout, batch_size, dim, seqlen);

    STD_TORCH_CHECK(x.stride(2) == 1 || x.stride(1) == 1);
    const bool is_channel_last = x.stride(1) == 1 && x.stride(2) > 1;
    if (!is_channel_last && dout.stride(2) != 1) { dout = torch::stable::contiguous(dout); }
    if (is_channel_last && dout.stride(1) != 1) {
        dout = torch::stable::contiguous(torch::stable::transpose(dout, -1, -2));
        dout = torch::stable::transpose(dout, -1, -2);
    }

    if (is_channel_last) {
        STD_TORCH_CHECK(dim % 8 == 0, "causal_conv1d only supports channel dimension divisible by 8 for now");
        STD_TORCH_CHECK(x.stride(2) % 8 == 0 && x.stride(0) % 8 == 0, "causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8");
        STD_TORCH_CHECK(dout.stride(2) % 8 == 0 && dout.stride(0) % 8 == 0, "causal_conv1d with channel last layout requires strides (dout.stride(0) and dout.stride(2)) to be multiples of 8");
    }

    if (bias_.has_value()) {
        auto bias = bias_.value();
        STD_TORCH_CHECK(bias.scalar_type() == weight_type);
        STD_TORCH_CHECK(bias.is_cuda());
        STD_TORCH_CHECK(bias.stride(last_dim(bias)) == 1);
        CHECK_SHAPE(bias, dim);
    }

    if (seq_idx_.has_value()) {
        STD_TORCH_CHECK(is_channel_last, "seq_idx only supported for channel last layout");
        auto seq_idx = seq_idx_.value();
        STD_TORCH_CHECK(seq_idx.scalar_type() == ScalarType::Int);
        STD_TORCH_CHECK(seq_idx.is_cuda());
        STD_TORCH_CHECK(seq_idx.is_contiguous());
        CHECK_SHAPE(seq_idx, batch_size, seqlen);
    }

    STD_TORCH_CHECK(dx.scalar_type() == input_type);
    STD_TORCH_CHECK(dx.is_cuda());
    CHECK_SHAPE(dx, batch_size, dim, seqlen);
    if (!is_channel_last) { STD_TORCH_CHECK(dx.stride(2) == 1); }
    if (is_channel_last) { STD_TORCH_CHECK(dx.stride(1) == 1); }

    // Otherwise the kernel will be launched from cuda:0 device
    torch::stable::accelerator::DeviceGuard device_guard(x.get_device_index());

    ConvParamsBwd params;
    set_conv_params_bwd(params, batch_size, dim, seqlen, width,
                        x, weight, bias_.has_value() ? bias_.value().data_ptr() : nullptr,
                        dout, dx, dweight, bias_.has_value() ? dbias_.value().data_ptr() : nullptr,
                        silu_activation);

    if (seq_idx_.has_value()) {
        params.seq_idx_ptr = seq_idx_.value().data_ptr();
    } else {
        params.seq_idx_ptr = nullptr;
    }

    if (initial_states_.has_value()) {
        STD_TORCH_CHECK(is_channel_last, "initial_states is only supported for channel last layout");
        auto initial_states = initial_states_.value();
        STD_TORCH_CHECK(initial_states.scalar_type() == input_type);
        STD_TORCH_CHECK(initial_states.is_cuda());
        CHECK_SHAPE(initial_states, batch_size, dim, width - 1);
        STD_TORCH_CHECK(initial_states.stride(1) == 1);
        params.initial_states_ptr = initial_states.data_ptr();
        params.initial_states_batch_stride = initial_states.stride(0);
        params.initial_states_c_stride = initial_states.stride(1);
        params.initial_states_l_stride = initial_states.stride(2);
    } else {
        params.initial_states_ptr = nullptr;
    }

    if (dfinal_states_.has_value()) {
        STD_TORCH_CHECK(is_channel_last, "dfinal_states is only supported for channel last layout");
        auto dfinal_states = dfinal_states_.value();
        STD_TORCH_CHECK(dfinal_states.scalar_type() == input_type);
        STD_TORCH_CHECK(dfinal_states.is_cuda());
        CHECK_SHAPE(dfinal_states, batch_size, dim, width - 1);
        params.dfinal_states_ptr = dfinal_states.data_ptr();
        params.dfinal_states_batch_stride = dfinal_states.stride(0);
        params.dfinal_states_c_stride = dfinal_states.stride(1);
        params.dfinal_states_l_stride = dfinal_states.stride(2);
    } else {
        params.dfinal_states_ptr = nullptr;
    }

    if (dinitial_states_.has_value()) {
        Tensor dinitial_states = dinitial_states_.value();
        STD_TORCH_CHECK(dinitial_states.stride(1) == 1);
        params.dinitial_states_ptr = dinitial_states.data_ptr();
        params.dinitial_states_batch_stride = dinitial_states.stride(0);
        params.dinitial_states_c_stride = dinitial_states.stride(1);
        params.dinitial_states_l_stride = dinitial_states.stride(2);
    } else {
        params.dinitial_states_ptr = nullptr;
    }

    params.deterministic = deterministic;

    Tensor dweight_workspace;
    Tensor dbias_workspace;

    if (deterministic) {
        if (!is_channel_last) {
            std::vector<int64_t> ws_shape = {batch_size, dim, width};
            dweight_workspace = torch::stable::new_zeros(x, IntHeaderOnlyArrayRef(ws_shape), ScalarType::Float);
            params.dweight_workspace_ptr = dweight_workspace.data_ptr();
            params.dweight_workspace_batch_stride = dweight_workspace.stride(0);
            params.dweight_workspace_dim_stride = dweight_workspace.stride(1);

            if (dbias_.has_value()) {
                std::vector<int64_t> db_shape = {batch_size, dim};
                dbias_workspace = torch::stable::new_zeros(x, IntHeaderOnlyArrayRef(db_shape), ScalarType::Float);
                params.dbias_workspace_ptr = dbias_workspace.data_ptr();
                params.dbias_workspace_batch_stride = dbias_workspace.stride(0);
            } else {
                params.dbias_workspace_ptr = nullptr;
                params.dbias_workspace_batch_stride = 0;
            }
        } else {
            const int kChunkSizeL = seqlen <= 128 ? 64 : 128;
            const int n_chunks_L = (seqlen + kChunkSizeL - 1) / kChunkSizeL;

            std::vector<int64_t> ws_shape = {batch_size, n_chunks_L, dim, width};
            dweight_workspace = torch::stable::new_zeros(x, IntHeaderOnlyArrayRef(ws_shape), ScalarType::Float);
            params.dweight_workspace_ptr = dweight_workspace.data_ptr();
            params.dweight_workspace_batch_stride = dweight_workspace.stride(0);
            params.dweight_workspace_dim_stride = dweight_workspace.stride(2);

            if (dbias_.has_value()) {
                std::vector<int64_t> db_shape = {batch_size, n_chunks_L, dim};
                dbias_workspace = torch::stable::new_zeros(x, IntHeaderOnlyArrayRef(db_shape), ScalarType::Float);
                params.dbias_workspace_ptr = dbias_workspace.data_ptr();
                params.dbias_workspace_batch_stride = dbias_workspace.stride(0);
            } else {
                params.dbias_workspace_ptr = nullptr;
                params.dbias_workspace_batch_stride = 0;
            }
        }
    } else {
        params.dweight_workspace_ptr = nullptr;
        params.dbias_workspace_ptr = nullptr;
        params.dweight_workspace_batch_stride = 0;
        params.dweight_workspace_dim_stride = 0;
        params.dbias_workspace_batch_stride = 0;
    }

    auto stream = current_cuda_stream(x);
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_bwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(weight.scalar_type(), "causal_conv1d_bwd", [&] {
            if (!is_channel_last) {
                causal_conv1d_bwd_cuda<input_t, weight_t>(params, stream);
            } else {
                causal_conv1d_channellast_bwd_cuda<input_t, weight_t>(params, stream);
            }
        });
    });

    if (deterministic) {
        // dst += src without add_ (absent from the stable ABI): numerically
        // identical, just a small extra alloc+copy on this tiny opt-in path.
        std::vector<int64_t> sum_dims = is_channel_last ? std::vector<int64_t>{0, 1}
                                                        : std::vector<int64_t>{0};
        Tensor dw_sum = torch::stable::sum(dweight_workspace, IntHeaderOnlyArrayRef(sum_dims));
        torch::stable::copy_(dweight, torch::stable::subtract(dweight, dw_sum, -1.0));
        if (dbias_.has_value()) {
            Tensor dbias = dbias_.value();
            Tensor db_sum = torch::stable::sum(dbias_workspace, IntHeaderOnlyArrayRef(sum_dims));
            torch::stable::copy_(dbias, torch::stable::subtract(dbias, db_sum, -1.0));
        }
    }
}

void
causal_conv1d_update(Tensor x,
                     Tensor conv_state,
                     Tensor weight,
                     std::optional<Tensor> bias_,
                     Tensor out,
                     bool silu_activation,
                     std::optional<Tensor> cache_seqlens_,
                     std::optional<Tensor> conv_state_indices_
                     ) {
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    STD_TORCH_CHECK(input_type == ScalarType::Float || input_type == ScalarType::Half || input_type == ScalarType::BFloat16);
    STD_TORCH_CHECK(weight_type == ScalarType::Float || weight_type == ScalarType::Half || weight_type == ScalarType::BFloat16);
    STD_TORCH_CHECK(conv_state.scalar_type() == input_type);

    STD_TORCH_CHECK(x.is_cuda());
    STD_TORCH_CHECK(conv_state.is_cuda());
    STD_TORCH_CHECK(weight.is_cuda());

    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int width = weight.size(last_dim(weight));
    const int conv_state_len = conv_state.size(2);
    STD_TORCH_CHECK(conv_state_len >= width - 1);

    CHECK_SHAPE(x, batch_size, dim, seqlen);
    CHECK_SHAPE(weight, dim, width);

    STD_TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        STD_TORCH_CHECK(bias.scalar_type() == weight_type);
        STD_TORCH_CHECK(bias.is_cuda());
        STD_TORCH_CHECK(bias.stride(last_dim(bias)) == 1);
        CHECK_SHAPE(bias, dim);
    }

    ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out,
                        bias_.has_value() ? bias_.value().data_ptr() : nullptr,
                        silu_activation);
    params.conv_state_ptr = conv_state.data_ptr();
    params.conv_state_len = conv_state_len;
    // All stride are in elements, not bytes.
    params.conv_state_batch_stride = conv_state.stride(0);
    params.conv_state_c_stride = conv_state.stride(1);
    params.conv_state_l_stride = conv_state.stride(2);

    if (conv_state_indices_.has_value()) {
        auto conv_state_indices = conv_state_indices_.value();
        STD_TORCH_CHECK(conv_state_indices.scalar_type() == ScalarType::Int);
        STD_TORCH_CHECK(conv_state_indices.is_cuda());
        STD_TORCH_CHECK(conv_state_indices.stride(0) == 1);
        CHECK_SHAPE(conv_state_indices, batch_size);

        int conv_state_entries = conv_state.size(0);
        CHECK_SHAPE(conv_state, conv_state_entries, dim, conv_state_len);

        params.conv_state_indices_ptr = static_cast<int32_t*>(conv_state_indices.data_ptr());
    } else {
        CHECK_SHAPE(conv_state, batch_size, dim, conv_state_len);
        params.conv_state_indices_ptr = nullptr;
    }

    if (cache_seqlens_.has_value()) {
        auto cache_seqlens = cache_seqlens_.value();
        STD_TORCH_CHECK(cache_seqlens.scalar_type() == ScalarType::Int);
        STD_TORCH_CHECK(cache_seqlens.is_cuda());
        STD_TORCH_CHECK(cache_seqlens.stride(last_dim(cache_seqlens)) == 1);
        CHECK_SHAPE(cache_seqlens, batch_size);
        params.cache_seqlens = static_cast<int32_t*>(cache_seqlens.data_ptr());
    } else {
        params.cache_seqlens = nullptr;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    torch::stable::accelerator::DeviceGuard device_guard(x.get_device_index());
    auto stream = current_cuda_stream(x);
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_update", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(weight.scalar_type(), "causal_conv1d_update", [&] {
            causal_conv1d_update_cuda<input_t, weight_t>(params, stream);
        });
    });
}

// Register `torch.ops.causal_conv1d` on load.
STABLE_TORCH_LIBRARY(causal_conv1d, m) {
    m.def("causal_conv1d_fwd(Tensor x, Tensor weight, Tensor? bias, Tensor? seq_idx, "
          "Tensor? initial_states, Tensor(a!) out, Tensor(b!)? final_states_out, "
          "bool silu_activation) -> ()");
    m.def("causal_conv1d_bwd(Tensor x, Tensor weight, Tensor? bias, Tensor dout, "
          "Tensor? seq_idx, Tensor? initial_states, Tensor? dfinal_states, "
          "Tensor(a!) dx, Tensor(b!) dweight, Tensor(c!)? dbias, "
          "Tensor(d!)? dinitial_states, bool silu_activation, bool deterministic) -> ()");
    m.def("causal_conv1d_update(Tensor x, Tensor(a!) conv_state, Tensor weight, "
          "Tensor? bias, Tensor(b!) out, bool silu_activation, Tensor? cache_seqlens, "
          "Tensor? conv_state_indices) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(causal_conv1d, CUDA, m) {
    m.impl("causal_conv1d_fwd", TORCH_BOX(&causal_conv1d_fwd));
    m.impl("causal_conv1d_bwd", TORCH_BOX(&causal_conv1d_bwd));
    m.impl("causal_conv1d_update", TORCH_BOX(&causal_conv1d_update));
}

extern "C" {

static PyModuleDef_Slot causal_conv1d_slots[] = {
#ifdef Py_GIL_DISABLED
    {Py_mod_gil, Py_MOD_GIL_NOT_USED}, // safe, extension holds no Python state.
#endif
    {0, nullptr},
};

static PyModuleDef causal_conv1d_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_C",
    .m_size = 0,
    .m_slots = causal_conv1d_slots,
};

PyMODINIT_FUNC PyInit__C(void) {
    return PyModuleDef_Init(&causal_conv1d_module);
}

}  // extern "C"
