// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "intel_gpu/primitives/reshape.hpp"

namespace ov {
namespace op {
namespace internal {
using SDPA = ov::intel_gpu::op::SDPA;
using IndirectSDPA = ov::intel_gpu::op::IndirectSDPA;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

constexpr size_t value_idx = cldnn::scaled_dot_product_attention::ScaledDotProductAttentionInputIdx::VALUE;
constexpr size_t mask_idx = cldnn::scaled_dot_product_attention::ScaledDotProductAttentionInputIdx::ATTN_MASK;
constexpr size_t scale_idx = cldnn::scaled_dot_product_attention::ScaledDotProductAttentionInputIdx::SCALE;
constexpr size_t sink_idx = cldnn::scaled_dot_product_attention::ScaledDotProductAttentionInputIdx::SINK;
constexpr size_t cnt_inputs_with_qkv = value_idx + 1;
constexpr size_t cnt_inputs_with_mask = mask_idx + 1;
constexpr size_t cnt_inputs_with_scale = scale_idx + 1;
constexpr size_t cnt_inputs_with_sink = sink_idx + 1;

static std::shared_ptr<ov::op::v0::Constant> GetScalarConstInput(const std::shared_ptr<ov::op::Op>& op, size_t idx) {
    std::shared_ptr<ov::op::v0::Constant> constOp = nullptr;
    if (op->get_input_size() > idx && !op->get_input_partial_shape(idx).is_dynamic() && ov::shape_size(op->get_input_shape(idx)) == 1) {
        constOp = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(idx));
    }
    return constOp;
}

static void ReshapeInput(ProgramBuilder& p, const std::shared_ptr<ov::op::Op>& op, std::vector<cldnn::input_info>& inputs) {
    if (!p.use_new_shape_infer()) {
        auto layer_name = layer_type_name_ID(op);
        auto output_pshape = op->get_output_partial_shape(0);
        auto output_rank = output_pshape.size() < 4 ? 4 : output_pshape.size();

        for (size_t idx = 0; idx < op->get_input_size(); ++idx) {
            if (op->get_input_partial_shape(idx).rank().get_length() < 4) {
                auto &input = inputs[idx];
                auto input_pshape = op->get_input_partial_shape(idx);
                auto input_rank = input_pshape.size();

                auto input_shape = op->get_input_shape(idx);
                input_shape.insert(input_shape.begin(), output_rank - input_rank, 1ul);

                auto target_input_shape = tensor_from_dims(input_shape);
                auto input_reshape_name = layer_name + "_input_" + std::to_string(idx) + "_reshape";
                auto input_reshape_prim = cldnn::reshape(input_reshape_name, input, target_input_shape);
                p.add_primitive(*op, input_reshape_prim);
                input.pid = input_reshape_name;
            }
        }
    }
}

static void GetNewOrder(ProgramBuilder&p, const std::shared_ptr<ov::op::internal::SDPA>& op, std::vector<std::vector<int64_t>>& transpose_orders) {
    transpose_orders[0] = op->get_input0_transpose_order();
    transpose_orders[1] = op->get_input1_transpose_order();
    transpose_orders[2] = op->get_input2_transpose_order();
    transpose_orders[3] = op->get_output_transpose_order();

    if (!p.use_new_shape_infer() && op->get_input_partial_shape(0).rank().get_length() < 4) {
        for (auto &order : transpose_orders) {
            for (auto &dim : order) ++dim;
            order.insert(order.begin(), 0);
        }
    }
}

// Read head_size / num_heads from an input's DECLARED ov-core PartialShape at op-conversion
// time, honouring the same `order` (transpose_order) convention SDPABase::get_jit_constants()
// uses at runtime (get_head_size/get_num_heads in sdpa_utils.hpp: head_size is the axis at
// order[order.size()-1], num_heads is at order[order.size()-3], or 1 for a 3D BLS input with no
// separate heads axis). Used as a static fallback there, which otherwise derives these from the
// LIVE per-call GPU-plugin layout -- a separate, lossier mechanism than ov-core's own shape
// inference that can report a dim as dynamic even when it is a genuine static per-model
// constant (e.g. downstream of a Slice whose *other* bound is a legitimate runtime value, such
// as a growing KV-cache length). -1 means "not statically known here either", so the fallback
// is simply not populated and behavior is unchanged from before this fix.
static void GetStaticHeadDims(const std::shared_ptr<ov::op::Op>& op,
                              size_t input_idx,
                              const std::vector<int64_t>& order,
                              int64_t& head_size,
                              int64_t& num_heads) {
    head_size = -1;
    num_heads = -1;
    if (op->get_input_size() <= input_idx) {
        return;
    }
    const auto pshape = op->get_input_partial_shape(input_idx);
    if (pshape.rank().is_dynamic()) {
        return;
    }
    const auto rank = pshape.size();
    if (rank < 3) {
        return;
    }
    const auto& axis_order = !order.empty() ? order : ov::op::internal::SDPA::default_order(rank);
    if (axis_order.size() != rank) {
        return;
    }
    const auto head_size_axis = static_cast<size_t>(axis_order[rank - 1]);
    if (pshape[head_size_axis].is_static()) {
        head_size = pshape[head_size_axis].get_length();
    }
    if (rank == 3) {
        num_heads = 1;
    } else {
        const auto num_heads_axis = static_cast<size_t>(axis_order[rank - 3]);
        if (pshape[num_heads_axis].is_static()) {
            num_heads = pshape[num_heads_axis].get_length();
        }
    }
}

static void CreateScaledDotProductAttentionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& op) {
    // if transpose fusion is disabled, this is used

    validate_inputs_count(op, {cnt_inputs_with_qkv, cnt_inputs_with_mask, cnt_inputs_with_scale, cnt_inputs_with_sink});
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);

    auto scalar_scale = GetScalarConstInput(op, scale_idx);
    auto scalar_attn_mask = GetScalarConstInput(op, mask_idx);

    ReshapeInput(p, op, inputs);

    bool is_causal = op->get_causal();
    auto order = ov::op::internal::SDPA::default_order(op->get_output_partial_shape(0).size());
    auto sdpa_prim = cldnn::scaled_dot_product_attention(layerName,
                                                         inputs,
                                                         is_causal,
                                                         -1,
                                                         order,
                                                         order,
                                                         order,
                                                         order);

    if (scalar_scale) {
        sdpa_prim.scale_val = scalar_scale->cast_vector<float>()[0];
    }

    if (scalar_attn_mask) {
        sdpa_prim.attn_mask_val = scalar_attn_mask->cast_vector<float>()[0];
    }

    int64_t unused_v_num_heads;
    GetStaticHeadDims(op, cldnn::scaled_dot_product_attention::ScaledDotProductAttentionInputIdx::QUERY, order, sdpa_prim.q_head_size, sdpa_prim.q_num_heads);
    GetStaticHeadDims(op, cldnn::scaled_dot_product_attention::ScaledDotProductAttentionInputIdx::KEY, order, sdpa_prim.k_head_size, sdpa_prim.k_num_heads);
    GetStaticHeadDims(op, cldnn::scaled_dot_product_attention::ScaledDotProductAttentionInputIdx::VALUE, order, sdpa_prim.v_head_size, unused_v_num_heads);

    p.add_primitive(*op, sdpa_prim);
}

static void CreateSDPAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::SDPA>& op) {
    validate_inputs_count(op, {cnt_inputs_with_qkv, cnt_inputs_with_mask, cnt_inputs_with_scale, cnt_inputs_with_sink});
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);

    auto scalar_scale = GetScalarConstInput(op, scale_idx);
    auto scalar_attn_mask = GetScalarConstInput(op, mask_idx);

    ReshapeInput(p, op, inputs);

    std::vector<std::vector<int64_t>> transpose_orders(4);
    GetNewOrder(p, op, transpose_orders);

    bool is_causal = op->get_causal();
    int64_t indirect_axis = -1;

    auto sdpa_prim = cldnn::scaled_dot_product_attention(layerName,
                                                         inputs,
                                                         is_causal,
                                                         indirect_axis,
                                                         transpose_orders[0],
                                                         transpose_orders[1],
                                                         transpose_orders[2],
                                                         transpose_orders[3]);
    if (scalar_scale) {
        sdpa_prim.scale_val = scalar_scale->cast_vector<float>()[0];
    }

    if (scalar_attn_mask) {
        sdpa_prim.attn_mask_val = scalar_attn_mask->cast_vector<float>()[0];
    }

    int64_t unused_v_num_heads;
    GetStaticHeadDims(op, cldnn::scaled_dot_product_attention::ScaledDotProductAttentionInputIdx::QUERY, transpose_orders[0], sdpa_prim.q_head_size, sdpa_prim.q_num_heads);
    GetStaticHeadDims(op, cldnn::scaled_dot_product_attention::ScaledDotProductAttentionInputIdx::KEY, transpose_orders[1], sdpa_prim.k_head_size, sdpa_prim.k_num_heads);
    GetStaticHeadDims(op, cldnn::scaled_dot_product_attention::ScaledDotProductAttentionInputIdx::VALUE, transpose_orders[2], sdpa_prim.v_head_size, unused_v_num_heads);

    p.add_primitive(*op, sdpa_prim);
}

static void CreateIndirectSDPAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::IndirectSDPA>& op) {
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);

    auto scalar_scale = GetScalarConstInput(op, scale_idx);
    auto scalar_attn_mask = GetScalarConstInput(op, mask_idx);

    ReshapeInput(p, op, inputs);

    std::vector<std::vector<int64_t>> transpose_orders(4);
    GetNewOrder(p, op, transpose_orders);

    bool is_causal = op->get_causal();
    const auto compression_inputs = op->get_compression_inputs_num();
    validate_inputs_count(op, {cnt_inputs_with_mask + compression_inputs,
                            cnt_inputs_with_scale + compression_inputs, cnt_inputs_with_sink + compression_inputs});

    int64_t indirect_axis = op->get_indirect_axis();
    auto sdpa_prim = cldnn::scaled_dot_product_attention(layerName,
                                                         inputs,
                                                         is_causal,
                                                         indirect_axis,
                                                         transpose_orders[0],
                                                         transpose_orders[1],
                                                         transpose_orders[2],
                                                         transpose_orders[3],
                                                         op->get_quantization_attrs(),
                                                         op->get_kv_compressed());
    if (scalar_scale) {
        sdpa_prim.scale_val = scalar_scale->cast_vector<float>()[0];
    }

    if (scalar_attn_mask) {
        sdpa_prim.attn_mask_val = scalar_attn_mask->cast_vector<float>()[0];
    }

    int64_t unused_v_num_heads;
    GetStaticHeadDims(op, cldnn::scaled_dot_product_attention::ScaledDotProductAttentionInputIdx::QUERY, transpose_orders[0], sdpa_prim.q_head_size, sdpa_prim.q_num_heads);
    GetStaticHeadDims(op, cldnn::scaled_dot_product_attention::ScaledDotProductAttentionInputIdx::KEY, transpose_orders[1], sdpa_prim.k_head_size, sdpa_prim.k_num_heads);
    GetStaticHeadDims(op, cldnn::scaled_dot_product_attention::ScaledDotProductAttentionInputIdx::VALUE, transpose_orders[2], sdpa_prim.v_head_size, unused_v_num_heads);

    p.add_primitive(*op, sdpa_prim);
}

REGISTER_FACTORY_IMPL(internal, SDPA);
REGISTER_FACTORY_IMPL(internal, IndirectSDPA);
REGISTER_FACTORY_IMPL(v13, ScaledDotProductAttention);

}  // namespace ov::intel_gpu
