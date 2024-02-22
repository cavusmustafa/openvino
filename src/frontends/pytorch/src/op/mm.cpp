// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"
#include "openvino/opsets/opset10.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_mm_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    auto dtype0 = context.get_input_type(0);
    auto dtype1 = context.get_input_type(1);
    std::cout << "DEBUG - translate_mm - lhs.shape: " << lhs.get_partial_shape() << ", rhs.shape: " << rhs.get_partial_shape() << std::endl;
    auto mm = context.mark_node(std::make_shared<opset10::MatMul>(lhs, rhs));
    return {mm};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
