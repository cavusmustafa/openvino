// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_squeeze(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    if (context.input_is_none(1)) {
        return {context.mark_node(std::make_shared<v0::Squeeze>(x))};
    }
    return {context.mark_node(std::make_shared<v0::Squeeze>(x, context.get_input(1)))};
};

OutputVector translate_unsqueeze(const NodeContext& context) {
    auto input_size = context.get_input_size();
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    auto out = context.mark_node(std::make_shared<v0::Unsqueeze>(x, y));
    return {out};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
