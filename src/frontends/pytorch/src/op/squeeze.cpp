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
    std::cout << "DEBUG - translate_session - convert_node - check - unsqueeze - begin" << std::endl;
    auto input_size = context.get_input_size();
    std::cout << "DEBUG - translate_session - convert_node - check - unsqueeze - input_size: " << input_size << std::endl;
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    std::cout << "DEBUG - translate_session - convert_node - check - unsqueeze - x: " << x.get_partial_shape().to_string() << std::endl;
    //std::cout << "DEBUG - translate_session - convert_node - check - unsqueeze - x.type_info: " << context.get_input_type(0).type_info().name() << std::endl;
    //if (context.get_input_type(0).is<std::vector<size_t>>()) {
    //    std::cout << "DEBUG - translate_session - convert_node - check - unsqueeze - x - true" << std::endl;
    //} else {
    //    std::cout << "DEBUG - translate_session - convert_node - check - unsqueeze - x - false" << std::endl;
    //}
    std::cout << "DEBUG - translate_session - convert_node - check - unsqueeze - y: " << y.get_partial_shape().to_string() << std::endl;
    //auto x_data = context.const_input<std::vector<size_t>>(0);
    //std::cout << "DEBUG - translate_session - convert_node - check - unsqueeze - x_data: " << x_data.size() << std::endl;
    auto out = context.mark_node(std::make_shared<v0::Unsqueeze>(x, y));
    std::cout << "DEBUG - translate_session - convert_node - check - unsqueeze - end" << std::endl;
    return {out};
    //{"aten.unsqueeze.default", op::translate_1to1_match_2_inputs<opset10::Unsqueeze>},
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
