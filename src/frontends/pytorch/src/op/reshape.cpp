// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_reshape(const NodeContext& context) {
    // Translation is used by both aten::view and aten::reshape.
    // Schema: aten::view(Tensor input, int[] shape) -> Tensor
    // Schema: aten::reshape(Tensor input, int[] shape) -> Tensor
    // For shape parameter, int[] is converted into single dimensional Tensor.
    num_inputs_check(context, 2, 2);
    auto in1_const = context.const_input<Shape>(1);
    //std::string in1_str = "[";
    //for (size_t i=0; i<in1_const.size(); i++)
    //    in1_str += in1_const[i] + ",";
    //in1_str += "]";
    //std::cout << "DEBUG - translate_reshape - input0.shape: " << context.get_input(0).get_partial_shape().to_string()
    //          << ", input1.shape: " << context.get_input(1).get_partial_shape().to_string()
    //          << ", input1.value: " << in1_const << std::endl;
    //if (in1_const.size() == 4 && (in1_const[0] == 18)) {
    //    in1_const[0] = 0;
    //    auto const_shape = context.mark_node(std::make_shared<ov::op::v0::Constant>(context.get_input(1).get_element_type(), context.get_input(1).get_shape(), in1_const));
    //    auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), const_shape, true);
    //    return {context.mark_node(reshape)};
    //}
    //if (in1_const.size() == 3 && (in1_const[1] == 18)) {
    //    in1_const[1] = 0;
    //    auto const_shape = context.mark_node(std::make_shared<ov::op::v0::Constant>(context.get_input(1).get_element_type(), context.get_input(1).get_shape(), in1_const));
    //    auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), const_shape, true);
    //    return {context.mark_node(reshape)};
    //}
    //if (in1_const.size() > 1 && (in1_const[0] == 18 || in1_const[0] == 19)) {
    if (in1_const.size() > 1 && (in1_const[0] == 17 || in1_const[0] == 48)) {
        in1_const[0] = 0;
        auto const_shape = context.mark_node(std::make_shared<ov::op::v0::Constant>(context.get_input(1).get_element_type(), context.get_input(1).get_shape(), in1_const));
        auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), const_shape, true);
        return {context.mark_node(reshape)};
    }
    auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), context.get_input(1), false);
    return {context.mark_node(reshape)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
