// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include "openvino/op/floor.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

//OutputVector translate_reshape(const NodeContext& context) {
//    // Translation is used by both aten::view and aten::reshape.
//    // Schema: aten::view(Tensor input, int[] shape) -> Tensor
//    // Schema: aten::reshape(Tensor input, int[] shape) -> Tensor
//    // For shape parameter, int[] is converted into single dimensional Tensor.
//    std::cout << "DEBUG - translate_reshape - num_inputs: " << context.get_input_size() << std::endl;
//    std::vector<int32_t> shape_vec;
//    for (size_t i = 0; i < context.get_input_size(); i++) {
//        if (context.get_input_type(i).is<type::List>()) {
//            std::cout << "\tDEBUG - translate_reshape - input - i: " << i << ", list" << std::endl;
//        } else {
//            std::cout << "\tDEBUG - translate_reshape - input - i: " << i << ", other" << std::endl;
//        }
//        if (i > 0) {
//            auto a = context.get_input_from_visible_context(i).get_node_shared_ptr();
//            if (std::dynamic_pointer_cast<ov::op::v0::Parameter>(a)) {
//                shape_vec.push_back(0);
//                std::cout << "\t\tDEBUG - translate_reshape - input - val[" << i << "]: Parameter" << std::endl;
//            } else if (std::dynamic_pointer_cast<ov::op::v0::Floor>(a)) {
//                shape_vec.push_back(0);
//                std::cout << "\t\tDEBUG - translate_reshape - input - val[" << i << "]: Floor" << std::endl;
//            } else if (std::dynamic_pointer_cast<ov::op::v1::Multiply>(a)) {
//                shape_vec.push_back(0);
//                std::cout << "\t\tDEBUG - translate_reshape - input - val[" << i << "]: Multiply" << std::endl;
//            } else {
//                auto val = context.const_input<int32_t>(i);
//                shape_vec.push_back(val);
//                std::cout << "\t\tDEBUG - translate_reshape - input - val[" << i << "]: " << val << std::endl;
//            }
//        }
//    }
//    std::cout << "DEBUG - translate_reshape - B - input_size: " << context.get_input_size()-1
//              << ", vec_size: " << shape_vec.size() << std::endl;
//    auto shape_const = ov::op::v0::Constant::create(element::i32, Shape{context.get_input_size()-1}, shape_vec);
//    //std::cout << "DEBUG - translate_reshape - C - shape1: " << context.get_input(0).get_shape()
//    //          << ", shape2: " << shape_const->get_shape() << std::endl;
//    //num_inputs_check(context, 2, 2);
//    //auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), context.get_input(1), false);
//    auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), shape_const, true);
//    std::cout << "DEBUG - translate_reshape - C" << std::endl;
//    return {context.mark_node(reshape)};
//};

OutputVector translate_reshape(const NodeContext& context) {
    // Translation is used by both aten::view and aten::reshape.
    // Schema: aten::view(Tensor input, int[] shape) -> Tensor
    // Schema: aten::reshape(Tensor input, int[] shape) -> Tensor
    // For shape parameter, int[] is converted into single dimensional Tensor.
    num_inputs_check(context, 2, 2);
    auto in1_const = context.const_input<Shape>(1);

    //if (in1_const.size() > 1 && (in1_const[0] == 17 || in1_const[0] == 48)) {
    //    in1_const[0] = 0;
    //    auto const_shape = context.mark_node(std::make_shared<ov::op::v0::Constant>(context.get_input(1).get_element_type(), context.get_input(1).get_shape(), in1_const));
    //    auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), const_shape, true);
    //    return {context.mark_node(reshape)};
    //}
    //if (in1_const.size() == 3 && (in1_const[0] == 32 && in1_const[1] == 144 && in1_const[2] == 128)) {
    //    in1_const[1] = -1;
    //    auto const_shape = context.mark_node(std::make_shared<ov::op::v0::Constant>(context.get_input(1).get_element_type(), context.get_input(1).get_shape(), in1_const));
    //    auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), const_shape, false);
    //    return {context.mark_node(reshape)};
    //}
    //if (in1_const.size() == 3 && (in1_const[0] == 32 && in1_const[1] == 128 && in1_const[2] == 144)) {
    //    in1_const[2] = -1;
    //    auto const_shape = context.mark_node(std::make_shared<ov::op::v0::Constant>(context.get_input(1).get_element_type(), context.get_input(1).get_shape(), in1_const));
    //    auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), const_shape, false);
    //    return {context.mark_node(reshape)};
    //}
    //if (in1_const.size() == 4 && (in1_const[0] == 1 && in1_const[1] == 32 && in1_const[2] == 1 && in1_const[3] == 144)) {
    //    in1_const[3] = -1;
    //    auto const_shape = context.mark_node(std::make_shared<ov::op::v0::Constant>(context.get_input(1).get_element_type(), context.get_input(1).get_shape(), in1_const));
    //    auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), const_shape, false);
    //    return {context.mark_node(reshape)};
    //}
    for (size_t i =0; i < in1_const.size(); i++) {
        if (in1_const[i] == 144) {
            in1_const[i] = -1;
            auto const_shape = context.mark_node(std::make_shared<ov::op::v0::Constant>(context.get_input(1).get_element_type(), context.get_input(1).get_shape(), in1_const));
            auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), const_shape, false);
            return {context.mark_node(reshape)};
        }
        if (in1_const[i] == 6) {
            in1_const[i] = -1;
            auto const_shape = context.mark_node(std::make_shared<ov::op::v0::Constant>(context.get_input(1).get_element_type(), context.get_input(1).get_shape(), in1_const));
            auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), const_shape, false);
            return {context.mark_node(reshape)};
        }
        if (in1_const[i] == 7) {
            in1_const[i] = -1;
            auto const_shape = context.mark_node(std::make_shared<ov::op::v0::Constant>(context.get_input(1).get_element_type(), context.get_input(1).get_shape(), in1_const));
            auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), const_shape, false);
            return {context.mark_node(reshape)};
        }
        if (in1_const[i] == 511) {
            in1_const[i] = -1;
            auto const_shape = context.mark_node(std::make_shared<ov::op::v0::Constant>(context.get_input(1).get_element_type(), context.get_input(1).get_shape(), in1_const));
            auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), const_shape, false);
            return {context.mark_node(reshape)};
        }
        if (in1_const[i] == 63) {
            in1_const[i] = -1;
            auto const_shape = context.mark_node(std::make_shared<ov::op::v0::Constant>(context.get_input(1).get_element_type(), context.get_input(1).get_shape(), in1_const));
            auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), const_shape, false);
            return {context.mark_node(reshape)};
        }
    }

    auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), context.get_input(1), false);
    return {context.mark_node(reshape)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
