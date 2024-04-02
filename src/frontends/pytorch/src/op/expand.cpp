// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector base_expand(const NodeContext& context, const Output<Node>& x, const Output<Node>& sizes) {
    auto shape = context.mark_node(std::make_shared<v0::Abs>(sizes));
    return {context.mark_node(std::make_shared<v3::Broadcast>(x, shape, BroadcastType::BIDIRECTIONAL))};
};
}  // namespace

OutputVector translate_expand(const NodeContext& context) {
    // aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto sizes = context.get_input(1);
    // TODO: figure out what implicit means
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(2) || context.const_input<bool>(2) == false,
                                "Unexpected value of implicit for expand operation");
    return base_expand(context, x, sizes);
};

OutputVector translate_expand_as(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(y, element::i32));
    return {context.mark_node(std::make_shared<v3::Broadcast>(x, shape, BroadcastType::BIDIRECTIONAL))};
};

//OutputVector translate_expand_fx(const NodeContext& context) {
//    // aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
//    std::cout << "DEBUG - translate_expand - A - num_inputs: " << context.get_input_size() << std::endl;
//    //num_inputs_check(context, 2, 3);
//    auto x = context.get_input(0);
//    // TODO: This is a temporary solution to optimize out Broadcast if the input and
//    // output shapes are same. This should be removed after a proper optimization is
//    // implemented.
//    std::cout << "DEBUG - translate_expand - D" << std::endl;
//    if (!(std::dynamic_pointer_cast<ov::op::v0::Parameter>(context.get_input_from_visible_context(1).get_node_shared_ptr())) && context.get_input_size() == 2) {
//        auto sizes_const = context.const_input<Shape>(1);
//        if (x.get_partial_shape().is_static() && x.get_shape() == sizes_const) {
//            std::cout << "DEBUG - translate_expand - E" << std::endl;
//            return {x};
//        }
//    }
//    std::cout << "DEBUG - translate_expand - F" << std::endl;
//    auto sizes = context.get_input(1);
//    std::cout << "DEBUG - translate_expand - G" << std::endl;
//    // TODO: figure out what implicit means
//    //PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(2) || context.const_input<bool>(2) == false,
//    //                            "Unexpected value of implicit for expand operation");
//    std::cout << "DEBUG - translate_expand - H" << std::endl;
//    std::vector<int32_t> shape_vec;
//    for (size_t i = 0; i < context.get_input_size(); i++) {
//        if (context.get_input_type(i).is<type::List>()) {
//            std::cout << "\tDEBUG - translate_expand - input - i: " << i << ", list" << std::endl;
//        } else {
//            std::cout << "\tDEBUG - translate_expand - input - i: " << i << ", other" << std::endl;
//        }
//        if (i > 0) {
//            auto a = context.get_input_from_visible_context(i).get_node_shared_ptr();
//            if (std::dynamic_pointer_cast<ov::op::v0::Parameter>(a)) {
//                shape_vec.push_back(-1);
//                std::cout << "\t\tDEBUG - translate_expand - input - val[" << i << "]: Parameter" << std::endl;
//            } else {
//                auto val = context.const_input<int32_t>(i);
//                shape_vec.push_back(val);
//                std::cout << "\t\tDEBUG - translate_expand - input - val[" << i << "]: " << val << std::endl;
//            }
//        }
//    }
//    std::cout << "DEBUG - translate_expand - I - input_size: " << context.get_input_size()-1
//              << ", vec_size: " << shape_vec.size() << std::endl;
//    auto shape_const = ov::op::v0::Constant::create(element::i32, Shape{context.get_input_size()-1}, shape_vec);
//    std::cout << "DEBUG - translate_expand - J" << std::endl;
//    return base_expand(context, x, shape_const);
//    //return base_expand(context, x, sizes);
//};

OutputVector translate_expand_fx(const NodeContext& context) {
    // aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    std::cout << "DEBUG - translate_expand - base_expand - broadcast - input_shape: " << x.get_partial_shape() << std::endl;
    // TODO: This is a temporary solution to optimize out Broadcast if the input and
    // output shapes are same. This should be removed after a proper optimization is
    // implemented.
    auto sizes_const = context.const_input<Shape>(1);
    std::cout << "DEBUG - translate_expand - base_expand - broadcast - target_shape: " << sizes_const << std::endl;
    //if (sizes_const.size() == 4 && sizes_const[0] == 1 && sizes_const[1] == 1 && sizes_const[2] == 1 && sizes_const[3] == 144) {
    //    std::cout << "\tDEBUG - translate_expand - base_expand - broadcast - detected" << std::endl;
    //    return {x};
    //}
    //if (sizes_const.size() == 4 && sizes_const[0] == 1 && sizes_const[1] == 32 && sizes_const[2] == 144 && sizes_const[3] == 128) {
    //    std::cout << "\tDEBUG - translate_expand - base_expand - broadcast - detected" << std::endl;
    //    return {x};
    //}
    //if (sizes_const.size() == 4 && sizes_const[0] == 1 && sizes_const[1] == 32 && sizes_const[2] == 128 && sizes_const[3] == 144) {
    //    std::cout << "\tDEBUG - translate_expand - base_expand - broadcast - detected" << std::endl;
    //    return {x};
    //}
    for (size_t i =0; i < sizes_const.size(); i++) {
        if (sizes_const[i] == 144) {
            return {x};
        }
    }
    if (x.get_partial_shape().is_static() && x.get_shape() == sizes_const) {
        return {x};
    }
    auto sizes = context.get_input(1);
    // TODO: figure out what implicit means
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(2) || context.const_input<bool>(2) == false,
                                "Unexpected value of implicit for expand operation");
    return base_expand(context, x, sizes);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov