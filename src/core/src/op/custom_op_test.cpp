// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/custom_op_test.hpp"

#include <algorithm>

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/custom_op_test.hpp"

namespace ov {
namespace op {
namespace custom_op_test {
namespace {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in, Tensor& out, const Shape& shape, const AxisSet& axes) {
        ov::reference::custom_op_test(in.data<const T>(), out.data<T>(), shape, axes);
        return true;
    }
};
}  // namespace
}  // namespace softmax

namespace v1 {
CustomOpTest::CustomOpTest(const Output<Node>& arg, const size_t axis) : Op({arg}), m_axis(axis) {
    constructor_validate_and_infer_types();
}

bool CustomOpTest::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_CustomOpTest_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void CustomOpTest::validate_and_infer_types() {
    OV_OP_SCOPE(v1_CustomOpTest_validate_and_infer_types);
    const auto& input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static())
        NODE_VALIDATION_CHECK(this,
                              m_axis < static_cast<size_t>(input_shape.rank().get_length()),
                              "Reduction axis (",
                              m_axis,
                              ") is out of bounds (argument shape: ",
                              input_shape,
                              ").");

    set_output_type(0, get_input_element_type(0), input_shape);
}

std::shared_ptr<Node> CustomOpTest::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_CustomOpTest_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<CustomOpTest>(new_args.at(0), m_axis);
}

bool CustomOpTest::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_CustomOpTest_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    const auto& input_shape = inputs[0].get_shape();
    outputs[0].set_shape(input_shape);
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v1_CustomOpTest_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, f64),
                                      custom_op_test::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      input_shape,
                                      AxisSet{m_axis});
}

bool CustomOpTest::has_evaluate() const {
    OV_OP_SCOPE(v1_CustomOpTest_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
        return true;
    default:
        return false;
    }
}
}  // namespace v1

}  // namespace op
}  // namespace ov
