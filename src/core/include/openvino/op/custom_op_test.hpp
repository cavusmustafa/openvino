// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Softmax operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API CustomOpTest : public Op {
public:
    OPENVINO_OP("CustomOpTest", "opset1", op::Op);

    CustomOpTest() = default;
    /// \brief Constructs a softmax operation.
    ///
    /// \param arg Node that produces the first input tensor.<br>
    /// `[d0, ...]`
    /// \param axis The axis position (0-based) on which to calculate the softmax.
    ///
    /// Output `[d0, ...]`
    ///
    CustomOpTest(const Output<Node>& arg, const size_t axis = 1);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    size_t get_axis() const {
        return m_axis;
    }
    void set_axis(const size_t axis) {
        m_axis = axis;
    }
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    size_t m_axis{0};
};
}  // namespace v1

}  // namespace op
}  // namespace ov
