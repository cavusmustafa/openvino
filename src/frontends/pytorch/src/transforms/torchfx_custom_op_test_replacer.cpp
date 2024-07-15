// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "torchfx_custom_op_test_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/custom_op_test.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;
using namespace ov::pass::pattern;

CustomOpTestReplacer::CustomOpTestReplacer() {
    auto add = wrap_type<v1::Add>({any_input(), any_input()});
    auto softmax = wrap_type<v8::Softmax>({add});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        auto softmax = m.get_match_root();
        const auto& pattern_map = m.get_pattern_value_map();
        auto add_node = pattern_map.at(add).get_node_shared_ptr();
        auto custom_op_test = std::make_shared<v1::CustomOpTest>(add_node->get_input_node_shared_ptr(0));
        copy_runtime_info_and_name(softmax, {custom_op_test});
        replace_node(softmax, custom_op_test);
        std::cout << "DEBUG - transformations - custom_op_test - replaced" << std::endl;
        return true;
    };

    auto m = std::make_shared<Matcher>(softmax, "ov::frontend::pytorch::pass::CustomOpTestReplacer");
    this->register_matcher(m, callback);
};


}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
