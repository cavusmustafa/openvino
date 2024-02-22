// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_stack_list_construct_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/bitwise_and.hpp"
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

AtenStackListConstructReplacer::AtenStackListConstructReplacer() {
    auto list_construct = wrap_type<ov::op::util::FrameworkNode>();
    auto axis = wrap_type<v0::Constant>();

    // We search for a pattern: ListConstruct -> aten::stack <- Constant
    auto stack = wrap_type<ov::op::util::FrameworkNode>({list_construct, axis});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        auto stack = cast_fw_node(m.get_match_root(), "aten::stack");
        if (!stack) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        auto input_node = pattern_map.at(list_construct).get_node_shared_ptr();
        auto axis_node = pattern_map.at(axis).get_node_shared_ptr();
        auto axis_const = std::dynamic_pointer_cast<v0::Constant>(axis_node);
        auto axis = axis_const->cast_vector<int64_t>();
        if (axis.size() != 1) {
            add_exception_to_fw_node(stack, "aten::stack has multiple axes, only one is supported.");
            return false;
        }
        // Check if ListConstruct is an input
        if (auto list_construct_node = cast_fw_node(input_node, "prim::ListConstruct")) {
            const auto& list_inputs = list_construct_node->input_values();
            std::shared_ptr<Node> node;
            if (auto compression = u4_compression_stack(list_inputs, axis[0])) {
                node = compression;
            } else {
                OutputVector node_vector;
                auto zero = v0::Constant::create(element::i32, Shape{}, {0});
                // Iterate over values in ListConstruct
                for (const auto& list_input : list_inputs) {
                    auto node = concat_list_construct(list_input);
                    auto unsqueezed_node = std::make_shared<v0::Unsqueeze>(node, axis_const);
                    node_vector.push_back(unsqueezed_node);
                }
                // Concat vectors on provided axis
                node = std::make_shared<v0::Concat>(node_vector, axis[0]);
            }

            copy_runtime_info_and_name(stack, {node}, {input_node});
            replace_node(stack, node);
            return true;
        }
        add_exception_to_fw_node(stack, "Unsupported case of aten::stack.");
        return false;
    };

    auto m = std::make_shared<Matcher>(stack, "ov::frontend::pytorch::pass::AtenStackListConstructReplacer");
    this->register_matcher(m, callback);
};

GPTQDecompressionReplacer::GPTQDecompressionReplacer() {
    auto const_1 = wrap_type<v0::Constant>();
    auto const_2 = wrap_type<v0::Constant>();
    auto unsqueeze_1 = wrap_type<v0::Unsqueeze>({const_1, const_2});
    //auto const_3 = wrap_type<v0::Constant>();
    //auto const_4 = wrap_type<v0::Constant>();
    //auto shape_of = wrap_type<ov::op::v3::ShapeOf>({const_3});
    //auto broadcast_1 = wrap_type<ov::op::v3::Broadcast>({const_4, shape_of});
    //auto equal = wrap_type<ov::op::v1::Equal>({const_3, broadcast_1});
    //auto const_5 = wrap_type<v0::Constant>();
    //auto broadcast_2 = wrap_type<ov::op::v3::Broadcast>({const_5, shape_of});
    //auto select = wrap_type<v1::Select>({equal, broadcast_2, const_3});
    //auto broadcast_3 = wrap_type<ov::op::v3::Broadcast>({unsqueeze_1, select});
    auto const_abs = wrap_type<v0::Constant>();
    auto abs = wrap_type<v0::Abs>({const_abs});
    auto broadcast_3 = wrap_type<ov::op::v3::Broadcast>({unsqueeze_1, abs});
    auto const_6 = wrap_type<v0::Constant>();
    auto const_7 = wrap_type<v0::Constant>();
    auto unsqueeze_2 = wrap_type<v0::Unsqueeze>({const_6, const_7});
    auto bitwise_right_shift = wrap_type<ov::op::util::FrameworkNode>({broadcast_3, unsqueeze_2});
    auto and_mask = wrap_type<v0::Constant>();
    auto align_types_1 = wrap_type<ov::op::util::FrameworkNode>({bitwise_right_shift, and_mask});
    auto bitwise_and = wrap_type<ov::op::v13::BitwiseAnd>({align_types_1, align_types_1});
    auto const_8 = wrap_type<v0::Constant>();
    auto align_types_2 = wrap_type<ov::op::util::FrameworkNode>({bitwise_and, const_8});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        //std::cout << "DEBUG - GPTQDecompressionReplacer - A" << std::endl;
        auto align_types_2 = m.get_match_root();
        //if (!stack) {
        //    return false;
        //}
        const auto& pattern_map = m.get_pattern_value_map();
        auto input_node = pattern_map.at(unsqueeze_1).get_node_shared_ptr();
        auto weights_u32 = std::dynamic_pointer_cast<v0::Constant>(input_node->get_input_node_shared_ptr(0));
        auto axis = std::dynamic_pointer_cast<v0::Constant>(input_node->get_input_node_shared_ptr(1));
        auto axis_data = axis->get_data_ptr<uint32_t>();
        //std::cout << "DEBUG - GPTQDecompressionReplacer - A - axis: " << axis_data[0] << std::endl;

        auto u8_shape = weights_u32->get_shape();
        size_t full_size = shape_size(u8_shape);
        auto src = weights_u32->get_data_ptr<uint32_t>();

        auto read_u4_data = [](const void *array, size_t index) {
            auto arr_u32 = reinterpret_cast<const uint32_t*>(array);
            size_t idx_u32 = index / 8;
            size_t offset_u32 = index % 8;
            uint32_t val = arr_u32[idx_u32];
            val = val >> (offset_u32*4);
            val = val & 15;
            return val;
        };
        auto write_u4_data = [](void *array, size_t index, uint32_t data) {
            auto arr_u32 = reinterpret_cast<uint32_t*>(array);
            size_t idx_u32 = index / 8;
            size_t offset_u32 = index % 8;
            uint32_t old_val = arr_u32[idx_u32];
            data = data << (offset_u32*4);
            uint32_t mask = 15;
            mask = ~(mask << (offset_u32*4));
            uint32_t new_val = (old_val & mask) | data;
            arr_u32[idx_u32] = new_val;
        };
        //std::cout << "DEBUG - GPTQDecompressionReplacer - A - input_shape: " << u8_shape << std::endl;
        //auto u4_shape = u8_shape;
        //u4_shape.push_back(8);
        ov::Shape u4_shape;
        bool dim_added = false;
        size_t stride = 1;
        size_t size_y = 1;
        for (int i=0; i<u8_shape.size(); i++) {
            if (axis_data[0] == i) {
                u4_shape.push_back(8);
                dim_added = true;
            }
            if (axis_data[0] <= i) {
                stride *= u8_shape[i];
            } else {
                size_y *= u8_shape[i];
            }
            u4_shape.push_back(u8_shape[i]);
        }
        if (!dim_added) {
            u4_shape.push_back(8);
        }

        auto new_const = std::make_shared<v0::Constant>(element::u4, u4_shape);
        auto dst = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(new_const->get_data_ptr()));

        //std::cout << "DEBUG - GPTQDecompressionReplacer - A - read_u4 - index: 0, val: " << std::hex << read_u4_data(src, 0) << std::dec << std::endl;
        //std::cout << "DEBUG - GPTQDecompressionReplacer - A - read_u4 - index: 1, val: " << std::hex << read_u4_data(src, 1) << std::dec << std::endl;
        //std::cout << "DEBUG - GPTQDecompressionReplacer - A - read_u4 - index: 2, val: " << std::hex << read_u4_data(src, 2) << std::dec << std::endl;
        //std::cout << "DEBUG - GPTQDecompressionReplacer - A - read_u4 - index: 0+size_x, val: " << std::hex << read_u4_data(src, 512) << std::dec << std::dec << std::endl;
        //std::cout << "DEBUG - GPTQDecompressionReplacer - A - read_u4 - index: 0+size_x*2, val: " << std::hex << read_u4_data(src, 2*512) << std::dec << std::endl;
        //std::cout << "DEBUG - GPTQDecompressionReplacer - A - read_u4 - index: 0+size_x*3, val: " << std::hex << read_u4_data(src, 3*512) << std::dec << std::endl;

        //std::cout << "DEBUG - GPTQDecompressionReplacer - A - u4_shape: " << u4_shape << ", size_x: " << stride << ", size_y: " << size_y << std::endl;
        //std::cout << "DEBUG - GPTQDecompressionReplacer - B - src.shape: " << u8_shape << ", src.val[0]: " << std::hex << src[0] << std::dec << std::endl;
        //std::cout << "DEBUG - GPTQDecompressionReplacer - B - src.shape: " << u8_shape << ", src.val[1]: " << std::hex << src[1] << std::dec << std::endl;
        //std::cout << "DEBUG - GPTQDecompressionReplacer - B - src.shape: " << u8_shape << ", src.val[2]: " << std::hex << src[2] << std::dec << std::endl;
        //std::cout << "DEBUG - GPTQDecompressionReplacer - B - src.shape: " << u8_shape << ", src.val[3]: " << std::hex << src[3] << std::dec << std::endl;

        //size_t out_idx = 0;
        //for (size_t y=0; y<size_y; y++) {
        //    for (size_t x=0; x<(full_size*8)/size_y; x++) {
        //        uint32_t val = read_u4_data(src, x*size_y+y);
        //        write_u4_data(dst, out_idx++, val);
        //    }
        //}
        size_t in_idx = 0;
        for (size_t y=0; y<size_y; y++) {
            size_t offset = y*stride*8;
            for (size_t x=0; x<stride; x++) {
                for (size_t z=0; z<8; z++) {
                    uint32_t val = read_u4_data(src, in_idx);
                    write_u4_data(dst, (offset+x+stride*z), val);
                    in_idx++;
                }
            }
        }

        //for (size_t x=0; x<stride; x++) {
        //    uint32_t mask = 15;
        //    for (size_t y=0; y<8; y++) {
        //        size_t val_count = 0;
        //        size_t dst_index = 0;
        //        for (size_t z=0; z<size_y; z++) {
        //            uint32_t dst_value = 0;
        //            uint32_t masked_value = src[x*size_y+z] & mask;
        //            uint32_t shifted_value = masked_value << (val_count*4);
        //            val_count = (val_count+1)%8;
        //            if (val_count == 0) dst_index++;
        //            dst[y*size_y+val_count] = shifted_value | dst[y*size_y+val_count];
        //        }
        //    }
        //    mask = mask << 4;
        //}
        //std::copy(src, src + full_size, dst);  // TODO: Avoid copying, reuse the same constant
        copy_runtime_info_and_name(weights_u32, {new_const}, {weights_u32, bitwise_and, bitwise_right_shift});
        //std::cout << "DEBUG - GPTQDecompressionReplacer - C - dst.shape: " << u4_shape << ", dst.val[0]: " << std::hex << dst[0] << std::dec << std::endl;
        //std::cout << "DEBUG - GPTQDecompressionReplacer - C - dst.shape: " << u4_shape << ", dst.val[1]: " << std::hex << dst[1] << std::dec << std::endl;
        //std::cout << "DEBUG - GPTQDecompressionReplacer - C - dst.shape: " << u4_shape << ", dst.val[2]: " << std::hex << dst[2] << std::dec << std::endl;
        //std::cout << "DEBUG - GPTQDecompressionReplacer - C - dst.shape: " << u4_shape << ", dst.val[3]: " << std::hex << dst[3] << std::dec << std::endl;
        //return new_const;


        //auto axis_node = pattern_map.at(axis).get_node_shared_ptr();
        //auto axis_const = std::dynamic_pointer_cast<v0::Constant>(axis_node);
        //auto axis = axis_const->cast_vector<int64_t>();
        //if (axis.size() != 1) {
        //    add_exception_to_fw_node(stack, "aten::stack has multiple axes, only one is supported.");
        //    return false;
        //}
        //// Check if ListConstruct is an input
        //if (auto list_construct_node = cast_fw_node(input_node, "prim::ListConstruct")) {
        //    const auto& list_inputs = list_construct_node->input_values();
        //    std::shared_ptr<Node> node;
        //    if (auto compression = u4_compression_stack(list_inputs, axis[0])) {
        //        node = compression;
        //    } else {
        //        OutputVector node_vector;
        //        auto zero = v0::Constant::create(element::i32, Shape{}, {0});
        //        // Iterate over values in ListConstruct
        //        for (const auto& list_input : list_inputs) {
        //            auto node = concat_list_construct(list_input);
        //            auto unsqueezed_node = std::make_shared<v0::Unsqueeze>(node, axis_const);
        //            node_vector.push_back(unsqueezed_node);
        //        }
        //        // Concat vectors on provided axis
        //        node = std::make_shared<v0::Concat>(node_vector, axis[0]);
        //    }

        copy_runtime_info_and_name(bitwise_and, {new_const}, {input_node});
        replace_node(align_types_2, new_const);
        return true;
        //}
        //add_exception_to_fw_node(stack, "Unsupported case of aten::stack.");
        //return false;
    };

    auto m = std::make_shared<Matcher>(bitwise_and, "ov::frontend::pytorch::pass::GPTQDecompressionReplacer");
    this->register_matcher(m, callback);
};

FXDecompressionReplacer::FXDecompressionReplacer() {
    auto const_1_1 = wrap_type<v0::Constant>();
    auto const_1_2 = wrap_type<v0::Constant>();
    auto add = wrap_type<v1::Add>({const_1_1, const_1_2});
    auto const_2 = wrap_type<v0::Constant>();
    auto reshape = wrap_type<v1::Reshape>({add, const_2});
    auto const_3 = wrap_type<v0::Constant>();
    auto subtract = wrap_type<v1::Subtract>({const_3, reshape});
    auto convert = wrap_type<v0::Convert>({subtract});
    auto const_4 = wrap_type<v0::Constant>();
    auto mult = wrap_type<v1::Multiply>({const_4, convert});

    ov::matcher_pass_callback callback = [=](Matcher& m) {


        auto read_u4_data = [](const void *array, size_t index) {
            auto arr_u32 = reinterpret_cast<const uint32_t*>(array);
            size_t idx_u32 = index / 8;
            size_t offset_u32 = index % 8;
            uint32_t val = arr_u32[idx_u32];
            val = val >> (offset_u32*4);
            val = val & 15;
            return val;
        };
        auto write_u4_data = [](void *array, size_t index, uint32_t data) {
            auto arr_u32 = reinterpret_cast<uint32_t*>(array);
            size_t idx_u32 = index / 8;
            size_t offset_u32 = index % 8;
            uint32_t old_val = arr_u32[idx_u32];
            data = data << (offset_u32*4);
            uint32_t mask = 15;
            mask = ~(mask << (offset_u32*4));
            uint32_t new_val = (old_val & mask) | data;
            arr_u32[idx_u32] = new_val;
        };

        std::cout << "DEBUG - FXDecompressionReplacer - A" << std::endl;
        //auto mult = m.get_match_root();
        //if (!stack) {
        //    return false;
        //}
        const auto& pattern_map = m.get_pattern_value_map();
        std::cout << "DEBUG - FXDecompressionReplacer - B" << std::endl;
        auto add_node = pattern_map.at(add).get_node_shared_ptr();
        auto reshape_node = pattern_map.at(reshape).get_node_shared_ptr();
        auto sub_node = pattern_map.at(subtract).get_node_shared_ptr();
        auto convert_node = pattern_map.at(convert).get_node_shared_ptr();
        auto mult_node = pattern_map.at(mult).get_node_shared_ptr();
        std::cout << "DEBUG - FXDecompressionReplacer - C" << std::endl;
        auto add_input0_const = std::dynamic_pointer_cast<v0::Constant>(add_node->get_input_node_shared_ptr(0));
        std::cout << "DEBUG - FXDecompressionReplacer - D" << std::endl;
        auto add_in0_ptr = add_input0_const->get_data_ptr<uint8_t>();
        //auto add_in0_ptr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(add_input0_const->get_data_ptr()));
        std::cout << "DEBUG - FXDecompressionReplacer - E" << std::endl;
        auto add_input1_const = std::dynamic_pointer_cast<v0::Constant>(add_node->get_input_node_shared_ptr(1));
        std::cout << "DEBUG - FXDecompressionReplacer - F" << std::endl;
        auto add_in1_ptr = add_input1_const->get_data_ptr<uint8_t>();
        std::cout << "DEBUG - FXDecompressionReplacer - G - in0[0]: " << std::hex << (uint32_t)(add_in0_ptr[0]) << std::dec << std::endl;
        std::cout << "DEBUG - FXDecompressionReplacer - H - in0[1]: " << std::hex << (uint32_t)(add_in1_ptr[0]) << std::dec << std::endl;
        auto add_in0_shape = add_input0_const->get_shape();
        size_t add_in0_size = shape_size(add_in0_shape);
        auto add_replace_const = std::make_shared<v0::Constant>(element::f32, reshape_node->get_shape());
        //auto add_replace_ptr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(add_replace_const->get_data_ptr()));
        auto add_replace_ptr = const_cast<float*>(reinterpret_cast<const float*>(add_replace_const->get_data_ptr()));
	uint32_t add_val = (uint32_t)(add_in1_ptr[0] & 0x0F);
        std::cout << "DEBUG - FXDecompressionReplacer - I - add_val: " << add_val << ", add_in0_shape: " << add_in0_shape << ", add_in0_size: " << add_in0_size << std::endl;
	for (size_t i=0; i<add_in0_size; i++) {
	    uint32_t val = read_u4_data(add_in0_ptr, i);
            //std::cout << "\tDEBUG - FXDecompressionReplacer - I - val: " << val << std::endl;
	    val = (val + add_val) & 0x0F;
	    //write_u4_data(add_replace_ptr, i, val);
	    add_replace_ptr[i] = (float)val;
	}
        std::cout << "DEBUG - FXDecompressionReplacer - J - in0[0]: " << std::hex << (uint32_t)(add_replace_ptr[0]) << std::dec << std::endl;
        std::cout << "DEBUG - FXDecompressionReplacer - K - shape: " << reshape_node->get_shape() << std::endl;
        //auto new_convert_node = std::make_shared<v0::Convert>(add_replace_const, element::f32);
        std::cout << "DEBUG - FXDecompressionReplacer - K.1" << std::endl;
        auto sub_input0_const = std::dynamic_pointer_cast<v0::Constant>(sub_node->get_input_node_shared_ptr(0));
        std::cout << "DEBUG - FXDecompressionReplacer - K.2" << std::endl;
        auto sub_in0_ptr = sub_input0_const->get_data_ptr<uint8_t>();
        std::cout << "DEBUG - FXDecompressionReplacer - K.3" << std::endl;
        auto sub_in0_shape = sub_input0_const->get_shape();
        std::cout << "DEBUG - FXDecompressionReplacer - K.4" << std::endl;
        size_t sub_in0_size = shape_size(sub_in0_shape);
        std::cout << "DEBUG - FXDecompressionReplacer - K.5" << std::endl;
        //auto replace_sub_in0_const = std::make_shared<v0::Constant>(element::f32, sub_in0_shape);
        //std::cout << "DEBUG - FXDecompressionReplacer - K.6" << std::endl;
        //auto sub_in0_replace_ptr = const_cast<float*>(reinterpret_cast<const float*>(replace_sub_in0_const->get_data_ptr()));
        //std::cout << "DEBUG - FXDecompressionReplacer - L - sub_in1_shape: " << sub_in0_shape << std::endl;
	//for (size_t i=0; i<sub_in0_size; i++) {
	//    uint32_t val = read_u4_data(sub_in0_ptr, i);
        //    //std::cout << "\tDEBUG - FXDecompressionReplacer - M - i: " << i << ", val: " << val << std::endl;
	//    sub_in0_replace_ptr[i] = (float)val;
	//}
        auto new_convert_node = std::make_shared<v0::Convert>(sub_input0_const, element::f32);
        auto new_sub_node = std::make_shared<v1::Subtract>(new_convert_node, add_replace_const);
        //auto new_sub_node = std::make_shared<v1::Subtract>(new_convert_node, replace_sub_in0_const);
        //auto new_sub_node = std::make_shared<v1::Subtract>(replace_sub_in0_const, new_convert_node);

        auto mult_scale_const = std::dynamic_pointer_cast<v0::Constant>(mult_node->get_input_node_shared_ptr(0));
        auto new_mult_node = std::make_shared<v1::Multiply>(new_sub_node, mult_scale_const);
        replace_node(mult_node, new_mult_node);

        //auto add_input0_u4 = std::dynamic_pointer_cast<v0::Constant>(add_node->get_input_node_shared_ptr(0));
        //std::cout << "DEBUG - FXDecompressionReplacer - D" << std::endl;
        //auto add_input1_u4 = std::dynamic_pointer_cast<v0::Constant>(add_node->get_input_node_shared_ptr(1));
        //std::cout << "DEBUG - FXDecompressionReplacer - E" << std::endl;

	//OutputVector add_out(add_node->get_output_size());
        //std::cout << "DEBUG - FXDecompressionReplacer - F" << std::endl;
	//add_node->constant_fold(add_out, {add_input0_u4, add_input1_u4});
        //std::cout << "DEBUG - FXDecompressionReplacer - G - add_out.size: " << add_out.size() << std::endl;

        //const auto friendly_name_from = [](const ov::Node& node, const size_t output_count, const size_t idx) {
        //    constexpr auto single_output = static_cast<size_t>(1);
        //
        //    if (single_output == output_count) {
        //        return node.get_friendly_name();
        //    } else {
        //        return node.get_friendly_name() + "." + std::to_string(idx);
        //    }
        //};

        //for (size_t i = 0; i < add_out.size(); ++i) {
        //    std::cout << "DEBUG - FXDecompressionReplacer - G.1" << std::endl;
        //    auto node_output = add->output(i);
        //    auto replacement = add_out.at(i);
        //    std::cout << "DEBUG - FXDecompressionReplacer - G.2" << std::endl;
        //    if (replacement.get_node_shared_ptr()) {
        //        std::cout << "DEBUG - FXDecompressionReplacer - G.2.1" << std::endl;
	//    }
        //    if (node_output != replacement) {
        //        std::cout << "DEBUG - FXDecompressionReplacer - G.2.2" << std::endl;
	//    }
        //    if (replacement.get_tensor_ptr()) {
        //        std::cout << "DEBUG - FXDecompressionReplacer - G.2.3" << std::endl;
	//    }
        //    if (replacement.get_node_shared_ptr() && (node_output != replacement)) {
        //        std::cout << "DEBUG - FXDecompressionReplacer - G.3" << std::endl;
        //        replacement.get_node()->set_friendly_name(friendly_name_from(*add_node, add_out.size(), i));

        //        node_output.replace(replacement);
        //        // Copy runtime info from source nodes
        //        // when it was not propogated during pre-calculation
        //        //copy_runtime_info_from_input_values(add_node);
        //        // Propagate runtime info attributes to replacement
        //        //copy_runtime_info(add_node, replacement.get_node_shared_ptr());

        //        std::cout << "DEBUG - FXDecompressionReplacer - G.4" << std::endl;
	//	copy_runtime_info_and_name(add_node, {replacement.get_node_shared_ptr()});
        //    }
        //    std::cout << "DEBUG - FXDecompressionReplacer - G.5" << std::endl;
        //}
	//const auto add_const = ov::as_type_ptr<v0::Constant>(add_out[0].get_node_shared_ptr());
        std::cout << "DEBUG - FXDecompressionReplacer - H" << std::endl;
        //auto axis_data = axis->get_data_ptr<uint32_t>();

        //auto u8_shape = weights_u32->get_shape();
        //size_t full_size = shape_size(u8_shape);
        //auto src = weights_u32->get_data_ptr<uint32_t>();
        //replace_node(add_node, add_const);
        std::cout << "DEBUG - FXDecompressionReplacer - I" << std::endl;
	return false;
    };

    auto m = std::make_shared<Matcher>(mult, "ov::frontend::pytorch::pass::FXDecompressionReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
