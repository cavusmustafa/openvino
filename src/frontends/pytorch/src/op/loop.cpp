// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/loop.hpp"

#include <limits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/opsets/opset10.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_loop(const NodeContext& context) {
    const auto& inputs = context.inputs();
    PYTORCH_OP_CONVERSION_CHECK(inputs.size() >= 2, "Loop must have at least 2 inputs.");
    auto loop = std::make_shared<ov::op::v5::Loop>(inputs[0], inputs[1]);
    auto decoder = context.get_decoder();
    PYTORCH_OP_CONVERSION_CHECK(decoder->get_subgraph_size() == 1, "Loop must have 1 subgraph.");
    auto subgraph_decoder = decoder->get_subgraph_decoder(0);
    auto body = context.convert_subgraph(0);
    loop->set_function(body);
    ov::op::v5::Loop::SpecialBodyPorts spec_ports{0, 0};
    loop->set_special_body_ports(spec_ports);

    // process outputs first
    auto session = context.get_session();
    auto body_results = body->get_results();
    PYTORCH_OP_CONVERSION_CHECK(body_results.size() > 0, "At least one output from loop is required - condition.");
    std::map<size_t, Output<Node>> output_idxs;
    // 0 output is condition, do not need to connect it
    for (size_t i = 1; i < body_results.size(); i++) {
        auto result = body_results[i];
        auto out_idx = session->decode_tensor_name(result->input(0).get_source_output());
        PYTORCH_OP_CONVERSION_CHECK(output_idxs.count(out_idx) == 0,
                                    "More than one body output with same tensor name.");
        output_idxs[out_idx] = result;
    }

    auto body_parameters = body->get_parameters();
    // #0 body parameter is counter;
    PYTORCH_OP_CONVERSION_CHECK(body_parameters.size() > 0, "At least one input to Loop body is required");
    // Set counter shape
    body_parameters[0]->set_partial_shape(PartialShape{});
    // #0 loop input is  trip_count, #1 loop input is condition
    // Connect other inputs
    for (size_t i = 2; i < inputs.size(); i++) {
        if (i <= subgraph_decoder->num_of_outputs()) {
            loop->set_merged_input(body_parameters[i - 1], inputs[i], body_results[i - 1]);
        } else {
            loop->set_invariant_input(body_parameters[i - 1], inputs[i]);
        }
    }
    // Connect inputs from external context
    for (auto i = inputs.size() - 1; i < body_parameters.size(); i++) {
        auto param = body_parameters[i];
        auto input_idx = session->decode_tensor_name(param->output(0));
        auto external_output = context.get_tensor_from_model_or_create_input(input_idx);
        if (output_idxs.count(input_idx)) {
            loop->set_merged_input(param, external_output, output_idxs.at(input_idx));
        } else {
            loop->set_invariant_input(param, external_output);
        }
    }

    // connect outputs
    for (size_t i = 1; i < body_results.size(); i++) {
        auto result = body_results[i];
        auto out_idx = session->decode_tensor_name(result->input(0).get_source_output());
        context.add_tensor_to_context(out_idx, loop->get_iter_value(result, -1));
    }
    loop->validate_and_infer_types();
    return {context.mark_node(loop)->outputs()};
};

OutputVector translate_while_loop_fx(const NodeContext& context) {
    // FX while_loop has structure:
    // - 2 subgraphs: cond_fn (index 0) and body_fn (index 1)
    // - inputs = carried_inputs (values that are updated each iteration)
    // - cond_fn takes carried_inputs, returns boolean scalar
    // - body_fn takes carried_inputs, returns new carried_inputs
    //
    // OpenVINO Loop requires:
    // - trip_count (max iterations)
    // - execution_condition (initial condition)
    // - body model with Parameters and Results where:
    //   - Parameter[0] is iteration counter
    //   - Result[0] is next condition
    //   - Other parameters/results are carried values

    auto decoder = context.get_decoder();
    PYTORCH_OP_CONVERSION_CHECK(
        decoder->get_subgraph_size() == 2,
        "while_loop must have 2 subgraphs (cond and body), got " + std::to_string(decoder->get_subgraph_size()));

    const auto& inputs = context.inputs();
    const size_t num_carried = inputs.size();
    PYTORCH_OP_CONVERSION_CHECK(num_carried > 0, "while_loop must have at least one carried input.");

    // Convert cond and body subgraphs
    auto cond_model = context.convert_subgraph(0);
    auto body_model = context.convert_subgraph(1);

    auto cond_params = cond_model->get_parameters();
    auto cond_results = cond_model->get_results();
    auto body_params = body_model->get_parameters();
    auto body_results = body_model->get_results();

    PYTORCH_OP_CONVERSION_CHECK(cond_results.size() == 1, "cond_fn must have exactly one output (boolean condition).");
    PYTORCH_OP_CONVERSION_CHECK(body_results.size() == num_carried,
                                "body_fn must return same number of outputs as carried inputs. Expected " +
                                    std::to_string(num_carried) + ", got " + std::to_string(body_results.size()));

    // Build combined body model for OpenVINO Loop:
    // Parameters: [counter, ...carried_inputs]
    // Results: [condition, ...new_carried_values]
    //
    // Implementation approach:
    // 1. Create parameters for [counter, carried_inputs]
    // 2. Clone body operations connected to new parameters
    // 3. Clone cond operations connected to body outputs
    // 4. Return [cond_output, body_outputs]

    // Create trip_count (large number for while loop)
    auto trip_count =
        ov::opset10::Constant::create(ov::element::i64, ov::Shape{}, {std::numeric_limits<int32_t>::max()});

    // Create initial condition (true)
    auto init_cond = ov::opset10::Constant::create(ov::element::boolean, ov::Shape{}, {true});

    // For FX while_loop, we need to construct a proper body
    // The body should be: [counter, x0, x1, ...] -> [cond(body(x0,x1,...)), body(x0,x1,...)]

    // Create new parameters for the loop body
    ov::ParameterVector loop_body_params;
    // counter
    auto loop_counter = std::make_shared<ov::opset10::Parameter>(ov::element::i64, ov::PartialShape{});
    loop_counter->set_friendly_name("loop_iteration");
    loop_body_params.push_back(std::move(loop_counter));

    // carried inputs
    for (size_t i = 0; i < num_carried; i++) {
        auto param =
            std::make_shared<ov::opset10::Parameter>(inputs[i].get_element_type(), inputs[i].get_partial_shape());
        param->set_friendly_name("loop_carried_" + std::to_string(i));
        loop_body_params.push_back(std::move(param));
    }

    // Map body model parameters to loop body parameters
    std::map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> body_param_map;
    for (size_t i = 0; i < body_params.size() && i < num_carried; i++) {
        body_param_map[body_params[i]] = loop_body_params[i + 1];
    }

    // Clone body model operations
    ov::NodeVector body_ops;
    std::map<ov::Node*, ov::Node*> node_map;
    for (const auto& orig_param : body_params) {
        if (body_param_map.count(orig_param)) {
            node_map[orig_param.get()] = body_param_map[orig_param].get();
        }
    }

    // Clone all body operations (except parameters and results)
    for (const auto& op : body_model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Parameter>(op) || ov::is_type<ov::opset10::Result>(op)) {
            continue;
        }
        ov::OutputVector new_inputs;
        for (const auto& input : op->inputs()) {
            auto source = input.get_source_output();
            auto source_node = source.get_node();
            if (node_map.count(source_node)) {
                new_inputs.push_back(node_map[source_node]->output(source.get_index()));
            } else {
                new_inputs.push_back(source);
            }
        }
        auto cloned = op->clone_with_new_inputs(new_inputs);
        cloned->set_friendly_name(op->get_friendly_name());
        node_map[op.get()] = cloned.get();
        body_ops.push_back(cloned);
    }

    // Get cloned body outputs
    std::vector<ov::Output<ov::Node>> cloned_body_outputs;
    for (const auto& result : body_results) {
        auto source = result->input(0).get_source_output();
        auto source_node = source.get_node();
        if (node_map.count(source_node)) {
            cloned_body_outputs.push_back(node_map[source_node]->output(source.get_index()));
        } else {
            cloned_body_outputs.push_back(source);
        }
    }

    // Now clone cond model and connect to cloned body outputs
    // Create separate maps for cond params -> body outputs and cloned operations
    std::map<ov::Node*, size_t> cond_param_to_idx;  // Maps cond param to index
    for (size_t i = 0; i < cond_params.size(); i++) {
        cond_param_to_idx[cond_params[i].get()] = i;
    }
    std::map<ov::Node*, ov::Node*> cond_cloned_map;  // Maps original cond ops to cloned ops

    // Clone cond operations
    for (const auto& op : cond_model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Parameter>(op) || ov::is_type<ov::opset10::Result>(op)) {
            continue;
        }
        ov::OutputVector new_inputs;
        for (const auto& input : op->inputs()) {
            auto source = input.get_source_output();
            auto source_node = source.get_node();
            if (cond_param_to_idx.count(source_node)) {
                // This input is from a cond parameter - connect to corresponding body output
                size_t param_idx = cond_param_to_idx.at(source_node);
                new_inputs.push_back(cloned_body_outputs[param_idx]);
            } else if (cond_cloned_map.count(source_node)) {
                // This input is from a previously cloned cond operation
                new_inputs.push_back(cond_cloned_map[source_node]->output(source.get_index()));
            } else if (node_map.count(source_node)) {
                // This input is from a body operation
                new_inputs.push_back(node_map[source_node]->output(source.get_index()));
            } else {
                // Keep original (e.g., constants from outside)
                new_inputs.push_back(source);
            }
        }
        auto cloned = op->clone_with_new_inputs(new_inputs);
        cloned->set_friendly_name(op->get_friendly_name() + "_cond");
        cond_cloned_map[op.get()] = cloned.get();
        body_ops.push_back(std::move(cloned));
    }

    // Get condition output
    ov::Output<ov::Node> cond_output;
    for (const auto& result : cond_results) {
        auto source = result->input(0).get_source_output();
        if (cond_cloned_map.count(source.get_node())) {
            cond_output = cond_cloned_map[source.get_node()]->output(source.get_index());
        } else {
            cond_output = source;
        }
    }

    // Ensure condition is scalar boolean
    if (cond_output.get_partial_shape().rank().is_static() && cond_output.get_partial_shape().rank().get_length() > 0) {
        cond_output = std::make_shared<ov::opset10::Squeeze>(cond_output);
    }

    // Create results: [condition, body_output_0, body_output_1, ...]
    ov::ResultVector loop_body_results;
    loop_body_results.push_back(std::make_shared<ov::opset10::Result>(cond_output));
    for (const auto& out : cloned_body_outputs) {
        loop_body_results.push_back(std::make_shared<ov::opset10::Result>(out));
    }

    // Create body model
    auto loop_body = std::make_shared<ov::Model>(loop_body_results, loop_body_params, "while_loop_body");

    // Create Loop operation
    auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, init_cond);
    loop->set_function(loop_body);

    // Set special body ports: counter at index 0, condition output at index 0
    ov::op::v5::Loop::SpecialBodyPorts spec_ports{0, 0};
    loop->set_special_body_ports(spec_ports);

    // Set merged inputs (carried values)
    for (size_t i = 0; i < num_carried; i++) {
        loop->set_merged_input(loop_body_params[i + 1], inputs[i], loop_body_results[i + 1]);
    }

    // Validate and get outputs
    loop->validate_and_infer_types();

    // Get outputs (skip condition result which is index 0)
    OutputVector outputs;
    for (size_t i = 0; i < num_carried; i++) {
        auto out = loop->get_iter_value(loop_body_results[i + 1], -1);
        outputs.push_back(std::move(out));
    }

    context.mark_node(loop);

    // In FX, while_loop returns a tuple that is accessed via getitem.
    // We wrap outputs in make_list_construct so getitem can extract individual elements.
    return {context.mark_node(make_list_construct(outputs))};
}

OutputVector translate_scan_fx(const NodeContext& context) {
    // FX torch.ops.higher_order.scan has structure:
    //   scan(combine_fn, init=[c0..], xs=[x0..], additional_inputs=[])
    // combine_fn(carry.., x_slice..) -> (new_carry.., y..)
    // The decoder unpacks init then xs into flat inputs(); subgraph 0 is combine_fn.
    //
    // Maps to OpenVINO Loop:
    //   - carries -> merged inputs (body result feeds back each iteration)
    //   - xs      -> sliced inputs (axis 0, one slice per iteration)
    //   - ys      -> concatenated slices (stacked along a new axis 0)
    // trip_count = number of slices along xs axis 0.
    auto decoder = context.get_decoder();
    PYTORCH_OP_CONVERSION_CHECK(decoder->get_subgraph_size() == 1,
                                "scan must have exactly 1 subgraph (combine_fn), got " +
                                    std::to_string(decoder->get_subgraph_size()));

    const auto& inputs = context.inputs();
    // num_init carries: query the decoder's schema is not available here; infer from body.
    auto body_model = context.convert_subgraph(0);
    auto body_params = body_model->get_parameters();
    auto body_results = body_model->get_results();

    const size_t num_total = inputs.size();
    const size_t num_body_params = body_params.size();
    PYTORCH_OP_CONVERSION_CHECK(num_body_params == num_total,
                                "scan combine_fn params must match init+xs inputs. Got params=" +
                                    std::to_string(num_body_params) + " inputs=" + std::to_string(num_total));
    // combine_fn returns (new_carry.., y..) and takes (carry.., x_slice..). By scan's contract
    // the leading carry inputs and leading carry results have identical shapes, and carries
    // precede xs in both. Derive num_carry by matching leading input shapes to leading result
    // shapes (a carry input's shape == its fed-back result's shape; an xs input is sliced along
    // axis 0 so its full shape does NOT equal any result). This avoids needing a decoder-level
    // init count.
    size_t num_carry = 0;
    for (size_t i = 0; i < num_total && i < body_results.size(); i++) {
        auto in_ps = inputs[i].get_partial_shape();
        auto res_ps = body_results[i]->get_output_partial_shape(0);
        if (in_ps.rank().is_static() && res_ps.rank().is_static() && in_ps.compatible(res_ps)) {
            num_carry++;
        } else {
            break;
        }
    }
    PYTORCH_OP_CONVERSION_CHECK(num_carry > 0 && num_carry < num_total,
                                "scan: could not determine carry count (num_carry=" + std::to_string(num_carry) +
                                    ", total=" + std::to_string(num_total) + ")");
    const size_t num_xs = num_total - num_carry;
    const size_t num_ys = body_results.size() - num_carry;
    PYTORCH_OP_CONVERSION_CHECK(body_results.size() > num_carry, "scan: combine_fn must return at least one y output.");

    // trip_count from the first xs input's dim 0.
    auto first_xs = inputs[num_carry];
    auto shape_of = std::make_shared<ov::opset10::ShapeOf>(first_xs, ov::element::i64);
    auto zero_i = ov::opset10::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto axis0 = ov::opset10::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto trip_count = std::make_shared<ov::opset10::Gather>(shape_of, zero_i, axis0);
    auto init_cond = ov::opset10::Constant::create(ov::element::boolean, ov::Shape{}, {true});

    // Build loop body: [counter, carry.., x_slice..] -> [cond=true, new_carry.., y..]
    ov::ParameterVector loop_params;
    auto counter = std::make_shared<ov::opset10::Parameter>(ov::element::i64, ov::PartialShape{});
    loop_params.push_back(counter);
    for (size_t i = 0; i < num_carry; i++) {
        loop_params.push_back(
            std::make_shared<ov::opset10::Parameter>(inputs[i].get_element_type(), inputs[i].get_partial_shape()));
    }
    // xs body params: per-iteration slice has the xs rank with dim0 == 1 (Loop slices keep the axis).
    for (size_t i = 0; i < num_xs; i++) {
        auto xs_ps = inputs[num_carry + i].get_partial_shape();
        ov::PartialShape slice_ps = xs_ps;
        if (slice_ps.rank().is_static() && slice_ps.rank().get_length() > 0) {
            slice_ps[0] = 1;
        }
        loop_params.push_back(
            std::make_shared<ov::opset10::Parameter>(inputs[num_carry + i].get_element_type(), slice_ps));
    }

    // Map combine body params -> loop body params (skip counter at index 0). The xs slice from a
    // Loop retains axis 0 of size 1; the combine expects it squeezed, so squeeze before feeding.
    std::map<ov::Node*, ov::Node*> node_map;
    for (size_t i = 0; i < num_carry; i++) {
        node_map[body_params[i].get()] = loop_params[i + 1].get();
    }
    ov::NodeVector body_ops;
    std::vector<std::shared_ptr<ov::Node>> squeezed_xs;
    for (size_t i = 0; i < num_xs; i++) {
        auto sq_axis = ov::opset10::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto squeezed = std::make_shared<ov::opset10::Squeeze>(loop_params[1 + num_carry + i], sq_axis);
        node_map[body_params[num_carry + i].get()] = squeezed.get();
        body_ops.push_back(sq_axis);
        body_ops.push_back(squeezed);
        squeezed_xs.push_back(squeezed);
    }

    for (const auto& op : body_model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Parameter>(op) || ov::is_type<ov::opset10::Result>(op)) {
            continue;
        }
        ov::OutputVector new_inputs;
        for (const auto& input : op->inputs()) {
            auto source = input.get_source_output();
            auto src_node = source.get_node();
            if (node_map.count(src_node)) {
                new_inputs.push_back(node_map[src_node]->output(source.get_index()));
            } else {
                new_inputs.push_back(source);
            }
        }
        auto cloned = op->clone_with_new_inputs(new_inputs);
        cloned->set_friendly_name(op->get_friendly_name());
        node_map[op.get()] = cloned.get();
        body_ops.push_back(cloned);
    }

    auto resolve = [&](const ov::Output<ov::Node>& src) -> ov::Output<ov::Node> {
        auto n = src.get_node();
        return node_map.count(n) ? node_map[n]->output(src.get_index()) : src;
    };

    ov::ResultVector loop_results;
    auto true_const = ov::opset10::Constant::create(ov::element::boolean, ov::Shape{}, {true});
    loop_results.push_back(std::make_shared<ov::opset10::Result>(true_const));  // condition
    std::vector<ov::Output<ov::Node>> new_carry_outs, y_outs;
    for (size_t i = 0; i < num_carry; i++) {
        auto out = resolve(body_results[i]->input(0).get_source_output());
        new_carry_outs.push_back(out);
        loop_results.push_back(std::make_shared<ov::opset10::Result>(out));
    }
    for (size_t i = 0; i < num_ys; i++) {
        // combine returns y with the per-step shape; add axis 0 so slices concat into [T, ...].
        auto y = resolve(body_results[num_carry + i]->input(0).get_source_output());
        auto unsq_axis = ov::opset10::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto y_unsq = std::make_shared<ov::opset10::Unsqueeze>(y, unsq_axis);
        y_outs.push_back(y_unsq);
        loop_results.push_back(std::make_shared<ov::opset10::Result>(y_unsq));
    }

    auto loop_body = std::make_shared<ov::Model>(loop_results, loop_params, "scan_body");
    auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, init_cond);
    loop->set_function(loop_body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // carries: merged inputs (result index in body = 1 + i, since result 0 is condition)
    for (size_t i = 0; i < num_carry; i++) {
        loop->set_merged_input(loop_params[1 + i], inputs[i], loop_results[1 + i]);
    }
    // xs: sliced inputs along axis 0, part_size 1
    for (size_t i = 0; i < num_xs; i++) {
        loop->set_sliced_input(loop_params[1 + num_carry + i], inputs[num_carry + i], 0, 1, 1, -1, 0);
    }
    loop->validate_and_infer_types();

    OutputVector outputs;
    // final carries
    for (size_t i = 0; i < num_carry; i++) {
        outputs.push_back(loop->get_iter_value(loop_results[1 + i], -1));
    }
    // stacked ys via concatenated slices along axis 0
    for (size_t i = 0; i < num_ys; i++) {
        outputs.push_back(loop->get_concatenated_slices(loop_results[1 + num_carry + i], 0, 1, 1, -1, 0));
    }
    context.mark_node(loop);
    return {context.mark_node(make_list_construct(outputs))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
