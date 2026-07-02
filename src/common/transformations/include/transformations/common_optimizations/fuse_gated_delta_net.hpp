// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Remove Concat of Loop
 */

class TRANSFORMATIONS_API RemoveConcatSliceAfterLoop : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RemoveConcatSliceAfterLoop");
    RemoveConcatSliceAfterLoop();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses a loop-based gated delta net sub-graph into an internal GatedDeltaNet operation.
 *
 * Expected Loop body semantics per step:
 * 1) Decay recurrent state: `h_decay = h * exp(g)`
 * 2) Compute projection and delta: `delta = v - reduce_sum(h_decay * k, axis=-2)`
 * 3) Apply beta and update state: `h_new = h_decay + k * unsqueeze(delta * beta, axis=-2)`
 * 4) Compute per-step output: `o = reduce_sum(h_new * q, axis=-2, keep_dims=true)` and scatter to time index
 *
 * The matcher validates this body shape/operation pattern before replacing the Loop with `GatedDeltaNet`.
 */

class TRANSFORMATIONS_API FuseGDNLoop : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseGDNLoop");
    FuseGDNLoop();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses a torch.export `scan`-generated Loop into an internal GatedDeltaNet operation.
 *
 * The OpenVINO PyTorch FX frontend lowers a PyTorch `scan` over the delta-rule recurrence to a
 * Loop that carries the recurrent state as a merged input and consumes q/k/v/g/beta as sliced
 * inputs (axis 0 = time), collecting the per-step output via concatenated slices. This differs
 * structurally from `FuseGDNLoop` (which expects a ScatterUpdate-into-buffer body), so this pass
 * matches the scan variant: a Loop with exactly one 4D merged state input and five sliced inputs
 * whose body implements `h*exp(g)`, delta, state update, and output reduction. The external sliced
 * inputs are the full sequence tensors, which are fed directly to `GatedDeltaNet`.
 */

class TRANSFORMATIONS_API FuseScanGDN : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseScanGDN");
    FuseScanGDN();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuse l2_norm into GatedDeltaNet
 */

class TRANSFORMATIONS_API FuseL2NormIntoGDN : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseL2NormIntoGDN");
    FuseL2NormIntoGDN();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Verifies that Q, K, and V inputs of GatedDeltaNet are connected to the same
 *        Split/Slice/Concat ancestor through a chain of
 *        Reshape/Transpose/Gather/GatherND/Broadcast/Unsqueeze/Squeeze operations.
 */

class TRANSFORMATIONS_API FuseGroupedQueryIntoGDN : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseGroupedQueryIntoGDN");
    FuseGroupedQueryIntoGDN();
};

/// This pass transforms a loop-based Gated Delta Net sub-graph to a single internal `GatedDeltaNet` operation.
///
/// Before:
///      ┌────────┐   ┌────────┐   ┌───────┐   ┌──────────────┐   ┌───────┐         ┌───────┐
///      │   Q    │   │   K    │   │   V   │   │ Initial State│   │   G   │         │  Beta │
///      └───┬────┘   └───┬────┘   └──┬────┘   └──────┬───────┘   └──┬────┘         └───┬───┘
///          │            │           │               │              │                  │
///          │            │           │               │              │                  │
///      ┌───┴────────────┴───┐   ┌───┴────────┐      │         ┌────┴────────┐   ┌─────┴────┐
///      │ L2Norm(Q)+QScale   │   │ L2Norm(K)  │      │         │ Transpose   │   │ Transpose│
///      │       (1/√d)       │   └────┬───────┘      │         └────┬────────┘   └───┬──────┘
///      └────────┬───────────┘        │              │              │                │
///               │                    │              │              │                │
///      ┌────────┴────────────────────┴──────────────┴──────────────┴────────────────┴────────┐
///      │                                Loop (recurrent body)                                │
///      └──────────────────────────────────────┬──────────────────────────────────────────────┘
///                                             │
///                              ┌──────────────┴──────────────┐
///                              │ Concat / Slice / Reshape(s) │
///                              └──────────────┬──────────────┘
///                                             │
///                               ┌─────────────┴─────────────┐
///                               │    CoreAttn, StateOut     │
///                               └───────────────────────────┘
///
/// After:
///      ┌────────┐   ┌────────┐   ┌────────┐   ┌──────────────┐   ┌────────┐   ┌────────┐
///      │   Q    │   │   K    │   │   V    │   │ Initial State│   │   G    │   │  Beta  │
///      └───┬────┘   └───┬────┘   └───┬────┘   └──────┬───────┘   └───┬────┘   └───┬────┘
///          │            │            │               │               │            │
///          │            │            │               │               │            │
///      ┌───┴────────────┴────────────┴───────────────┴───────────────┴────────────┴───────┐
///      │                                  GatedDeltaNet                                   │
///      └────────────────────────────────────────┬─────────────────────────────────────────┘
///                                               │
///                                ┌──────────────┴──────────────┐
///                                │     CoreAttn, StateOut      │
///                                └─────────────────────────────┘

class TRANSFORMATIONS_API GatedDeltaNetFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("GatedDeltaNetFusion");
    GatedDeltaNetFusion() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::pass