# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestViewSplitDim(PytorchLayerTest):
    """Views whose lowering needs the symbolic numel check in infer_size.

    Under torch.compile these used to abort the whole compilation from Dynamo's
    guard codegen (IndexError in symbol_to_source), because the backend's
    make_fx retrace shared Dynamo's ShapeEnv and allocated symbols there that
    Dynamo had no source for. Splitting a hidden dim into heads x head_dim is
    the shape every attention block produces, so keep it covered.
    """

    def _prepare_input(self):
        return (self.random.uniform(0, 50, (1, 8, 96), dtype="float32"),)

    def create_model(self, shape):
        import torch

        class aten_view(torch.nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.shape = shape

            def forward(self, x):
                return x.view(self.shape)

        return aten_view(shape), "aten::view"

    @pytest.mark.parametrize("shape", [
        [1, 8, 6, 16],   # split one dim into two
        [1, 8, 2, 3, 16],  # split one dim into three
        [1, 768],        # merge all dims
        [1, 8, 6, -1],   # split with an inferred dim
        [8, 96],         # drop a leading 1
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_view_split_dim(self, shape, ie_device, precision, ir_version):
        self._test(*self.create_model(shape), ie_device, precision, ir_version)


class TestViewSplitDimAttention(PytorchLayerTest):
    """The attention-shaped case end to end: split, permute, merge back."""

    def _prepare_input(self):
        return (self.random.uniform(0, 50, (2, 4, 64), dtype="float32"),)

    def create_model(self):
        import torch

        class aten_view_heads(torch.nn.Module):
            def forward(self, x):
                b, s, d = x.shape
                n_heads, head_dim = 8, 8
                y = x.view(b, s, n_heads, head_dim)
                y = y.permute(0, 2, 1, 3)
                return y.reshape(b, n_heads * s, head_dim)

        return aten_view_heads(), "aten::view"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_view_split_dim_attention(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)
