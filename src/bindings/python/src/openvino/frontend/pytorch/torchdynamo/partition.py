# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

from typing import Dict

import torch
from torch.nn import Module
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition

from torch.fx.experimental.proxy_tensor import DecompositionInterpreter
from torch._decomp import decomposition_table
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
from openvino.frontend.pytorch.torchdynamo.op_support import OperatorSupport

import typing as t
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Partitioner:
    def __init__(self):
        self.supported_ops = OperatorSupport()

    def fx_serialize(self, graph_module: GraphModule, *args, **kwargs):
        fx_gm = make_fx(graph_module)(*args)
        return fx_gm

    def add_get_attr_inputs(self, partitions: t.List[Partition]):
        # TODO: Find a more efficient way to include input
        # "get_attr" nodes to the partitions.
        getattr_to_merge: Dict[Node, Node] = {}
        for partition in partitions:
            for pnode in partition.nodes:
                for pnode_input in pnode.all_input_nodes:
                    if pnode_input.op in ["get_attr"] and pnode_input.op not in getattr_to_merge:
                        getattr_to_merge[pnode_input] = partition
        for getattr_node, getattr_part in getattr_to_merge.items():
            getattr_part.add_node(getattr_node)

    def check_fully_supported(self, graph_module: GraphModule) -> bool:
        num_fused = 0
        for node in graph_module.graph.nodes:
            if node.op == "call_module" and "fused_" in node.name:
                num_fused += 1
            elif node.op != "placeholder" and node.op != "output":
                return False
        if num_fused == 1:
            return True
        return False

    def filter_nodes(self, partitions: t.List[Partition]):
        include_nodes = ["pow_1",
                         "mean",
                         "add",
                         "rsqrt",
                         "mul",
                         "mul_1",
                         "view",
                         "unsqueeze",
                         "expand",
                         "unsqueeze_1",
                         "bitwise_right_shift",
                         "_to_copy",
                         "bitwise_and",
                         "add_1",
                         "view_1",

                         "view_2",
                         "unsqueeze_2",
                         "expand_1",
                         "unsqueeze_3",
                         "bitwise_right_shift_1",
                         "_to_copy_1",
                         "bitwise_and_1",
                         "view_3",
                         "sub",
                         "mul_2",
                         "view_4",
                         "mm"]
        for partition in partitions:
            nodes_to_remove = []
            for pnode in partition.nodes:
                #print("DEBUG - filter_nodes - node.name: ", pnode.name, ", lookup")
                if pnode.name in include_nodes:
                    print("\tDEBUG - filter_nodes - node.name: ", pnode.name, ", found: yes")
                else:
                #    print("\tDEBUG - filter_nodes - node.name: ", pnode.name, ", found: no")
                    nodes_to_remove.append(pnode)
            for rm_node in nodes_to_remove:
                partition.remove_node(rm_node)

    def make_partitions(self, graph_module: GraphModule) -> GraphModule:
        partitioner = CapabilityBasedPartitioner(
            graph_module, self.supported_ops, allows_single_node_partition=False)
        partitions = partitioner.propose_partitions()
        new_partitions = []
        min_num_nodes = 0
        if os.getenv("OPENVINO_TORCH_MIN_NUM_NODES") is not None:
            min_num_nodes = int(os.getenv("OPENVINO_TORCH_MIN_NUM_NODES"))
        for part in partitions:
            print("DEBUG - partitions - partition_size: ", len(part.nodes))
            if len(part.nodes) > min_num_nodes:
                new_partitions.append(part)
        #self.filter_nodes(new_partitions)
        self.add_get_attr_inputs(new_partitions)
        print("DEBUG - partitions - num_partitions: ", len(new_partitions))
        fused_graph_module = partitioner.fuse_partitions(new_partitions)

        return fused_graph_module
