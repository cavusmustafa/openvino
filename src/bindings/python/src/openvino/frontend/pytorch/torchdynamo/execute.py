# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from warnings import warn

import torch
import torch.overrides

from torch.fx import GraphModule
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.torchdynamo.compile import openvino_compile
from openvino.runtime import Core, Type, PartialShape, Tensor, Shape
from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_cache_dir, _get_device, _get_aot_autograd
from openvino.frontend.pytorch.utils import make_constant, fetch_attr, pt_to_ov_type_map, torch_tensor_to_ov_const

import time

from typing import Callable, Optional, Any

from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


DEFAULT_OPENVINO_PYTHON_CONFIG = MappingProxyType(
    {
        "use_python_fusion_cache": True,
        "allow_single_op_fusion": True,
    },
)

compiled_cache = {}
req_cache = {}
max_openvino_partitions = 0
partitioned_modules = {}


def execute(
    gm: GraphModule,
    *args,
    executor: str = "openvino",
    executor_parameters: Optional[dict] = None,
    options: Optional[Any] = None,
):
    if executor == "openvino":
        return openvino_execute_partitioned(gm, *args, executor_parameters=executor_parameters, options=options)
    elif executor == "strictly_openvino":
        return openvino_execute(gm, *args, executor_parameters=executor_parameters)

    msg = "Received unexpected value for 'executor': {0}. Allowed values are: openvino, strictly_openvino.".format(executor)
    raise ValueError(msg)


import numpy as np


def execute_cached(compiled_model, *args):
    ov_inputs = [a.detach().cpu().numpy() for a in args]
    ov_inputs.reverse()
    res = compiled_model(ov_inputs)
    result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
    return result


#def openvino_execute(gm: GraphModule, *args, executor_parameters=None, partition_id, options):
#
#    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG
#
#    use_cache = executor_parameters.get(
#        "use_python_fusion_cache",
#        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
#    )
#    global compiled_cache
#
#    model_hash_str = executor_parameters.get("model_hash_str", None)
#    if model_hash_str is not None:
#        fully_supported = False
#        if len(model_hash_str) > 3 and model_hash_str[-3:] == "_fs":
#            fully_supported = True
#        if not fully_supported:
#            model_hash_str = model_hash_str + "_p" + str(partition_id)
#
#    if use_cache and (partition_id in compiled_cache):
#        compiled = compiled_cache[partition_id]
#        req = req_cache[partition_id]
#    else:
#        compiled = openvino_compile(gm, *args, model_hash_str=model_hash_str, options=options)
#        compiled_cache[partition_id] = compiled
#        req = compiled.create_infer_request()
#        req_cache[partition_id] = req
#
#    flat_args, _ = tree_flatten(args)
#    ov_inputs = []
#    for arg in flat_args:
#        ov_inputs.append((arg if isinstance(arg, int) else arg.detach().cpu().numpy()))
#
#    res = req.infer(ov_inputs, share_inputs=True, share_outputs=True)
#
#    results1 = [torch.from_numpy(res[out]) for out in compiled.outputs]
#    if len(results1) == 1:
#        return results1[0]
#    return results1

def openvino_execute(gm: GraphModule, *args, executor_parameters=None, partition_id, options):

    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    use_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    global compiled_cache

    model_hash_str = executor_parameters.get("model_hash_str", None)
    if model_hash_str is not None:
        fully_supported = False
        if len(model_hash_str) > 3 and model_hash_str[-3:] == "_fs":
            fully_supported = True
        if not fully_supported:
            model_hash_str = model_hash_str + "_p" + str(partition_id)


    if use_cache and (partition_id in compiled_cache):
        compiled = compiled_cache[partition_id]
        req = req_cache[partition_id]
    else:
        compiled = openvino_compile(gm, *args, model_hash_str=model_hash_str, options=options)
        compiled_cache[partition_id] = compiled
        req = compiled.create_infer_request()
        req_cache[partition_id] = req

    flat_args, _ = tree_flatten(args)
    ov_inputs = []
    for arg in flat_args:
        if isinstance(arg, int):
            ov_inputs.append(arg)
        else:
            data_ptr = arg.detach().cpu().data_ptr()

            ov_tensor = Tensor(data_ptr, Shape(arg.detach().cpu().shape), pt_to_ov_type_map[str(arg.detach().cpu().dtype)])
            ov_inputs.append(ov_tensor)

        #if isinstance(arg, torch.nn.parameter.Parameter):
        #    if arg.data.dtype == torch.bfloat16:
        #        arg.data = arg.data.view(dtype=torch.float16)
        #elif isinstance(arg, torch.Tensor):
        #    if arg.dtype == torch.bfloat16:
        #        arg = arg.view(dtype=torch.float16)
        #ov_inputs.append((arg if isinstance(arg, int) else arg.detach().cpu().numpy()))

    time_1 = time.time()
    res = req.infer(ov_inputs, share_inputs=True, share_outputs=True)
    time_2 = time.time()

    #print("DEBUG - openvino_execute - res - type: ", type(res))
    #pinfo = req.get_profiling_info()
    #print("DEBUG - openvino_execute - pinfo - type: ", type(pinfo))
    #num_reorders = 0
    #for pc in pinfo:
    #    #print("\tDEBUG - openvino_execute - pc - type: ", pc.node_type, ", exec_type: ", pc.exec_type, ", name: ", pc.node_name, ", cpu_time: ", pc.cpu_time, ", real_time: ", pc.real_time, ", status: ", pc.status)
    #    if "Reorder" in pc.node_type:
    #        num_reorders += 1
    #print("DEBUG - openvino_execute - pinfo - num_args: ", len(flat_args), ", num_reorders: ", num_reorders, ", time: ", (time_2-time_1))
    print("DEBUG - openvino_execute - pinfo - num_args: ", len(flat_args), ", time: ", (time_2-time_1))

    results1 = [torch.from_numpy(res[out]) for out in compiled.outputs]
    r_idx = 0
    for res in results1:
        if res.dtype == torch.float16:
            res = res.view(dtype=torch.bfloat16)
            results1[r_idx] = res
        r_idx += 1

    if len(results1) == 1:
        return results1[0]
    return results1

class OpenVINOGraphModule(torch.nn.Module):
    def __init__(self, gm, partition_id, use_python_fusion_cache, model_hash_str: str = None, options=None):
        super().__init__()
        self.gm = gm
        self.partition_id = partition_id
        self.executor_parameters = {"use_python_fusion_cache": use_python_fusion_cache,
                                    "model_hash_str": model_hash_str}
        self.perm_fallback = False
        self.options = options

    def __call__(self, *args):
        if self.perm_fallback:
            return self.gm(*args)

        result = openvino_execute(self.gm, *args, executor_parameters=self.executor_parameters, partition_id=self.partition_id, options=self.options)
        #try:
        #    result = openvino_execute(self.gm, *args, executor_parameters=self.executor_parameters, partition_id=self.partition_id, options=self.options)
        #except Exception:
        #    logger.debug("OpenVINO execution failed. Falling back to native PyTorch execution.")
        #    self.perm_fallback = True
        #    return self.gm(*args)

        return result


def partition_graph(gm: GraphModule, use_python_fusion_cache: bool, model_hash_str: str = None, options=None):
    global max_openvino_partitions
    partition_id = max_openvino_partitions
    for node in gm.graph.nodes:
        # TODO: use a better way to identify fused submodule
        if node.op == "call_module" and "fused_" in node.name:
            openvino_submodule = getattr(gm, node.name)
            gm.delete_submodule(node.target)
            gm.add_submodule(
                node.target,
                OpenVINOGraphModule(openvino_submodule, partition_id, use_python_fusion_cache,
                                    model_hash_str=model_hash_str, options=options),
            )
            partition_id = partition_id + 1

    max_openvino_partitions = partition_id

    return gm


def openvino_execute_partitioned(gm: GraphModule, *args, executor_parameters=None, options=None):
    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    global partitioned_modules

    use_python_fusion_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    model_hash_str = executor_parameters.get("model_hash_str", None)

    signature = str(id(gm))
    if (not _get_aot_autograd(options)):
        for idx, input_data in enumerate(args):
            if isinstance(input_data, torch.Tensor):
                signature = signature + "_" + str(idx) + ":" + str(input_data.type())[6:] + ":" + str(input_data.size())[11:-1].replace(" ", "")
            else:
                signature = signature + "_" + str(idx) + ":" + type(input_data).__name__ + ":val(" + str(input_data) + ")"

    if signature not in partitioned_modules:
        partitioned_modules[signature] = partition_graph(gm, use_python_fusion_cache=use_python_fusion_cache,
                                                         model_hash_str=model_hash_str, options=options)
    return partitioned_modules[signature](*args)


def clear_caches():
    global partitioned_modules
    global compiled_cache

    compiled_cache.clear()
    partitioned_modules.clear()
