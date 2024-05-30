.. {#pytorch_2_0_torch_compile}

PyTorch Deployment via "torch.compile"
======================================



The ``torch.compile`` feature enables you to use OpenVINO for PyTorch-native applications.
It speeds up PyTorch code by JIT-compiling it into optimized kernels.
By default, Torch code runs in eager-mode, but with the use of ``torch.compile`` it goes through the following steps:

1. **Graph acquisition** - the model is rewritten as blocks of subgraphs that are either:

   * compiled by TorchDynamo and "flattened",
   * falling back to the eager-mode, due to unsupported Python constructs (like control-flow code).

2. **Graph lowering** - all PyTorch operations are decomposed into their constituent kernels specific to the chosen backend.
3. **Graph compilation** - the kernels call their corresponding low-level device-specific operations.



How to Use
####################

To use ``torch.compile``, you need to define the ``openvino`` backend.
This way Torch FX subgraphs will be directly converted to OpenVINO representation without
any additional PyTorch-based tracing/scripting.


.. code-block:: sh

   ...
   model = torch.compile(model, backend='openvino')
   ...

For OpenVINO versions below 2024.1, an additional import was needed to use ``openvino`` backend. However, OpenVINO package is now configured with `torch_dynamo_backends entrypoint <https://pytorch.org/docs/stable/torch.compiler_custom_backends.html#registering-custom-backends>`__. The additional import is not needed, if OpenVINO is installed as a pip package. For other install channels such as conda, import statement can be used as below.

.. code-block:: sh

   import openvino.torch

   ...
   model = torch.compile(model, backend='openvino')
   ...


Execution diagram:

.. image:: ../assets/images/torch_compile_backend_openvino.svg
   :width: 992px
   :height: 720px
   :scale: 60%
   :align: center

Options
++++++++++++++++++++

It is possible to use additional arguments for ``torch.compile`` to set the backend device,
enable model caching, set the cache directory etc. You can use a dictionary of the available options:

* ``device`` - enables selecting a specific hardware device to run the application.
  By default, the OpenVINO backend for ``torch.compile`` runs PyTorch applications
  on CPU. If you set this variable to ``GPU.0``, for example, the application will
  use the integrated graphics processor instead.
* ``aot_autograd`` - enables aot_autograd graph capture. The aot_autograd graph capture
  is needed to enable dynamic shapes or to finetune a model. For models with dynamic
  shapes, it is recommended to set this option to ``True``. By default, aot_autograd 
  is set to ``False``.
* ``model_caching`` - enables saving the optimized model files to a hard drive,
  after the first application run. This makes them available for the following
  application executions, reducing the first-inference latency. By default, this
  variable is set to ``False``. Set it to ``True`` to enable caching.
* ``cache_dir`` - enables defining a custom directory for the model files (if
  ``model_caching`` is set to ``True``). By default, the OpenVINO IR is saved
  in the cache sub-directory, created in the application's root directory.
* ``decompositions`` - enables defining additional operator decompositions. By 
  default, this is an empty list. For example, to add a decomposition for 
  an operator ``my_op``, add ``'decompositions': [torch.ops.aten.my_op.default]``
  to the options. 
* ``disabled_ops`` - enables specifying operators that can be disabled from
  openvino execution and make it fall back to native PyTorch runtime. For 
  example, to disable an operator ``my_op`` from OpenVINO execution, add 
  ``'disabled_ops': [torch.ops.aten.my_op.default]`` to the options. By 
  default, this is an empty list.
* ``config`` - enables passing any OpenVINO configuration option as a dictionary
  to this variable. For details on the various options, refer to the
  :ref:`OpenVINO Advanced Features <openvino-advanced-features>`.

See the example below for details:

.. code-block:: python

   model = torch.compile(model, backend="openvino", options = {"device" : "CPU", "model_caching" : True, "cache_dir": "./model_cache"})

You can also set OpenVINO specific configuration options by adding them as a dictionary under ``config`` key in ``options``:

.. code-block:: python

   opts = {"device" : "CPU", "config" : {"PERFORMANCE_HINT" : "LATENCY"}}
   model = torch.compile(model, backend="openvino", options=opts)


Windows support
+++++++++++++++++++++

Currently, PyTorch does not support ``torch.compile`` feature on Windows officially. However, it can be accessed by running
the below instructions:

1. Install the PyTorch nightly wheel file - `2.1.0.dev20230713 <https://download.pytorch.org/whl/nightly/cpu/torch-2.1.0.dev20230713%2Bcpu-cp38-cp38-win_amd64.whl>`__ ,
2. Update the file at ``<python_env_root>/Lib/site-packages/torch/_dynamo/eval_frames.py``
3. Find the function called ``check_if_dynamo_supported()``:

   .. code-block:: console

      def check_if_dynamo_supported():
          if sys.platform == "win32":
              raise RuntimeError("Windows not yet supported for torch.compile")
          if sys.version_info >= (3, 11):
              raise RuntimeError("Python 3.11+ not yet supported for torch.compile")

4. Put in comments the first two lines in this function, so it looks like this:

   .. code-block:: console

      def check_if_dynamo_supported():
       #if sys.platform == "win32":
       #    raise RuntimeError("Windows not yet supported for torch.compile")
       if sys.version_info >= (3, 11):
           `raise RuntimeError("Python 3.11+ not yet supported for torch.compile")

Support for PyTorch 2 export quantization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PyTorch 2 export quantization is supported by OpenVINO backend in ``torch.compile``. This feature can be accessed be following the steps below:

1. Use ``torch._dynamo.export`` function to trace the model into an FX graph of flattened ATen operators.

   .. code-block:: python

      import torch
      exported_graph_module, guards = torch._dynamo.export(
          model,
          input_tensor,
          pre_dispatch=True,
          aten_graph=True,
      )

2. Initialize and prepare the quantizer. At this time, torch.compile with OpenVINO backend is only verified with ``X86InductorQuantizer``.

   .. code-block:: python

      import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
      from torch.ao.quantization.quantize_pt2e import prepare_pt2e
      quantizer = xiq.X86InductorQuantizer()
      operator_config = xiq.get_default_x86_inductor_quantization_config()
      quantizer.set_global(operator_config)
      prepared_graph_module = prepare_pt2e(exported_graph_module, quantizer)

3. Quantize the model and move the quantized model to eval mode. To be able to benefit from the optimizations in OpenVINO backend, constant folding in should be disabled in quantization when using ``torch.compile``. As provided below, this can be done passing ``fold_quantize=False`` parameter into the ``convert_pt2e`` function.

   .. code-block:: python

      from torch.ao.quantization.quantize_pt2e import convert_pt2e
      converted_graph_module = convert_pt2e(prepared_graph_module, fold_quantize=False)
      torch.ao.quantization.move_exported_model_to_eval(converted_graph_module)

4. Set torch.compile backend as OpenVINO and execute the model.

   .. code-block:: python

      ov_optimized_model_int8 = torch.compile(converted_graph_module, backend="openvino")
      ov_optimized_model_int8(example_inputs[0])

Torchserve Integration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Support for Automatic1111 Stable Diffusion WebUI
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Automatic1111 Stable Diffusion WebUI is an open-source repository that hosts a browser-based interface for the Stable Diffusion
based image generation. It allows users to create realistic and creative images from text prompts.
Stable Diffusion WebUI is supported on Intel CPUs, Intel integrated GPUs, and Intel discrete GPUs by leveraging OpenVINO
``torch.compile`` capability. Detailed instructions are available in
`Stable Diffusion WebUI repository. <https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon>`__


Architecture
#################

The ``torch.compile`` feature is part of PyTorch 2.0, and is based on:

* **TorchDynamo** - a Python-level JIT that hooks into the frame evaluation API in CPython,
  (PEP 523) to dynamically modify Python bytecode right before it is executed (PyTorch operators
  that cannot be extracted to FX graph are executed in the native Python environment).
  It maintains the eager-mode capabilities using
  `Guards <https://pytorch.org/docs/stable/dynamo/guards-overview.html>`__ to ensure the
  generated graphs are valid.

* **AOTAutograd** - generates the backward graph corresponding to the forward graph captured by TorchDynamo.
* **PrimTorch** - decomposes complicated PyTorch operations into simpler and more elementary ops.
* **TorchInductor** - a deep learning compiler that generates fast code for multiple accelerators and backends.


When the PyTorch module is wrapped with ``torch.compile``, TorchDynamo traces the module and
rewrites Python bytecode to extract sequences of PyTorch operations into an FX Graph,
which can be optimized by the OpenVINO backend. The Torch FX graphs are first converted to
inlined FX graphs and the graph partitioning module traverses inlined FX graph to identify
operators supported by OpenVINO.

All the supported operators are clustered into OpenVINO submodules, converted to the OpenVINO
graph using OpenVINO's PyTorch decoder, and executed in an optimized manner using OpenVINO runtime.
All unsupported operators fall back to the native PyTorch runtime on CPU. If the subgraph
fails during OpenVINO conversion, the subgraph falls back to PyTorch's default inductor backend.



Additional Resources
############################

* `PyTorch 2.0 documentation <https://pytorch.org/docs/stable/index.html>`_

