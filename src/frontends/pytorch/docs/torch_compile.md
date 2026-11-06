# torch.compile backend internals

`torch.compile(model, backend="openvino")` is implemented in
[torchdynamo/](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/) on top of the FX decoder. The
user-facing options are documented in the
[torch.compile article](../../../../docs/articles_en/openvino-workflow/torch-compile.rst). This page covers the
internal flow and the parts the vLLM integration relies on.

## Flow

| Step | File | What happens |
|---|---|---|
| Entry | [backend.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/backend.py) | `openvino()` receives the dynamo graph and passes it to `fx_openvino`, through `aot_autograd` when that option is set. With `aot_autograd`, weights are folded into Constants; without it, `make_fx` applies the decompositions |
| Partitioning | [partition.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/partition.py), [op_support.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/op_support.py) | `CapabilityBasedPartitioner` groups supported nodes into `fused_*` submodules; unsupported nodes stay in PyTorch |
| Conversion | [compile.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/compile.py) | Each partition: `TorchFXPythonDecoder` -> `fe.convert()` -> input shapes -> `core.compile_model` |
| Execution | [execute.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/execute.py) | `OpenVINOGraphModule` runs the compiled partition, with caches of compiled models and infer requests |

**Weight freezing.** Dynamo passes the weights as graph inputs, so PyTorch's `replace_params_with_constants` finds
nothing to freeze. `_freeze_static_inputs` in backend.py freezes by `fw_metadata.static_input_indices` instead and
falls back to the upstream helper when those indices aren't available; its docstring explains why.

## Fallback

* A failure while partitioning or compiling (`fx_openvino`) falls back to `compile_fx` (inductor).
* A failure while executing a partition falls back to the partition's PyTorch graph, permanently for that
  partition.

The `no_fallback` option turns both into errors; it is resolved through the vLLM preset when that is present.

## Compiled-model caches

[execute.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/execute.py) keeps compiled models
per partition and input signature. Two additions avoid recompiling on dynamo re-traces:

* **Structural cache.** The key hashes the graph's structure, so it survives re-traces that only rename or
  reorder placeholders. Literal arguments and `options["model_id"]` are part of the key, so models with the same
  architecture don't share a compiled model. On by default only under the vLLM preset (`OV_STRUCTURAL_CACHE`
  overrides).
* **Shape-agnostic compile.** When every integer graph input can be rebuilt from a tensor dimension, one dynamic
  model serves all shapes and the key keeps only dtypes and ranks (`OV_SHAPE_AGNOSTIC_CACHE=0` disables it).

`execute.clear_caches()` drops these caches together with the vLLM binding caches, which are keyed by `id()` of
the dropped objects. Compiles made during vLLM's profile run are freed right after use
(`OV_EVICT_PROFILE_COMPILE=0` keeps them).

## Hooks for integrations

The generic files call the [vLLM package](vllm_integration.md) at fixed points, and each call is a no-op for
graphs and options without vLLM markers:

| Call site | Hook | Purpose |
|---|---|---|
| `op_support.OperatorSupport` | `vllm.maybe_register_pa_op` | Mark `ov_ext` attention operations supported |
| `Partitioner.make_partitions` | `vllm.maybe_rewrite_paged_attention`, `vllm.keep_paged_attention_partitioned` | Rewrite vLLM operations before partitioning; keep attention in OV after it |
| `compile.openvino_compile` | `compile_hooks.decoder_default_float_type` | Decoder `default_float_type`, see [narrow default float type](default_float_type.md) |
| | `compile_hooks.apply_post_convert` | `vllm_model` rt_info, register side-channel Parameters, fix Concat ranks |
| | `compile_hooks.apply_input_shapes` | Replace integer inputs with `ShapeOf` of their source tensor, set input shapes |
| | `compile_hooks.apply_post_config` | CPU config defaults, KV-cache Parameter types |
| `execute.openvino_execute` | `runtime_hooks.run_pa_infer` | Bind side-channel inputs and run inference |
