# Custom operations of the vLLM path

The [vLLM rewrites](vllm_integration.md#graph-rewrites) replace vLLM's attention and Gated DeltaNet operations
with two `ov_ext` custom operations that the FX frontend translates directly. Both are registered only in the FX
operation table ([op_table.cpp](../src/op_table.cpp), namespace `op::ext`). The GDN translator produces
internal operations (`ov::op::internal`), so a converted model that contains them can't always be serialized to
IR.

| FX operation | Python definition | Translator | Produces |
|---|---|---|---|
| `ov_ext.paged_attention.default` | [paged_attention.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/vllm/paged_attention.py) | [op/paged_attention.cpp](../src/op/paged_attention.cpp) | `PagedAttentionExtension` |
| `ov_ext.gdn_attention.default` | [gated_delta_net.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/vllm/gated_delta_net.py) | [op/gdn_attention.cpp](../src/op/gdn_attention.cpp) | `PagedCausalConv1D`, `PagedGatedDeltaNet` |

The Python operations are `torch.library.custom_op`s with a fake (meta) implementation for tracing. Their eager
bodies call vLLM's own operations; they only run when a call isn't converted, which the vLLM path prevents for
attention (see [KV-cache ownership](vllm_integration.md#kv-cache-ownership)).

## Side-channel Parameters

Neither operation receives its cache, block tables or sequence metadata as FX inputs. The translators create
Parameters for them instead, named by a fixed convention:

* `__pa__<layer>__<field>`: one per attention layer;
* `__pa__shared__<field>`: one per conversion, reused by every layer;
* `__pa__gdn__<layer>__<field>` and `__pa__gdn__shared__<field>`: the same for GDN.

The names are the contract with the binder in
[side_channel.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/vllm/side_channel.py), which
resolves them at every step ([sources](vllm_integration.md#side-channel)); rename a field in both places
together. The Parameters aren't part of the decoded graph: they are registered with
`NodeContext::add_external_parameter`, have no numeric tensor name, and keep their types through
`CanonicalizeFloatPrecision`.

## `ov_ext::paged_attention`

```
paged_attention(query, key, value, layer_name: str, scale: float, kv_sharing_target: str) -> Tensor
```

| PagedAttentionExtension input | Built from |
|---|---|
| `query`, `key`, `value` | FX inputs, flattened to rank 2 (`[tokens, heads * head_size]`) |
| `key_cache`, `value_cache` | `__pa__<layer>__key_cache` / `value_cache`; the backend retypes them to `KV_CACHE_PRECISION` |
| `past_lens`, `max_context_len` | Derived in the graph from `__pa__shared__seq_lens` and `query_start_loc` |
| `subsequence_begins` | `__pa__shared__query_start_loc` |
| `block_indices`, `block_indices_begins` | Per layer: KV-cache groups have their own block tables |
| `scale` | The `scale` argument; when it is 0, `1/sqrt(head_size)` |
| `sliding_window` | `__pa__<layer>__sliding_window`, so hybrid models can mix sliding and full attention |
| Every other input | Empty or zero Constants: no ALiBi, sinks, rotation, xattention or adaptive RKV |

* The output is converted back to the type of `query`.
* `layer_name` must constant-fold. If it doesn't, each call gets its own `unknown_layer_N` tag, so that layers
  with different cache shapes don't share one Parameter.
* The per-layer K/V head geometry is attached as rt_info of the operation, so `ConvertPagedAttnInputs` can size
  each layer's cache independently.
* **KV sharing.** A layer with a non-empty `kv_sharing_target` reuses the K/V its target built in the same
  conversion. vLLM hands such a layer raw K/V, and PagedAttention always writes, so the write must be
  idempotent. The backend refuses graphs where the two layers are in different partitions.
* Features the operation has no input for (logits soft-capping, non-decoder attention) or that are passed empty
  (ALiBi, sinks) are refused at model load, see [KV-cache ownership](vllm_integration.md#kv-cache-ownership).
* `OV_DBG_PA_TRANS=1` prints the inputs and the emitted operation for every call.

## `ov_ext::gdn_attention`

```
gdn_attention(mixed_qkv, b, a, conv_weight, conv_bias, A_log, dt_bias,
              layer_name: str, num_k_heads: int, num_v_heads: int, head_k_dim: int, head_v_dim: int) -> Tensor
```

The translator builds the Gated DeltaNet layer of Qwen3.5-style models:

* `PagedCausalConv1D` over `mixed_qkv`, followed by SiLU, with its state in `__pa__gdn__<layer>__gdn_conv_state`;
* `PagedGatedDeltaNet` on the q, k and v heads split from that result, with q/k L2 normalization and its state in
  `__pa__gdn__<layer>__gdn_recurrent_state`. Its gate and beta follow vLLM's `fused_gdn_gating`.

Both kernels run in f32 with f32 states, whatever the activation type, and the result is converted back to the
type of `mixed_qkv`. Sequence metadata is shared by all GDN layers. `conv_weight` must be a static rank-2 tensor
and `layer_name` a constant string.

## Adding an `ov_ext` operation

1. Define the `torch.library.custom_op` with a fake implementation and an eager body that is correct on its own.
2. Rewrite the vLLM call site into it from `maybe_rewrite_paged_attention`, behind its own option.
3. Mark it supported in `maybe_register_pa_op`, so the partitioner keeps it in OV.
4. Add the translator under `op::ext` and register it in `get_supported_ops_fx()`.
5. Bind any side-channel Parameters in `side_channel.py`.
6. Add an end-to-end test that compares against vLLM eager, see [testing](tests.md).
