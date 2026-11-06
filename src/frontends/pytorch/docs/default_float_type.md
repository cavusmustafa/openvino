# Narrow default float type

A model that PyTorch built with a bf16 or f16 default dtype arrives at the frontend with every activation and
weight in that type. OpenVINO's standard form is different: weights are compressed behind a decompression
`Convert`, activations are f32, and the execution precision is chosen by the `INFERENCE_PRECISION_HINT` property.
When a decoder reports a narrow *default float type*, the frontend converts the model into that standard form.
Today only the `torch.compile` vLLM path does this; other conversions are unaffected unless their decoder is
given the type explicitly.

## Decoder interface

* C++: `TorchDecoder::get_default_float_type()`, which returns `element::f32` unless overridden.
* Python: `TorchDecoderBase.get_default_float_type()`. `TorchFXPythonDecoder` and `TorchScriptPythonDecoder`
  take a `default_float_type` argument: a torch float dtype or an OpenVINO type.

Only the top-level decoder is read. The `torch.compile` backend passes `options["default_float_type"]` to the
decoder when the `canonical_precision` option is on, as it is under the
[vLLM preset](vllm_integration.md#the-vllm-preset).

## Conversion

`TranslateSession` records a non-f32 type in the model's rt_info as `default_float_type`. `FrontEnd::normalize()`
reads it back and, for a narrow type, runs `CanonicalizeFloatPrecision`, then marks the new Converts as
decompression and re-enables their constant folding, so the model is in the same state as one read from IR. The
rt_info entry stays on the model, so the caller can pick the inference precision from it.

[CanonicalizeFloatPrecision](../src/transforms/canonicalize_float_precision.hpp) keeps large weights narrow behind a
`Convert(f32)`, folds small Constants to f32, retypes the rest of the graph to f32 and gives the model inputs,
outputs and `__pa__` side-channel Parameters their original types. Models with subgraph operations are left
unchanged. Its header describes each step.

**Explicit rounding.** A cast to bf16/f16, or a narrow Range, keeps its rounding when the value reaches an
operation whose result jumps on a one-ulp difference: an integer or boolean output, or a rounding operation such
as Floor. The check follows widening casts and data movement, so `x.to(torch.bfloat16).float().to(torch.int32)`
still returns 2 for 1.999. Narrow *arithmetic* is not rounded: float math runs in f32 by design.
`is_rounding_sensitive` and `is_value_preserving` in
[canonicalize_float_precision.cpp](../src/transforms/canonicalize_float_precision.cpp) define the exact sets.

[MarkCompressedWeightsCast](../src/transforms/mark_compressed_weights_cast.hpp) runs in every `normalize()`,
independent of the default float type. It marks the trailing precision cast of a weight-decompression chain as
decompression, so the CPU plugin folds the chain into a compressed FullyConnected.

Tests are listed in [testing](tests.md#pytorch-frontend-tests).
