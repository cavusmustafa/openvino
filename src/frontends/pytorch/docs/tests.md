# Testing the torch.compile and vLLM paths

| Suite | Location | Covers |
|---|---|---|
| vLLM tests | [tests/vllm_tests](../../../../tests/vllm_tests/) | vLLM + OpenVINO end to end, and unit tests of the plugin package |
| PyTorch frontend tests | [test_torch_frontend.py](../../../../tests/layer_tests/py_frontend_tests/test_torch_frontend.py) | Decoder `default_float_type`, `CanonicalizeFloatPrecision`, `MarkCompressedWeightsCast` |
| Transformation tests | `ov_transformations_tests` | `NormalizeVLLMRoPE` and its `vllm_model` gate |
| CPU plugin tests | `ov_cpu_unit_tests`, `ov_cpu_func_tests` | MLP and QKV projection fusions on the flattened `[tokens, hidden]` graphs this path produces |

## vLLM tests

The suite needs `vllm-cpu` and its own validated PyTorch, which can't be installed together with the other test
requirements, so it has its own requirements file,
[tests/requirements_vllm](../../../../tests/requirements_vllm). Use a separate virtual environment:

```sh
python3 -m venv venv_vllm && . venv_vllm/bin/activate
pip install -r tests/requirements_vllm
pip install <openvino wheel built from this tree>
OV_LAZY_KVCACHE=1 python -m pytest tests/vllm_tests -m precommit -v
```

* End-to-end tests download a small model from the Hugging Face Hub (`MODEL_ID` in `conftest.py`). Set
  `HF_HUB_OFFLINE=1` once it is cached.
* `OV_LAZY_KVCACHE=1` allocates the KV-cache pool lazily, which keeps the suite's memory use down.
* On multi-socket machines, pin the run to one NUMA node.
* `conftest.py` sets up vLLM's import-time environment and provides `new_openvino_llm()`, which builds a model
  with the compilation config the backend requires. Each test builds and deletes its own instance.

The tests fall into two groups:

* **End to end** (`test_run.py`, `test_dynamic_shapes.py`, `test_advanced_features.py`, `test_tiny_models.py`,
  the reload tests): OpenVINO output compared with vLLM eager.
* **Unit** (the rest): plugin logic on hand-built FX graphs or stand-in objects, with no model.

New tests that should run in CI need the `precommit` marker. Prefer unit tests where no model is needed. Tiny
random models give flat logits, so compare their tokens with the tie-aware rule of `test_tiny_models.py`.
CI runs the marked tests through [job_vllm_tests.yml](../../../../.github/workflows/job_vllm_tests.yml).

## PyTorch frontend tests

```sh
cd tests/layer_tests
python -m pytest py_frontend_tests/test_torch_frontend.py -k "float_precision or default_float_type or cast_is"
```

They need the usual layer-test requirements and no vLLM. Rounding cases of `CanonicalizeFloatPrecision` are
parametrized in `test_canonicalize_float_precision_keeps_rounding_before_sensitive_consumer`; add a new pattern
there with an input whose narrow rounding changes the integer result.
