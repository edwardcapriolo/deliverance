# Gemma4 HF Text Test Coverage

Date: 2026-06-19

Source tests:

- `/ai-code/transformers/tests/models/gemma4/test_modeling_gemma4.py`
- `/ai-code/transformers/tests/causal_lm_tester.py`
- `/ai-code/transformers/tests/test_modeling_common.py`
- `/ai-code/transformers/tests/generation/test_utils.py`
- `/ai-code/transformers/tests/test_configuration_common.py`

Deliverance tests:

- `core/src/test/java/io/teknek/deliverance/model/gemma4/Gemma4HfTextModelPortedTest.java`
- reusable HF-style JUnit mixins under `core/src/test/java/io/teknek/deliverance/model/hf/`

## Reusable Mixins

Deliverance now ports feasible HF common tests through public JUnit 5 interfaces:

- `HfTextCommonTestAdapter`
- `HfConfigTesterMixinPort`
- `HfModelTesterMixinPort`
- `HfGenerationTesterMixinPort`
- `HfUnsupportedMixinPort`

Model tests implement the adapter by supplying a tiny checkpoint writer, model loader, config loader, sample token ids, and model-specific config round-trip checks. Gemma4 is the first adapter implementation.

## Active Gemma4-Specific Coverage

- HF `Gemma4TextModelTester` four-layer/shared-KV setup.
- HF `Gemma4TextModelTester` config overrides are represented in the tiny fixture:
  - `num_hidden_layers=4`.
  - `num_kv_shared_layers=2`.
  - layer types `sliding/full/sliding/full`.
  - `vocab_size_per_layer_input=99`.
  - `hidden_size_per_layer_input=16`.
  - `enable_moe_block=true`.
  - `moe_intermediate_size=16`.
  - `top_k_experts=2`.
  - `use_bidirectional_attention="vision"` by default, with a separate `"all"` test.
- HF `test_all_bidirectional_attention_uses_bidirectional_mask` equivalent.
- HF slow text-only integration equivalents:
  - text-only forward.
  - states sharing with and without cache.
  - generation beyond sliding window.
- Internal forward invariants used to make the HF equivalents actionable:
  - forward shape.
  - deterministic forward.
  - batch prefill vs token-by-token prefill.
  - decode vs cold replay.

## Active Common/Mixin Coverage

- `ConfigTester` feasible equivalents:
  - common config properties.
  - config JSON round-trip through the model-specific config shape.
  - save/reload config equivalent.
- `CausalLMModelTest.test_model` feasible equivalent:
  - tiny model forward shape and finite output.
- `ModelTesterMixin` feasible equivalents:
  - same checkpoint loaded twice produces identical output.
  - deterministic repeated forward.
  - past KV tensor shape/finite-value sanity.
- `GenerationTesterMixin` feasible equivalents:
  - continuation from cached KV matches cold replay.
  - embedding-only forward is explicitly rejected for Gemma4 PLE when token-derived per-layer inputs are unavailable.

## MoE Coverage

- HF Gemma4 text tester enables MoE. Deliverance Gemma4 now loads router/expert weights and routes through the Gemma4-specific MoE branch when `enable_moe_block=true`.
- `hfTextModelTesterMoeExpertsAffectForwardOutput` changes only MoE weights and expects output drift.
- `denseGemma4ConfigDoesNotRequireMoeWeights` verifies dense Gemma4 configs still load without router/expert weights.

## Disabled Directional Tests

- HF `test_generate_from_random_inputs_embeds` is empty in Gemma4 because upstream skips the inherited common test. Deliverance keeps it disabled but now includes a directional body based on HF `GenerationTesterMixin.test_generate_from_random_inputs_embeds`.
- HF RoPE scaling common tests remain disabled because upstream skips them for Gemma4's per-layer-type RoPE.
- HF SDPA/FlashAttention/torch-compile tests remain disabled with upstream skip reasons where HF skips them.

## Unsupported Or Not Text-Only

- HF CausalLM inherited classification/QA head tests: Deliverance Gemma4 exposes causal-LM generation, not sequence/token-classification or QA heads.
- HF `PipelineTesterMixin`: targets transformers pipelines; Deliverance does not implement HF pipelines.
- HF `TrainingTesterMixin`: requires autograd/training APIs; Deliverance Gemma4 tests are inference-only.
- HF `TensorParallelTesterMixin`: targets transformers `tp_plan`/device mesh APIs; Deliverance TP uses separate runtime and transport APIs.
- HF advanced generation tests for beam/sample/assistant/speculative/cache-implementation behavior require HF `generate` features not currently present in Deliverance generation.
- HF mutable torch module tests for resize/tie/offload/meta-device behavior do not map directly to Deliverance safetensors checkpoint loading.
- HF attention backend tests for eager/SDPA/Flash/Flex dispatch do not map directly to Deliverance's Java attention implementation.
- Gemma4 audio, image, video, processor, and multimodal feature tests are outside this text-only port.

## Remaining Work

- Validate Gemma4 MoE numerics against a Python HF reference dump for a tiny deterministic checkpoint.
- Continue expanding active common tests where Deliverance has matching APIs, especially save/load parity and generation API behavior.
- Convert unsupported areas from broad disabled documentation into narrower disabled directional tests when a concrete Deliverance API boundary exists.
- Reuse the public `io.teknek.deliverance.model.hf` mixins for other model families by adding model-specific tiny checkpoint adapters.
