# Qwen3 Support

Qwen3 is the next Deliverance text-generation target after Qwen2. The goal is to make Qwen3 the first reliable
thinking-capable model family in Deliverance, without assuming the older Qwen2 implementation is correct.

## Current Status

Implemented:

- `Qwen3Config` parsing for Hugging Face `model_type=qwen3` configs.
- `Qwen3Model` registered in `ModelSupport`.
- Qwen3-specific attention hook for per-head Q/K RMSNorm before RoPE.
- HF-style tiny checkpoint tests using generated safetensors weights.
- Tokenizer-only smoke coverage for `Qwen/Qwen3-0.6B`.
- Dense real-model smoke test for `Qwen/Qwen3-0.6B`.

Verified focused command:

```sh
mvn -q -pl core -am -Dtest=Qwen3ConfigTest,Qwen3HfTextModelPortedTest,Qwen3SmallIT -Dsurefire.failIfNoSpecifiedTests=false test
```

Observed real-model smoke output:

```text
QWEN3_06B_SHORT=1<|im_end|>
QWEN3_06B_THINKING=<think>\nOkay, so the question is 1 plus 1. Hmm,
```

The `0.6B` model is a bring-up target, not the final quality bar for thinking or tool use.

## Architecture Notes

The implementation was compared against Hugging Face:

- `transformers/src/transformers/models/qwen3/configuration_qwen3.py`
- `transformers/src/transformers/models/qwen3/modeling_qwen3.py`
- `transformers/tests/models/qwen3/test_modeling_qwen3.py`

Important Qwen3 details:

- Uses decoder-only causal LM architecture.
- Uses Qwen3 RMSNorm.
- Uses GQA when `num_key_value_heads < num_attention_heads`.
- Applies `q_norm` and `k_norm` after Q/K projection and before RoPE.
- Uses standard MLP form: `down(act(gate(x)) * up(x))`.
- Supports optional sliding-window layer typing in config.
- Uses Qwen tokenizer/chat template family with `<think>` / `</think>` tokens for thinking output.

Deliverance adds a family-specific hook in `CausalSelfAttention`:

```java
protected void normalizeQueryKey(AbstractTensor queryBatch, AbstractTensor keyBatch)
```

Qwen3 overrides this hook to apply per-head Q/K RMSNorm in the same location as Hugging Face.

## Tests

Qwen3 focused tests:

- `Qwen3ConfigTest`
- `Qwen3HfTextModelPortedTest`
- `Qwen3SmallIT`
- `HfQwen3TokenizerSmokeTest`

`Qwen3HfTextModelPortedTest` follows the same pattern as the Gemma4 HF ported tests:

```java
public class Qwen3HfTextModelPortedTest implements
        HfConfigTesterMixinPort,
        HfModelTesterMixinPort,
        HfGenerationTesterMixinPort,
        HfUnsupportedMixinPort
```

The tiny tests use generated safetensors checkpoints rather than mocks for model internals.

## Current Limitations

- Qwen3 quantized checkpoints are not yet generated or validated.
- Larger Qwen3 thinking models have not yet been validated.
- Tool-call quality has not yet been validated on a capable Qwen3 checkpoint.
- Qwen3 prompt rendering currently uses a narrow fallback because the upstream template uses Python-Jinja syntax that
  Jinjava does not parse, including `messages[::-1]`.
- `Qwen/Qwen3-0.6B` is useful for fast bring-up, but it should not be treated as evidence that thinking/tool quality is
  good enough.

## Next Steps

1. Quantize `Qwen/Qwen3-0.6B` locally and compare dense vs Q4 first-token/top-k behavior.
2. Bring up a larger Qwen3 thinking model after the 0.6B path is stable.
3. Add Qwen3 tool-call prompt and parser tests.
4. Replace the Qwen3 template fallback with broader Python-Jinja compatibility or an explicit Qwen3 renderer if needed.
