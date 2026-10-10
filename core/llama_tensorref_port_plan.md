# Llama TensorRef/KV2 Port Plan

## 1. Baseline

- Run existing Llama JQ4 prompt/generation and logits tests.
- Capture provider, TTFT, decode tok/s, and stage profiles.
- Confirm model cache, tokenizer, and prompt-template behavior.

## 2. Scope Boundary

- This slice targets the concrete Llama model family only.
- Do not change inherited behavior for Qwen2, Mistral, Mixtral, Gemma3, Gemma4, Nemotron, or other Llama subclasses.
- Do not make `LlamaModel`'s new TensorRef behavior implicitly apply to subclasses.
- Use an explicit concrete-Llama opt-in or an exact Llama configuration guard before enabling TensorRef/KV2.
- Add scope assertions proving each excluded subclass remains on its current execution path.

## 3. Static Port

- Add `LlamaModel.loadTransformerBlockWeights2()`.
- Map each layer to `CausalSelfAttention2`, `MLPBlock2`, `RmsNorm2`, and `TransformerBlock2`.
- Load all weights through `TensorRef` and `loadRef`.
- Preserve Llama GQA, RoPE, residual, norm ordering, activation, and tied-output-head behavior.
- Enable `usesTensorRefExecution()` and `usesKvCache2Generation()` only after the new block path exists.
- Do not add a production fallback or dual execution switch.

## 4. Characterization

- Add tiny Llama legacy-vs-TensorRef stage parity tests.
- Compare embeddings, Q/K/V projections, RoPE, attention output, residuals, norms, MLP stages, layer output, and logits slices.
- Exercise F32/BF16 and Q4 where existing test infrastructure supports them.
- Strengthen `Llama32ThreeBPromptIT` into a real output acceptance suite:
  - factual question with an asserted answer;
  - constrained Java generation with asserted symbols and method names;
  - short conversational response with nonblank, non-garbled output;
  - repeated-token and raw special-token rejection checks.
- Keep the tests in `Llama32ThreeBSuite` so the model is loaded once in the IDE and profiler output remains available.

## 5. Integration

- Add Llama JQ4 real-model generation smoke coverage.
- Validate KV2 prefix reuse with repeated prompts.
- Record TTFT, prefill tok/s, decode tok/s, and provider counters.

## 6. Cleanup

- Remove obsolete Llama legacy production block loading once parity passes.
- Retain legacy code only where another family still depends on it.
- Run focused Llama tests, then the broader model-family suite.

## Acceptance

- Tiny stage parity passes within defined numerical tolerances.
- Real Llama JQ4 generation produces coherent output.
- KV2 prefix hits are observed on repeated prompts.
- No legacy Llama execution is selected in production.
