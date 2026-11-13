# Phase 3: Source And Test Analysis

## Status

Complete for operator review. No model implementation changes have been made.

## Transformers Source

Source files reviewed:

- `src/transformers/models/granite/configuration_granite.py`
- `src/transformers/models/granite/modular_granite.py`
- Generated `src/transformers/models/granite/modeling_granite.py`
- `tests/models/granite/test_modeling_granite.py`

The Granite implementation is structurally close to Llama:

```text
GraniteModel extends LlamaModel
GraniteDecoderLayer extends LlamaDecoderLayer
GraniteAttention extends LlamaAttention
GraniteForCausalLM extends LlamaForCausalLM
```

Granite-specific forward behavior:

- Embeddings are multiplied by `embedding_multiplier`.
- Attention scaling uses `attention_multiplier` instead of the usual default scale.
- Attention uses GQA with `num_attention_heads / num_key_value_heads` KV repetition.
- Residual attention and MLP branches are multiplied by `residual_multiplier`.
- Final logits are divided by `logits_scaling`.
- Projection bias is controlled by `attention_bias` and `mlp_bias`.
- RMSNorm computes variance in F32 and casts the result back to the input dtype.
- RoPE uses `rope_parameters` and the configured `rope_theta`.
- The model supports Transformers cache objects even though the config default has `use_cache=false`.

No new architectural mechanism was found in the Granite forward implementation.

## LLMON Source And Tests

Sources reviewed:

- `generative-computing/llmon/llmon/lang/README.md`
- `generative-computing/llmon/llmon/lang/chat/render.py`
- `generative-computing/llmon/llmon/lang/tests/test_chat_render.py`
- The checkpoint's `chat_template.jinja`

The Java implementation must reproduce the template behavior, including:

- `content` shorthand for an executed instruction.
- Structured `instruction`, `data`, and `requirement` items.
- Documents with optional titles.
- Tools flattened into addressable leaf blocks.
- Tool results rendered as untrusted `data` blocks.
- Two-character IDs drawn from the model's 75-character single-token ID alphabet.
- Optional pinned IDs and duplicate-ID validation.
- `exec` bindings that distinguish authoritative instructions from input data.
- Assistant tool-call rendering.
- EOS placement after assistant turns.
- `split_prompt_target` behavior.
- Stop tokens consisting of the tokenizer EOS token and `<|open_end|>`.

The LLMON repository provides `llmon/lang/tests/chat/render_golden.json` with exact deterministic examples, including:

- Plain single-user prompts with and without a generation prompt.
- Structured instruction/data/requirement fields with mixed `exec` flags.
- Content plus data.
- System messages across multiple turns.
- Top-level documents and tools.
- Message-level tools.
- Assistant tool calls and multiple tool results.
- Tool-call continuation cycles.
- Empty/null content and non-string tool results.
- Unknown roles and template leniency.

`test_chat_render.py` additionally tests:

- Exact golden output for deterministic random IDs.
- Unique-ID mode across 300 blocks and 20 seeds.
- Pinned IDs and rejection of malformed or repeated pins.
- Prompt/target splitting.
- Assistant tool-call body rendering.
- EOS defaults and alternate EOS tokens.
- Stop token IDs.
- Invalid conversation structure.

## Test Port Matrix

| Source coverage | Reckoning target | Status |
| --- | --- | --- |
| Granite config common tests | Granite typed config test | required |
| Granite config multiplier round-trip | Config serialization test | required |
| Granite tiny model shape test | Tiny Granite characterization test | required |
| Granite 3B logits reference | Granite Dead Reckoning baseline | required |
| Granite inherited generation tests | Granite generation IT | required |
| LLMON golden chat renders | Java renderer golden tests | required |
| Random and unique block IDs | Renderer ID tests | required |
| Pinned and invalid IDs | Renderer validation tests | required |
| Instruction/data/requirement binding | Structured prompt tests | required |
| Documents and tool flattening | Structured prompt tests | required |
| Tool result and assistant tool calls | Conversation continuation tests | required |
| Prompt split and EOS/stop IDs | Tokenization/generation tests | required |
| JSON/shorthand converter tests | Only if Java exposes those converter APIs | assess |

## Granite Source-Test Details

The Transformers Granite suite contains:

- Common configuration tests.
- Integer multiplier save/load round-trip for `embedding_multiplier` and `logits_scaling`.
- Tiny model construction and hidden-state shape checks.
- Generic generation and pipeline mixins.
- Slow BF16/FP16 logits integration checks for an older Granite 3B checkpoint.

The slow integration logits checks are useful as a pattern, but they are not a direct oracle for this exact LLMON 3B checkpoint. The model-specific Dead Reckoning baseline will be the exact checkpoint oracle.

Sentence Transformers source/tests are not applicable to this text-generation checkpoint.

## Main Implementation Risk

The highest-risk first component is not the transformer block. It is a Java LLMON prompt model/renderer that produces the same string and token IDs as the bundled Transformers template.

## Required First Deliverable

Build a user-facing Java LLMON DSL/input model so callers do not construct JSON objects or raw LLMON markup by hand.

The DSL must represent:

- Instructions.
- Data.
- Requirements.
- Documents.
- Tools and tool results.
- `exec` bindings.
- Optional stable block IDs.

The DSL renders the exact LLMON string required by the model. It will be usable by Java callers and by the raw `/v1/completions` client path.

## Required LLMON Tests

- DSL-to-rendered-string golden tests.
- Structured instruction/data/requirement tests.
- `exec: true` and `exec: false` binding tests.
- Document and tool flattening tests.
- Tool-call and tool-result continuation tests.
- Stable ID and random-ID tests.
- Invalid structure and duplicate-ID tests.
- Exact tokenizer input-ID tests.
- Web raw-completions integration test using a rendered DSL prompt.

The first implementation should therefore target the renderer and raw completions path before model generation wiring.

## First Slice Result

Implemented the first Java DSL slice at:

```text
core/src/main/java/io/teknek/reckoning/safetensors/prompt/llmon/LlmOnPrompt.java
```

It currently supports typed system, user, instruction, data, requirement, document, tool, assistant-tool-call, and tool-result messages, deterministic valid IDs, pinned IDs, execution bindings, JSON/`JsonNode` values, and raw completion prompt rendering.

Focused characterization tests are at:

```text
core/src/test/java/io/teknek/reckoning/safetensors/prompt/llmon/LlmOnPromptTest.java
```

Validation:

```text
Tests run: 25, Failures: 0, Errors: 0
```

The renderer now has named characterization coverage for the repository's structured, plain, multi-turn, document, tool, tool-call cycle, null-content, and generation-prompt examples. The tests also cover every character in the Granite ID alphabet, malformed IDs, and duplicate pinned IDs, with explicit sunny-day and rainy-day assertions. Exact byte-for-byte golden parity for random-seeded IDs, prompt splitting, stop-token IDs, unknown-role leniency, and the raw web completion path remains outstanding.

## Next Step After Approval

Build the Java LLMON input model/renderer characterization tests and compare its output against the checkpoint's Transformers template behavior.
