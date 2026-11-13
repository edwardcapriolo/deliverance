# Config Field Analysis

Source: `ibm-granite/granite-4.1-8b-llmon-preview/config.json`

This is the initial inventory. `new to Reckoning` means the field is not yet confirmed as mapped by the generic `Config` or an existing compatible model config.

| Field | Type | Value | Reckoning status | Runtime question |
| --- | --- | --- | --- | --- |
| `architectures` | array<string> | `GraniteForCausalLM` | known metadata; model mapping required | Which target model class owns this architecture? |
| `attention_bias` | boolean | `false` | new/unconfirmed | Are Q/K/V/O projection biases absent? |
| `attention_dropout` | number | `0.0` | existing in Granite-specific config; not confirmed for this model | Is it inference-only metadata? |
| `attention_multiplier` | number | `0.0078125` | existing generic field | Where is it applied in the attention score path? |
| `bos_token_id` | integer | `100257` | existing generic field | Prompt construction and generation behavior. |
| `dtype` | string | `bfloat16` | new/unconfirmed config alias | Is this Transformers metadata or runtime dtype selection? |
| `embedding_multiplier` | number | `12.0` | existing generic field | Must embedding output be scaled before the first block? |
| `eos_token_id` | integer | `100257` | existing generic field | Confirm scalar EOS handling. |
| `hidden_act` | string | `silu` | existing activation mapping | Confirm exact MLP activation. |
| `hidden_size` | integer | `4096` | existing generic field | Embedding and residual width. |
| `initializer_range` | number | `0.1` | new/unconfirmed inference relevance | Likely training metadata; verify no runtime use. |
| `intermediate_size` | integer | `12800` | existing generic field | MLP intermediate width. |
| `logits_scaling` | number | `16.0` | existing Granite-specific field | Confirm final logits scaling location and direction. |
| `max_position_embeddings` | integer | `131072` | existing generic field | Context limit and RoPE range. |
| `mlp_bias` | boolean | `false` | new/unconfirmed | Confirm all MLP projections are bias-free. |
| `model_type` | string | `granite` | new model dispatch key for this checkpoint | Must map to the correct Granite implementation. |
| `num_attention_heads` | integer | `32` | existing generic field | Query head count. |
| `num_hidden_layers` | integer | `40` | existing generic field | Transformer block count. |
| `num_key_value_heads` | integer | `8` | existing generic field | GQA with four query heads per KV head. |
| `pad_token_id` | integer | `100256` | existing tokenizer/config metadata | Padding behavior in prompt encoding. |
| `residual_multiplier` | number | `0.22` | existing generic field | Confirm residual scaling at each block branch. |
| `rms_norm_eps` | number | `1e-05` | existing generic field | RMS norm epsilon. |
| `rope_parameters` | object | `rope_theta=10000000`, `rope_type=default` | existing map concept; Granite mapping unconfirmed | Confirm Transformers RoPE resolution and application. |
| `rope_theta` | integer | `10000000` | legacy/duplicate alias; precedence unconfirmed | Determine whether `rope_parameters` wins. |
| `tie_word_embeddings` | boolean | `false` | existing model metadata concept; runtime mapping unconfirmed | Output head must not alias input embeddings. |
| `transformers_version` | string | `5.0.1.dev0` | source provenance | Pin the upstream implementation version. |
| `use_cache` | boolean | `false` | new/unusual generation setting | Determine whether this is a model default or an LLMON/inference constraint. |
| `vocab_size` | integer | `100352` | existing generic field | Embedding and logits width. |
| `torch_dtype` | string | `bfloat16` | existing metadata concept; runtime mapping unconfirmed | Confirm stored and working dtype behavior. |

## Initial New Or Unconfirmed Fields

- `attention_bias`
- `dtype`
- `initializer_range`
- `mlp_bias`
- `model_type=granite` dispatch for this exact architecture
- `logits_scaling` application semantics
- `rope_parameters` precedence versus `rope_theta`
- `tie_word_embeddings`
- `use_cache=false`
- `torch_dtype`

These fields require the upstream Transformers source/config class before a target `Config` class is finalized.

## LLMON Finding

No LLMON-specific field appears in `config.json`.

LLMON behavior is represented by the tokenizer vocabulary and `chat_template.jinja`, plus post-training behavior. It is not currently evidence of a new transformer computation block.
