# Config Field Analysis

Source: `ibm-granite/granite-4.1-3b-llmon-preview/config.json`

| Field | Type | Value | Status |
| --- | --- | --- | --- |
| `architectures` | array<string> | `GraniteForCausalLM` | known metadata; model mapping required |
| `attention_bias` | boolean | `false` | new/unconfirmed |
| `attention_dropout` | number | `0.0` | existing Granite concept; source verification required |
| `attention_multiplier` | number | `0.015625` | existing generic field; exact application required |
| `bos_token_id` | integer | `100257` | known |
| `dtype` | string | `bfloat16` | new/unconfirmed config alias |
| `embedding_multiplier` | number | `12.0` | existing generic field; exact application required |
| `eos_token_id` | integer | `100257` | known |
| `hidden_act` | string | `silu` | known activation |
| `hidden_size` | integer | `2560` | known |
| `initializer_range` | number | `0.1` | likely training metadata; source verification required |
| `intermediate_size` | integer | `8192` | known |
| `logits_scaling` | number | `10.0` | existing Granite concept; exact application required |
| `max_position_embeddings` | integer | `131072` | known |
| `mlp_bias` | boolean | `false` | new/unconfirmed |
| `model_type` | string | `granite` | new model dispatch verification required |
| `num_attention_heads` | integer | `40` | known |
| `num_hidden_layers` | integer | `40` | known |
| `num_key_value_heads` | integer | `8` | known GQA field |
| `pad_token_id` | integer | `100256` | tokenizer/config metadata |
| `residual_multiplier` | number | `0.22` | existing generic field; exact application required |
| `rms_norm_eps` | number | `1e-05` | known |
| `rope_parameters` | object | `rope_theta=10000000`, `rope_type=default` | source mapping required |
| `rope_theta` | integer | `10000000` | duplicate alias; precedence required |
| `tie_word_embeddings` | boolean | `false` | output-head behavior required |
| `transformers_version` | string | `5.0.1.dev0` | source provenance |
| `use_cache` | boolean | `false` | unusual generation setting; semantics required |
| `vocab_size` | integer | `100352` | known |
| `torch_dtype` | string | `bfloat16` | metadata/runtime mapping required |

No LLMON-specific config field appears. LLMON behavior is represented by the tokenizer vocabulary and chat template plus post-training behavior.
