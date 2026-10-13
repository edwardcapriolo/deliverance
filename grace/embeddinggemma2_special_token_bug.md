# Grace EmbeddingGemma 2 Special-Token Bug

## Defect

Grace resolves EmbeddingGemma 2's BOS, EOS, PAD, image, video, and audio token
IDs, but normal encoding does not insert BOS/EOS. `encode(text)` calls
`buildInputsWithSpecialTokens` when special tokens are enabled, while the base
implementation returns the input IDs unchanged. `GemmaTokenizer` does not
override that method, and Grace does not apply the tokenizer JSON post-processor
used by Transformers.

Transformers configures EmbeddingGemma 2 with both `add_bos_token=True` and
`add_eos_token=True`, producing:

```text
BOS + token_ids + EOS
```

With `add_special_tokens=False`, Transformers leaves the token IDs unchanged.

## Impact

EmbeddingGemma 2 receives a different sequence from the reference tokenizer.
This changes attention positions and embedding output, so model parity cannot
be established until this is fixed.

The fix must be configuration-driven. Gemma 4 and other models must not gain
BOS/EOS insertion merely because they use `GemmaTokenizer`.

## Reproduction

The regression test is:

```text
EmbeddingGemma2TokenizerLoadTest.defaultEncodingAddsBosAndEos
```

It currently observes token `818` as the first token instead of BOS ID `2`,
and the special-token opt-out test observes no two-token difference.

## Reference

- Transformers `tokenization_python.py:865-950`
- Transformers `tokenization_utils_tokenizers.py:626-650`
- Transformers `convert_embedding_gemma2_weights.py:877-878`
- Deliverance `PreTrainedTokenizerBase.encode`
- Deliverance `PreTrainedTokenizerBase.buildInputsWithSpecialTokens`
- Deliverance `GemmaTokenizer`

## Completion Criteria

- The focused regression test fails before the production fix.
- The test passes after the fix.
- `add_special_tokens=False` remains unchanged.
- Existing Grace tokenizer tests pass.
- Existing Gemma 4 and other model tokenizer behavior does not regress.
