# Phase 1: Documentation And Model Selection

## Status

Complete for operator review. Phase 2 is not started.

## Selected Checkpoint

`ibm-granite/granite-4.1-3b-llmon-preview`

The 3B instruction-tuned LLMON checkpoint is preferred for the first implementation because it has the same LLMON behavior and Granite architecture as the 8B target while requiring substantially less storage and CPU time.

## Config Summary

- Architecture: `GraniteForCausalLM`
- Model type: `granite`
- 40 layers
- Hidden size: `2560`
- Intermediate size: `8192`
- Attention heads: `40`
- KV heads: `8`
- Context: `131072`
- Configured dtype: BF16
- Vocabulary: `100352`
- BF16 parameters: approximately 3.66B

## LLMON Summary

The model card and LLMON materials describe LLMON as an LLM-native markup and post-training format.

The model ships a chat template that renders structured instructions, data, requirements, documents, tools, tool results, and execution pointers using reserved tokens.

The transformer architecture is expected to remain Granite 4.1; LLMON is learned prompt and post-training behavior.

## Red Test Evidence

The opt-in `Granite41LlmonPreviewPortIT` was run with the `large-model` exclusion cleared.

The test successfully downloaded the complete checkpoint and supporting files into the Reckoning model cache.

It then failed at the intended first unsupported stage:

```text
java.lang.IllegalArgumentException: GRANITE not found in AutoModelForCausaLm
```

This confirms that model acquisition works and the next implementation gate is Granite config/model dispatch.

## Source Material Reviewed

- Model card and exact config.
- LLMON repository and language documentation.
- LLMON language paper.
- LLMON post-training paper.
- IBM Granite technical blog.

## Known Risks

- `rope_parameters` and duplicate `rope_theta` precedence.
- Exact application of attention, embedding, residual, and logits multipliers.
- `attention_bias=false` and `mlp_bias=false` weight selection.
- Meaning of `use_cache=false` for generation.
- Exact tokenizer special-token and chat-template behavior.

## Next Step After Approval

Download and inventory the complete 3B model repository using Java, then inspect its tokenizer, chat template, checkpoint tensors, and generation configuration.

## Port Gate

The opt-in red integration test is:

```text
core/src/test/java/io/teknek/reckoning/integration/granite/Granite41LlmonPreviewPortIT.java
```

It downloads the exact checkpoint, verifies `config.json` exists, and then attempts the real Reckoning model loader.

It is tagged `large-model` and `model-port-red`; it is expected to fail until the Granite config and model dispatch are implemented.
