# Generation Mixin Port

This document tracks the line-by-line port of Hugging Face generation concepts into Deliverance. The goal is not to copy
Python implementation style. The goal is to model the same generation concepts with explicit Java types, production-grade
ownership rules, and no untyped kwargs bag.

## Ground Rules

* Work method by method and line by line.
* Do not jump ahead to Gemma4 symptoms or speculative fixes.
* Avoid raw strings when the domain is known.
* Avoid dynamic maps for internal model execution state.
* Add typed fields for supported concepts.
* Explicitly leave unsupported concepts out rather than carrying a generic escape hatch.
* Implement only when requested; otherwise keep notes/planning separate.

## Completed So Far

### `adjust_generation_fn(...)`

Current understanding:

* It sets or loads generation configuration.
* It can replace `self.generate` with trusted remote custom generation code.
* It carries Hugging Face auto-class and pipeline context that Deliverance does not currently have.

Deliverance decision:

* Do not support remote Python code execution.
* If a future API exposes `trust_remote_code`, it should fail explicitly.
* Custom generation, if needed, should be a Java strategy/interface, not downloaded Python.

No implementation was kept for this method.

### `prepare_inputs_for_generation(...)`

Implemented first typed slice in `AbstractModel`:

```java
protected GenerationStepInputs prepareInputsForGeneration(...)
```

Supporting types added:

* `GenerationStepInputs`
* `GenerationInputNames`
* `GenerationInputKey`
* `GenerationInputPreparer`
* `PastKeyValues`
* `ModelInputName`

Config/runtime support added:

* `Config.isEncoderDecoder`, defaulting to `false`
* `AbstractModel.getMainInputName()`, defaulting to `ModelInputName.INPUT_IDS`

Implemented behavior:

* decoder-only vs encoder-decoder input names
* enum-backed generation input keys instead of strings
* `inputIds` suffix slicing by `nextSequenceLength`
* `inputsEmbeds` path for decoder-only first iteration
* full `inputsEmbeds` path when `nextSequenceLength == null`
* `attentionMask` slicing to prepared sequence length
* `positionIds` slicing to prepared sequence length
* `tokenTypeIds` slicing to prepared sequence length
* `mmTokenTypeIds` slicing to prepared sequence length
* `encoderAttentionMask` retention for encoder-decoder models
* decoder-only models ignore `encoderAttentionMask`
* no kwargs escape hatch
* no remote-code `cache_position` compatibility

Tests added/updated:

* `GenerationInputNamesTest`
* `GenerationInputPreparerTest`

User reported all tests pass.

## Important Design Decisions

### Typed Inputs Instead Of Kwargs

Hugging Face builds a dictionary of model inputs. Deliverance uses `GenerationStepInputs` instead.

Decision:

* No internal `Map<String, Object>` for generation execution state.
* Every supported input must be an explicit typed field.

### Enums Instead Of Raw Strings

`GenerationInputKey` and `ModelInputName` avoid stringly typed control flow.

Decision:

* Use enums while the domain is small and known.
* If input names later control broad behavior, promote to a trait/sealed interface.

### Main Input Name Belongs To The Model

`main_input_name` is not a raw checkpoint config value. It is a runtime model capability.

Decision:

* Add `AbstractModel.getMainInputName()`.
* Default is `ModelInputName.INPUT_IDS`.
* Non-text models can override later.

### Past Key Values Are Not Just `KvBuffer`

HF `past_key_values` and Deliverance `KvBufferCache.KvBuffer` serve the same broad purpose, but they are not equivalent.

Decision:

* Introduce `PastKeyValues` as a generation-boundary concept.
* Do not blindly rename or wrap `KvBuffer` and call it done.
* Future implementation must define committed length, read-only views, scoped writes, and immutability of committed KV.

## Known Incomplete Areas

### `PastKeyValues`

Current status:

* Interface exists with `sequenceLength()` and `close()` only.
* It has JavaDoc explaining committed sequence length.

Still needed:

* read-only view API
* scoped write session API
* explicit commit semantics
* tests for immutable committed positions
* integration with `KvBufferCache.KvBuffer`

### Batch Inputs

Current status:

* `inputIds` is still `int[]`.
* This represents one sequence.

Future consideration:

* `int[][]` or a typed batch token structure if Deliverance generation supports multiple prompts in one call.

### Compileable Cache / 4D Masks

HF has a block that creates 4D masks for compileable caches.

Current decision:

* Not implemented yet.
* Deliverance needs typed mask and cache capability abstractions before this can be represented cleanly.

### Remote Code Compatibility

HF has a remote-code compatibility block for `cache_position`.

Current decision:

* Not supported.
* Do not add this unless Deliverance gains a safe local Java equivalent.

## Next Method To Continue

Next Hugging Face method:

```python
def _prepare_model_inputs(
    self,
    inputs: torch.Tensor | None,
    bos_token_id: torch.Tensor | None,
    model_kwargs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, str | None, dict[str, torch.Tensor]]:
```

First concept in that method:

```python
if (
    self.config.is_encoder_decoder
    and hasattr(self, "encoder")
    and self.encoder.main_input_name != self.main_input_name
):
    input_name = self.encoder.main_input_name
else:
    input_name = self.main_input_name
```

Deliverance concepts already started for this:

* `Config.isEncoderDecoder`
* `AbstractModel.getMainInputName()`
* `ModelInputName`

Questions for next session:

* How should Deliverance represent an encoder-side model/capability?
* Does `AbstractModel` need `getEncoderMainInputName()` before actual encoder-decoder support exists?
* What should replace HF's `inputs` plus `model_kwargs` split without introducing a kwargs map?

## Recommended Next Steps

1. Continue `_prepare_model_inputs(...)` line by line.
2. Define a typed replacement for HF's `inputs` argument.
3. Decide whether encoder-specific main input name needs a hook now or later.
4. Do not wire `prepareInputsForGeneration(...)` into `generate(...)` until the surrounding generation-prep methods are understood.
5. Return to `PastKeyValues` after the input-preparation boundary is mapped, because it needs a production-grade ownership design before integration.
