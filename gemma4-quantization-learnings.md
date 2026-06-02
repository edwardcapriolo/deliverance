# Gemma4, Quantization, and Kernel Learnings

This document is a working summary of what we learned while bringing up Gemma 4 support, tokenizer parity, model quantization, and BF16/Q4 execution paths in Deliverance.

It is intended as a practical engineering note so we do not re-learn the same lessons by trial and error.

## 1. Gemma4 High-Level Status

### What is now known to be true

- Gemma 4 is not just "Gemma 2 with new weights".
- `google/gemma-4-E2B-it` uses:
  - hybrid layer types (`sliding_attention`, `full_attention`)
  - per-layer input embeddings (PLE)
  - shared KV layers (`num_kv_shared_layers = 20`)
  - full-attention proportional RoPE
- The runtime prompt tokenization path can now match Hugging Face for the target Gemma 4 prompt when using `grace`.

### Major fixes that materially changed behavior

- switched Gemma 4 runtime prompt/tokenization to prefer `grace`
- changed `AbstractModel.encodeText(...)` to use:
  - `EncodeOptions.defaults().withoutSpecialTokens()`
- removed the legacy token-rendering layer
- fixed Gemma 4 attention scaling mismatch:
  - upstream/vLLM use normalized attention with scaling `1.0`
  - Deliverance had incorrectly applied `1/sqrt(head_dim)`
- fixed Gemma 4 block/final RMSNorm semantics:
  - upstream uses `RMSNorm(x) * weight`
  - Deliverance had been doing `RMSNorm(x) * (1 + weight)`

### What changed in outputs over time

We observed a sequence of increasingly better failure modes:

1. punctuation/control-token garbage
   - `## // ~~ ...`
2. coherent but wrong meta-analysis
   - `The user is asking ...`
3. coherent but factually wrong answer
   - `The capital of New York is **New York City`

This progression strongly suggests that tokenizer/prompt and some attention math issues were real and worth fixing.

## 2. Prompt and Tokenizer Learnings

### Critical discovery

The legacy `core` tokenizer path did **not** tokenize Gemma 4 chat template control tokens correctly.

For the rendered prompt:

```text
<|turn>system
You are a concise assistant.<turn|>
<|turn>user
What is the capital of New York?<turn|>
<|turn>model
```

Hugging Face token ids are:

```text
[105, 9731, 107, 3048, 659, 496, 63510, 16326, 236761, 106,
 107, 105, 2364, 107, 3689, 563, 506, 5279, 529, 1799,
 3773, 236881, 106, 107, 105, 4368, 107]
```

`grace` now matches that for the golden prompt.

### Important nuance

Prompt rendering and tokenization are separate concerns:

- rendered prompt string can look correct
- token ids can still be wrong

The final Deliverance runtime prompt path should be checked at the token-id level.

### Final validated state for the target prompt

- `LEGACY_PROMPT_TOKEN_IDS`: wrong
- `GRACE_PROMPT_TOKEN_IDS`: matches Hugging Face
- `RUNTIME_PROMPT_TOKEN_IDS`: matches `grace`
- `FINAL_PROMPT_TOKEN_IDS`: equals `[bos] + runtimePromptTokens`

This means Gemma 4 prompt rendering + prompt tokenization + BOS insertion are now essentially validated for that prompt.

## 3. grace Module Learnings

### Why grace matters

The old `core` tokenizer interface is too weak for modern HF behavior.

Examples of things that matter:

- added token handling
- special token ids
- chat templates
- decode cleanup
- family-specific BPE behavior
- skipping special tokens

### Important grace bugs that were fixed

1. merge-array parsing in `tokenizer.json`
   - Gemma-style BPE merges can be stored as two-element arrays
2. pretokenizer gap preservation
   - unmatched regex spans were being dropped
3. Gemma-specific BPE path
   - spaces must normalize to `▁`
   - byte-level encoding should not be forced for Gemma
4. Gemma decode cleanup
   - `▁` must decode back to spaces

### grace family split

One tokenizer implementation was too broad.

Now there is a clearer split between:

- `GemmaTokenizer`
- `ByteLevelBpeTokenizer`
- `Quen2Tokenizer`
- `BertTokenizer`

And `AutoTokenizer` dispatch uses tokenizer/json traits rather than routing everything through Gemma.

### Testing pattern that worked

The most effective tokenizer work loop was:

1. make one HF-backed golden test fail
2. patch one concrete tokenizer behavior
3. rerun that single test
4. add narrower regression tests for the exact bug

Good focused tests now exist for:

- Gemma prompt golden ids
- merge parsing
- regex gap preservation
- Gemma turn-token atomicity
- Gemma decode space cleanup

## 4. Guided Choice Learnings

### Old problem

Guided choice logic was using decoded text fragments to decide whether a generated token still matched a target string prefix:

```java
String decodedToken = abstractModel.decodeToken(i);
String entire = current + decodedToken;
if (currentChoices.stream().anyMatch(ch -> ch.startsWith(entire))) ...
```

This is brittle because decode behavior is formatting-sensitive and tokenizer-dependent.

### New approach

Guided choice should use token ids, not decoded text.

Fixes applied:

- `ChoiceEncoded` now uses `AbstractModel.encodeText(...)`
- added `ChoiceEncoded.anyStartsWith(List<Integer> prefix)`
- `GuidedChoiceSampler` now uses `ResponseContext.generatedTokens`
- stop condition was fixed so a one-token guided choice stops immediately after the first token instead of generating an extra junk token like `<unused99>`

## 5. Gemma4 Decoder Learnings

### Shared-KV

We confirmed:

- shared-KV matters a lot for E2B
- disabling it makes outputs much worse

We also learned:

- generic KV cache page reconstruction is probably not broken in the simple synthetic case
- source row written to cache can match exactly
- but later full-state reconstruction in Gemma 4 debugging still looked suspicious and slow to iterate on

Shared-KV remains a meaningful suspect, but not the only one.

### Full-attention debugging

Useful focused debugging stages were:

- `q_after_proj`
- `q_after_norm`
- `q_after_rope`
- `k_after_proj`
- `k_after_norm`
- `k_after_rope` or `k_shared`
- `v_after_proj`
- `v_after_norm`
- `attn_presoftcap`
- `attn_postsoftmax`
- `o_proj`

One major learning from side-by-side comparison:

- Deliverance had been scaling Gemma 4 attention by `1/sqrt(head_dim)`
- Hugging Face / vLLM use normalized attention with `scaling = 1.0`
- removing that extra scale materially improved behavior

### RMSNorm

Another important side-by-side discovery:

- block/final norms for Gemma 4 should use `RMSNorm(x) * weight`
- not `RMSNorm(x) * (1 + weight)`

Also found:

- `RmsNorm` had a latent bug dividing by `embeddingLength` instead of `length`
- this was corrected

### PLE

PLE is now extracted into a helper support path and has some direct tests, but it is probably not the dominant remaining bug.

The key formula that should remain pinned is:

```text
(identity + projected) / sqrt(2)
```

## 6. Quantization Learnings

### What the working external Gemma 2 Q4 model taught us

Good external Q4 policy looks like:

#### Quantized
- large 2D projection matrices:
  - `self_attn.q_proj.weight`
  - `self_attn.k_proj.weight`
  - `self_attn.v_proj.weight`
  - `self_attn.o_proj.weight`
  - `mlp.gate_proj.weight`
  - `mlp.up_proj.weight`
  - `mlp.down_proj.weight`

#### Kept dense
- embeddings
- norms / layernorms
- `lm_head`
- non-2D tensors
- scalar calibration tensors

#### Writer convention
- 1D dense vectors should be canonicalized to `[1, N]`

### Quantizer updates that mattered

- explicit allowlist of quantized matrix suffixes
- embeddings excluded by suffix
- norm weights excluded by suffix
- non-2D tensors kept dense
- scalar `shape: []` support in loader
- split logical parent names skipped in quantizer iteration
- `.qb` sidecars classified as expected in comparison reports

### Important reality check

Producing a Q4 checkpoint does **not** automatically mean it will run fast.

If runtime paths dequantize or use fallback kernels, Q4 can still be slow.

## 7. BF16 × Q4 Learnings

### The old fallback path

At one point, `BF16 x Q4` in Panama worked by:

- materializing the activation tensor to dense `F32`
- then using the existing `F32 x Q4` path

That was slow, but gave more coherent results.

### The direct kernel path

A direct `BF16 x Q4` Panama implementation was attempted.

What we learned:

- small direct tests passing do **not** prove real-model correctness
- production Gemma 4 projections are much harsher than tiny toy cases
- if direct BF16×Q4 yields garbled model output while the old fallback yields coherent output, the direct kernel is wrong or incomplete

### What was actually wrong

The original direct `BF16 x Q4` path had a real kernel correctness bug, not a quantization-margin issue.

Important evidence:

- `PanamaTensorOperationsTest.batchDotProductBf16Q4WithOffsetsAndRowChunk` compared Panama against `NaiveTensorOperations` on the **same** BF16/Q4 tensors
- the mismatch was large and structured, not small floating-point drift
- the output deltas repeated by output column across multiple rows
- that pattern pointed to deterministic packed-lane / block-group misalignment in the direct kernel

In practice, the suspect area was the direct SIMD `GemmerBF16Q4` implementation, especially implicit lane-group assumptions while pairing BF16 values with packed Q4 nibbles.

### What fixed correctness

The stable rebuild path was:

1. treat `NaiveTensorOperations` as the arithmetic oracle for BF16/Q4
2. stop relying on packed SIMD lane reinterpretation for correctness
3. decode Q4 explicitly from flat tensor offsets and raw packed bytes
4. use a block-aware reference gemmer that works directly from:
   - `a.getMemorySegment()` for BF16 activations
   - `b.getMemorySegment()` for packed Q4 bytes
   - `b.getFactorForIndex(row, column)` for per-block scale

This corrected the gibberish output and restored coherent Gemma 4 JQ4 generation.

### Important semantic lesson

For Q4 tensors, correctness depends on three things being aligned at the same time:

- row/window semantics
- logical column offset semantics
- packed block layout semantics

Using `b.get(row, col)` as a generic fallback was too naive for the hot path and also easy to get wrong around views/shapes. The more stable reference implementation worked from flat offsets and decoded the Q4 block structure directly.

### Better test strategy

### Better test strategy

Toy tests are necessary but not sufficient.

Useful direct kernel tests should include:

- multi-row inputs
- multi-column outputs
- nonzero offsets
- chunked row selection
- comparison against a trusted reference path

The two most useful BF16/Q4 tensor tests during this work were:

- `PanamaTensorOperationsTest.batchDotProductBf16Q4WithOffsetsAndRowChunk`
- `PanamaTensorOperationsTest.batchDotProductBf16Q4MultiRowMultiCol`

These exposed bugs that smaller happy-path tests did not.

### Current known-good runtime shape

The current known-good BF16/Q4 path is no longer the old "materialize BF16 activations to F32 and reuse F32xQ4" fallback.

Instead it is:

- a correctness-first, block-aware BF16/Q4 reference path
- plus ARM tiling improvements layered on top of that reference layout

That was enough to move from:

- gibberish model output

to:

- coherent JQ4 output like:
  - `The capital of New York is **New York City`

### Performance learnings from the rebuild

On the ARM path, the rebuild progressed roughly like this:

- correctness-first reference block decoder: coherent output, but slow
- added `1x4` shared-block reuse: major improvement
- added `4x1`: additional improvement
- added `4x4`: another meaningful improvement

This showed that for BF16/Q4, reuse across both rows and columns matters materially.

### Optimization ideas worth revisiting later

One later optimization pass was **not** a clear win overall and was reverted.

That experiment tried to:

- hoist more repeated math inside the aligned block loops
- reduce repeated `bf16ToFloat(...)` structure overhead
- process 2 packed Q4 bytes at a time
- accumulate unscaled block sums and apply scale once per block

Observed behavior:

- decode after the first token looked faster
- but `timeToFirstToken` got much worse in end-to-end runs
- this suggests the optimization may have helped steady-state decode while hurting prompt/prefill enough to lose overall

So the idea itself should **not** be discarded, but it needs dedicated profiling before being trusted.

If someone has dedicated profiling time later, good candidates to revisit are:

- inner-loop hoisting of repeated address math in the aligned 32-wide block paths
- reducing BF16 conversion overhead further
- processing 2 packed Q4 bytes at a time, but only if it helps real TTFT not just post-first-token decode
- tile-selection heuristics for `1x4`, `4x1`, and `4x4`
- phase-aware heuristics if prefill and decode benefit from different tiles

The big lesson is: end-to-end `timeToFirstToken` must stay the primary metric. A micro-optimization that improves post-first-token decode but hurts prefill can look good in a hot loop and still be a regression for real requests.

### API semantics lesson

`NaiveTensorOperations` is a reference harness.
The project is built around the Panama/native semantics, so if `Naive` disagrees on API behavior like result-column placement, `Naive` should be aligned to Panama rather than treated as the main truth source for those semantics.

## 8. Inspection / Comparison Tooling

The `safetensors` module now has model inspection/comparison tooling to answer questions like:

- which tensors were quantized?
- which stayed dense?
- which are logical split parents?
- what `.qb` sidecars exist?
- how does an original checkpoint compare to a quantized checkpoint?

This is useful for reducing blind debugging in the quantization path.

## 9. Fetcher Learnings

Fetcher behavior should be:

- one cache root:
  - `~/.deliverance/<owner>_<model>/`
- tokenizer-only fetch should use the same directory
- local complete directories should be used offline without first touching Hugging Face
- simple completeness checks are enough initially:
  - missing file -> download
  - zero-byte file -> redownload
  - wrong size -> redownload

This is now implemented for model and tokenizer fetchers.

## 10. Useful Testing / Build Tricks

### 1. Run one test at a time
When chasing a tokenizer or kernel bug, single-test loops were far more effective than full-suite runs.

### 2. `-Xint` can be useful when the JDK or vector/JIT path gets in the way
Running Maven and Surefire in interpreted mode helped keep the JVM from crashing while iterating on tokenizer work:

```bash
JAVA_TOOL_OPTIONS='-Xint' mvn ... -DargLine='-Xint' test
```

It is slower, but useful when the toolchain itself is unstable.

### 3. Use external oracles when possible
For tokenizer work, the Hugging Face tokenizer output was the best oracle.
For quantization policy, a known-good external Q4 checkpoint was the best oracle.

## 11. Current Practical Guidance

### For Gemma 4 correctness work

- keep using the `grace` runtime tokenizer path
- keep prompt rendering/tokenization comparisons around as a sanity tool
- avoid falling back to the legacy `core` tokenizer for Gemma 4

### For Gemma 4 Q4 work

- the quantizer policy is now close to the known-good Gemma 2 Q4 policy
- runtime speed still depends on real Q4 projection kernels, not just disk format
- if the direct BF16×Q4 kernel is not trustworthy, keep a known-good reference path while debugging it
- prefer block-aware BF16/Q4 implementations that decode Q4 layout explicitly over clever lane-reinterpretation tricks
- benchmark whole-request behavior, not just a kernel micro-loop
- when testing optimizations, track at least:
  - `timeToFirstTokenMs`
  - `totalTimeMs`
  - tokens/sec after first token

### For debugging

Prefer:
- small direct tests
- one failing golden test
- external oracles

Avoid:
- widening scope mid-debug
- mixing tokenizer, quantizer, decoder, and kernel changes in one loop if possible

## 12. Short Checklist Of Hard-Won Lessons

- Gemma 4 runtime prompt token ids must match HF, not just decode roundtrip.
- `grace` is a better foundation than the legacy core tokenizer path for modern HF models.
- Added tokens and tokenizer-family dispatch matter a lot.
- Shared-KV matters for E2B; disabling it is not a fix.
- Gemma 4 attention should not apply extra `1/sqrt(head_dim)` scaling.
- Gemma 4 block/final RMSNorm should not use `1 + weight`.
- Q4 on disk is not the same as fast Q4 at runtime.
- `.qb` right-only tensors in comparisons are expected sidecars, not bugs.
- If one code path is the project’s real execution model, make the test harness match it.
