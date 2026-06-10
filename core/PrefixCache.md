## Prefix Cache

Prefix cache is incredibly useful when you have large **system/root prompts**. Here is an example where we have a set
of directives that we will give to every prompt. 


```
PromptContext ctx = m.promptSupport().get().builder()
    .addSystemMessage("You are an assistant that produces concise, production-grade software.")
    .addSystemMessage("Output java code.")
    .addSystemMessage("Refrain from editorializing your reply.")
    .addSystemMessage("Generate java code into the package 'io.teknek.shape' .")
    .addSystemMessage("Do not import java.awt")
```

Obviously if we can "reuse" the prompt we are winning. This is what the
prefix cache does.

### KV Cache settings

The settings are applied when the model is initialized. (not per query)
```
KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withMaxPrefixTokensPerPrompt(512)
                .withMaxEntries(10_000)
                .withBlockSize(32);
try (AbstractModel m = AutoModelForCausaLm.newBuilder(fetch).withWorkingQuantType(DType.I8)
        .withKvBufferCacheSettings(settings)
        .build()){
```

* maxPrefixTokensPerPrompt - Regardless of how long a prompt is sent only consider this many tokens for the cache
* maxEntries - the size of the cache. IF this is a memory cache eviction happens passed this limit
* withBlockSize - the number of tokens in a block (explained below)

#### Block size
It is possible to store the KVs at each token, but that is a bad idea with limited value. Some words can be multiple
tokens and the KV information is large (MBs). We get the most value from a long complete match, so matches only occur
at block boundaries.

With a 32-token block size, a 33-token prompt can reuse the first 32 prompt tokens from cache. The thirty-third prompt token
must still run through the model at position 32, and the first generated token must be decoded at position 33. Prefix
length should never reduce the generation token budget.

### Invariants

There are two different levels of correctness. Keep them separate when reviewing or testing this feature.

#### Mechanical invariants

These must always hold, even if generated text is not identical between cold and cache-hit paths:

* cache hits are block-aligned
* uncached suffix prompt tokens are still processed at their original positions
* decoding begins after the full prompt length, not after `prefix_length + prompt_length`
* copied KV buffers preserve stored key/value tensor rows

These invariants are covered by focused unit tests such as `GenerationCursorTest` and `KvBufferCachePrefixTest`.

#### Output-preservation invariant

This is the stronger user-facing contract documented by systems such as vLLM and OpenAI prompt caching:

```
generated output with cache == generated output without cache
```

Deliverance should only claim this when the relevant model and tensor-provider configuration also satisfies split-prefill
equivalence:

```
batchForward(allPromptTokens, 0)
```

is numerically equivalent to:

```
batchForward(prefixTokens, 0)
batchForward(suffixTokens, prefixLength)
```

If split-prefill equivalence fails, prefix cache may still be mechanically correct, but it is not transparent in the
vLLM/OpenAI sense for that runtime configuration.

### What Deliverance supports today

Deliverance prefix cache currently supports the mechanical cache behavior: block-aligned lookup, KV copy round trips,
and correct decode-start/token-budget math. This is useful for reducing prefill work on shared prompt prefixes.

The disk KV backend is separate from prefix-cache identity. `KvBufferCacheSettings(File)` stores active KV pages as
memory-mapped page files for live `KvBuffer` instances; it does not make those files durable prefix-cache entries. When
disk KV is enabled, Deliverance skips prefix snapshot storage entirely. This is intentional: the current prefix cache is
copy-snapshot based, and storing every block-aligned prefix snapshot as disk-backed KV pages can create quadratic disk
growth. See `core/DiskKvBackend.md` for the active disk-page storage contract, cleanup behavior, and metrics.

Deliverance does not currently provide a documented deterministic or batch/chunk-invariant inference mode. That means
the project does not currently promise exact generated-token or generated-text equality between cold full-prefill and
cache-hit split-prefill paths for every model and tensor-provider configuration.

### Comparison points

External serving systems document prefix caching as output-preserving. vLLM's Automatic Prefix Caching design document
says prefix caching is widely used because it "won't change model outputs" and notes that it only caches full blocks:

https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html

OpenAI's Prompt Caching FAQ says prompt caching "does not influence the generation of output tokens or the final
response" and that the generated output is identical regardless of whether caching is used:

https://platform.openai.com/docs/guides/prompt-caching

Matrix multiplication, attention, RMSNorm, and activation quantization can all produce different numerical results when
the same request is sliced into different batch or chunk shapes. This can happen even with temperature 0 and a fixed
seed. OpenAI's seed reproducibility cookbook also describes deterministic sampling as best effort rather than guaranteed:

https://cookbook.openai.com/examples/reproducible_outputs_with_the_seed_parameter

For more background see Thinking Machines, "Defeating Nondeterminism in LLM Inference":

https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

Adding a deterministic, batch/chunk-invariant mode would be a valuable future direction for someone interested in
working on the inference engine. That work would need targeted tests for full-prefill versus split-prefill equivalence
across model families, tensor providers, quantization modes, and attention implementations.

### Testing guidance

Use three categories of tests:

* Mechanical tests: assert block-aligned lookup, KV copy round trips, and decode-start/token-budget math.
* Disk KV boundary tests: assert disk-backed active pages do not populate prefix-cache snapshots.
* Split-prefill tests: compare full prefill against prefix-plus-suffix prefill before involving the cache.
* Output-preservation tests: compare generated tokens/text with and without cache only after split-prefill equivalence is known to hold for that model and tensor-provider configuration.

A generated-text mismatch is only a prefix-cache bug if the lower-level mechanical and split-prefill invariants already
hold. Otherwise the mismatch may be caused by ordinary chunk-shape numerical differences in the model execution path.

### How do you know it is working

A number of metrics were added to the feature, but we are so proud of it we even log information every prompt

```
[main] INFO io.teknek.deliverance.model.AbstractModel - time_to_first_token=811.371745 prefix_length=24
```

The higher the prefix_length the larger KV match you found. time_to_first_token should be a lower number when 
there is any matching.
