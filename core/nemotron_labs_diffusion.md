# Nemotron Labs Diffusion Support

Nemotron Labs Diffusion support brings a different kind of text generation into Deliverance: one checkpoint can run as a normal autoregressive language model, as a diffusion language model, and as a linear self-speculation model. It is not just another Llama-shaped port. It is a three-way generation system sharing one Ministral-style transformer backbone and one output head.

## Models

Primary upstream repositories:

- `nvidia/Nemotron-Labs-Diffusion-3B-Base`
- `nvidia/Nemotron-Labs-Diffusion-3B`

Current practical benchmark target:

- `edwardcapriolo/Nemotron-Labs-Diffusion-3B-JQ4`

The Base checkpoint is useful for loader, tensor inventory, and architecture work. The non-Base instruction checkpoint is the better target for chat/instruction benchmark quality.

Important config facts:

- `model_type = nemotron_labs_diffusion`
- architecture `NemotronLabsDiffusionModel`
- upstream implementation wraps a `Ministral3Model`
- hidden size `3072`
- layers `26`
- attention heads `32`
- KV heads `8`
- head dim `128`
- KV width `1024`
- mask token id `100`
- block size `32`
- YaRN RoPE

## What AR Means

Autoregressive generation is the familiar language-model loop:

```text
prompt tokens -> prefill KV cache -> sample one token -> append token -> repeat
```

Each generated token depends on the prompt and all prior generated tokens. The model sees a causal attention mask, so token `n` cannot look ahead to token `n + 1`.

Nemotron supports AR because the same transformer can be run in causal mode. In Deliverance, AR mode is useful as:

- a loader and logits sanity check
- a baseline for the shared weights, RoPE, KV cache, and output head
- a fallback for simple completion behavior

AR mode is selected with generation options:

```json
{
  "generationOptions": {
    "mode": "ar"
  }
}
```

Existing config:

- `benchmarks/configs/nemotron-labs-diffusion-3b-base-jq4-ar.json`

Existing script:

```sh
./benchmarks/run-nemotron-ar-q4-benchmark.sh
```

## What Diffusion Means

Diffusion text generation does not have to commit to one token at a time. Instead, it works over a block of masked positions.

Conceptually:

```text
causal prefix -> append a block of mask tokens -> denoise masked block -> accept confident tokens -> verify/update cache
```

The block can use bidirectional information inside the active block during denoising. That is the big difference from AR: a token being filled inside the block can use information from neighboring block positions that are also being denoised.

In practice, Deliverance currently focuses on Nemotron's linear self-speculation path because it maps well onto the existing generation and KV-cache machinery:

```text
causal prefill
draft a block without updating committed KV
verify causally with cache update
accept matching prefix plus bonus token
crop cache back to accepted length if needed
repeat
```

The model still returns ordinary generated text, but the runtime counters tell a different story from AR. Watch:

- function evaluations / NFE
- accepted tokens per block
- transferred tokens
- block length buckets
- attention time
- logits block projection time

## Generation Modes In Deliverance

Nemotron reads mode settings from `AutoModelConfig.generationOptions`.

Supported mode values:

- `ar`: causal autoregressive generation
- `linear_spec`: linear self-speculation diffusion-style generation
- `diffusion`: currently routes to the linear-spec path

Diffusion/linear-spec options:

```json
{
  "generationOptions": {
    "mode": "linear_spec",
    "blockLength": 32,
    "threshold": 0.0
  }
}
```

The current fast QOD diffusion config also enables the optimized output-head and attention options:

```json
{
  "outputHeadQuantization": "Q4",
  "gpuDiffusionBlockProjection": true,
  "packedBlockAttention": true,
  "packedPrefill": true,
  "generationOptions": {
    "mode": "linear_spec",
    "blockLength": 32,
    "threshold": 0.0
  }
}
```

Existing configs:

- `benchmarks/configs/nemotron-labs-diffusion-3b-base-jq4-diffusion.json`
- `benchmarks/configs/nemotron-labs-diffusion-3b-jq4-diffusion-turboquant-kv.json`

Existing script:

```sh
./benchmarks/run-nemotron-diffusion-q4-benchmark.sh
```

## Java Usage

AR mode:

```java
ModelFetcher fetcher = new ModelFetcher("edwardcapriolo", "Nemotron-Labs-Diffusion-3B-JQ4");
try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher)
        .withOutputHeadQuantization(DType.Q4)
        .withGenerationOptions(Map.of("mode", "ar"))
        .buildLocalTransformerModel()) {
    PromptContext prompt = model.promptSupport().orElseThrow().builder()
            .addUserMessage("Answer in one sentence: what is Paris?")
            .build();
    Response response = model.generate(UUID.randomUUID(), prompt,
            new GeneratorParameters().withMaxTokens(64), new DoNothingGenerateEvent());
    System.out.println(response.responseText);
}
```

Linear-spec diffusion mode:

```java
ModelFetcher fetcher = new ModelFetcher("edwardcapriolo", "Nemotron-Labs-Diffusion-3B-JQ4");
try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher)
        .withOutputHeadQuantization(DType.Q4)
        .withGpuDiffusionBlockProjection(true)
        .withPackedBlockAttention(true)
        .withGenerationOptions(Map.of(
                "mode", "linear_spec",
                "blockLength", 32,
                "threshold", 0.0))
        .buildLocalTransformerModel()) {
    PromptContext prompt = model.promptSupport().orElseThrow().builder()
            .addUserMessage("Solve step by step: if x / 2 + 6 = 25, what is x?")
            .build();
    Response response = model.generate(UUID.randomUUID(), prompt,
            new GeneratorParameters().withMaxTokens(128), new DoNothingGenerateEvent());
    System.out.println(response.responseText);
}
```

The uploaded Deliverance QOD/JQ4 artifact is available at:

```text
https://huggingface.co/edwardcapriolo/Nemotron-Labs-Diffusion-3B-JQ4
```

Use that owner/model pair in benchmarks and examples:

```java
new ModelFetcher("edwardcapriolo", "Nemotron-Labs-Diffusion-3B-JQ4")
```

## Runtime Internals

Deliverance's Nemotron path is built around KVCache2 because diffusion and speculation need explicit cache modes.

Core cache modes:

- `PREFILL_UPDATE_CACHE`: causal prompt prefill
- `DECODE_UPDATE_CACHE`: normal one-token AR decode
- `DENOISE_BLOCK_NO_UPDATE`: read prefix/block state without committing denoising KV
- `VERIFY_AND_UPDATE_CACHE`: causal verification/update after a draft block

Core attention patterns:

- `CAUSAL`: AR and verification
- `BIDIRECTIONAL`: full bidirectional blocks where applicable
- `PREFIX_CAUSAL_PLUS_BIDIRECTIONAL_BLOCK`: cached causal prefix plus denoising block visibility

Important implementation pieces:

- `NemotronLabsDiffusionModel`: owns AR, diffusion, and linear-spec generation orchestration
- `KvCacheSelfAttention`: KVCache2-backed attention for causal and no-update phases
- `PackedBlockAttention`: block score/value helper used by diffusion-style block execution
- `KvCacheSession`: request-local KVCache2 session
- `KvBlock`: immutable committed KV block
- `MutableKvBlock`: writable active block
- `TrackedReadOnlyTensor`: debug guard for borrowed non-copying KV views

## TurboQuant KV

Nemotron can opt into KVCache2 TurboQuant committed blocks through model config:

```json
{
  "kvBufferCache": {
    "kvBlockStoragePolicy": "MSE_TURBOQUANT",
    "kvTurboQuantBits": 4
  }
}
```

This is a memory-capacity feature first. It compresses immutable committed KV blocks and decodes rows/ranges as needed. It is not yet a fully fused TurboQuant attention kernel.

Related docs:

- [KV TurboQuant Plan](KVTurboQuant.md)
- [KVCache2 Roadmap](KVCache2Roadmap.md)
- [Prefix Cache MSE TurboQuant](prefix_cache_turboquant.md)

## Benchmarks

Run AR:

```sh
./benchmarks/run-nemotron-ar-q4-benchmark.sh
```

Run diffusion/linear-spec:

```sh
./benchmarks/run-nemotron-diffusion-q4-benchmark.sh
```

Run diffusion/linear-spec with TurboQuant KV config:

```sh
./benchmarks/run-nemotron-diffusion-q4-benchmark.sh \
  --model-config benchmarks/configs/nemotron-labs-diffusion-3b-jq4-diffusion-turboquant-kv.json
```

Profile lines to watch:

- `nemotron_labs_diffusion.diffusion.denoise_step`
- `nemotron_labs_diffusion.logits_block_projection`
- `nemotron_labs_diffusion.logits_block_projection.provider_gpu`
- `packedblockattention.score_value`
- `kvcacheselfattention.attention`
- `kvcacheselfattention.pack_kv`
- `kvcache.v2.turboquant.*` when TurboQuant KV is enabled

## Related Assets

- [Nemotron Labs Diffusion Port Plan](nemotron_labs_diffusion_port_plan.md)
- [Benchmarking](benchmarking.md)
- [KV TurboQuant Plan](KVTurboQuant.md)
- [KVCache2 Roadmap](KVCache2Roadmap.md)
- [DiffusionGemma Support Map](diffusion_gemma_support.md)
- `benchmarks/run-nemotron-ar-q4-benchmark.sh`
- `benchmarks/run-nemotron-diffusion-q4-benchmark.sh`
- `benchmarks/configs/nemotron-labs-diffusion-3b-base-jq4-ar.json`
- `benchmarks/configs/nemotron-labs-diffusion-3b-base-jq4-diffusion.json`
- `benchmarks/configs/nemotron-labs-diffusion-3b-jq4-diffusion-turboquant-kv.json`
