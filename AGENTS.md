# AGENTS.md

## Collaboration Rules

- For non-trivial requests, sketch a plan first, get user agreement, then implement.
- I need to do what people ask not estimate what they ask beause I believe I am smarter then them and can anticipate their needs.
- LLM memory is lossy. When planning, verify claims against repo source, docs, or web/source references before relying on them.
- Do not assume this project works like another project; many implementations here are bespoke. Verify behavior locally before mapping concepts from HuggingFace, Spring, vLLM, etc.
- Do not present a proposed fix as "the fix" unless there is strong evidence it addresses the issue. Say "hypothesis" or "candidate fix" when that is what it is.
- Avoid guess-and-check loops. Adding tests, running, adding printlns, and rerunning repeatedly should be a last resort; first isolate invariants and inspect the relevant implementation.
- Always prefer production-grade implementations over quick scaffolding: use clear lifecycle management, explicit invariants, defensive validation, appropriate concurrency primitives, observability, and deterministic tests.
- Build algorithms that will scale to the real target model sizes. Single-threaded scalar shortcuts for tensor/math paths have no chance of working in production; if an implementation is only a tiny-test scaffold, say so explicitly and do not present it as real model support.
- Add appropriate tests for behavior changes. Unit tests are preferred for mechanics; ITs are appropriate when model loading or reflection paths are the actual risk.
- Add JavaDoc or markdown for non-intuitive behavior and project-specific terms. Do not assume every programmer knows domain shorthand such as matmul = matrix multiply.

## First Files To Read

- Start with `README.md`, root `pom.xml`, and the module POM for the area you touch.
- Existing high-value docs are under `core/`: `inference_flow.md`, `PrefixCache.md`, `generator_sampling.md`, `tool_parser.md`, and `gemma4_support.md`.
- `grace/README.md` explains the newer tokenizer path; do not assume the older core tokenizer behavior is authoritative.

## Build And Test Commands

- Requires JDK 25 for most modules. `client` overrides to Java 17 because it is generated OpenAPI client code.
- Fast compile check for core and dependencies: `MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -pl core -am -DskipTests compile`.
- Use `-am` when compiling `core`; otherwise Maven may try to resolve local snapshot artifacts such as `native` from remote repositories.
- In this Alpine/QEMU/JDK 25 environment, normal tiered JIT has crashed inside the JVM during Maven/Javac/Surefire. Use `MAVEN_OPTS="-XX:TieredStopAtLevel=1"` for compile/package runs if you see SIGSEGVs in JDK internals.
- Focused single test examples: `mvn -pl tensor -Dtest=TensorCopyFromTest test`, `mvn -pl core -Dtest=GenerationCursorTest test`, `mvn -pl core -Dtest=KvBufferCachePrefixTest test`.
- Many integration tests download HuggingFace models and can take seconds to minutes depending on cache state. Avoid broad IT runs unless requested.

## Testing Discipline

- When asked to add a test, add the test and run it, then report the result. If the test exposes a bug, stop and let the user review/confirm the failure before changing production code.
- Do not mix feature/improvement test work with bug fixes unless the user explicitly approves fixing the discovered bug in the same pass.
- For tensor-operation parameterized tests, cover Panama and native SIMD when available. Naive operations are acceptable as an oracle/reference, but they are test-only and too slow for normal user paths.
- Real users usually run either the optimal native path or the Java Panama fallback path, so provider parity tests should exercise those providers directly instead of only comparing native against naive.
- Panama is the production Java baseline and must be correct first. Use small naive/reference calculations to prove Panama math on edge cases, then compare native SIMD and GPU accelerators against Panama on realistic shapes.
- Tensor tests must cover tails, offsets, row chunks, block boundaries, odd dimensions, and non-divisible batch sizes. A tensor library that cannot reliably do math on edge cases is not a usable foundation.
- Where possible, exercise all Panama machine-spec branches (`AVX_128`, `AVX_256`, `AVX_512`, `ARM_128`) from tests. The Java Vector API paths are selectable even when the current CPU is not that exact architecture; they may not be optimal, but they should still compute correctly.
- When porting a Hugging Face model with tiny synthetic checkpoints, first inspect the local Transformers source/tests and the real model `config.json` plus `model.safetensors.index.json` metadata. Tiny checkpoints must use the real checkpoint tensor names and shape formulas scaled down from real config fields; do not invent packed tensors or alternate layouts just because they are convenient for tests. Add tests that assert representative real tensor keys and shape formulas, then assert the tiny checkpoint writes matching scaled shapes. If the real model is too large to run locally, say that explicitly; do not present tiny synthetic tests as proof of real-model viability or performance.

## Module Boundaries

- `math`: vector/math helpers.
- `tensor`: tensor storage and operations providers (`Naive`, `Panama`, native SIMD integration points).
- `safetensors`: model weight/config loading.
- `grace`: newer tokenizer and prompt-processing implementation.
- `core`: inference engine, model implementations, generation, embeddings, classifiers.
- `web`: Spring Boot HTTP server and OpenAPI spec source.
- `client`: OpenAPI-generated Retrofit client.

## Reflection And Model Constructors

- Model classes are instantiated reflectively in `ModelSupport`; constructor signature changes will not always fail at obvious call sites.
- If you change `AbstractModel` constructor parameters, update every model class and every `ModelSupport.getConstructor(...)` signature together: Llama, Gemma2, Gemma3, Gemma4, Qwen2, Mistral, Mixtral, GPT2, and BERT.
- `AutoModelForCausaLm.applyTuning(...)` still handles model-family runtime tuning such as tool-call parsers.

## Tokenization

- The project is moving toward `grace.PreTrainedTokenizer` as the runtime tokenizer. Avoid adding new behavior to the old bespoke tokenizer path.
- Modern tokenizer behavior belongs in `grace`, including special tokens, chat templates, decode cleanup, byte-level BPE, and WordPiece behavior.
- `ResponseContext` should rely on tokenizer decode output directly; do not reintroduce token-rendering cleanup layers outside tokenizer implementations.

## Prefix Cache

- Read `core/PrefixCache.md` before changing prefix-cache behavior.
- Keep separate the mechanical invariants from output equivalence. Mechanical invariants: block-aligned hits, suffix prompt tokens keep original positions, decode starts after full prompt length, and copied KV rows round-trip.
- Do not treat generated text mismatch between cold and cache-hit paths as a prefix-cache bug unless split-prefill equivalence is already proven for that model/tensor-provider configuration.
- `GenerationCursorTest` protects decode-start/token-budget math. `KvBufferCachePrefixTest` protects KV store/lookup/copy value round trips.

## OpenAPI Generated Client

- `client` generates sources from `web/deliverance_specification.yaml` during `generate-sources`.
- Generated OpenAPI tests can confuse IDEs and compilation. `client/pom.xml` disables generated API/model tests and deletes `target/generated-sources/openapi/src/test` during generation; preserve that behavior.

## Embeddings

- BERT/sentence-transformers parity is not fully solved. Do not casually change BERT attention, transformer block ordering, or pooling expected values without checking HuggingFace source and a Python reference.
- `PoolingType.AVG` and `PoolingType.MODEL` have historically differed from sentence-transformers outputs; keep embedding parity work focused and separately verified.

## Native SIMD

- `AutoModelForCausaLm` attempts native SIMD first through `NativeSimdTensorOperations`; logs show the chosen tensor provider at model startup.
- If native libraries are missing, tests may still run using Panama operations, but performance and numeric paths can differ.
