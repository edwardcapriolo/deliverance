# AGENTS.md

## Collaboration Rules

- For non-trivial requests, sketch a plan first, get user agreement, then implement.
- I need to do what people ask not estimate what they ask beause I believe I am smarter then them and can anticipate their needs.
- LLM memory is lossy. When planning, verify claims against repo source, docs, or web/source references before relying on them.
- Do not assume this project works like another project; many implementations here are bespoke. Verify behavior locally before mapping concepts from HuggingFace, Spring, vLLM, etc.
- Do not present a proposed fix as "the fix" unless there is strong evidence it addresses the issue. Say "hypothesis" or "candidate fix" when that is what it is.
- Avoid guess-and-check loops. Adding tests, running, adding printlns, and rerunning repeatedly should be a last resort; first isolate invariants and inspect the relevant implementation.
- Always prefer production-grade implementations over quick scaffolding: use clear lifecycle management, explicit invariants, defensive validation, appropriate concurrency primitives, observability, and deterministic tests.
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
