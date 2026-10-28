# XLM-RoBERTa Sequence Classification Support

This is a cross-module port. Most implementation work belongs in `grace`, which is the
repository's Hugging Face tokenizer equivalent. The model execution and classification
changes belong in `core`.

## Mission

Deliver a faithful port that can load and run:

```text
papluca/xlm-roberta-base-language-detection
```

The completed model must classify ordinary sentences by language through the existing Java
classification API. The first user-visible proof should be a small, readable test using
five or six sentences, for example:

```text
"This is English."       -> en
"Bonjour tout le monde." -> fr
"Das ist Deutsch."       -> de
"Hola, ¿cómo estás?"     -> es
"これは日本語です。"       -> ja
"Привет, как дела?"      -> ru
```

These smoke-test examples demonstrate the goal; they do not replace parity coverage.
Where Hugging Face `transformers` or `tokenizers` provides built-in examples, fixtures, or
expected values, those upstream cases are the preferred source of confidence. The Java
tests should preserve those cases and add the six-language classification test as a clear
end-to-end acceptance test.

## Porting Shape Principle

Faithfulness does not require reproducing Python's class inheritance literally.

The priority is to match the upstream outward-facing shape and conceptual boundaries:

- Configuration fields and model-family identification
- Tokenizer inputs, outputs, special tokens, and masks
- Model tensor names and mathematical data flow
- Classification-head inputs, outputs, and label behavior
- Test class and method intent
- Performance-critical operation boundaries

Inside those boundaries, Java may use a more idiomatic design when it improves reuse,
resource ownership, type safety, or performance. The existing `grace` layer deliberately
started by matching the outward-facing tokenizer shape because every additional matching
surface makes faithful ports of future Hugging Face models easier.

For this port, HF's `Bert -> Roberta -> XLM-R` hierarchy is therefore a strong guide to
shared responsibilities, not a requirement to reproduce those exact Java classes. The
implementation should avoid duplication while choosing the clearest Java abstraction for
the shared encoder and classification head.

## Definition of Done

The mission is complete only when:

- The model downloads or loads from its Hugging Face model directory without manual file
  rewriting.
- The tokenizer produces the expected XLM-R input IDs and attention masks.
- The encoder and classification head match the upstream model within documented numeric
  tolerances.
- The public Java API returns the expected language labels for the smoke-test sentences.
- The faithful upstream test ports pass, including the relevant tokenizer and model tests.
- The implementation uses the repository's production tensor and SIMD paths rather than a
  scalar-only test shortcut.

## Ownership Split

### `grace`

- XLM-R tokenizer discovery and construction
- SentencePiece/BPE tokenizer behavior required by XLM-R
- Special-token handling
- Added-token behavior
- Padding and truncation semantics
- Attention-mask construction
- Tokenizer parity tests

### `core`

- XLM-R model registration and configuration loading
- XLM-R encoder weight naming and execution
- RoBERTa-family position-ID semantics used by XLM-R
- First-token representation selection
- XLM-R classification-head execution
- Classification result and label mapping changes
- Tensor-provider and SIMD parity tests

## Current Code Cross-Reference

The following is the current gap assessment against `grace` and `core`.

### HF Inheritance Findings

The checked-out Hugging Face modular sources establish this inheritance chain:

```text
BertEmbeddings
  -> RobertaEmbeddings

BertModel
  -> RobertaModel
      -> XLMRobertaModel

RobertaForSequenceClassification
  -> XLMRobertaForSequenceClassification
```

The RoBERTa specialization changes the embedding position behavior and model naming while
reusing the BERT encoder layers. XLM-R then remains a thin specialization of RoBERTa. The
classification-head behavior is also shared semantically: the same `<s>`-token,
dropout, dense, `tanh`, dropout, and `out_proj` sequence is used. The generated XLM-R file
contains copied class definitions, but the modular source makes clear that this is a
generation artifact, not an architectural reason to duplicate the implementation in Java.

The Java design should follow the modular architecture, not the generated-file layout:

- One shared BERT-family encoder implementation
- A thin RoBERTa model specialization for prefixes and position IDs
- A thin XLM-R specialization for tokenizer/model registration
- One shared RoBERTa classification head implementation

### `grace` Findings

`grace/src/main/java/io/teknek/deliverance/grace/AutoTokenizer.java`

- Does not recognize `XLMRobertaTokenizer`.
- Its generic BPE path is selected mainly for byte-level BPE tokenizers.
- XLM-R uses SentencePiece-style `U+2581` word markers and its own special-token
  post-processing, so routing it through `ByteLevelBpeTokenizer` would not be faithful.

`grace/src/main/java/io/teknek/deliverance/grace/PreTrainedTokenizerBase.java`

- Already provides useful shared behavior for encoding, attention masks, padding,
  truncation, special-token masks, added tokens, and batch encoding.
- Its generic `applyBpe` path starts with `ByteLevelCodec.encode`, which is not the XLM-R
  SentencePiece encoding path.
- The generic special-token builder can insert BOS/EOS, but XLM-R needs an explicit
  `<s> tokens </s>` post-processor and must preserve the model's token IDs.

`grace/src/main/java/io/teknek/deliverance/grace/BytePairEncodingModel.java`

- Stores only the current generic BPE vocabulary, merge ranks, pre-tokenizer config, and
  unknown token.
- It has no representation for the XLM-R tokenizer's model/post-processor semantics.

`grace/src/main/java/io/teknek/deliverance/grace/Encoding.java` and
`PreTrainedTokenizerBase.pad(...)`

- Already expose the `inputIds`, `attentionMask`, and special-token mask needed by core.
- Padding and mask behavior should be retained, then verified against XLM-R fixtures.

### `core` Findings

`core/src/main/java/io/teknek/deliverance/model/ModelSupport.java`

- Registers `BERT` but not `XLM-ROBERTA`.
- `detectModel` will therefore reject the target checkpoint before model construction.

`core/src/main/java/io/teknek/deliverance/model/bert/BertConfig.java`

- Parses most shared encoder fields and `label2id`.
- Does not currently model the XLM-R-specific `bos_token_id`, `eos_token_id`, or the
  distinction between RoBERTa-family padding-index position defaults and ordinary BERT
  position defaults.

`core/src/main/java/io/teknek/deliverance/model/bert/BertModel.java`

- Already contains the production TensorRef/SIMD-capable encoder implementation.
- Weight lookup accepts `""` and `"bert."` prefixes, not `"roberta."`.
- Input loading assumes BERT tensor names and requires token-type embeddings.
- `BertInput` defaults position IDs to `0..sequenceLength-1`; RoBERTa-family models use
  positions that start at `padding_idx + 1` for non-padding tokens.
- The existing embedding path can therefore not be reused unchanged for XLM-R.
- `timedEmbedding(..., PoolingType.MODEL)` selects the last token and applies a BERT-style
  pooler. XLM-R sequence classification selects the first `<s>` representation.
- `loadClassifierWeights()` expects `classifier.weight` and `classifier.bias`.
- XLM-R sequence classification uses the same semantic head as RoBERTa:
  `classifier.dense`, activation, then `classifier.out_proj`, with dropout around the
  dense projection.

`core/src/main/java/io/teknek/deliverance/model/AbstractModel.java`

- The public `classify(String, PoolingType)` API already performs vectorized classifier
  scoring and softmax.
- Its current contract assumes one embedding vector followed by one linear classifier.
- XLM-R requires a model-specific representation/head path, so the API should remain
  stable while the internal classification hook is generalized rather than bypassed.

## Actionable Implementation Plan

### Phase 1: Freeze Upstream Fixtures

1. Record the checked-out `transformers` and `sentence-transformers` revisions.
2. Download or use the target model files without rewriting them.
3. Capture the target model's tokenizer outputs for the upstream smoke sentences:
   token strings, token IDs, special-token placement, attention masks, and decoded text.
4. Capture Hugging Face reference tensors for the encoder and classification logits for
   the same inputs.
5. Add these fixtures before changing production code so failures identify the layer that
   diverges.

### Phase 2: Port XLM-R Tokenization in `grace`

1. Add an XLM-R tokenizer implementation or a properly factored SentencePiece/BPE path;
   do not make the byte-level tokenizer pretend to be XLM-R.
2. Add `XLMRobertaTokenizer` dispatch in `AutoTokenizer`.
3. Implement XLM-R normalization, `U+2581` word-boundary behavior, merge/token lookup,
   unknown-token behavior, and decode behavior from the upstream tokenizer source.
4. Implement `<s> ... </s>` special-token construction for single inputs and the exact
   pair-input behavior from the upstream tokenizer.
5. Verify truncation occurs with the same special-token accounting as Hugging Face.
6. Port the direct XLM-R tokenizer test class and common tokenizer assertions.

### Phase 3: Share the Encoder Path in `core`

1. Register `XLM-ROBERTA` and deserialize its configuration without altering the source
   checkpoint.
2. Refactor the existing `BertModel` so its encoder blocks, attention, MLP, and tensor
   execution are reusable by a thin `RobertaModel` specialization. Do not copy the block
   implementations.
3. Add model-family hooks for weight prefixes, embedding tensor names, token-type table
   size, and RoBERTa-family position-ID generation.
4. Implement RoBERTa-family position IDs exactly: non-padding positions begin at
   `pad_token_id + 1`, while padding positions retain `pad_token_id`.
5. Keep token-type IDs available as an all-zero input for XLM-R, but do not assume a
   two-entry BERT token-type table.
6. Port the XLM-R base encoder integration test and the shared RoBERTa position tests.

### Phase 4: Implement the Classification Head

1. Add one shared RoBERTa-family classification head that loads
   `classifier.dense`, `classifier.out_proj`, and their biases. XLM-R should reuse it.
2. Select the first `<s>` hidden state, matching `XLMRobertaForSequenceClassification`.
3. Apply the exact shared head sequence from upstream: dropout, dense projection, `tanh`,
   dropout, and output projection.
4. Generalize the internal classification hook in `AbstractModel` so the existing public
   `classify(String, PoolingType)` API remains usable without forcing XLM-R through the
   BERT pooler path.
5. Preserve vectorized matrix operations for both classifier projections and use the
   existing softmax implementation for probabilities.
6. Port a fixed-logit/head test before running the full model, then add full checkpoint
   parity tests.

### Phase 5: End-to-End Acceptance

1. Load `papluca/xlm-roberta-base-language-detection` through
   `AutoModelForSequenceClassification`.
2. Classify five or six multilingual sentences and assert the expected labels.
3. Compare Java logits/probabilities with Hugging Face for those same sentences.
4. Run the Sentence Transformers feature-extraction and sequence-classification contract
   tests that apply to the implementation.
5. Run core tests with the Panama provider and native SIMD provider where available.
6. Benchmark classification latency only after parity passes.

## HF Architecture Reference

The checked-out Hugging Face implementation confirms the intended structure:

- `XLMRobertaModel` is a thin specialization of `RobertaModel`.
- XLM-R uses the shared RoBERTa embedding and encoder behavior, including
  `padding_idx + 1` position IDs.
- `XLMRobertaForSequenceClassification` reuses the RoBERTa sequence-classification
  semantics and constructs the XLM-R base model without a pooler.
- The shared head takes `features[:, 0, :]`, then applies dropout, dense, `tanh`, dropout,
  and `out_proj`.
- `XLMRobertaTokenizer` is a separate tokenizer class backed by XLM-R's tokenizer files.

Therefore the actionable design is:

> Share a BERT-family encoder implementation and optimized tensor kernels; add a thin
> RoBERTa specialization, reuse its classification head, and add the XLM-R tokenizer and
> registration. Do not attempt to support XLM-R by changing only the `roberta.` weight
> prefix, and do not duplicate the full encoder or classification head.

## The Process Is

1. Inspect the already checked-out `transformers` and `sentence-transformers` sources.
2. Identify the exact upstream tests covering XLM-R, tokenization, encoder behavior, pooling, and sequence classification.
3. Port those tests to Java while preserving:
   - Test class names
   - Test method names, adapted only to Java conventions
   - Fixtures and expected values
   - Control flow and behavioral coverage
4. Implement the model faithfully, not approximately:
   - Same tensor shapes and data flow
   - Same masking, positions, pooling, and classifier-head semantics
   - Same numerical behavior
   - Equivalent performance design
   - SIMD/vectorized paths where the reference implementation uses optimized kernels
5. Run the Java port against the same expected results and model fixtures.
6. Only call the model supported once the port passes parity tests.

This is a **faithful functional and architectural port**, not a minimal implementation
that merely produces plausible classifications.

## Target Model

Target model:

```text
papluca/xlm-roberta-base-language-detection
```

The model is an `XLMRobertaForSequenceClassification` checkpoint with:

- 12 encoder layers
- 768 hidden dimensions
- 12 attention heads
- 3,072 intermediate dimensions
- 20 language labels
- Single-label classification
- Maximum model sequence length of 512 tokens

Expected public use remains the existing classification API:

```java
Map<String, Float> result =
    AutoModelForSequenceClassification
        .fromPretrained(new ModelFetcher(
            "papluca", "xlm-roberta-base-language-detection"))
        .classify("Bonjour tout le monde.", PoolingType.MODEL);
```

## Upstream Sources

Before implementation, locate and record the exact revisions of the checked-out:

- Hugging Face `transformers` implementation and tests
- `sentence-transformers` implementation and tests relevant to encoder and pooling behavior
- Model `config.json`
- Tokenizer configuration and tokenizer files
- Safetensors metadata and tensor names

The port must follow the upstream source rather than an inferred or simplified version of
the architecture. Any intentional deviation must be documented with its reason and a
corresponding parity test.

## Test Inventory

The checked-out upstream repositories are:

```text
/ai-code/transformers
/ai-code/sentence-transformers
```

### Direct XLM-R Tests

Port the following tests from `transformers` first:

`transformers/tests/models/xlm_roberta/test_tokenization_xlm_roberta.py`

- `XLMRobertaTokenizationTest`
- `integration_expected_tokens`
- `integration_expected_token_ids`
- `expected_tokens_from_ids`
- `integration_expected_decoded_text`
- The inherited `TokenizerTesterMixin` behavior exercised by the class

`transformers/tests/models/xlm_roberta/test_modeling_xlm_roberta.py`

- `XLMRobertaModelIntegrationTest.test_xlm_roberta_base`
- `XLMRobertaModelIntegrationTest.test_xlm_roberta_base_sdpa`
- `XLMRobertaModelIntegrationTest.test_xlm_roberta_large` is architecture coverage, but
  is not required for the initial base-checkpoint mission

### Shared RoBERTa Model Tests

XLM-R follows the RoBERTa encoder family, so the relevant shared behavioral tests come
from `transformers/tests/models/roberta/test_modeling_roberta.py`:

- `RobertaModelTester.create_and_check_model`
- The shared `ModelTesterMixin` sequence-classification coverage exercised because
  `RobertaForSequenceClassification` is in `RobertaModelTest.all_model_classes`
- `RobertaModelTest.test_config`
- `RobertaModelTest.test_model`
- `RobertaModelTest.test_inputs_embeds_matches_input_ids`
- `RobertaModelTest.test_create_position_ids_respects_padding_index`
- `RobertaModelTest.test_create_position_ids_from_inputs_embeds`
- `RobertaModelIntegrationTest.test_inference_no_head`
- `RobertaModelIntegrationTest.test_inference_classification_head`

The generic `ModelTesterMixin` assertions used by `RobertaModelTest` are also relevant
where their inputs and outputs apply to the Java model. Causal-LM, masked-LM, token
classification, multiple-choice, question-answering, decoder, and FlashAttention tests
are not part of the initial sequence-classification mission unless implementation work
reveals a shared invariant that requires them.

Do not port the byte-level RoBERTa tokenizer tests as XLM-R tokenizer tests. XLM-R has
its own tokenizer test class and SentencePiece/BPE behavior; use the direct XLM-R suite
and the common tokenizer mixin instead.

### Sentence Transformers Tests

The relevant Sentence Transformers tests are:

`sentence-transformers/tests/base/modules/test_transformer.py`

- `test_infer_flatten_position_offset`
- The `TINY_XLMR` fixture declaration and its expected padding offset of `2`

`sentence-transformers/tests/base/modules/transformer/test_feature_extraction.py`

- The parameterized architecture loading and feature-extraction coverage for
  `xlm-roberta`
- `TestTransformerArchitectures.test_get_embedding_dimension`
- `TestTransformerArchitectures.test_save_load`
- `TestTransformerArchitectures.test_modalities_property`
- `TestTransformerArchitectures.test_supports_consistency`
- `TestTransformerArchitectures.test_inference_with_supported_modalities`

`sentence-transformers/tests/base/modules/transformer/test_sequence_classification.py`

- The parameterized `xlm-roberta` sequence-classification architecture coverage
- `TestSequenceClassificationArchitectures.test_get_embedding_dimension`
- `TestSequenceClassificationArchitectures.test_save_load`
- `TestSequenceClassificationArchitectures.test_module_output_name`
- `TestSequenceClassificationArchitectures.test_inference_with_supported_modalities`
- `TestSequenceClassificationArchitectures.test_inference_with_supported_modality_pairs`

These tests validate the Transformer-module contract around loading, preprocessing,
forwarding, output dimensions, and classification scores. They complement the direct
Hugging Face model parity tests; they do not replace them.

### Tests Not Selected

The following are useful references but are not initial port targets:

- Sentence Transformers cross-encoder pretrained score tables: they cover many RoBERTa
  models but not this language-detection checkpoint.
- Generic Sentence Transformers training, evaluator, loss, and multi-vector suites:
  they test higher-level products rather than the tokenizer and encoder path needed here.
- RoBERTa masked-LM, causal-LM, multiple-choice, token-classification, and QA tests:
  they are outside the target task unless shared implementation requires them.

### Port Order

1. XLM-R tokenizer integration and common tokenizer tests.
2. XLM-R position-ID and padding-index tests.
3. XLM-R base encoder forward parity.
4. XLM-R sequence-classification head parity.
5. Sentence Transformers feature-extraction and sequence-classification contract tests.
6. The `papluca/xlm-roberta-base-language-detection` six-language end-to-end acceptance test.

## Functional Scope

The port must cover the complete path used by this checkpoint:

- XLM-R configuration parsing and model registration
- XLM-R tokenizer loading and encoding
- Special tokens, padding, truncation, and attention masks
- XLM-R position IDs and padding offset behavior
- Embedding lookup and normalization
- Self-attention and feed-forward encoder blocks
- Classification representation selection
- XLM-R classification head, including activation and output projection
- Label mapping from `id2label` / `label2id`
- Probability normalization and public result ordering

The implementation must not route this model through causal-language-model generation or
pretend that BERT's pooling and classifier head are equivalent to XLM-R's.

## Performance Requirements

Correct output alone is insufficient. The port must preserve the performance intent of the
reference implementation and this repository's tensor architecture:

- Use the existing tensor provider abstraction rather than scalar-only loops.
- Add or reuse SIMD-capable operations for the hot paths.
- Keep tensor shapes, row layouts, and memory movement compatible with vectorized kernels.
- Exercise both the Panama/vector path and native SIMD path where available.
- Avoid test-only shortcuts that cannot scale to the full XLM-R base checkpoint.
- Measure classification latency and memory behavior after numerical parity is established.

## Execution Paths

The port must preserve both existing execution paths during this effort:

- Tensor2 / `TensorRef` execution is the primary production path for BERT-family forward
  inference.
- Tensor1 / `AbstractTensor` execution remains supported for existing model utilities,
  direct embedding tests, legacy block tests, and other model families sharing the base
  lifecycle.

The shared BERT-family abstraction must therefore keep both weight-loading paths and both
embedding/forward paths behaviorally aligned. RoBERTa and XLM-R model-family hooks must be
applied to both paths; the tensor2 path must not be correct only because the tensor1 path is
left untouched.

## Test Porting Rules

For each selected upstream test:

- Keep the upstream test class identity recognizable.
- Keep method names recognizable, translating only syntax that has no Java equivalent.
- Preserve test ordering where it communicates setup or dependency.
- Preserve fixtures and expected values instead of replacing them with loose assertions.
- Use the same tokenizer inputs and model outputs whenever the fixture is portable.
- Add explicit tolerances only where the Java numeric path requires them.
- Explain every omitted upstream test in the document or test source.

Tests must include multilingual examples, at minimum:

```text
Hello, how are you?
Bonjour tout le monde.
これは日本語です。
Hola, ¿cómo estás?
Привет, как дела?
```

## Verification Gates

The model is not considered supported until all gates pass:

1. Configuration and tokenizer fixtures load without special-case manual setup.
2. Representative tokenizer outputs match the upstream reference.
3. Encoder intermediate or final representations match the upstream reference within the
   documented tolerance.
4. Classification logits and probabilities match the upstream reference.
5. Label names and scores are returned through the public Java API correctly.
6. Ported tests pass on the standard Java tensor path.
7. Provider parity tests pass for Panama and native SIMD where available.
8. The full relevant `core` test set passes.
9. A benchmark records classification throughput and latency on the development machine.

## Fast Regression Loop

Every change made during this effort must run these two fast BERT checks before proceeding:

```bash
MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -pl core -Dtest=ClassificationTest#classifyAuto test
MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -pl core -Dtest=BertModelingPortedTest test
```

They protect both the real BERT sequence-classification path and the existing BERT model
unit coverage while the shared encoder and classification infrastructure is being changed.
The first test asserts stable smoke probabilities for the jailbreak classifier; the second
checks the tiny-model encoder, embeddings, TensorPlan, classification shape, and residual
behavior.

## Explicit Non-Goals

- A plausible classifier with unverified output parity
- A tokenizer implemented only for the example sentences
- A scalar reference implementation presented as production support
- Reusing BERT behavior where XLM-R semantics differ
- Declaring support based only on successful model loading
