# Embedding Showdown

This harness compares Deliverance's Java EmbeddingGemma 2 text path against the
upstream Sentence Transformers implementation on the same local checkpoint and
the same controlled corpus.

It is both a correctness gate and a throughput benchmark. The benchmark does
not compare different models, token limits, prompts, or corpora.

## Workload

The controlled workload is:

```text
core/src/test/resources/embedding-showdown.jsonl
```

It contains fixed cases covering short retrieval text, punctuation, whitespace,
Unicode, code, structured text, numbers, a longer paragraph, and query/document
pairs. The file is versioned so benchmark changes are visible in code review.

Both runners use:

- the same checkpoint directory
- the same JSONL case order
- the same tokenizer special-token behavior
- mean pooling and float32 L2 normalization
- CPU execution
- one embedding per timed call
- independent warmup before timed iterations

## Requirements

Java requires the normal Deliverance build environment. Python requires:

```sh
python3 -m pip install sentence-transformers numpy pillow torchvision
```

The checkpoint must already exist locally. Set `MODEL_DIR` to the directory
containing `config.json`, tokenizer files, and SafeTensors weights.

## Run

From the repository root:

```sh
MODEL_DIR="$HOME/.deliverance/google_embeddinggemma-2" \
PYTHON=python3 \
WARMUP=3 \
ITERATIONS=10 \
sh core/scripts/run-embedding-showdown.sh
```

The script writes:

```text
core/target/embedding-showdown-java.json
core/target/embedding-showdown-python.json
```

Override the workload size or output paths:

```sh
MODEL_DIR="$HOME/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/<revision>" \
CORPUS=core/src/test/resources/embedding-showdown.jsonl \
WARMUP=10 \
ITERATIONS=100 \
POOL_SIZE=1 \
sh core/scripts/run-embedding-showdown.sh
```

## Correctness Gates

For a quick single-document comparison before running the full showdown:

```sh
python3 core/scripts/embedding_gemma2_reference.py \
  --model-dir "$HOME/.deliverance/google_embeddinggemma-2"
```

The matching Java runner is `EmbeddingGemma2ReferenceMain` and emits
`core/target/embedding-gemma2-reference-java.json`. It uses the same default
text, tokenizer call, mean pooling, projection, and normalization.

To create the checked-in full-vector fixture used by
`EmbeddingGemma2ReferenceParityTest`:

```sh
python3 core/scripts/embedding_gemma2_reference.py \
  --model-dir "$HOME/.deliverance/google_embeddinggemma-2" \
  --output core/src/test/resources/embed_gemma2.json
```

The parity test then compares every token ID, all 768 vector values, dimension,
and final L2 norm.

The comparator fails if any of these conditions fail:

- case IDs or order differ
- tokenizer token IDs differ exactly
- embedding dimensions differ
- maximum absolute vector error exceeds `0.02`
- cosine similarity falls below `0.9999`
- vectors differ after quantization to `1e-4` precision

The raw float-bit SHA-256 is also recorded for diagnostics. It is not expected
to match across Java and PyTorch because provider order, BF16 conversion, and
floating-point reduction order can differ. Exact token IDs plus the explicit
vector tolerance and quantized hash are the reproducible parity contract.

Tighten the gate for a stricter run:

```sh
python3 core/scripts/compare_embedding_showdown.py \
  --max-abs-error 0.005 \
  --min-cosine 0.99999 \
  --quantum 0.00001
```

## Output

Each runner prints mean, p50, p95, and p99 latency plus embeddings/sec and
tokens/sec. The final comparator prints:

```text
parity.cases=12
parity.max_abs_error=...
parity.min_cosine=...
java.embeddings_s=...
python.embeddings_s=...
python_vs_java.speedup=...
parity.status=PASS
```

The Python side is the upstream reference for this comparison. A passing run
means the same controlled inputs produced matching tokenization and embedding
vectors within the documented numerical contract before speed is considered.

## Fairness Notes

- Run both sides on the same machine with the same checkpoint revision.
- Keep `POOL_SIZE=1` when comparing single-thread latency.
- Use a larger `ITERATIONS` value for stable throughput numbers.
- Do not compare a Sentence Transformers GPU run against Java CPU output.
- Native provider selection is printed by the Java model startup logs and should
  be recorded with benchmark results.
