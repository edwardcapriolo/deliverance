#!/usr/bin/env python3
"""Run the Sentence Transformers side of the EmbeddingGemma 2 showdown."""

import argparse
import hashlib
import json
import pathlib
import struct
import time

import numpy as np
from sentence_transformers import SentenceTransformer


def float_bits_sha256(values):
    digest = hashlib.sha256()
    for value in np.asarray(values, dtype=np.float32).reshape(-1):
        digest.update(struct.pack(">f", float(value)))
    return digest.hexdigest()


def read_cases(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_one(model, text):
    return np.asarray(
        model.encode([text], batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0],
        dtype=np.float32,
    ).reshape(-1)


def tokenizer_ids(model, text):
    encoded = model.tokenizer(text, add_special_tokens=True, truncation=False, return_attention_mask=False)
    return [int(value) for value in encoded["input_ids"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--corpus", default="core/src/test/resources/embedding-showdown.jsonl")
    parser.add_argument("--output", default="core/target/embedding-showdown-python.json")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    corpus = read_cases(pathlib.Path(args.corpus))
    model = SentenceTransformer(args.model_dir, device="cpu")
    for _ in range(args.warmup):
        for entry in corpus:
            encode_one(model, entry["text"])

    results = []
    samples = []
    token_count = 0
    for entry in corpus:
        vector = encode_one(model, entry["text"])
        results.append(
            {
                "id": entry["id"],
                "tokenIds": tokenizer_ids(model, entry["text"]),
                "embedding": vector.tolist(),
                "embeddingBitsSha256": float_bits_sha256(vector),
            }
        )

    for _ in range(args.iterations):
        for entry in corpus:
            start = time.perf_counter_ns()
            encode_one(model, entry["text"])
            samples.append(time.perf_counter_ns() - start)
            token_count += len(tokenizer_ids(model, entry["text"]))

    samples.sort()
    seconds = sum(samples) / 1_000_000_000.0

    def percentile(value):
        return samples[min(len(samples) - 1, max(0, int(np.ceil(value * len(samples))) - 1))] / 1_000_000.0

    report = {
        "engine": "sentence-transformers-python",
        "model": str(pathlib.Path(args.model_dir)),
        "corpus": args.corpus,
        "cases": results,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "stats": {
            "count": len(samples),
            "mean_ms": seconds * 1000.0 / len(samples),
            "p50_ms": percentile(0.50),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
            "embeddings_s": len(samples) / seconds,
            "tokens_s": token_count / seconds,
        },
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats = report["stats"]
    print(
        f"python cases={len(corpus)} iterations={args.iterations} mean_ms={stats['mean_ms']:.3f} "
        f"p50_ms={stats['p50_ms']:.3f} p95_ms={stats['p95_ms']:.3f} p99_ms={stats['p99_ms']:.3f} "
        f"embeddings_s={stats['embeddings_s']:.1f} tokens_s={stats['tokens_s']:.1f} output={output}"
    )


if __name__ == "__main__":
    main()
