#!/usr/bin/env python3
"""Compare Java and Sentence Transformers EmbeddingGemma 2 showdown output."""

import argparse
import hashlib
import json
import math
import pathlib
import struct

import numpy as np


def quantized_sha256(values, quantum):
    result = hashlib.sha256()
    scaled = np.asarray(values, dtype=np.float64) / quantum
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5)).astype(np.int64)
    for value in rounded:
        result.update(struct.pack(">q", int(value)))
    return result.hexdigest()


def cosine(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--java", default="core/target/embedding-showdown-java.json")
    parser.add_argument("--python", dest="python_report", default="core/target/embedding-showdown-python.json")
    parser.add_argument("--max-abs-error", type=float, default=0.02)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--quantum", type=float, default=1.0e-4)
    args = parser.parse_args()

    java = json.loads(pathlib.Path(args.java).read_text(encoding="utf-8"))
    python = json.loads(pathlib.Path(args.python_report).read_text(encoding="utf-8"))
    java_cases = {entry["id"]: entry for entry in java["cases"]}
    python_cases = {entry["id"]: entry for entry in python["cases"]}
    if list(java_cases) != list(python_cases):
        raise SystemExit("FAIL case IDs/order differ")

    worst_abs = 0.0
    worst_cosine = 1.0
    for case_id in java_cases:
        left = java_cases[case_id]
        right = python_cases[case_id]
        if left["tokenIds"] != right["tokenIds"]:
            raise SystemExit(f"FAIL token IDs differ for {case_id}")
        left_vector = np.asarray(left["embedding"], dtype=np.float64)
        right_vector = np.asarray(right["embedding"], dtype=np.float64)
        if left_vector.shape != right_vector.shape:
            raise SystemExit(f"FAIL embedding shape differs for {case_id}")
        absolute = float(np.max(np.abs(left_vector - right_vector)))
        similarity = cosine(left_vector, right_vector)
        worst_abs = max(worst_abs, absolute)
        worst_cosine = min(worst_cosine, similarity)
        if absolute > args.max_abs_error or similarity < args.min_cosine:
            raise SystemExit(
                f"FAIL {case_id} max_abs_error={absolute:.8f} cosine={similarity:.8f}"
            )
        if quantized_sha256(left_vector, args.quantum) != quantized_sha256(right_vector, args.quantum):
            raise SystemExit(f"FAIL quantized embedding hash differs for {case_id}")

    java_stats = java["stats"]
    python_stats = python["stats"]
    print(f"parity.cases={len(java_cases)}")
    print(f"parity.max_abs_error={worst_abs:.8f}")
    print(f"parity.min_cosine={worst_cosine:.8f}")
    print(f"java.embeddings_s={java_stats['embeddings_s']:.1f}")
    print(f"python.embeddings_s={python_stats['embeddings_s']:.1f}")
    print(f"python_vs_java.speedup={python_stats['embeddings_s'] / java_stats['embeddings_s']:.3f}x")
    print("parity.status=PASS")


if __name__ == "__main__":
    main()
