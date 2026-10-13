#!/usr/bin/env python3
"""Emit one deterministic Sentence Transformers EmbeddingGemma 2 reference vector."""

import argparse
import json
import pathlib

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_TEXT = "This is a test document about machine learning"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output", default="core/target/embedding-gemma2-reference-python.json")
    args = parser.parse_args()

    model = SentenceTransformer(args.model_dir, device="cpu")
    tokenized = model.tokenizer(args.text, add_special_tokens=True, truncation=False,
                                return_attention_mask=True)
    vector = np.asarray(model.encode([args.text], batch_size=1, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)[0], dtype=np.float32)
    result = {
        "text": args.text,
        "tokenIds": [int(value) for value in tokenized["input_ids"]],
        "embedding": vector.tolist(),
        "dimension": int(vector.size),
        "l2Norm": float(np.linalg.norm(vector.astype(np.float64))),
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
