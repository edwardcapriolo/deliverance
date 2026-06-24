#!/usr/bin/env python3
import argparse
import hashlib
import pathlib
import time

from tokenizers import Tokenizer


def ids_sha256(ids):
    h = hashlib.sha256()
    for token_id in ids:
        h.update(int(token_id).to_bytes(4, byteorder="big", signed=True))
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text-file", default="grace/src/test/resources/tokenizer-showdown.txt")
    parser.add_argument("--repeat", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    text = pathlib.Path(args.text_file).read_text(encoding="utf-8") * args.repeat
    text_bytes = len(text.encode("utf-8"))
    tokenizer = Tokenizer.from_file(str(pathlib.Path(args.model_dir) / "tokenizer.json"))
    ids = tokenizer.encode(text).ids
    print(f"hf.ids_sha256={ids_sha256(ids)}")
    print(f"hf.tokens={len(ids)}")
    print(f"hf.chars={len(text)}")
    print(f"hf.bytes={text_bytes}")

    for _ in range(args.warmup):
        tokenizer.encode(text)

    start = time.perf_counter_ns()
    token_count = 0
    for _ in range(args.iterations):
        token_count += len(tokenizer.encode(text).ids)
    elapsed = time.perf_counter_ns() - start
    seconds = elapsed / 1_000_000_000
    mean_ms = (elapsed / 1_000_000) / args.iterations
    chars_s = (len(text) * args.iterations) / seconds
    tokens_s = token_count / seconds
    print(f"hf iterations={args.iterations} mean_ms={mean_ms:.3f} chars_s={chars_s:.1f} tokens_s={tokens_s:.1f}")


if __name__ == "__main__":
    main()
