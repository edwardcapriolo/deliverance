#!/usr/bin/env python3
"""Print Hugging Face GraniteMoeHybrid reference logits for a local checkpoint.

Usage from the Deliverance repo root:

    PYTHONPATH=/path/to/transformers/src \
      python core/src/test/resources/granite_moe_hybrid_reference.py \
      --model-dir /path/to/tiny-granite-checkpoint

The script intentionally works from a local checkpoint directory so Java tests can
write deterministic synthetic safetensors first, then use Transformers as the
independent oracle for expected logits.
"""

import argparse
import json
import pathlib

import torch
from transformers import GraniteMoeHybridForCausalLM


def parse_input_ids(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Local HF checkpoint directory")
    parser.add_argument("--input-ids", default="3,4,5,6", help="Comma-separated token ids")
    parser.add_argument("--slice", type=int, default=8, help="Number of final-token logits to print")
    args = parser.parse_args()

    model_dir = pathlib.Path(args.model_dir).expanduser().resolve()
    input_ids = parse_input_ids(args.input_ids)

    model = GraniteMoeHybridForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.float32,
        device_map="cpu",
        local_files_only=True,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(torch.tensor([input_ids], dtype=torch.long))

    logits = outputs.logits[0].float()
    result = {
        "model_dir": str(model_dir),
        "input_ids": input_ids,
        "logits_shape": list(outputs.logits.shape),
        "last_token_logits_slice": logits[-1, : args.slice].tolist(),
        "mean_logits_by_position": logits.mean(dim=-1).tolist(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
