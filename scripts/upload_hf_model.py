#!/usr/bin/env python3
"""Create a Hugging Face model repo if needed and upload a local model directory."""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Hugging Face repo id, for example edwardcapriolo/model-name")
    parser.add_argument("local_dir", help="Local model directory to upload")
    parser.add_argument("--private", action="store_true", help="Create the model repo as private")
    parser.add_argument("--message", default="Upload Deliverance model", help="Commit message")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    local_dir = Path(args.local_dir).expanduser().resolve()
    if not local_dir.is_dir():
        raise SystemExit(f"local_dir is not a directory: {local_dir}")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        commit_message=args.message,
    )
    print(f"Uploaded {local_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
