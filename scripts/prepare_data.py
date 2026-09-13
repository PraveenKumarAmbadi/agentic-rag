#!/usr/bin/env python3
"""Prepare the SciFact dataset for a fresh clone.

The data/ directory is gitignored, so a fresh clone has no corpus. This
script restores it in one of two ways:

1. If data/scifact.zip already exists locally, extract it.
2. Otherwise download the official BEIR scifact.zip and extract it.

Usage::
    python scripts/prepare_data.py
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ZIP_PATH = DATA_DIR / "scifact.zip"
CORPUS_PATH = DATA_DIR / "scifact" / "corpus.jsonl"

BEIR_SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/scifact.zip"
)


def main() -> int:
    if CORPUS_PATH.exists():
        print(f"Dataset already present at {CORPUS_PATH.parent}. Nothing to do.")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print(f"Downloading SciFact dataset from BEIR ...")
        urllib.request.urlretrieve(BEIR_SCIFACT_URL, ZIP_PATH)
        print(f"Saved to {ZIP_PATH}")

    print(f"Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)

    if not CORPUS_PATH.exists():
        print(
            f"ERROR: extraction finished but {CORPUS_PATH} is missing. "
            "Check the zip layout.",
            file=sys.stderr,
        )
        return 1

    print(f"Done. Corpus ready at {CORPUS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
