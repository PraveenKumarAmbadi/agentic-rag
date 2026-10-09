#!/usr/bin/env python3
"""Download the SciFact dataset (BEIR) if it is not present.

The data/ directory is gitignored, so a fresh clone has no corpus. This
script fetches the official BEIR scifact.zip and extracts it so that the
configured data.corpus_path exists (default data/scifact/corpus.jsonl).
Idempotent: does nothing when the corpus is already there.

Usage::
    python -m scripts.prepare_data
"""

import logging
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from configs.loader import load_config

BEIR_SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/scifact.zip"
)

logger = logging.getLogger(__name__)


def ensure_scifact(corpus_path: Path, url: str = BEIR_SCIFACT_URL) -> Path:
    """Make sure the SciFact corpus exists at `corpus_path`.

    The zip contains a top-level `scifact/` folder, so it is extracted into
    the parent of the corpus' folder. The download goes to a temporary file
    and is deleted afterwards, so an interrupted download never leaves a
    corrupt archive behind.

    Raises:
        RuntimeError: if extraction does not produce `corpus_path`.
    """
    corpus_path = Path(corpus_path)
    if corpus_path.is_file():
        logger.info("SciFact already present at %s", corpus_path.parent)
        return corpus_path

    extract_to = corpus_path.parent.parent
    extract_to.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "scifact.zip"
        logger.info("Downloading SciFact from %s ...", url)
        urllib.request.urlretrieve(url, archive)
        logger.info("Extracting into %s ...", extract_to)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_to)

    if not corpus_path.is_file():
        raise RuntimeError(
            f"Extraction finished but {corpus_path} is missing; check "
            "data.corpus_path against the archive layout (scifact/corpus.jsonl)."
        )
    logger.info("SciFact ready at %s", corpus_path.parent)
    return corpus_path


def main() -> int:
    from tracing import configure_logging

    configure_logging(json_format=False)
    try:
        ensure_scifact(Path(load_config()["data"]["corpus_path"]))
    except Exception as exc:
        logger.error("Could not prepare SciFact: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
