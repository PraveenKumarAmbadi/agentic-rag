"""Centralized YAML configuration loader."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Load config.yaml from the configs/ package.

    Args:
        path: Optional override for the config file location.

    Returns:
        Parsed config as a nested dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            "Expected configs/config.yaml in the repository root."
        )

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
