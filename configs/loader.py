"""Centralized, validated YAML configuration loader.

Resolution order (later wins):
  1. defaults in configs/schema.py
  2. the YAML file (configs/config.yaml, or $AGENTIC_RAG_CONFIG)
  3. environment overrides for deployment-specific values (ENV_OVERRIDES)

The result is validated and returned as a plain nested dict, so call sites
keep using ``cfg["section"]["key"]`` while every key is guaranteed present.
"""

import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from configs.schema import AppConfig

CONFIG_PATH = Path(__file__).parent / "config.yaml"
CONFIG_PATH_ENV = "AGENTIC_RAG_CONFIG"

# env var -> (section, key). Only values that legitimately differ between
# deployments belong here; secrets (API keys) are read where they are used
# and never enter the config dict.
ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "ES_HOST": ("elasticsearch", "host"),
    "ES_PORT": ("elasticsearch", "port"),
    "LLM_BASE_URL": ("llm", "base_url"),
    "LLM_MODEL": ("llm", "model"),
}


class ConfigError(ValueError):
    """The configuration file is missing, unreadable, or invalid."""


def validate_config(raw: dict) -> dict:
    """Validate a raw config mapping and fill defaults.

    Raises:
        ConfigError: listing every invalid or unknown key.
    """
    try:
        return AppConfig.model_validate(raw).model_dump()
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration:\n{exc}") from exc


def _apply_env_overrides(raw: dict, environ: dict[str, str]) -> dict:
    for env_name, (section, key) in ENV_OVERRIDES.items():
        value = environ.get(env_name)
        if value:
            raw.setdefault(section, {})[key] = value
    return raw


def load_config(
    path: Path | str | None = None, environ: dict[str, str] | None = None
) -> dict:
    """Load, override, and validate the application config.

    Args:
        path: Explicit config file; defaults to $AGENTIC_RAG_CONFIG, then
            configs/config.yaml.
        environ: Environment mapping (defaults to os.environ; injectable for tests).

    Raises:
        ConfigError: if the file is missing, not a mapping, or fails validation.
    """
    environ = os.environ if environ is None else environ
    path = Path(path or environ.get(CONFIG_PATH_ENV) or CONFIG_PATH)
    if not path.is_file():
        raise ConfigError(
            f"Config file not found: {path}. Run from the repository root or "
            f"set {CONFIG_PATH_ENV}."
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at top level.")
    return validate_config(_apply_env_overrides(raw, environ))
