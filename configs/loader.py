"""Centralized, validated YAML configuration loader.

Resolution order (later wins):
  1. defaults in configs/schema.py
  2. the YAML file (configs/config.yaml, or $AGENTIC_RAG_CONFIG)
  3. environment overrides:
       - short aliases for common deployment values (ENV_OVERRIDES)
       - any key via AGENTIC_RAG__<SECTION>__<KEY>, e.g.
         AGENTIC_RAG__AGENT__REQUEST_TIMEOUT_S=600 (case-insensitive;
         unknown keys are an error, like in the YAML)

The result is validated and returned as a plain nested dict, so call sites
keep using ``cfg["section"]["key"]`` while every key is guaranteed present.
"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from configs.schema import AppConfig

CONFIG_PATH = Path(__file__).parent / "config.yaml"
CONFIG_PATH_ENV = "AGENTIC_RAG_CONFIG"
GENERIC_ENV_PREFIX = "AGENTIC_RAG__"

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


def _resolve_path(env_name: str) -> list[str]:
    """Map AGENTIC_RAG__AGENT__TOP_K to the schema path ["agent", "top_k"]."""
    parts = env_name[len(GENERIC_ENV_PREFIX):].split("__")
    model: type[BaseModel] | None = AppConfig
    path = []
    for part in parts:
        fields = model.model_fields if model is not None else {}
        name = next((f for f in fields if f.lower() == part.lower()), None)
        if name is None:
            raise ConfigError(f"{env_name}: no config key {'.'.join(path + [part.lower()])}")
        path.append(name)
        annotation = fields[name].annotation
        model = annotation if isinstance(annotation, type) and issubclass(annotation, BaseModel) else None
    if model is not None:
        raise ConfigError(f"{env_name}: {'.'.join(path)} is a section, not a key")
    return path


def _apply_env_overrides(raw: dict, environ: dict[str, str]) -> dict:
    for env_name, (section, key) in ENV_OVERRIDES.items():
        value = environ.get(env_name)
        if value:
            raw.setdefault(section, {})[key] = value
    for env_name in sorted(environ):
        if not env_name.startswith(GENERIC_ENV_PREFIX) or environ[env_name] == "":
            continue
        *parents, key = _resolve_path(env_name)
        node = raw
        for name in parents:
            node = node.setdefault(name, {})
            if not isinstance(node, dict):
                raise ConfigError(f"{env_name}: {name} is not a section in the config file")
        node[key] = environ[env_name]
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
