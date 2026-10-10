"""Agent tool registry — JSON-schema-validated tool calls, hand-rolled.

The agent loop asks the LLM to reply with a single JSON object naming a tool
and its arguments. This module is the single source of truth for:

- the tool vocabulary exposed to the model (name, description, params),
- parsing a raw model reply into a validated ``Action``,
- ``extract_json()``, shared by the rescue and critique parsers.

Validation is hand-rolled (no ``jsonschema`` dependency): every param declares
a JSON type and required-ness, and ``validate_action`` enforces exactly that —
wrong types, missing required params, and unknown params are all rejected so
a malformed tool call can never reach an executor.
"""

import json
from dataclasses import dataclass


class ToolCallError(ValueError):
    """A tool call was found in model output but failed validation."""


# Descriptions say what a tool DOES. When to choose each one is decision
# policy and lives in one place: DECIDE_SYSTEM_PROMPT (agent/prompts.py).
TOOL_SPECS: list[dict] = [
    {
        "name": "answer",
        "description": (
            "Hand the retrieved documents to the writer, who reads them in "
            "full and answers from them (or refuses if, read in full, they do "
            "not address the question)."
        ),
        # qwen2.5 volunteers this on almost every answer decision; with no
        # declared param, every answer decision was rejected as invalid.
        "params": {
            "documents": {
                "type": "array",
                "items": "integer",
                "required": False,
                "description": "numbers of the documents that are about the question",
            },
        },
    },
    {
        "name": "hybrid_search",
        "description": (
            "Search the corpus again with a new query. The user's input has "
            "already been searched; results are merged with the documents "
            "you already have."
        ),
        "params": {
            "query": {
                "type": "string",
                "required": True,
                "description": (
                    "a reformulation: different terms, synonyms or a narrower "
                    "focus than every earlier search"
                ),
            },
        },
    },
    {
        "name": "insufficient_evidence",
        "description": "Stop and tell the user the corpus cannot answer.",
        "params": {
            "reason": {
                "type": "string",
                "required": True,
                "description": "what specifically is missing",
            },
        },
    },
]

TOOL_NAMES = [spec["name"] for spec in TOOL_SPECS]

_PY_TYPES = {"string": str, "boolean": bool, "integer": int, "array": list}


def _is_type(value, type_name: str) -> bool:
    # bool is a subclass of int in Python; JSON true is not an integer.
    if type_name == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, _PY_TYPES[type_name])


@dataclass(frozen=True)
class Action:
    """One validated tool call parsed from model output."""

    tool: str
    args: dict


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of model output.

    Models wrap JSON in prose ("Here is my decision: {...}"), so scan from
    every ``{`` and try to decode. Returns ``None`` if no object parses.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def validate_action(obj: dict) -> Action:
    """Validate a parsed JSON object against the tool specs."""
    tool = obj.get("tool")
    if not isinstance(tool, str):
        raise ToolCallError("action must have a string 'tool' field")
    spec = next((s for s in TOOL_SPECS if s["name"] == tool), None)
    if spec is None:
        raise ToolCallError(
            f"unknown tool {tool!r}; expected one of {TOOL_NAMES}"
        )
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        raise ToolCallError("'args' must be an object")
    unknown = set(args) - set(spec["params"])
    if unknown:
        raise ToolCallError(f"unknown params for {tool}: {sorted(unknown)}")
    cleaned: dict = {}
    for name, pspec in spec["params"].items():
        if name not in args:
            if pspec["required"]:
                raise ToolCallError(f"missing required param {name!r} for {tool}")
            continue
        value = args[name]
        if not _is_type(value, pspec["type"]):
            raise ToolCallError(
                f"param {name!r} of {tool} must be a {pspec['type']}"
            )
        if pspec["type"] == "array" and not all(_is_type(v, pspec["items"]) for v in value):
            raise ToolCallError(
                f"param {name!r} of {tool} must be an array of {pspec['items']}s"
            )
        if isinstance(value, str) and not value.strip():
            raise ToolCallError(f"param {name!r} of {tool} must be non-empty")
        cleaned[name] = value
    return Action(tool=tool, args=cleaned)


def parse_action(text: str) -> Action | None:
    """Parse model output into a validated ``Action``.

    Returns ``None`` when the output contains no JSON object at all (treated
    by the agent loop as "fall through to generation"). Raises
    ``ToolCallError`` when a JSON object was found but is not a valid call —
    that distinction keeps parse errors measurable in the trace.
    """
    obj = extract_json(text)
    if obj is None:
        return None
    return validate_action(obj)


def _type_label(pspec: dict) -> str:
    if pspec["type"] == "array":
        return f"array of {pspec['items']}s"
    return pspec["type"]


def render_tool_docs() -> str:
    """Render the tool vocabulary for the decide-turn system prompt.

    Tools and their params only: the reply format and the decision policy
    are stated once, in DECIDE_SYSTEM_PROMPT.
    """
    lines = []
    for spec in TOOL_SPECS:
        lines.append(f"- {spec['name']}: {spec['description']}")
        for name, pspec in spec["params"].items():
            need = "required" if pspec["required"] else "optional"
            lines.append(
                f"    args.{name} ({_type_label(pspec)}, {need}): {pspec['description']}"
            )
    return "\n".join(lines)
