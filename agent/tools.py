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


TOOL_SPECS: list[dict] = [
    {
        "name": "hybrid_search",
        "description": (
            "Search the corpus with BM25+SPLADE+dense fusion. The first "
            "search runs automatically with the user's question; on later "
            "calls reformulate the query (different terminology, more "
            "specific) when the documents retrieved so far are clearly "
            "off-topic for the question."
        ),
        "params": {
            "query": {
                "type": "string",
                "required": True,
                "description": "The search query.",
            },
        },
    },
    {
        "name": "answer",
        "description": (
            "Answer the question from the currently retrieved documents. "
            "Prefer this whenever ANY retrieved document plausibly bears on "
            "the question, even partially or for one aspect."
        ),
        "params": {},
    },
    {
        "name": "insufficient_evidence",
        "description": (
            "Declare that the retrieved documents cannot answer the "
            "question. Only choose this after reviewing documents whose "
            "topics genuinely do not address the question: refusing when a "
            "retrieved document addresses the question is a worse failure "
            "than an imperfect answer."
        ),
        "params": {
            "reason": {
                "type": "string",
                "required": True,
                "description": "What specifically is missing.",
            },
        },
    },
]

TOOL_NAMES = [spec["name"] for spec in TOOL_SPECS]

_PY_TYPES = {"string": str, "boolean": bool, "integer": int}


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
        if not isinstance(value, _PY_TYPES[pspec["type"]]):
            raise ToolCallError(
                f"param {name!r} of {tool} must be a {pspec['type']}"
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


def render_tool_docs() -> str:
    """Render the tool vocabulary for the decide-turn system prompt."""
    lines = [
        "Available tools — reply with exactly ONE JSON object of the form "
        '{"tool": "<name>", "args": {...}}:'
    ]
    for spec in TOOL_SPECS:
        params = ", ".join(
            f'{name} ({pspec["type"]}, {"required" if pspec["required"] else "optional"})'
            for name, pspec in spec["params"].items()
        )
        sig = f"{spec['name']}({params})" if params else spec["name"] + "()"
        lines.append(f"- {sig}: {spec['description']}")
    return "\n".join(lines)
