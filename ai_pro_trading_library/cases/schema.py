"""Lightweight JSON Schema validation for case.json files.

Avoids the `jsonschema` dependency by implementing the small subset of
draft-2020-12 used in `case.schema.json` (required, type, enum, pattern,
additionalProperties).
"""

from __future__ import annotations

import json
import re
from importlib.resources import files
from typing import Any


_SCHEMA_CACHE: dict[str, Any] = {}


class CaseValidationError(ValueError):
    pass


def _load_schema() -> dict[str, Any]:
    if "case" not in _SCHEMA_CACHE:
        text = files("ai_pro_trading_library.cases").joinpath("case.schema.json").read_text()
        _SCHEMA_CACHE["case"] = json.loads(text)
    return _SCHEMA_CACHE["case"]


_TYPE_MAP = {
    "string": str,
    "object": dict,
    "array": list,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def _check_type(value: Any, expected: str, path: str) -> None:
    py = _TYPE_MAP.get(expected)
    if py is None:
        return
    if not isinstance(value, py) or (expected == "integer" and isinstance(value, bool)):
        raise CaseValidationError(f"{path}: expected {expected}, got {type(value).__name__}")


def _validate(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    if "type" in schema:
        _check_type(value, schema["type"], path)
    if "enum" in schema and value not in schema["enum"]:
        raise CaseValidationError(f"{path}: {value!r} not in enum {schema['enum']}")
    if "pattern" in schema and isinstance(value, str):
        if not re.search(schema["pattern"], value):
            raise CaseValidationError(f"{path}: {value!r} does not match pattern {schema['pattern']}")
    if schema.get("type") == "object" and isinstance(value, dict):
        for required in schema.get("required", []):
            if required not in value:
                raise CaseValidationError(f"{path}: missing required field {required!r}")
        for key, child_schema in schema.get("properties", {}).items():
            if key in value:
                _validate(value[key], child_schema, f"{path}.{key}")
    if schema.get("type") == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if item_schema is not None:
            for i, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{i}]")


def validate_case(case: dict[str, Any]) -> None:
    """Raise CaseValidationError if `case` does not match case.schema.json."""
    _validate(case, _load_schema())
