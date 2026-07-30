"""Generate ``strategies/_standard/signal.schema.v2.json`` from the v2 dataclasses.

``cyqnt.signal/v2`` is defined in Python (``standard_bot/core/signal_contract.py``),
which a JavaScript execution layer cannot read. This walks those dataclasses and
emits an equivalent JSON Schema so the contract is machine-checkable on both
sides, and so it cannot drift: regenerate, and a changed dataclass changes the
schema.

    python docs/gen_signal_schema_v2.py [out.json]
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import sys
import typing

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cyqnt_trd.standard_bot.core import signal_contract as SC  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "strategies/_standard/signal.schema.v2.json")

_PRIM = {str: "string", int: "integer", float: "number", bool: "boolean"}

#: Fields whose meaning is not obvious from the name. Everything else relies on
#: the dataclass docstrings, which are emitted as the object description.
FIELD_DOCS = {
    "decision_time": "epoch ms; equals the snapshot's decision_as_of. The bar/quote "
                     "this decision was made on — not when it was sent.",
    "intent": "The position lifecycle instruction. close_long and open_short are "
              "DIFFERENT: one reduces existing exposure, the other creates new "
              "opposite exposure, and on spot only the first is possible.",
    "reduce_only": "Derived from intent; an executor must honour it.",
    "score": "0-100 strength.",
    "confidence": "0-1 confidence in the evidence, independent of strength.",
    "candidates": "Selection bots only: the ranked shortlist. A trade signal "
                  "leaves this empty, and vice versa — the shape is the same.",
    "dedup_key": "Stable across re-emissions of the same decision, so a consumer "
                 "can suppress duplicates. NOT an execution idempotency key: "
                 "that is instance:node:event_ref and only the executor knows it.",
    "exchange_managed": "True when the stop is placed at the venue. When False the "
                        "position is unprotected if the process dies.",
}


def _unwrap_optional(tp):
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]  # noqa: E721
        if len(args) == 1:
            return args[0], True
    return tp, False


def _allow_null(sch: dict) -> dict:
    """An ``Optional[X]`` field serializes to ``null``, so the schema must say so.

    Without this every optional field (entry / exit_plan / advisory_action /
    candidates[].trade / every nullable price) fails validation against the real
    JSON, which is emitted by ``to_dict()`` and keeps the Nones.
    """
    if "$ref" in sch:
        # A ref cannot carry a type; wrap it so null stays legal.
        return {"anyOf": [sch, {"type": "null"}]}
    t = sch.get("type")
    if t is None:
        return sch
    types = list(t) if isinstance(t, list) else [t]
    if "null" not in types:
        types.append("null")
    out = {**sch, "type": types}
    if "enum" in out and None not in out["enum"]:
        out["enum"] = list(out["enum"]) + [None]
    return out


def _schema_for(tp, defs):
    tp, optional = _unwrap_optional(tp)
    origin = typing.get_origin(tp)
    if optional:
        return _allow_null(_schema_for_inner(tp, defs))
    return _schema_for_inner(tp, defs)


def _schema_for_inner(tp, defs):
    origin = typing.get_origin(tp)

    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return {"type": "string", "enum": [m.value for m in tp]}
    if tp in _PRIM:
        return {"type": _PRIM[tp]}
    if origin in (list, tuple):
        args = [a for a in typing.get_args(tp) if a is not Ellipsis]
        item = _schema_for(args[0], defs) if args else {}
        return {"type": "array", "items": item}
    if origin is dict:
        return {"type": "object", "additionalProperties": True}
    if dataclasses.is_dataclass(tp):
        name = tp.__name__
        if name not in defs:
            defs[name] = {}          # reserve first: guards self-reference
            defs[name] = _object_schema(tp, defs)
        return {"$ref": "#/definitions/%s" % name}
    return {}


def _object_schema(cls, defs):
    # signal_contract uses `from __future__ import annotations`, so
    # dataclasses.fields() hands back annotation STRINGS. Resolve them against
    # the defining module or every nested dataclass silently degrades to {}.
    hints = typing.get_type_hints(cls, vars(sys.modules[cls.__module__]))
    props, required = {}, []
    for f in dataclasses.fields(cls):
        if f.name.startswith("_"):
            continue
        sub = _schema_for(hints.get(f.name, f.type), defs)
        doc = FIELD_DOCS.get(f.name)
        if doc:
            sub = {**sub, "description": doc}
        props[f.name] = sub
        has_default = (f.default is not dataclasses.MISSING
                       or f.default_factory is not dataclasses.MISSING)  # type: ignore[misc]
        if not has_default:
            required.append(f.name)
    out = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    if cls.__doc__:
        out["description"] = " ".join(cls.__doc__.split())
    return out


def build() -> dict:
    defs: dict = {}
    root = _object_schema(SC.StandardSignal, defs)
    # StandardSignal.to_dict() flattens the intent into three derived strings and
    # emits schema/enum values rather than objects; mirror that here, because the
    # JSON a consumer receives comes from to_dict(), not from asdict().
    root["properties"]["schema"] = {"const": SC.SCHEMA_VERSION}
    # Derive the enums from PositionIntent rather than hand-listing them: the
    # hand-written list silently omitted "flat", so a valid close_* signal failed
    # its own schema.
    for name, doc in (
        ("target_side", "The side this signal moves TOWARD."),
        ("closes_side", "The side being reduced or closed."),
        ("order_side", "The exchange-level order side."),
    ):
        values = sorted({getattr(intent, name) for intent in SC.PositionIntent},
                        key=lambda v: (v is None, v))
        root["properties"][name] = {
            "type": ["string", "null"], "enum": values, "description": doc,
        }
    # ``reduce_only`` is emitted by to_dict() from the intent but has no
    # StandardSignal field, so field introspection alone left it out of the
    # schema entirely: valid output failed validation, and the executor had no
    # documented way to know the flag exists.
    root["properties"]["reduce_only"] = {
        "type": "boolean",
        "description":
            "True when the instruction may only shrink an existing position. A "
            "property of the intent, never an independent claim — REDUCE_*, "
            "CLOSE_* and FLAT are reduce-only; everything else is not.",
    }
    # ``kind`` is Optional on the dataclass only so that "unset" can mean
    # "derive it from the payload". __post_init__ always resolves it, so the
    # wire format never carries null and the schema must not permit one.
    root["properties"]["kind"] = {
        "type": "string",
        "enum": [item.value for item in SC.SignalKind if item is not SC.SignalKind.NOOP],
        "description":
            "Which shape of answer this is, and the ONE field to switch on: "
            "trade fills entry/exit_plan/size, selection fills candidates/"
            "universe_size, alert fills advisory_action/summary. Derived from the "
            "payload, so it cannot disagree with the fields it labels. `noop` "
            "exists in the enum for v1 compatibility and is never emitted — "
            "`intent=hold` already says no-change.",
    }
    root["required"] = ["schema", "kind", "bot_id", "decision_time", "intent", "provenance"]
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": SC.SCHEMA_VERSION,
        "title": "cyqnt standard bot signal (v2)",
        "description":
            "One shape for every bot kind. A trade signal fills entry/exit_plan/size; "
            "a selection signal fills candidates/universe_size; an advisory signal "
            "fills advisory_action/summary. The key set is identical — consumers "
            "parse one schema and branch on `kind`. "
            "Generated from standard_bot/core/signal_contract.py by "
            "docs/gen_signal_schema_v2.py; do not hand-edit.",
        "definitions": defs,
        **root,
    }


if __name__ == "__main__":
    schema = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("wrote %s" % OUT)
    print("  definitions: %d" % len(schema["definitions"]))
    print("  top-level properties: %d" % len(schema["properties"]))
