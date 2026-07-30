"""Generate ``strategies/_standard/input.schema.v1.json`` from ``core/input_contract.py``.

The output side already has a machine-checkable contract
(``signal.schema.v2.json``). The input side had the Python definitions and three
JSON *samples*, but no schema — so a JS/외부 producer feeding a bot had nothing to
validate against. This closes that: ``cyqnt.input/v1`` describes the
``BotContext`` envelope plus the seven canonical frame shapes, generated from the
same dataclasses the runtime enforces, so the two cannot drift.

    python docs/gen_input_schema_v1.py [out.json]
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cyqnt_trd.standard_bot.core import input_contract as IC  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "strategies/_standard/input.schema.v1.json")

#: Columns whose meaning is load-bearing and routinely got confused.
COLUMN_DOCS = {
    "event_time": "When the thing HAPPENED (epoch ms).",
    "available_time": "When we could first have KNOWN it (epoch ms). The PIT gate: "
                      "a backtest may only read rows whose available_time <= the "
                      "decision time. Conflating this with event_time is how a "
                      "walk-forward test silently reads the future.",
    "instrument_id": "Canonical symbol, e.g. BTCUSDT. Null on a market-wide row.",
    "metric": "Name of the measured quantity in a long-form MetricFrame, so "
              "funding / OI / long-short all share one column vocabulary.",
    "confirmed": "False for an in-flight bar. A bot must not decide on one.",
    "side": "long | short for a position; bid | ask for a book level.",
}


def _column_schema(name: str, *, numeric: set, times: set) -> dict:
    if name in times:
        out = {"type": ["integer", "null"], "description": "epoch ms"}
    elif name in numeric:
        out = {"type": ["number", "null"]}
    else:
        out = {"type": ["string", "number", "boolean", "null"]}
    if name in COLUMN_DOCS:
        out = {**out, "description": COLUMN_DOCS[name]}
    return out


def _frame_schema(fs) -> dict:
    numeric = set(getattr(fs, "numeric_columns", ()) or ())
    times = set(getattr(fs, "time_columns", ()) or ())
    required = list(fs.required)
    props = {c: _column_schema(c, numeric=numeric, times=times)
             for c in list(required) + list(fs.optional)}
    return {
        "type": "array",
        "description": "%s — row grain: %s. %s" % (
            fs.name, fs.row_grain, " ".join((fs.description or "").split())),
        "items": {
            "type": "object",
            "properties": props,
            "required": required,
        },
    }


def _frame_envelope(shapes) -> dict:
    """One ``frames[node]`` entry: the shape it claims plus the rows in it.

    The wrapper is not packaging. ``status`` / ``reason`` is the only thing that
    keeps "I could not read it" apart from "I read it and it was empty" — the
    distinction a strategy needs in order to abstain instead of treating a dead
    source as a zero. ``source_fallback`` records that a substitute transport
    answered, without which two runs are silently incomparable.

    ``rows`` is then bound to the declared ``shape`` with one if/then per shape,
    which is what makes the schema actually check something: a bare
    ``additionalProperties: true`` accepted a BarFrame carrying
    ``{"close": "not-a-number"}`` without complaint.
    """
    conditionals = [
        {
            "if": {"properties": {"shape": {"const": name}},
                   "required": ["shape"]},
            "then": {"properties": {"rows": {"$ref": "#/definitions/%s" % ref}}},
        }
        for ref, name in sorted(shapes.items(), key=lambda item: item[1])
    ]
    return {
        "type": "object",
        "required": ["shape", "rows"],
        "properties": {
            "shape": {
                "type": "string",
                "description": "canonical shape id; RawFrame@1.0 when the node "
                               "declares none.",
                "anyOf": [{"enum": sorted(shapes.values())},
                          {"const": "RawFrame@1.0"}],
            },
            "rows": {"type": "array"},
            "status": {"enum": ["ok", "empty", "error", "unnormalised"]},
            "reason": {"type": "string",
                       "description": "why the node could not be read; present "
                                      "exactly when status is error/unnormalised."},
            "availability": {"type": "string"},
            "pit_hazard": {"type": "string"},
            "source_fallback": {
                "type": "string",
                "description": "set when a substitute source answered instead of "
                               "the primary one — two runs served by different "
                               "transports are not comparable.",
            },
            "available_time_inferred": {"type": "boolean"},
            "window_start": {"type": ["integer", "null"]},
            "window_end": {"type": ["integer", "null"]},
            "rows_after_decision_time_dropped": {"type": "integer"},
            "rows_before_window_dropped": {"type": "integer"},
            "pit_safe": {"type": "boolean"},
            "source_timeframe": {"type": "string"},
            "full_row_count": {"type": "integer"},
        },
        "allOf": conditionals,
    }


def build() -> dict:
    defs = {}
    shapes = {}
    for key, fs in IC.SCHEMAS.items():
        ref = fs.name.replace("@", "_").replace(".", "_")
        defs[ref] = _frame_schema(fs)
        shapes[ref] = fs.name
    defs["Frame"] = _frame_envelope(shapes)

    frame_names = sorted(fs.name for fs in IC.SCHEMAS.values())
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": IC.INPUT_SCHEMA_VERSION,
        "title": "cyqnt standard bot input (BotContext, v1)",
        "description":
            "The complete input a bot decides on. `frames` holds each declared "
            "data node exactly as its source returned it; `typed` holds the same "
            "data normalised to one of the canonical frame shapes below, so a bot "
            "reading funding, OI and news uses one column vocabulary instead of "
            "three. `source_status` covers EVERY declared node including failures, "
            "so \"not read\" and \"read and empty\" stay distinguishable. "
            "Generated from standard_bot/core/input_contract.py by "
            "docs/gen_input_schema_v1.py; do not hand-edit.",
        "type": "object",
        "required": ["schema", "decision_time"],
        "properties": {
            "schema": {"const": IC.INPUT_SCHEMA_VERSION},
            "bot_id": {"type": "string"},
            "decision_time": {
                "type": "integer",
                "description": "epoch ms. The single point in time this whole "
                               "context is valid at; equals the snapshot's "
                               "decision_as_of. Every frame is PIT-gated to it.",
            },
            "snapshot_id": {"type": "string"},
            "run_id": {"type": "string"},
            "trace_id": {"type": "string"},
            "equity": {"type": ["number", "null"],
                       "description": "Account equity, when the caller supplies it. "
                                      "Required to resolve an absolute size mode."},
            "positions": {
                "type": "object",
                "description": "Signed exposure per symbol: >0 long, <0 short, "
                               "absent/0 flat. A bot that does not read this may "
                               "not emit close/reduce/flip intents.",
                "additionalProperties": {"type": "number"},
            },
            "frames": {
                "type": "object",
                "description":
                    "node key -> one normalised frame. Each entry names its "
                    "canonical `shape` and carries the `rows` in it, plus the "
                    "metadata a reader needs to trust them: `status` (ok | empty | "
                    "error | unnormalised), `reason` when it failed, "
                    "`source_fallback` when a substitute source answered, and the "
                    "node's `availability` / `pit_hazard`. `rows` is validated "
                    "against the shape named in `shape`, so a frame cannot claim "
                    "to be a BarFrame and carry something else.",
                "additionalProperties": {"$ref": "#/definitions/Frame"},
            },
            "source_status": {
                "type": "object",
                "description": "node key -> ok | empty | stale | error. Present for "
                               "every DECLARED node, including the ones that failed.",
                "additionalProperties": {"type": "string"},
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
            "config": {"type": "object", "additionalProperties": True},
        },
        "definitions": defs,
    }


if __name__ == "__main__":
    schema = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("wrote %s" % OUT)
    shapes = {name: body for name, body in schema["definitions"].items()
              if "items" in body}
    print("  frame shapes: %d (+ the Frame envelope)" % len(shapes))
    for name in sorted(shapes):
        print("    %-22s required=%d"
              % (name, len(shapes[name]["items"]["required"])))
