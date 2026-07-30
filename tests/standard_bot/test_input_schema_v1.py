"""``cyqnt.input/v1`` — one shape, and a schema that actually rejects things.

Two failures this guards against, both of which had already happened:

**Three dialects, one id.** ``input_bundle_example.json`` used
``frames[node] = {shape, rows}`` with integer ms; ``bot_context.json`` nested rows
under a differently-keyed wrapper with ISO strings; the schema itself described a
third form (bare arrays under ``typed``). All three announced
``"schema": "cyqnt.input/v1"``, so an integrator validating against it got a
failure from whichever artifact they happened to hold.

**A schema that validated nothing.** ``frames`` was
``additionalProperties: true``, and every real producer wrote ``frames`` rather
than ``typed`` — so the seven shape definitions never ran. A bundle whose klines
rows were ``[{"close": "not-a-number"}]`` passed clean. The pollution test below
is the part that matters: a schema nobody can fail is not a contract.
"""

from __future__ import annotations

import copy
import json
import os

import pytest

jsonschema = pytest.importorskip("jsonschema")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA_PATH = os.path.join(REPO, "strategies/_standard/input.schema.v1.json")
SAMPLES = os.path.join(REPO, "docs/standard_bot_io/samples")

#: every artifact that claims to be cyqnt.input/v1
INPUT_SAMPLES = ("live_bundle_example.json", "input_bundle_example.json",
                 "input_bundle_preview.json", "bot_context.json")


def _schema():
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _validator():
    schema = _schema()
    jsonschema.Draft7Validator.check_schema(schema)
    return jsonschema.Draft7Validator(schema)


def _sample(name):
    with open(os.path.join(SAMPLES, name), encoding="utf-8") as fh:
        return json.load(fh)


def test_the_schema_is_itself_valid():
    _validator()


@pytest.mark.parametrize("name", INPUT_SAMPLES)
def test_every_input_sample_validates(name):
    errors = sorted(_validator().iter_errors(_sample(name)),
                    key=lambda e: list(e.path))
    assert not errors, "%s: %s" % (name, [
        "%s: %s" % ("/".join(str(p) for p in e.path), e.message) for e in errors[:5]])


@pytest.mark.parametrize("name", INPUT_SAMPLES)
def test_every_input_sample_uses_the_one_frame_shape(name):
    """No dialects: rows live under ``frames[node].rows``, times are integers."""
    sample = _sample(name)
    assert sample["schema"] == "cyqnt.input/v1"
    assert "typed" not in sample, (
        "%s still carries a parallel 'typed' section — one artifact, one place "
        "for the rows" % name)
    frames = sample.get("frames") or {}
    assert frames, "%s has no frames" % name
    for key, entry in frames.items():
        assert isinstance(entry, dict), "%s/%s must be an object, not a bare array" % (name, key)
        assert "shape" in entry and "rows" in entry, "%s/%s" % (name, key)
        for row in entry["rows"][:5]:
            for column, value in row.items():
                if column.endswith("_time") and value is not None:
                    assert isinstance(value, int), (
                        "%s/%s.%s is %r — the contract types every time column as "
                        "an integer epoch-ms" % (name, key, column, value))


#: (label, mutation) — each must be rejected. If any of these passes, the schema
#: has gone back to being decorative.
POLLUTION = [
    ("a numeric column carrying a string",
     lambda d: d["frames"]["klines"]["rows"][0].update({"close": "not-a-number"})),
    ("a required column removed",
     lambda d: d["frames"]["klines"]["rows"][0].pop("instrument_id")),
    ("rows that do not match the declared shape",
     lambda d: d["frames"]["klines"].update({"shape": "MetricFrame@1.0"})),
    ("an undefined status value",
     lambda d: d["frames"]["klines"].update({"status": "probably_fine"})),
    ("a frame with no shape",
     lambda d: d["frames"]["klines"].pop("shape")),
    ("a frame whose rows are not an array",
     lambda d: d["frames"]["klines"].update({"rows": {"close": 1.0}})),
    ("a decision_time that is not epoch ms",
     lambda d: d.update({"decision_time": "2026-08-06T07:00:00Z"})),
    ("the wrong schema id",
     lambda d: d.update({"schema": "cyqnt.input/v2"})),
]


@pytest.mark.parametrize("label,mutate", POLLUTION, ids=[p[0] for p in POLLUTION])
def test_the_schema_rejects_polluted_input(label, mutate):
    sample = copy.deepcopy(_sample("live_bundle_example.json"))
    assert not list(_validator().iter_errors(sample)), "baseline must be clean"
    mutate(sample)
    assert list(_validator().iter_errors(sample)), (
        "the schema accepted %s — it is not checking the rows" % label)


def test_the_schema_is_generated_not_hand_edited(tmp_path):
    """Regenerating must be a no-op, or the contract and the code have drifted.

    Writes into pytest's own tmp_path rather than a fixed path in the repo: a
    shared filename races with any other run (including the doc generator, which
    invokes the suite itself) and produces a failure that reproduces nowhere.
    """
    import subprocess
    import sys

    out = str(tmp_path / "input.schema.v1.check.json")
    subprocess.run([sys.executable, os.path.join(REPO, "docs/gen_input_schema_v1.py"), out],
                   cwd=REPO, check=True, capture_output=True)
    with open(out, encoding="utf-8") as fh:
        regenerated = json.load(fh)
    assert regenerated == _schema(), (
        "strategies/_standard/input.schema.v1.json is stale — run "
        "python docs/gen_input_schema_v1.py")
