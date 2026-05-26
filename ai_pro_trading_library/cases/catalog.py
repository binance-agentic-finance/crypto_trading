"""Case catalog loader."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def load_catalog() -> dict[str, Any]:
    catalog_path = files("ai_pro_trading_library.cases").joinpath("catalog.json")
    return json.loads(catalog_path.read_text())


def list_case_ids() -> tuple[str, ...]:
    return tuple(case["case_id"] for case in load_catalog()["cases"])

