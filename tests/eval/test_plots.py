"""The offline report must show missing evidence and keep supplied text inert."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

EVAL = Path(__file__).resolve().parents[2] / "eval"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from factor_eval.plots import GROUPS, render_diagnostics  # noqa: E402


def test_empty_diagnostics_still_outputs_every_group_and_portable_html(tmp_path):
    card = SimpleNamespace(name="empty", verdict="HOLD_INCOMPLETE", diagnostics={})
    paths = render_diagnostics(card, tmp_path / "report")
    assert set(paths["images"]) == {key for key, _, _ in GROUPS}
    html = Path(paths["html"]).read_text()
    assert "N/A" in html and "HOLD_INCOMPLETE" in html
    for filename in paths["images"].values():
        path = Path(filename)
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert f'src="{path.name}"' in html
    assert "https://" not in html and "http://" not in html
    assert str(tmp_path) not in html


def test_report_escapes_names_verdicts_and_missing_data_reasons(tmp_path):
    hostile = '<script>alert("text")</script>'
    card = SimpleNamespace(name=hostile, verdict=hostile,
                           diagnostics={"ic_vs_n": {"status": "N/A", "reason": hostile}})
    paths = render_diagnostics(card, tmp_path)
    html = Path(paths["html"]).read_text()
    assert hostile not in html
    assert "&lt;script&gt;alert(&quot;text&quot;)&lt;/script&gt;" in html
    assert "ic_vs_n.png" in html


def test_report_rejects_a_non_mapping_diagnostics_payload(tmp_path):
    card = SimpleNamespace(name="bad", verdict="HOLD", diagnostics=[{"status": "ok"}])
    with pytest.raises(TypeError, match="mapping"):
        render_diagnostics(card, tmp_path)
