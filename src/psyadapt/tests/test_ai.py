import json
import os
from types import SimpleNamespace

import pytest


def test_add_ai_interpretation_no_key_does_nothing(monkeypatch):
    from psyadapt.ai import AIConfig, add_ai_interpretation

    report = {"summary": {"primary_estimator": "CronbachAlpha", "primary_score": 0.73}}
    cfg = AIConfig(api_key=None)

    out = add_ai_interpretation(report, enabled=True, config=cfg)
    assert out is report
    assert "ai" not in report


def test_add_ai_interpretation_parses_output_text(monkeypatch):
    from psyadapt.ai import AIConfig, add_ai_interpretation

    report = {"summary": {"primary_estimator": "CronbachAlpha", "primary_score": 0.73}}

    class _FakeResp:
        def __init__(self, payload: dict):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=30):
        payload = {"output_text": "Коротка інтерпретація українською."}
        return _FakeResp(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    cfg = AIConfig(api_key="sk-test", model="o4-mini")
    add_ai_interpretation(report, enabled=True, config=cfg)

    assert "ai" in report
    assert report["ai"]["provider"] == "openai"
    assert report["ai"]["model"] == "o4-mini"
    assert "інтерпретація" in report["ai"]["text"].lower()
