from __future__ import annotations

"""Optional AI interpretation layer for PsyAdapt.

The library can optionally generate a short, plain-language interpretation of
the computed reliability report.

Key design points:
* **Opt-in** only (nothing is sent anywhere unless enabled).
* **No hard dependency** on the `openai` Python SDK.
* Uses `OPENAI_API_KEY` or `PSYADAPT_OPENAI_API_KEY` when no key is passed.

The produced text is intended for Ukrainian psychologists who don't want to
dig into psychometrics details.
"""

import json
import os
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


@dataclass(frozen=True)
class AIConfig:
    provider: str = "openai"
    model: str = "o4-mini"
    api_key: Optional[str] = None
    timeout_s: float = 30.0
    # If True, store failure diagnostics into report["ai"] instead of silently skipping.
    debug: bool = False


def _get_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key:
        return explicit_key
    return os.getenv("PSYADAPT_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def _build_prompt_uk(report: Dict[str, Any]) -> str:
    """Create a compact prompt for the AI.

    We avoid sending any raw respondent data. Only aggregate metrics + diagnostics.
    """
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    inputs = report.get("inputs", {}) if isinstance(report, dict) else {}
    results = report.get("results", {}) if isinstance(report, dict) else {}

    # Pull a few relevant bits if present
    primary = summary.get("primary_estimator")
    primary_score = summary.get("primary_score")
    icc_score = summary.get("icc_score")
    n_files = inputs.get("n_files")
    n_objects = summary.get("n_objects")
    n_items = summary.get("n_items")

    # Cronbach diagnostics are stored inside results.primary.details in this project.
    # If present, highlight suggested reverse/remove items.
    diag = None
    try:
        primary_rep = results.get("primary", {})
        diag = (primary_rep.get("details") or {}).get("diagnostics")
    except Exception:
        diag = None

    reverse_items = []
    remove_items = []
    if isinstance(diag, dict):
        items = diag.get("items") or []
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                nm = it.get("item_name") or it.get("name") or it.get("item")
                if not nm:
                    continue
                if bool(it.get("suggest_reverse")):
                    reverse_items.append(str(nm))
                if bool(it.get("remove_candidate")) or it.get("flag") == "problematic":
                    remove_items.append(str(nm))

    # Keep it short and structured.
    payload = {
        "n_files": n_files,
        "n_objects": n_objects,
        "n_items": n_items,
        "primary_estimator": primary,
        "primary_score": primary_score,
        "icc_score": icc_score,
        "reverse_candidates": reverse_items[:15],
        "remove_candidates": remove_items[:15],
    }

    return textwrap.dedent(
        f"""
        Ти — асистент для українського психолога.
        Поясни результати надійності методики простими словами (без формул),
        у 5–10 коротких пунктів.

        Правила:
        - Пиши українською.
        - Будь обережним у висновках: це лише статистичні індикатори, не діагноз.
        - Якщо показники хороші — скажи, що адаптація виглядає надійною.
        - Якщо слабкі — скажи, що є ризики і що саме краще перевірити.
        - Якщо є кандидати на реверс/видалення — вкажи їх як пріоритет.
        - Дай практичні поради: що робити далі.

        Ось агрегований звіт (JSON):
        {json.dumps(payload, ensure_ascii=False)}
        """
    ).strip()


def _openai_responses_call(*, api_key: str, model: str, prompt: str, timeout_s: float) -> str:
    """Call OpenAI Responses API using stdlib urllib.

    Returns: plain text response.
    """
    body = json.dumps(
        {
            "model": model,
            "input": prompt,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    req = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)

    # Preferred convenience field
    if isinstance(data, dict) and isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()

    # Fallback: traverse outputs
    out = []
    for block in (data.get("output") or []):
        if not isinstance(block, dict):
            continue
        for c in (block.get("content") or []):
            if isinstance(c, dict) and c.get("type") in {"output_text", "text"} and isinstance(c.get("text"), str):
                out.append(c["text"])
    text = "\n".join(out).strip()
    return text


def add_ai_interpretation(
    report: Dict[str, Any],
    *,
    enabled: bool = False,
    config: Optional[AIConfig] = None,
) -> Dict[str, Any]:
    """Optionally enrich a report with an AI interpretation.

    If disabled or no API key is available, returns report unchanged.

    Adds:
      report["ai"] = {"provider", "model", "text"}
    """
    if not enabled:
        return report
    if not isinstance(report, dict):
        return report

    cfg = config or AIConfig()
    if cfg.provider != "openai":
        # Future-proof hook for other providers.
        return report

    api_key = _get_api_key(cfg.api_key)
    if not api_key:
        return report

    try:
        prompt = _build_prompt_uk(report)
        text = _openai_responses_call(
            api_key=api_key,
            model=cfg.model,
            prompt=prompt,
            timeout_s=float(cfg.timeout_s),
        )
    except urllib.error.HTTPError as e:
        if cfg.debug:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            report["ai"] = {
                "provider": cfg.provider,
                "model": cfg.model,
                "status": "failed",
                "error": f"HTTP {getattr(e, 'code', '?')}: {body[:1200]}",
            }
        return report
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as e:
        if cfg.debug:
            report["ai"] = {
                "provider": cfg.provider,
                "model": cfg.model,
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
            }
        return report
    except Exception:
        # Never crash the core pipeline because of AI.
        if cfg.debug:
            report["ai"] = {
                "provider": cfg.provider,
                "model": cfg.model,
                "status": "failed",
                "error": "Unexpected error while calling AI provider.",
            }
        return report

    if text:
        report["ai"] = {
            "provider": cfg.provider,
            "model": cfg.model,
            "text": text,
        }
    return report
