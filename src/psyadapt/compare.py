from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional

from psyadapt.io.csv import (
    load_csv_dataframe,
    load_csv_matrix,
    load_csv_raters,
    load_csv_raters_with_names,
)
from psyadapt.reliability import (
    AutoReliability,
    CronbachAlpha,
    ICC,
    CohenKappa,
    FleissKappa,
    KrippendorffAlpha,
)

from psyadapt.validate import validate_single_test_csv, validate_raters_csvs

from psyadapt.ai import AIConfig, add_ai_interpretation

RatingScale = Literal["nominal", "ordinal", "interval"]
Agg = Literal["mean", "median"]


@dataclass(frozen=True)
class CompareConfig:
    rating_scale: RatingScale = "nominal"
    aggregate: Agg = "mean"
    icc_model: Literal["oneway", "twoway"] = "twoway"
    icc_type: Literal["agreement", "consistency"] = "agreement"
    icc_unit: Literal["single", "average"] = "single"


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_list(x: Any, length: int, fill: Any = None) -> List[Any]:
    if isinstance(x, list) and len(x) == length:
        return x
    return [fill] * length


def _build_per_item_from_reports(
    *,
    n_items: int,
    primary_report: Dict[str, Any],
    icc_report: Optional[Dict[str, Any]] = None,
    item_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    per_item: List[Dict[str, Any]] = [
        {
            "item_index": i,
            "item_name": (
                item_names[i]
                if isinstance(item_names, list) and len(item_names) == n_items
                else None
            ),
        }
        for i in range(n_items)
    ]

    p_details = (primary_report or {}).get("details", {}) if isinstance(primary_report, dict) else {}
    primary_est = (primary_report or {}).get("estimator", None)

    kappa = _safe_list(p_details.get("kappa_per_item"), n_items, None)
    alpha = _safe_list(p_details.get("alpha_per_item"), n_items, None)

    n_pairs = _safe_list(p_details.get("n_pairs_per_item"), n_items, None)
    agreement = _safe_list(p_details.get("agreement_per_item"), n_items, None)
    expected_agreement = _safe_list(p_details.get("expected_agreement_per_item"), n_items, None)

    do = _safe_list(p_details.get("do_per_item"), n_items, None)
    de = _safe_list(p_details.get("de_per_item"), n_items, None)

    for i in range(n_items):
        per_item[i]["primary"] = {
            "estimator": primary_est,
            "kappa": kappa[i],
            "alpha": alpha[i],
            "n_pairs": n_pairs[i],
            "agreement": agreement[i],
            "expected_agreement": expected_agreement[i],
            "do": do[i],
            "de": de[i],
        }

    if icc_report is not None and isinstance(icc_report, dict):
        icc_details = icc_report.get("details", {})
        icc_est = icc_report.get("estimator", "ICC")
        icc_per_item = _safe_list(icc_details.get("icc_per_item"), n_items, None)
        n_targets_used = _safe_list(icc_details.get("n_targets_used_per_item"), n_items, None)

        for i in range(n_items):
            per_item[i]["icc"] = {
                "estimator": icc_est,
                "icc": icc_per_item[i],
                "n_targets_used": n_targets_used[i],
            }

    return per_item


def _make_estimator(cls, *, rating_scale: str, aggregate: str):
    """
    Compatibility factory: different estimator classes may name params differently.
    Tries common variants; falls back to parameterless init.
    """
    # 1) rating_scale + aggregate
    try:
        return cls(rating_scale=rating_scale, aggregate=aggregate)
    except TypeError:
        pass
    # 2) scale + aggregate
    try:
        return cls(scale=rating_scale, aggregate=aggregate)
    except TypeError:
        pass
    # 3) only rating_scale
    try:
        return cls(rating_scale=rating_scale)
    except TypeError:
        pass
    # 4) only scale
    try:
        return cls(scale=rating_scale)
    except TypeError:
        pass
    # 5) only aggregate
    try:
        return cls(aggregate=aggregate)
    except TypeError:
        pass
    # 6) nothing
    return cls()


def compare_reliability_csv(
    paths: Iterable[str],
    *,
    rating_scale: RatingScale = "nominal",
    aggregate: Agg = "mean",
    icc_model: Literal["oneway", "twoway"] = "twoway",
    icc_type: Literal["agreement", "consistency"] = "agreement",
    icc_unit: Literal["single", "average"] = "single",
    primary: Optional[Literal["auto", "cohen", "fleiss", "krippendorff", "icc", "cronbach"]] = None,
    ai: bool = False,
    ai_provider: Literal["openai"] = "openai",
    ai_model: str = "o4-mini",
    ai_api_key: Optional[str] = None,
    ai_timeout_s: float = 30.0,
    ai_debug: bool = False,
) -> Dict[str, Any]:
    paths = list(paths)

    if len(paths) == 0:
        return {
            "meta": {"generated_at_utc": _utc_ts(), "schema_version": "1.0"},
            "inputs": {
                "n_files": 0,
                "files": [],
                "rating_scale": rating_scale,
                "aggregate": aggregate,
                "icc": {"model": icc_model, "icc_type": icc_type, "unit": icc_unit},
            },
            "status": "error",
            "errors": [{
                "severity": "error",
                "code": "paths.empty",
                "message": "No CSV paths provided.",
                "file": None,
                "column": None,
            }],
            "warnings": [],
            "results": {},
            "per_item": [],
            "summary": {},
        }

    primary_norm = "auto" if primary is None else str(primary).strip().lower()
    if primary_norm == "":
        primary_norm = "auto"

    if len(paths) == 1 and primary_norm not in {"auto", "cronbach"}:
        return {
            "meta": {"generated_at_utc": _utc_ts(), "schema_version": "1.0"},
            "inputs": {
                "n_files": len(paths),
                "files": paths,
                "rating_scale": rating_scale,
                "aggregate": aggregate,
                "icc": {"model": icc_model, "icc_type": icc_type, "unit": icc_unit},
                "primary": primary_norm,
            },
            "status": "error",
            "errors": [{
                "severity": "error",
                "code": "primary.incompatible",
                "message": "Primary estimator override requires 2+ CSV files (except 'cronbach').",
                "file": None,
                "column": None,
            }],
            "warnings": [],
            "results": {},
            "per_item": [],
            "summary": {},
        }

    if len(paths) >= 2 and primary_norm == "cronbach":
        return {
            "meta": {"generated_at_utc": _utc_ts(), "schema_version": "1.0"},
            "inputs": {
                "n_files": len(paths),
                "files": paths,
                "rating_scale": rating_scale,
                "aggregate": aggregate,
                "icc": {"model": icc_model, "icc_type": icc_type, "unit": icc_unit},
                "primary": primary_norm,
            },
            "status": "error",
            "errors": [{
                "severity": "error",
                "code": "primary.incompatible",
                "message": "CronbachAlpha is only valid for a single CSV test file.",
                "file": None,
                "column": None,
            }],
            "warnings": [],
            "results": {},
            "per_item": [],
            "summary": {},
        }

    if len(paths) == 1:
        v = validate_single_test_csv(paths[0], allow_missing=True)
    else:
        v = validate_raters_csvs(paths, allow_missing=True)

    issues = v.get("issues", [])
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]

    CRITICAL_CODES = {
        "file.not_found",
        "file.not_a_file",
        "csv.read_failed",
        "raters.shape_mismatch",
        "raters.columns_mismatch",
        "csv.non_numeric_column",
        "csv.too_few_rows",
        "csv.too_few_cols",
        "csv.duplicate_columns",
    }
    critical_errors = [e for e in errors if e.get("code") in CRITICAL_CODES]

    if critical_errors:
        return {
            "meta": {"generated_at_utc": _utc_ts(), "schema_version": "1.0"},
            "inputs": {
                "n_files": len(paths),
                "files": paths,
                "rating_scale": rating_scale,
                "aggregate": aggregate,
                "icc": {"model": icc_model, "icc_type": icc_type, "unit": icc_unit},
            },
            "status": "error",
            "errors": critical_errors,
            "warnings": warnings + [e for e in errors if e.get("code") not in CRITICAL_CODES],
            "results": {},
            "per_item": [],
            "summary": {},
        }

    cfg = CompareConfig(
        rating_scale=rating_scale,
        aggregate=aggregate,
        icc_model=icc_model,
        icc_type=icc_type,
        icc_unit=icc_unit,
    )

    report: Dict[str, Any] = {
        "meta": {"generated_at_utc": _utc_ts(), "schema_version": "1.0"},
        "inputs": {
            "n_files": len(paths),
            "files": paths,
            "rating_scale": cfg.rating_scale,
            "aggregate": cfg.aggregate,
            "primary": primary_norm,
            "icc": {"model": cfg.icc_model, "icc_type": cfg.icc_type, "unit": cfg.icc_unit},
        },
        "status": "ok",
        "summary": {},
        "results": {},
        "per_item": [],
        "warnings": [],
    }

    # ---------- 1 file: Cronbach ----------
    if len(paths) == 1:
        df = load_csv_dataframe(paths[0])
        est = CronbachAlpha().fit(df)
        primary_report = est.report()

        report["summary"] = {
            "primary_estimator": primary_report.get("estimator"),
            "primary_score": primary_report.get("score"),
            "n_raters": 1,
            "n_objects": est.fit_info_.n_objects if est.fit_info_ else None,
            "n_items": est.fit_info_.n_items if est.fit_info_ else None,
        }

        report["results"]["primary"] = primary_report
        report["warnings"] = list(est.warnings_)

        report["per_item"] = (
            [
                {
                    "item_index": i,
                    "item_name": (str(df.columns[i]) if hasattr(df, "columns") and i < len(df.columns) else None),
                }
                for i in range(est.fit_info_.n_items)
            ]
            if est.fit_info_
            else []
        )
        # Optional AI interpretation (opt-in)
        if ai:
            cfg_ai = AIConfig(
                provider=ai_provider,
                api_key=ai_api_key,
                model=ai_model,
                timeout_s=float(ai_timeout_s),
                debug=bool(ai_debug),
            )
            add_ai_interpretation(report, enabled=True, config=cfg_ai)

        return report

    # ---------- 2+ files ----------
    X3, item_names = load_csv_raters_with_names(paths)
    n_raters, n_objects, n_items = map(int, X3.shape)

    primary_meta: Dict[str, Any]
    primary_report: Dict[str, Any]
    primary_est: Any
    primary_wrapper: Any = None  # may stay None

    if primary_norm == "auto":
        primary_wrapper = AutoReliability(
            rating_scale=cfg.rating_scale,
            aggregate=cfg.aggregate,
        ).fit(X3)

        primary_est = primary_wrapper.estimator_ if getattr(primary_wrapper, "estimator_", None) is not None else primary_wrapper
        primary_report = primary_est.report()
        primary_meta = {
            "selected_by": "AutoReliability",
            "selected_rule": primary_wrapper.details_.get("selected_rule") if getattr(primary_wrapper, "details_", None) else None,
            "forced": False,
            "wrapper_estimator": "AutoReliability",
        }

    elif primary_norm == "icc":
        primary_est = ICC(
            model=cfg.icc_model,
            icc_type=cfg.icc_type,
            unit=cfg.icc_unit,
            aggregate=cfg.aggregate,
        ).fit(X3)
        primary_report = primary_est.report()
        primary_meta = {
            "selected_by": "forced",
            "selected_rule": None,
            "forced": True,
            "wrapper_estimator": None,
        }



    elif primary_norm in {"cohen", "fleiss", "krippendorff"}:

        # Force a specific agreement estimator directly (no AutoReliability dependency)

        if primary_norm == "cohen":

            if n_raters != 2:
                return {

                    "meta": {"generated_at_utc": _utc_ts(), "schema_version": "1.0"},

                    "inputs": {

                        "n_files": len(paths),

                        "files": paths,

                        "rating_scale": rating_scale,

                        "aggregate": aggregate,

                        "icc": {"model": icc_model, "icc_type": icc_type, "unit": icc_unit},

                        "primary": primary_norm,

                    },

                    "status": "error",

                    "errors": [{

                        "severity": "error",

                        "code": "primary.incompatible",

                        "message": "CohenKappa requires exactly 2 CSV files/raters.",

                        "file": None,

                        "column": None,

                    }],

                    "warnings": [],

                    "results": {},

                    "per_item": [],

                    "summary": {},

                }

            primary_est = _make_estimator(CohenKappa, rating_scale=cfg.rating_scale, aggregate=cfg.aggregate).fit(X3)

        elif primary_norm == "fleiss":

            primary_est = _make_estimator(FleissKappa, rating_scale=cfg.rating_scale, aggregate=cfg.aggregate).fit(X3)


        else:  # krippendorff

            primary_est = _make_estimator(KrippendorffAlpha, rating_scale=cfg.rating_scale, aggregate=cfg.aggregate).fit(X3)

        # Safety guard (so it NEVER becomes None silently)

        if primary_est is None or not hasattr(primary_est, "report"):
            return {

                "meta": {"generated_at_utc": _utc_ts(), "schema_version": "1.0"},

                "inputs": {

                    "n_files": len(paths),

                    "files": paths,

                    "rating_scale": rating_scale,

                    "aggregate": aggregate,

                    "icc": {"model": icc_model, "icc_type": icc_type, "unit": icc_unit},

                    "primary": primary_norm,

                },

                "status": "error",

                "errors": [{

                    "severity": "error",

                    "code": "primary.failed",

                    "message": "Forced primary estimator returned None or missing report().",

                    "file": None,

                    "column": None,

                }],

                "warnings": [],

                "results": {},

                "per_item": [],

                "summary": {},

            }

        primary_report = primary_est.report()

        primary_meta = {

            "selected_by": "forced",

            "selected_rule": None,

            "forced": True,

            "wrapper_estimator": None,

        }


    else:
        return {
            "meta": {"generated_at_utc": _utc_ts(), "schema_version": "1.0"},
            "inputs": {
                "n_files": len(paths),
                "files": paths,
                "rating_scale": rating_scale,
                "aggregate": aggregate,
                "icc": {"model": icc_model, "icc_type": icc_type, "unit": icc_unit},
                "primary": primary_norm,
            },
            "status": "error",
            "errors": [{
                "severity": "error",
                "code": "primary.unknown",
                "message": "Unknown primary override. Use: auto, cohen, fleiss, krippendorff, icc, cronbach.",
                "file": None,
                "column": None,
            }],
            "warnings": [],
            "results": {},
            "per_item": [],
            "summary": {},
        }

    # ---- ICC: reuse if primary is ICC, else compute separately ----
    if primary_norm == "icc":
        icc_est = primary_est
        icc_report = primary_report
    else:
        icc_est = ICC(
            model=cfg.icc_model,
            icc_type=cfg.icc_type,
            unit=cfg.icc_unit,
            aggregate=cfg.aggregate,
        ).fit(X3)
        icc_report = icc_est.report()

    # ---- results (ALWAYS) ----
    report["results"]["primary"] = primary_report
    report["results"]["primary_meta"] = primary_meta
    report["results"]["icc"] = icc_report

    # ---- warnings (merge & de-dup) ----
    merged: List[str] = []
    primary_warnings: List[str] = []
    if primary_wrapper is not None and hasattr(primary_wrapper, "warnings_"):
        primary_warnings = list(primary_wrapper.warnings_)
    elif hasattr(primary_est, "warnings_"):
        primary_warnings = list(primary_est.warnings_)

    icc_warnings = list(icc_est.warnings_) if hasattr(icc_est, "warnings_") else []
    for w in primary_warnings + icc_warnings:
        if w not in merged:
            merged.append(w)
    report["warnings"] = merged

    # ---- summary (ALWAYS) ----
    report["summary"] = {
        "primary_estimator": primary_report.get("estimator"),
        "primary_score": primary_report.get("score"),
        "icc_estimator": icc_report.get("estimator"),
        "icc_score": icc_report.get("score"),
        "n_raters": n_raters,
        "n_objects": n_objects,
        "n_items": n_items,
    }

    # ---- per_item (ALWAYS) ----
    report["per_item"] = _build_per_item_from_reports(
        n_items=n_items,
        primary_report=primary_report,
        icc_report=icc_report,
        item_names=item_names,
    )

    # Optional AI interpretation (opt-in)
    if ai:
        cfg_ai = AIConfig(
            provider=ai_provider,
            api_key=ai_api_key,
            model=ai_model,
            timeout_s=float(ai_timeout_s),
            debug=bool(ai_debug),
        )
        add_ai_interpretation(report, enabled=True, config=cfg_ai)

    return report
