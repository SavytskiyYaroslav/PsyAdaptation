from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from psyadapt.compare import compare_reliability_csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="psyadapt",
        description="Psychometrics reliability toolkit (CSV -> JSON report).",
    )

    p.add_argument(
        "paths",
        nargs="+",
        help="CSV file paths. Use 1 file for Cronbach, 2+ for inter-rater metrics.",
    )

    p.add_argument(
        "--scale",
        choices=["nominal", "ordinal", "interval"],
        default="nominal",
        help="Rating scale used for Krippendorff (>10 raters) and metadata. Default: nominal.",
    )
    p.add_argument(
        "--aggregate",
        choices=["mean", "median"],
        default="mean",
        help="Aggregate per-item metrics. Default: mean.",
    )
    p.add_argument(
        "--primary",
        choices=["auto", "cronbach", "cohen", "fleiss", "krippendorff"],
        default="auto",
        help="Primary reliability estimator (default: auto).",
    )

    # ICC params (always computed additionally for 2+ files)
    p.add_argument("--icc-model", choices=["oneway", "twoway"], default="twoway")
    p.add_argument("--icc-type", choices=["agreement", "consistency"], default="agreement")
    p.add_argument("--icc-unit", choices=["single", "average"], default="single")

    p.add_argument(
        "--out",
        default=None,
        help="Output JSON file path. If omitted, prints JSON to stdout.",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (indent=2).",
    )

    # Optional AI interpretation (opt-in)
    p.add_argument(
        "--ai",
        action="store_true",
        help="Add plain-language AI interpretation to the report (requires API key).",
    )
    p.add_argument(
        "--ai-provider",
        choices=["openai"],
        default="openai",
        help="AI provider to use (default: openai).",
    )
    p.add_argument(
        "--ai-model",
        default="o4-mini",
        help="Model name for AI interpretation (default: o4-mini).",
    )
    p.add_argument(
        "--ai-key",
        default=None,
        help="API key (optional). If omitted, uses OPENAI_API_KEY/PSYADAPT_OPENAI_API_KEY.",
    )

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report = compare_reliability_csv(
        args.paths,
        rating_scale=args.scale,
        aggregate=args.aggregate,
        icc_model=args.icc_model,
        icc_type=args.icc_type,
        icc_unit=args.icc_unit,
        primary=args.primary,
        ai=args.ai,
        ai_provider=args.ai_provider,
        ai_model=args.ai_model,
        ai_api_key=args.ai_key,
    )

    if args.pretty:
        payload = json.dumps(report, ensure_ascii=False, indent=2)
    else:
        payload = json.dumps(report, ensure_ascii=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        sys.stdout.write(payload)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
