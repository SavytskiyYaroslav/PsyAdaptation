from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Optional

from psyadapt import compare_reliability_csv, compare_reliability_to_excel


def _expand_paths(patterns: List[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        if any(ch in p for ch in ["*", "?", "["]):
            out.extend(sorted(glob.glob(p)))
        else:
            out.append(p)
    return out


def _first_existing(paths: List[str]) -> List[str]:
    return [p for p in paths if Path(p).exists()]


def run_case(
    *,
    title: str,
    paths: List[str],
    out_xlsx: Path,
    scale: str,
    aggregate: str,
    primary: Optional[str],
    ai_key: str,
    ai_model: str,
    ai_timeout_s: float,
) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    rep = compare_reliability_csv(
        paths,
        rating_scale=scale,   # type: ignore[arg-type]
        aggregate=aggregate,  # type: ignore[arg-type]
        primary=primary,      # type: ignore[arg-type]
        ai=True,
        ai_api_key=ai_key,
        ai_model=ai_model,
        ai_timeout_s=ai_timeout_s,
        ai_debug=True,
    )

    summ = rep.get("summary", {})
    ai = rep.get("ai")
    print(f"\n=== {title} ===")
    print("FILES:", ", ".join(paths))
    print("SUMMARY:", summ)
    if isinstance(ai, dict) and ai.get("text"):
        print("AI: present")
    elif isinstance(ai, dict) and ai.get("error"):
        # Show a compact diagnostic to speed up debugging (401/404/429, etc.).
        err = str(ai.get("error"))
        print("AI: failed ->", err[:250] + ("..." if len(err) > 250 else ""))
    else:
        print("AI: missing")

    compare_reliability_to_excel(
        paths,
        out_path=str(out_xlsx),
        rating_scale=scale,   # type: ignore[arg-type]
        aggregate=aggregate,  # type: ignore[arg-type]
        primary=primary,      # type: ignore[arg-type]
        ai=True,
        ai_api_key=ai_key,
        ai_model=ai_model,
        ai_timeout_s=ai_timeout_s,
        ai_debug=True,
    )
    print("Excel saved:", out_xlsx.as_posix())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Manual smoke test: run PsyAdapt with AI interpretation enabled and export XLSX reports."
    )
    ap.add_argument(
        "--key",
        default=os.getenv("PSYADAPT_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key. If omitted, reads PSYADAPT_OPENAI_API_KEY or OPENAI_API_KEY from env.",
    )
    ap.add_argument("--model", default="o4-mini", help="OpenAI model for AI interpretation (default: o4-mini).")
    ap.add_argument("--timeout", type=float, default=30.0, help="AI request timeout in seconds (default: 30).")

    ap.add_argument("--scale", default="nominal", choices=["nominal", "ordinal", "interval"])
    ap.add_argument("--aggregate", default="mean", choices=["mean", "median"])

    ap.add_argument(
        "--out-dir",
        default="data/reports_ai",
        help="Where to save XLSX files (default: data/reports_ai).",
    )

    # Inputs (you can override with your real files)
    ap.add_argument(
        "--cronbach",
        default="data/leary/vertical_workbook_MATRIX.csv",
        help="Single CSV for CronbachAlpha (1-file case).",
    )
    ap.add_argument(
        "--cohen",
        nargs="*",
        default=["data/leary/leary_t1.csv", "data/leary/leary_t2.csv"],
        help="Two CSVs for CohenKappa + ICC. Supports globs.",
    )
    ap.add_argument(
        "--fleiss",
        nargs="*",
        default=["data/leary/leary_t1.csv", "data/leary/leary_t2.csv", "data/leary/leary_t3.csv"],
        help="Three CSVs for FleissKappa + ICC. Supports globs.",
    )
    ap.add_argument(
        "--kripp",
        nargs="*",
        default=["data/leary/leary_t1.csv", "data/leary/leary_t2.csv", "data/leary/leary_t3.csv"],
        help="Three CSVs for forced KrippendorffAlpha + ICC. Supports globs.",
    )

    args = ap.parse_args()

    if not args.key:
        print("ERROR: No API key. Provide --key or set PSYADAPT_OPENAI_API_KEY / OPENAI_API_KEY.")
        return 2

    out_dir = Path(args.out_dir)

    # 1) Cronbach (single CSV)
    cron = _first_existing([args.cronbach])
    if cron:
        run_case(
            title="CronbachAlpha (1 CSV) + AI",
            paths=cron,
            out_xlsx=out_dir / "cronbach_ai.xlsx",
            scale=args.scale,
            aggregate=args.aggregate,
            primary="cronbach",
            ai_key=args.key,
            ai_model=args.model,
            ai_timeout_s=args.timeout,
        )
    else:
        print(f"SKIP: Cronbach input not found: {args.cronbach}")

    # 2) Cohen (2 CSV)
    cohen_paths = _first_existing(_expand_paths(args.cohen))
    if len(cohen_paths) >= 2:
        run_case(
            title="CohenKappa (2 CSV) + ICC + AI",
            paths=cohen_paths[:2],
            out_xlsx=out_dir / "cohen_icc_ai.xlsx",
            scale=args.scale,
            aggregate=args.aggregate,
            primary="cohen",
            ai_key=args.key,
            ai_model=args.model,
            ai_timeout_s=args.timeout,
        )
    else:
        print(f"SKIP: Cohen inputs not found (need 2): {args.cohen}")

    # 3) Fleiss (3 CSV)
    fleiss_paths = _first_existing(_expand_paths(args.fleiss))
    if len(fleiss_paths) >= 3:
        run_case(
            title="FleissKappa (3 CSV) + ICC + AI",
            paths=fleiss_paths[:3],
            out_xlsx=out_dir / "fleiss_icc_ai.xlsx",
            scale=args.scale,
            aggregate=args.aggregate,
            primary="fleiss",
            ai_key=args.key,
            ai_model=args.model,
            ai_timeout_s=args.timeout,
        )
    else:
        print(f"SKIP: Fleiss inputs not found (need 3): {args.fleiss}")

    # 4) Forced Krippendorff (3 CSV)
    kripp_paths = _first_existing(_expand_paths(args.kripp))
    if len(kripp_paths) >= 3:
        run_case(
            title="KrippendorffAlpha forced (3 CSV) + ICC + AI",
            paths=kripp_paths[:3],
            out_xlsx=out_dir / "krippendorff_icc_ai.xlsx",
            scale=args.scale,
            aggregate=args.aggregate,
            primary="krippendorff",
            ai_key=args.key,
            ai_model=args.model,
            ai_timeout_s=args.timeout,
        )
    else:
        print(f"SKIP: Krippendorff inputs not found (need 3): {args.kripp}")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
