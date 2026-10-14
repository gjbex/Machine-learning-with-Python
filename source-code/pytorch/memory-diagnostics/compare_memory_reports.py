#!/usr/bin/env python3
"""Compare JSON reports produced by memory_diagnostics.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_arguments() -> argparse.Namespace:
    """Parse report paths."""

    parser = argparse.ArgumentParser(
        description="Tabulate CUDA-memory diagnosis indicators.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--show-phases",
        action="store_true",
        help="also show rank-maximum allocated memory at first-step boundaries",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    """Load and minimally validate one exercise report."""

    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    required = {"case", "preset", "batch_size_per_device", "rank_reports"}
    missing = required.difference(report)
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(sorted(missing))}")
    if not report["rank_reports"]:
        raise ValueError(f"{path}: contains no rank reports")
    return report


def maximum_indicator(report: dict[str, Any], name: str) -> float:
    """Return the largest indicator across ranks."""

    return max(rank["indicators"][name] for rank in report["rank_reports"])


def maximum_phase_allocation(
    report: dict[str, Any], step: int, phase: str
) -> float:
    """Return the largest allocated-memory sample for a phase across ranks."""

    values = []
    for rank in report["rank_reports"]:
        values.extend(
            sample["allocated_mib"]
            for sample in rank["samples"]
            if sample["step"] == step and sample["phase"] == phase
        )
    if not values:
        raise ValueError(f"missing phase sample: step={step}, phase={phase}")
    return max(values)


def main() -> int:
    """Print side-by-side evidence without automatically naming a diagnosis."""

    arguments = parse_arguments()
    try:
        loaded = [(path, load_report(path)) for path in arguments.reports]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    headings = (
        "report",
        "case",
        "preset",
        "batch",
        "first persistent MiB",
        "step growth MiB",
        "forward increase MiB",
        "peak MiB",
    )
    rows = []
    for path, report in loaded:
        rows.append(
            (
                path.name,
                str(report["case"]),
                str(report["preset"]),
                str(report["batch_size_per_device"]),
                f"{maximum_indicator(report, 'first_step_persistent_change_mib'):.1f}",
                f"{maximum_indicator(report, 'between_step_growth_mib'):.1f}",
                f"{maximum_indicator(report, 'largest_forward_increase_mib'):.1f}",
                f"{maximum_indicator(report, 'maximum_peak_allocated_mib'):.1f}",
            )
        )

    widths = [
        max(len(headings[column]), *(len(row[column]) for row in rows))
        for column in range(len(headings))
    ]
    print("  ".join(value.ljust(width) for value, width in zip(headings, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))

    if arguments.show_phases:
        phase_columns = (
            (-1, "after_input", "input"),
            (0, "after_forward", "forward"),
            (0, "after_backward", "backward"),
            (0, "after_optimizer_step", "optimizer"),
            (0, "after_clear_gradients", "cleared"),
        )
        phase_headings = ("report", *(label for _, _, label in phase_columns))
        phase_rows = []
        try:
            for path, report in loaded:
                phase_rows.append(
                    (
                        path.name,
                        *(
                            f"{maximum_phase_allocation(report, step, phase):.1f}"
                            for step, phase, _ in phase_columns
                        ),
                    )
                )
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        phase_widths = [
            max(
                len(phase_headings[column]),
                *(len(row[column]) for row in phase_rows),
            )
            for column in range(len(phase_headings))
        ]
        print("\nFirst-step allocated memory (MiB; maximum across ranks)")
        print(
            "  ".join(
                value.ljust(width) for value, width in zip(phase_headings, phase_widths)
            )
        )
        print("  ".join("-" * width for width in phase_widths))
        for row in phase_rows:
            print(
                "  ".join(
                    value.ljust(width) for value, width in zip(row, phase_widths)
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
