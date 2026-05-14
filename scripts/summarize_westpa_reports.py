from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


METRICS = ("auroc", "auprc", "top1_enrichment", "top5_enrichment", "top10_enrichment", "top25_enrichment")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate WESTPA evaluation report metrics across seeds or runs."
    )
    parser.add_argument("report_dirs", nargs="+", type=Path, help="Report directories with metrics.csv.")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/reports/summary.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/reports/summary.md"))
    args = parser.parse_args()

    rows = load_metric_rows(args.report_dirs)
    summary_rows = summarize_metric_rows(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, summary_rows)
    write_markdown(args.output_md, summary_rows)

    print(f"Reports: {len(args.report_dirs)}")
    print(f"Rows: {len(rows)}")
    print(f"Summary CSV: {args.output_csv}")
    print(f"Summary Markdown: {args.output_md}")


def load_metric_rows(report_dirs: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for report_dir in report_dirs:
        metrics_path = report_dir / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.csv: {metrics_path}")
        with open(metrics_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["report_dir"] = str(report_dir)
                rows.append(row)
    return rows


def summarize_metric_rows(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    rankers = sorted({row["ranker"] for row in rows})
    summary: list[dict[str, float | str]] = []
    for ranker in rankers:
        ranker_rows = [row for row in rows if row["ranker"] == ranker]
        out: dict[str, float | str] = {"ranker": ranker, "n": float(len(ranker_rows))}
        for metric in METRICS:
            values = np.asarray(
                [_to_float(row.get(metric, "nan")) for row in ranker_rows],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            out[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            out[f"{metric}_std"] = float(np.std(finite)) if finite.size else float("nan")
        summary.append(out)

    stride = next((row for row in summary if row["ranker"] == "STRIDE"), None)
    if stride is not None:
        baselines = [row for row in summary if row["ranker"] not in {"STRIDE", "random"}]
        for metric in ("auroc", "auprc", "top25_enrichment"):
            best = max(
                (
                    float(row[f"{metric}_mean"])
                    for row in baselines
                    if np.isfinite(float(row[f"{metric}_mean"]))
                ),
                default=float("nan"),
            )
            stride[f"{metric}_delta_vs_best_baseline"] = (
                float(stride[f"{metric}_mean"]) - best
                if np.isfinite(best)
                else float("nan")
            )
    return summary


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, float | str]]) -> None:
    lines = [
        "# WESTPA Report Summary",
        "",
        "| Ranker | n | AUROC mean | AUROC std | AUPRC mean | AUPRC std | Top 25% mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {ranker} | {n:.0f} | {auroc_mean:.4g} | {auroc_std:.4g} | "
            "{auprc_mean:.4g} | {auprc_std:.4g} | {top25:.4g} |".format(
                ranker=row["ranker"],
                n=float(row["n"]),
                auroc_mean=float(row["auroc_mean"]),
                auroc_std=float(row["auroc_std"]),
                auprc_mean=float(row["auprc_mean"]),
                auprc_std=float(row["auprc_std"]),
                top25=float(row["top25_enrichment_mean"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    main()
