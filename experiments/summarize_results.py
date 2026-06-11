# Agrega el CSV de resultados crudos y produce un resumen por variante
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_NUMERIC_COLS = [
    "success",
    "valid",
    "total_weeks",
    "total_difficulty",
    "plan_length",
    "coverage",
    "expanded_nodes",
    "max_frontier_size",
    "runtime_seconds",
    "mc_success_probability",
    "mc_expected_weeks",
    "mc_expected_failed_courses",
    "llm_global_quality",
    "final_score",
]

_SUMMARY_FIELDS = ["variant", "count"] + [
    f"{col}__{stat}"
    for col in _NUMERIC_COLS
    for stat in ("mean", "std", "min", "max")
]


def _safe_float(value: str) -> float | None:
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0.0
    return {
        "mean": round(mean, 6),
        "std": round(variance ** 0.5, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def summarize(input_path: Path) -> list[dict]:
    raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row.get("variant", "unknown")
            for col in _NUMERIC_COLS:
                v = _safe_float(row.get(col, ""))
                if v is not None:
                    raw[variant][col].append(v)

    summary_rows = []
    for variant in sorted(raw):
        row: dict = {"variant": variant}
        counts = [len(raw[variant][c]) for c in _NUMERIC_COLS if raw[variant][c]]
        row["count"] = max(counts) if counts else 0
        for col in _NUMERIC_COLS:
            s = _stats(raw[variant][col])
            for stat, val in s.items():
                row[f"{col}__{stat}"] = val
        summary_rows.append(row)

    return summary_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agrega resultados de experimentos por variante.")
    parser.add_argument("--input", default="experiments/results/raw_results.csv")
    parser.add_argument("--output", default="experiments/results/summary_results.csv")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: no existe el archivo de entrada '{input_path}'.")
        return 1

    rows = summarize(input_path)
    if not rows:
        print("No se encontraron datos para resumir.")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resumen guardado en: {output_path.resolve()}")

    # Imprimir tabla compacta en terminal
    print(f"\n{'Variante':<16} {'N':>4} {'valid%':>7} {'weeks_mean':>11} {'mc_success':>10} {'llm_qual':>9} {'score_mean':>11}")
    print("-" * 72)
    for r in rows:
        valid_pct = r.get("valid__mean", float("nan"))
        print(
            f"{r['variant']:<16} "
            f"{r['count']:>4} "
            f"{valid_pct * 100:>6.1f}% "
            f"{r.get('total_weeks__mean', float('nan')):>11.1f} "
            f"{r.get('mc_success_probability__mean', float('nan')):>10.3f} "
            f"{r.get('llm_global_quality__mean', float('nan')):>9.3f} "
            f"{r.get('final_score__mean', float('nan')):>11.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
