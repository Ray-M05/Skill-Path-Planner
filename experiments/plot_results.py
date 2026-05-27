from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _load_raw(path: Path) -> dict[str, dict[str, list[float]]]:
    """Devuelve {variant: {col: [values...]}}."""
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row.get("variant", "unknown")
            for col, val in row.items():
                try:
                    v = float(val)
                    if math.isfinite(v):
                        data[variant][col].append(v)
                except (ValueError, TypeError):
                    pass
    return data


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def _bar_chart(
    variants: list[str],
    means: list[float],
    stds: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    color: str = "#4C72B0",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(variants))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=color, alpha=0.85, error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    for bar, mean_val in zip(bars, means):
        if math.isfinite(mean_val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05 + 0.001,
                f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {output_path.name}")


PLOTS = [
    {
        "col": "valid",
        "title": "Tasa de exito por variante",
        "ylabel": "Fraccion de instancias con plan valido",
        "filename": "success_rate_by_variant.png",
        "color": "#55A868",
    },
    {
        "col": "runtime_seconds",
        "title": "Tiempo de ejecucion por variante",
        "ylabel": "Segundos (promedio ± std)",
        "filename": "runtime_by_variant.png",
        "color": "#C44E52",
    },
    {
        "col": "expanded_nodes",
        "title": "Nodos expandidos por variante",
        "ylabel": "Nodos (promedio ± std)",
        "filename": "expanded_nodes_by_variant.png",
        "color": "#8172B2",
    },
    {
        "col": "total_weeks",
        "title": "Semanas totales del plan por variante",
        "ylabel": "Semanas (promedio ± std)",
        "filename": "total_weeks_by_variant.png",
        "color": "#CCB974",
    },
    {
        "col": "mc_success_probability",
        "title": "Probabilidad de exito Monte Carlo por variante",
        "ylabel": "P(exito) (promedio ± std)",
        "filename": "mc_success_by_variant.png",
        "color": "#4C72B0",
    },
    {
        "col": "llm_global_quality",
        "title": "Calidad global LLM Evaluator por variante",
        "ylabel": "Calidad global [0-1] (promedio ± std)",
        "filename": "llm_quality_by_variant.png",
        "color": "#DD8452",
    },
    {
        "col": "final_score",
        "title": "Score final combinado por variante",
        "ylabel": "Score final (promedio ± std)",
        "filename": "final_score_by_variant.png",
        "color": "#4C72B0",
    },
]


def main(argv: list[str] | None = None) -> int:
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib no esta instalado. Ejecuta: pip install matplotlib")
        return 1

    parser = argparse.ArgumentParser(description="Genera graficas comparativas del experimento.")
    parser.add_argument("--input", default="experiments/results/raw_results.csv")
    parser.add_argument("--output-dir", default="experiments/results/plots")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: no existe '{input_path}'. Ejecuta run_experiments.py primero.")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_raw(input_path)
    variants = sorted(data.keys())
    print(f"Variantes encontradas: {variants}")
    print(f"Generando {len(PLOTS)} graficas en '{output_dir}' ...\n")

    for plot_cfg in PLOTS:
        col = plot_cfg["col"]
        means = [_mean(data[v][col]) for v in variants]
        stds = [_std(data[v][col]) for v in variants]

        has_data = any(math.isfinite(m) for m in means)
        if not has_data:
            print(f"  [SKIP] {plot_cfg['filename']} — sin datos para '{col}'")
            continue

        _bar_chart(
            variants=variants,
            means=means,
            stds=stds,
            title=plot_cfg["title"],
            ylabel=plot_cfg["ylabel"],
            output_path=output_dir / plot_cfg["filename"],
            color=plot_cfg["color"],
        )

    print(f"\nTodas las graficas guardadas en: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())