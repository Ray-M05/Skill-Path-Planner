from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Settings, load_settings
from src.dataset.loader import load_dataset
from src.llm.client import LLMClient
from src.llm.evaluator import evaluate_plan_with_llm
from src.llm.interpreter import validate_goal_spec
from src.llm.prompts import GOAL_INTERPRETER_SYSTEM_PROMPT, build_goal_interpreter_user_prompt
from src.main import _enrich_plan, build_mock_goal_response, build_profile_from_goal, run_planner_k
from src.metrics import extract_metrics_row, rows_to_csv_string
from src.scoring import rank_plans

DATASET = load_dataset("data")

EXAMPLES = [
    "Sé Python básico y quiero ser data analyst en máximo 30 semanas, 8 horas por semana",
    "Conozco Python básico y estadística básica, quiero ser ML engineer, tengo 12 horas semanales",
    "No sé nada, quiero ser analista de datos en 6 meses a ritmo lento y cursos fáciles",
    "Sé SQL básico y Python intermedio, quiero ser data analyst lo más rápido posible",
    "Quiero trabajar con machine learning, tengo algo de python, máximo 40 semanas y 10 horas semanales",
]

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)


def _fmt_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def run_pipeline(
    query: str,
    provider: str,
    planner: str,
    k_plans: int,
    use_monte_carlo: bool,
    mc_runs: int,
    use_evaluate: bool,
    max_nodes: int,
) -> tuple[str, str, str, str, str]:
    """Ejecuta el pipeline completo y devuelve 5 outputs para los tabs."""
    if not query.strip():
        empty = "— Escribe una consulta y presiona Generar plan —"
        return empty, empty, empty, empty, empty

    try:
        settings = load_settings()
        real_provider = provider.lower()

        # --- Interpretación del goal ---
        user_prompt = build_goal_interpreter_user_prompt(query, DATASET)
        if real_provider == "mock":
            raw = build_mock_goal_response(query, DATASET)
        else:
            real_settings = Settings(
                llm_provider=real_provider,
                llm_model=settings.llm_model,
                llm_api_key=settings.llm_api_key,
                llm_temperature=settings.llm_temperature,
                llm_max_output_tokens=settings.llm_max_output_tokens,
                llm_timeout_seconds=settings.llm_timeout_seconds,
                llm_cache=settings.llm_cache,
            )
            client = LLMClient(real_settings)
            raw = client.complete_json(GOAL_INTERPRETER_SYSTEM_PROMPT, user_prompt)

        goal = validate_goal_spec(raw, DATASET)
        profile = build_profile_from_goal(goal)

        # --- Planificador ---
        monte_carlo_runs = mc_runs if use_monte_carlo else 0
        plans = run_planner_k(planner, k_plans, goal, DATASET, profile, max_nodes=max_nodes)

        if not plans:
            msg = "No se encontró ninguna trayectoria válida con los parámetros dados."
            return msg, msg, msg, msg, msg

        enriched = []
        for plan in plans:
            enriched.append(
                _enrich_plan(
                    plan,
                    goal,
                    DATASET,
                    profile,
                    real_provider,
                    settings,
                    monte_carlo_runs,
                    42,
                    use_evaluate,
                )
            )

        ranked = rank_plans(enriched)
        best = ranked[0]
        best_dict = asdict(best)

        # --- Tab 1: Plan completo ---
        plan_out = _fmt_json(best_dict)

        # --- Tab 2: Score breakdown ---
        if best.score_breakdown:
            bd = best.score_breakdown
            sig = bd.get("signals", {})
            breakdown_lines = [
                f"**Score final: {best.final_score}**\n",
                f"| Componente | Valor |",
                f"|---|---|",
                f"| Requeridas cubiertas | `{bd['required']:+.4f}` |",
                f"| Recomendadas cubiertas | `{bd['recommended']:+.4f}` |",
                f"| Penalización tiempo | `{bd['time_penalty']:+.4f}` |",
                f"| Penalización dificultad | `{bd['difficulty_penalty']:+.4f}` |",
                f"| Éxito Monte Carlo | `{bd['mc_success']:+.4f}` |",
                f"| Calidad LLM | `{bd['llm_quality']:+.4f}` |",
                f"\n**Señales activas**",
                f"- Monte Carlo: {'✅' if sig.get('has_mc') else '❌'}",
                f"- Evaluación LLM: {'✅' if sig.get('has_llm') else '❌'}",
                f"- Peso requeridas (w): `{sig.get('w_required', 0)}`",
            ]
            score_out = "\n".join(breakdown_lines)
        else:
            score_out = "Score breakdown no disponible (plan inválido)."

        # --- Tab 3: Monte Carlo ---
        mc = best.monte_carlo
        if mc and not mc.get("skipped"):
            mc_lines = [
                f"**Simulación con {mc.get('runs', 0)} corridas**\n",
                f"| Métrica | Valor |",
                f"|---|---|",
                f"| Probabilidad de éxito | `{mc['success_probability']:.1%}` |",
                f"| IC 95% | `[{mc['success_ci_95'][0]:.1%}, {mc['success_ci_95'][1]:.1%}]` |",
                f"| Riesgo | `{mc['risk_score']:.4f}` |",
                f"| Semanas esperadas | `{mc['expected_weeks']:.1f}` |",
                f"| Semanas P10 / P50 / P90 | `{mc['weeks_p10']:.1f}` / `{mc['weeks_p50']:.1f}` / `{mc['weeks_p90']:.1f}` |",
                f"| Cursos fallidos esperados | `{mc['expected_failed_courses']:.2f}` |",
            ]
            mc_out = "\n".join(mc_lines)
        else:
            mc_out = "Monte Carlo no activado. Marca **Simulación Monte Carlo** y elige el número de corridas."

        # --- Tab 4: Evaluación LLM ---
        llm = best.llm_evaluation
        if llm and not llm.get("skipped"):
            strengths = "\n".join(f"- {s}" for s in llm.get("main_strengths", []))
            weaknesses = "\n".join(f"- {w}" for w in llm.get("main_weaknesses", []))
            llm_lines = [
                f"**Calidad global: {llm['global_quality']:.0%}**\n",
                f"| Dimensión | Puntuación |",
                f"|---|---|",
                f"| Alineación con el rol | `{llm['goal_alignment']:.0%}` |",
                f"| Coherencia pedagógica | `{llm['pedagogical_coherence']:.0%}` |",
                f"| Realismo de perfil | `{llm['profile_realism']:.0%}` |",
                f"| No redundancia | `{llm['non_redundancy']:.0%}` |",
                f"| Valor profesional | `{llm['professional_value']:.0%}` |",
                f"\n**Fortalezas**\n{strengths}",
                f"\n**Debilidades**\n{weaknesses}",
                f"\n**Justificación**\n{llm.get('justification', '')}",
            ]
            llm_out = "\n".join(llm_lines)
        else:
            llm_out = "Evaluación LLM no activada. Marca **Evaluar con LLM** para obtener análisis cualitativo."

        # --- Tab 5: Ranking (si k > 1) ---
        if len(ranked) > 1:
            ranking_lines = [f"**Top {len(ranked)} planes encontrados**\n", "| Rank | Cursos | Semanas | Score |", "|---|---|---|---|"]
            for i, p in enumerate(ranked):
                courses_str = " → ".join(p.course_ids)
                score_str = f"`{p.final_score}`" if p.final_score is not None else "n/a"
                ranking_lines.append(f"| {i+1} | {courses_str} | {p.total_weeks} | {score_str} |")
            ranking_out = "\n".join(ranking_lines)
        else:
            ranking_out = "Usa **K planes** > 1 con el planificador A* para ver alternativas rankeadas."

        return plan_out, score_out, mc_out, llm_out, ranking_out

    except Exception as exc:
        err = f"❌ Error: {exc}"
        return err, err, err, err, err


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Skill Path Planner") as demo:

        gr.Markdown(
            """
# 🎓 Skill Path Planner
**Planificador inteligente de trayectorias profesionales**
Describe tu situación actual y tu meta en lenguaje natural. El sistema encontrará la secuencia óptima de cursos.
"""
        )

        with gr.Row():
            # --- Columna izquierda: inputs ---
            with gr.Column(scale=2):
                query = gr.Textbox(
                    label="¿Qué quieres aprender o en qué quieres trabajar?",
                    placeholder=(
                        "Ejemplos:\n"
                        "• Sé Python básico y quiero ser data analyst en 30 semanas\n"
                        "• Quiero ser ML engineer, tengo 10 horas por semana\n"
                        "• No sé nada, quiero trabajar con datos, cursos fáciles"
                    ),
                    lines=4,
                    max_lines=6,
                )

                gr.Examples(
                    examples=[[e] for e in EXAMPLES],
                    inputs=[query],
                    label="💡 Sugerencias de consulta",
                    examples_per_page=5,
                )

                with gr.Group():
                    gr.Markdown("### ⚙️ Configuración del planificador")

                    with gr.Row():
                        provider = gr.Radio(
                            choices=["Mock", "Gemini"],
                            value="Mock",
                            label="Proveedor LLM",
                            info="Mock: sin API key. Gemini: requiere GEMINI_API_KEY en .env",
                        )
                        planner = gr.Radio(
                            choices=["astar", "greedy", "ucs"],
                            value="astar",
                            label="Algoritmo planificador",
                            info="A* recomendado. Greedy/UCS más rápidos pero subóptimos.",
                        )

                    with gr.Row():
                        k_plans = gr.Slider(
                            minimum=1, maximum=3, step=1, value=1,
                            label="K planes alternativos",
                            info="Solo A* genera múltiples alternativas",
                        )
                        max_nodes = gr.Number(
                            value=0, minimum=0, step=500,
                            label="Límite de nodos A*",
                            info="0 = sin límite. Reduce para búsquedas más rápidas.",
                        )

                    with gr.Row():
                        use_monte_carlo = gr.Checkbox(
                            label="Simulación Monte Carlo",
                            value=False,
                            info="Estima probabilidad de éxito del plan",
                        )
                        mc_runs = gr.Slider(
                            minimum=50, maximum=500, step=50, value=200,
                            label="Corridas Monte Carlo",
                            visible=False,
                        )

                    use_evaluate = gr.Checkbox(
                        label="Evaluar plan con LLM",
                        value=False,
                        info="Análisis cualitativo del plan (consume 1 llamada extra a la API)",
                    )

                use_monte_carlo.change(
                    fn=lambda v: gr.update(visible=v),
                    inputs=use_monte_carlo,
                    outputs=mc_runs,
                )

                run_btn = gr.Button("🚀 Generar plan", variant="primary", size="lg")

            # --- Columna derecha: outputs ---
            with gr.Column(scale=3):
                gr.Markdown("### 📊 Resultados")
                with gr.Tabs():
                    with gr.Tab("Plan"):
                        plan_out = gr.Code(
                            label="Plan generado",
                            language="json",
                            lines=30,
                        )
                    with gr.Tab("Score"):
                        score_out = gr.Markdown(
                            value="— Ejecuta una consulta para ver el desglose de score —"
                        )
                    with gr.Tab("Monte Carlo"):
                        mc_out = gr.Markdown(
                            value="— Activa Monte Carlo para ver la simulación de riesgo —"
                        )
                    with gr.Tab("Evaluación LLM"):
                        llm_out = gr.Markdown(
                            value="— Activa 'Evaluar con LLM' para ver el análisis cualitativo —"
                        )
                    with gr.Tab("Ranking"):
                        ranking_out = gr.Markdown(
                            value="— Usa K > 1 con A* para ver planes alternativos —"
                        )

        run_btn.click(
            fn=run_pipeline,
            inputs=[query, provider, planner, k_plans, use_monte_carlo, mc_runs, use_evaluate, max_nodes],
            outputs=[plan_out, score_out, mc_out, llm_out, ranking_out],
        )

        gr.Markdown(
            """
---
**Roles disponibles:** Analista de datos · Ingeniero de Machine Learning
**Habilidades:** Python básico/intermedio · SQL · Estadística · Análisis de datos · ML · Deep Learning · MLOps
"""
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(inbrowser=True, theme=THEME)
