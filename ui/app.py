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
from src.llm.interpreter import validate_goal_spec
from src.llm.prompts import GOAL_INTERPRETER_SYSTEM_PROMPT, build_goal_interpreter_user_prompt
from src.main import (
    DEFAULT_MAX_NODES,
    _enrich_plan,
    build_mock_goal_response,
    build_profile_from_goal,
    run_planner_k,
)
from src.scoring import rank_plans
from ui.graph_view import build_course_graph_figure, build_course_table_markdown

_DEFAULT_DATA_DIR = "data"

EXAMPLE_CHIPS = [
    ("Convertirme en Data Analyst", "Sé Python básico y quiero ser data analyst en máximo 30 semanas, 8 horas por semana"),
    ("Transición a ML Engineer",    "Conozco Python básico y estadística básica, quiero ser ML engineer, tengo 12 horas semanales"),
]

THEME = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="amber",
    neutral_hue="stone",
    radius_size="lg",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#F4EDE0",
    block_background_fill="#EDE7D8",
    body_background_fill_dark="#2C2010",
    block_background_fill_dark="#1E180C",
    block_border_color="#EDE7D8",
    block_border_width="1px",
    body_text_color="#2C1E0E",
    block_label_text_color="#7A6040",
    block_title_text_color="#2C1E0E",
    input_background_fill="#EDE7D8",
    input_border_color="#C8BCA8",
    input_border_color_focus="#C5811E",
    input_placeholder_color="#B0A088",
    button_primary_background_fill="#C5811E",
    button_primary_background_fill_hover="#A96D18",
    button_primary_text_color="white",
    button_primary_border_color="#C5811E",
    button_secondary_background_fill="#E4DAC8",
    button_secondary_background_fill_hover="#D9D0BC",
    button_secondary_border_color="#C8BCA8",
    button_secondary_text_color="#3D2B10",
    slider_color="#C5811E",
    color_accent="#C5811E",
    color_accent_soft="#E4DAC8",
    table_even_background_fill="#EDE7D8",
    table_odd_background_fill="#E8E2D6",
    table_border_color="#C8BCA8",
    loader_color="#C5811E",
)


def _fmt_json(obj: dict) -> str:
    def _default(o: object) -> object:
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    return json.dumps(obj, ensure_ascii=False, indent=2, default=_default)


def _fmt_score_md(best) -> str:
    if not best.score_breakdown:
        return "_Score breakdown no disponible (plan inválido)._"
    bd = best.score_breakdown
    sig = bd.get("signals", {})
    lines = [
        f"**Puntuación final: {best.final_score}**",
        "",
        "| | |",
        "|:---|---:|",
        f"| Requisitos cubiertos | **{bd['required']:.0%}** |",
        f"| Recomendados cubiertos | **{bd['recommended']:.0%}** |",
        f"| Penalización de tiempo | **{bd['time_penalty']:.2f}** |",
        f"| Penalización de dificultad | **{bd['difficulty_penalty']:.2f}** |",
        f"| Éxito Monte Carlo | **{bd['mc_success_raw']:.0%}** _(aporte: {bd['mc_success']:+.4f})_ |",
        f"| Calidad LLM | **{bd['llm_quality_raw']:.0%}** _(aporte: {bd['llm_quality']:+.4f})_ |",
        "",
        (
            f"`Monte Carlo {'✅' if sig.get('has_mc') else '❌'}` "
            f"`Eval LLM {'✅' if sig.get('has_llm') else '❌'}` "
            f"`Peso (w): {sig.get('w_required', 0)}`"
        ),
    ]
    return "\n".join(lines)


def run_pipeline(
    query: str,
    provider: str,
    planner: str,
    k_plans: int,
    use_monte_carlo: bool,
    mc_runs: int,
    use_evaluate: bool,
    data_dir: str = _DEFAULT_DATA_DIR,
) -> tuple[str, str, str, str, str]:
    if not query.strip():
        empty = "— Escribe una consulta y presiona Generar plan —"
        return empty, empty, empty, empty, empty

    try:
        dataset = load_dataset(data_dir.strip() or _DEFAULT_DATA_DIR)
        settings = load_settings()
        real_provider = provider.lower()

        user_prompt = build_goal_interpreter_user_prompt(query, dataset)
        if real_provider == "mock":
            raw = build_mock_goal_response(query, dataset)
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

        goal = validate_goal_spec(raw, dataset)
        profile = build_profile_from_goal(goal)

        monte_carlo_runs = mc_runs if use_monte_carlo else 0
        plans = run_planner_k(planner, k_plans, goal, dataset, profile, max_nodes=DEFAULT_MAX_NODES)

        if not plans:
            msg = "No se encontró ninguna trayectoria válida con los parámetros dados."
            return msg, msg, msg, msg, msg

        enriched = []
        for plan in plans:
            enriched.append(
                _enrich_plan(
                    plan, goal, dataset, profile,
                    real_provider, settings, monte_carlo_runs, 42, use_evaluate,
                )
            )

        ranked = rank_plans(enriched)
        best = ranked[0]

        # Tab 1: Plan
        plan_out = _fmt_json(asdict(best))

        # Tab 2: Puntuación
        score_out = _fmt_score_md(best)

        # Tab 3: Monte Carlo
        mc = best.monte_carlo
        if mc and not mc.get("skipped"):
            mc_lines = [
                f"**Simulación con {mc.get('runs', 0)} corridas**",
                "",
                "| Métrica | Valor |",
                "|:---|---:|",
                f"| Probabilidad de éxito | **{mc['success_probability']:.1%}** |",
                f"| IC 95% | `[{mc['success_ci_95'][0]:.1%}, {mc['success_ci_95'][1]:.1%}]` |",
                f"| Riesgo | **{mc['risk_score']:.4f}** |",
                f"| Semanas esperadas | **{mc['expected_weeks']:.1f}** |",
                f"| P10 / P50 / P90 | `{mc['weeks_p10']:.1f}` / `{mc['weeks_p50']:.1f}` / `{mc['weeks_p90']:.1f}` |",
                f"| Cursos fallidos esperados | **{mc['expected_failed_courses']:.2f}** |",
            ]
            mc_out = "\n".join(mc_lines)
        else:
            mc_out = "Monte Carlo no activado. Marca **Simulación Monte Carlo** y elige el número de corridas."

        # Tab 4: Evaluación
        llm = best.llm_evaluation
        if llm and not llm.get("skipped"):
            strengths = "\n".join(f"- {s}" for s in llm.get("main_strengths", []))
            weaknesses = "\n".join(f"- {w}" for w in llm.get("main_weaknesses", []))
            llm_lines = [
                f"**Calidad global: {llm['global_quality']:.0%}**",
                "",
                "| Dimensión | Puntuación |",
                "|:---|---:|",
                f"| Alineación con el rol | **{llm['goal_alignment']:.0%}** |",
                f"| Coherencia pedagógica | **{llm['pedagogical_coherence']:.0%}** |",
                f"| Realismo de perfil | **{llm['profile_realism']:.0%}** |",
                f"| No redundancia | **{llm['non_redundancy']:.0%}** |",
                f"| Valor profesional | **{llm['professional_value']:.0%}** |",
                f"\n**Fortalezas**\n{strengths}",
                f"\n**Debilidades**\n{weaknesses}",
                f"\n**Justificación**\n{llm.get('justification', '')}",
            ]
            llm_out = "\n".join(llm_lines)
        else:
            llm_out = "Evaluación LLM no activada. Marca **Evaluar con LLM** para obtener análisis cualitativo."

        # Tab 5: Ranking
        if len(ranked) > 1:
            ranking_lines = [
                f"**Top {len(ranked)} planes encontrados**",
                "",
                "| Rank | Cursos | Semanas | Score |",
                "|:---:|:---|:---:|:---:|",
            ]
            for i, p in enumerate(ranked):
                courses_str = " → ".join(p.course_ids)
                score_str = f"**{p.final_score}**" if p.final_score is not None else "n/a"
                ranking_lines.append(f"| {i+1} | {courses_str} | {p.total_weeks} | {score_str} |")
            ranking_out = "\n".join(ranking_lines)
        else:
            ranking_out = "Usa **K planes** > 1 con el planificador A* para ver alternativas rankeadas."

        return plan_out, score_out, mc_out, llm_out, ranking_out

    except Exception as exc:
        err = f"❌ Error: {exc}"
        return err, err, err, err, err


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Skill Path Planner", theme=THEME) as demo:

        gr.Markdown("# Skill Path Planner\n**Planificador inteligente de trayectorias**")

        with gr.Tabs():
            with gr.Tab("🧭 Planificador"):
                with gr.Row():
                    # Columna izquierda
                    with gr.Column(scale=2):
                        query = gr.Textbox(
                            show_label=False,
                            placeholder=(
                                "• Describe cual es tu objetivo profesional y habilidades que posees\n"
                                "• Coloca limites de horas semnas y cantidad de semanas disponibles para mas detalle"
                            ),
                            lines=4,
                            max_lines=6,
                        )

                        with gr.Column(visible=True) as chips_col:
                            with gr.Row():
                                chip1 = gr.Button("Convertirme en Data Analyst", variant="secondary", size="sm")
                                chip2 = gr.Button("Transición a ML Engineer",    variant="secondary", size="sm")

                        with gr.Group():
                            gr.Markdown("##### ⚙ CONFIGURATION")

                            data_dir_input = gr.Textbox(
                                value=_DEFAULT_DATA_DIR,
                                label="Carpeta del dataset",
                                placeholder="data  o  data/generated",
                                info="Ruta relativa a la raíz del proyecto",
                            )

                            with gr.Row():
                                provider = gr.Radio(
                                    choices=["Mock", "Gemini"],
                                    value="Mock",
                                    label="Proveedor LLM",
                                    info="Mock: sin API key. Gemini: requiere GEMINI_API_KEY en .env",
                                )
                                planner = gr.Radio(
                                    choices=[("A*", "astar"), ("Greedy", "greedy"), ("UCS", "ucs")],
                                    value="astar",
                                    label="Algoritmo",
                                    info="A* recomendado. Greedy/UCS más rápidos pero subóptimos.",
                                )

                            k_plans = gr.Slider(
                                minimum=1, maximum=3, step=1, value=1,
                                label="Planes alternativos (K)",
                                info="Solo A* genera múltiples alternativas",
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
                                label="Evaluar con LLM",
                                value=False,
                                info="Análisis cualitativo del plan (consume 1 llamada extra a la API)",
                            )

                        run_btn = gr.Button("✨ Generar plan", variant="primary", size="lg")

                    # Columna derecha
                    with gr.Column(scale=3):
                        gr.Markdown("### 📊 Resultados")
                        with gr.Tabs():
                            with gr.Tab("Plan"):
                                plan_out = gr.Code(label="Plan generado", language="json", lines=30)
                            with gr.Tab("Puntuación"):
                                score_out = gr.Markdown(
                                    value="— Ejecuta una consulta para ver el desglose de puntuación —"
                                )
                            with gr.Tab("Monte Carlo"):
                                mc_out = gr.Markdown(
                                    value="— Activa Monte Carlo para ver la simulación de riesgo —"
                                )
                            with gr.Tab("Evaluación"):
                                llm_out = gr.Markdown(
                                    value="— Activa 'Evaluar con LLM' para ver el análisis cualitativo —"
                                )
                            with gr.Tab("Ranking"):
                                ranking_out = gr.Markdown(
                                    value="— Usa K > 1 con A* para ver planes alternativos —"
                                )

                        gr.Markdown(
                            "**Roles disponibles:** Analista de Datos · Ingeniero ML  \n"
                            "**Habilidades:** Python · SQL · Estadística · ML · Deep Learning · MLOps"
                        )

            with gr.Tab("🗺 Mapa de cursos"):
                gr.Markdown(
                    "### Grafo de precedencia de cursos\n"
                    "Cada nodo es un curso; las flechas indican que la habilidad de salida del "
                    "curso origen es prerrequisito del curso destino. Los chips muestran si la "
                    "habilidad producida es **requerida (✓)** o **recomendada (○)** para cada rol."
                )
                with gr.Row():
                    refresh_graph_btn = gr.Button(
                        "🔄 Recargar desde la base de datos", variant="primary", size="sm"
                    )
                graph_plot = gr.Plot(value=build_course_graph_figure(_DEFAULT_DATA_DIR), show_label=False)
                with gr.Accordion("Tabla detallada de cursos", open=False):
                    courses_table = gr.Markdown(value=build_course_table_markdown(_DEFAULT_DATA_DIR))

                def _reload_graph(data_dir: str) -> tuple:
                    d = data_dir.strip() or _DEFAULT_DATA_DIR
                    return build_course_graph_figure(d), build_course_table_markdown(d)

                refresh_graph_btn.click(
                    fn=_reload_graph,
                    inputs=data_dir_input,
                    outputs=[graph_plot, courses_table],
                )

        # Event wiring

        def _on_dataset_change(new_dir: str) -> tuple:
            d = new_dir.strip() or _DEFAULT_DATA_DIR
            is_default = d == _DEFAULT_DATA_DIR
            try:
                fig = build_course_graph_figure(d)
                table = build_course_table_markdown(d)
            except Exception as exc:
                import matplotlib.pyplot as plt
                fig = plt.figure()
                table = f"❌ No se pudo cargar el dataset desde `{d}`: {exc}"
            return fig, table, gr.update(visible=is_default)

        data_dir_input.change(
            fn=_on_dataset_change,
            inputs=data_dir_input,
            outputs=[graph_plot, courses_table, chips_col],
        )

        use_monte_carlo.change(
            fn=lambda v: gr.update(visible=v),
            inputs=use_monte_carlo,
            outputs=mc_runs,
        )

        run_btn.click(
            fn=run_pipeline,
            inputs=[query, provider, planner, k_plans, use_monte_carlo, mc_runs, use_evaluate, data_dir_input],
            outputs=[plan_out, score_out, mc_out, llm_out, ranking_out],
        )

        for chip, (_, full_text) in zip([chip1, chip2], EXAMPLE_CHIPS):
            chip.click(fn=lambda t=full_text: t, outputs=query)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(inbrowser=True)
