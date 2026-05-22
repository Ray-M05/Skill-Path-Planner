# Skill Path Planner

Sistema de planificacion de trayectorias profesionales con busqueda, validacion formal, simulacion Monte Carlo y evaluacion LLM.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
python -m pytest
```

## Gemini

La integracion real con Gemini pero los tests normales no hacen llamadas al API. Para activar pruebas reales configura `.env` con `GEMINI_API_KEY` y `LLM_LIVE_TESTS=true`.

## Modos LLM

Modo mock para tests y desarrollo sin coste:

```text
LLM_PROVIDER=mock
LLM_MODEL=mock
```

En este modo el codigo debe pasar un `mock_response` a `LLMClient`, o usar un cliente fake en tests.

Modo Gemini real:

```text
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash-lite
GEMINI_API_KEY=your_api_key_here
```

La interpretacion de objetivos usa un vocabulario controlado construido desde `data/roles.json` y `data/skills.json`. El LLM extrae desde el texto del usuario el rol objetivo, habilidades iniciales y restricciones. Debe devolver JSON, y `src/llm/interpreter.py` valida que los IDs existan antes de crear un `GoalSpec`.

## CLI preliminar

Ejemplo sin coste usando mock y A* automatico:

```bash
python -m src.main --provider mock --planner astar --goal "Quiero ser ingeniero de machine learning en 18 meses, se Python basico y puedo dedicar 10 horas semanales."
```

Ejemplo con Gemini real y validacion de una trayectoria manual, usando `GEMINI_API_KEY` en `.env`:

```bash
python -m src.main --provider gemini --goal "Quiero ser analista de datos en 12 semanas, se Python basico y SQL basico, y puedo dedicar 8 horas semanales." --courses "course_python_intermediate,course_statistics_basic,course_data_analysis_basic"
```

La CLI imprime el texto del usuario, el prompt exacto enviado al LLM, la respuesta cruda, el `GoalSpec` validado y el resultado del validador formal. Si no pasas `--courses`, genera la trayectoria con `--planner greedy`, `--planner ucs` o `--planner astar`.
