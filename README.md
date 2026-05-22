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
