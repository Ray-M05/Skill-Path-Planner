# Skill Path Planner

Sistema de planificacion de trayectorias profesionales con busqueda heuristica (A*, UCS, Greedy), validacion formal, simulacion Monte Carlo y evaluacion LLM.

## Setup

```bash
# 1. Crear y activar el entorno virtual
py -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env y poner tus valores (ver seccion "Modos LLM" abajo)

# 4. Verificar instalacion
py -m pytest
```

## Modos LLM

Abre `.env` y elige el modo segun lo que necesites.

**Modo mock** (sin API key, para desarrollo y tests):

```text
LLM_PROVIDER=mock
```

**Modo Gemini real** (requiere API key):

```text
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash-lite
GEMINI_API_KEY=tu_api_key_aqui
```

La interpretacion de objetivos usa un vocabulario controlado construido desde `data/roles.json` y `data/skills.json`. El LLM extrae el rol objetivo, habilidades iniciales y restricciones del texto del usuario. La respuesta debe ser JSON valido; `src/llm/interpreter.py` valida que los IDs existan antes de crear el `GoalSpec`. Si ningun rol encaja, puede crear un rol a medida con prefijo `custom_`.

Para habilitar las pruebas que llaman al API real, agrega tambien:

```text
LLM_LIVE_TESTS=true
```

## CLI

Planificacion automatica con mock (sin coste, sin API key):

```bash
py -m src.main --provider mock --planner astar --goal "Quiero ser ingeniero de machine learning en 18 meses, se Python basico y puedo dedicar 10 horas semanales."
```

Con Monte Carlo para estimar probabilidad de exito:

```bash
py -m src.main --provider mock --planner astar --monte-carlo-runs 500 --monte-carlo-seed 42 --goal "Quiero ser ingeniero de machine learning en 18 meses, se Python basico y puedo dedicar 10 horas semanales."
```

Validacion de una trayectoria manual con Gemini real:

```bash
py -m src.main --provider gemini --goal "Quiero ser analista de datos en 12 semanas, se Python basico y SQL basico, y puedo dedicar 8 horas semanales." --courses "course_python_intermediate,course_statistics_basic,course_data_analysis_basic"
```

Planificadores disponibles: `astar` (recomendado), `ucs`, `greedy`.
Flags utiles: `--k 3` (hasta 3 planes alternativos, solo A*), `--evaluate` (evaluacion LLM cualitativa), `--metrics` (imprime CSV de metricas).

## Dataset generado (dominio Musica)

El sistema incluye un generador de instancias sinteticas con un dominio real y coherente (Musica). Produce un catalogo completo (skills, cursos, roles, perfiles, instancias) que puede usarse directamente con el CLI y el runner de experimentos.

Generar el dataset:

```bash
py -m src.simulation.instance_generator --seed 42 --n-instances 15 --out data/generated
```

El catalogo generado tiene 16 habilidades en 5 pistas (Teoria musical, Instrumento, Composicion, Produccion musical, Canto), 5 roles (Interprete musical, Compositor, Productor musical, Profesor de musica, Cantante) y 6 perfiles de estudiante. Los parametros de cada curso (duracion, dificultad, probabilidad de aprobacion) se muestrean con un generador de numeros pseudoaleatorios sembrado, escalados por el nivel de la habilidad en el grafo de prerrequisitos. La estructura del grafo es curada (no aleatoria) para que cada arista tenga sentido en el dominio.

Usar el dataset generado con el CLI:

```bash
py -m src.main --provider mock --data-dir data/generated --planner astar --goal "Quiero ser productor musical"
```

```bash
py -m src.main --provider gemini --data-dir data/generated --goal "Me gustaria componer musica, ya tengo algo de teoria"
```

El modo mock funciona con el dataset generado sin API key: el interprete elige el rol existente mas cercano por nombre y alias.

## Experimentos

Comparar variantes (greedy, ucs, astar, astar_mc, astar_mc_llm) sobre el dataset principal:

```bash
py experiments/run_experiments.py --config experiments/configs/base.json
```

Sobre el dataset generado de Musica:

```bash
py experiments/run_experiments.py --data-dir data/generated --instances data/generated/instances.json --variants greedy ucs astar astar_mc --output experiments/results/generated_raw_results.csv
```

Resumir resultados y generar graficas:

```bash
py experiments/summarize_results.py --input experiments/results/raw_results.csv
py experiments/plot_results.py
```

## Interfaz grafica (UI)

```bash
py ui/app.py
```

## Tests

```bash
py -m pytest          # todos los tests
py -m pytest tests/test_planners.py -v     # un archivo
py -m pytest -k "astar"                    # por patron
```
