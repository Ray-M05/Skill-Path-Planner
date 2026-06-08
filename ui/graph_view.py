"""Vista de grafo de precedencia de cursos.

Construye un grafo dirigido donde los nodos son cursos y existe una arista
A -> B si el curso B requiere una habilidad que produce el curso A.  Cada nodo
recibe un color pastel unico; todas sus aristas salientes comparten ese mismo
color para que sea facil trazar de donde viene cada dependencia.

Siempre se recarga el dataset desde disco al invocar `build_course_graph_figure`.
"""

from __future__ import annotations

import colorsys
import random
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.loader import load_dataset
from src.models import Course, Dataset, Role


PALETTE = {
    "bg": "#F4EDE0",
    "node_fill": "#FFFFFF",
    "node_border": "#C8BCA8",
    "edge": "#9B8466",
    "text": "#2C1E0E",
    "subtle": "#7A6040",
    "accent": "#C5811E",
}

def _assign_node_colors(graph: nx.DiGraph, seed: int = 42) -> dict[str, str]:
    """Asigna un color pastel unico a cada nodo del grafo.

    Distribuye los matices uniformemente en el espacio HSV y luego los mezcla
    con una semilla fija para que el resultado sea estable entre recargas pero
    visualmente 'aleatorio'.  Saturacion baja (0.38) y brillo alto (0.93)
    producen colores pasteles que contrastan bien con texto oscuro.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    hues = [i / max(n, 1) for i in range(n)]
    rng = random.Random(seed)
    rng.shuffle(hues)
    colors: dict[str, str] = {}
    for node, hue in zip(nodes, hues):
        r, g, b = colorsys.hsv_to_rgb(hue, 0.38, 0.93)
        colors[node] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return colors

ROLE_STYLE = {
    "role_ml_engineer": {
        "label": "ML",
        "required_fill": "#C5811E",
        "recommended_fill": "#F1D9A8",
        "text_required": "#FFFFFF",
        "text_recommended": "#5A3E10",
    },
    "role_data_analyst": {
        "label": "DA",
        "required_fill": "#3D7A8C",
        "recommended_fill": "#BFD7DF",
        "text_required": "#FFFFFF",
        "text_recommended": "#1F3D45",
    },
}

# Dimensiones del nodo (unidades del eje)
NODE_W = 2.8
NODE_H = 0.82
NODE_HW = NODE_W / 2  # half-width
NODE_HH = NODE_H / 2  # half-height


def _build_graph(dataset: Dataset) -> nx.DiGraph:
    graph = nx.DiGraph()
    skill_producer: dict[str, str] = {}
    for course in dataset.courses.values():
        for skill_id in course.outcomes:
            skill_producer[skill_id] = course.id
        graph.add_node(course.id, course=course)

    for course in dataset.courses.values():
        for prereq in course.prerequisites:
            producer = skill_producer.get(prereq)
            if producer and producer != course.id:
                graph.add_edge(producer, course.id)
    return graph


def _assign_layers(graph: nx.DiGraph) -> dict[str, int]:
    layers: dict[str, int] = {}
    for layer, nodes in enumerate(nx.topological_generations(graph)):
        for node in nodes:
            layers[node] = layer
    return layers


def _role_badges_for(course: Course, roles: dict[str, Role]) -> list[tuple[str, str]]:
    """Devuelve [(role_id, 'required'|'recommended')] para los skills de salida."""
    badges: list[tuple[str, str]] = []
    for role_id, role in roles.items():
        kind: str | None = None
        for skill_id in course.outcomes:
            if skill_id in role.required_skills:
                kind = "required"
                break
            if skill_id in role.recommended_skills and kind is None:
                kind = "recommended"
        if kind is not None:
            badges.append((role_id, kind))
    return badges



def _layered_positions(
    graph: nx.DiGraph, layers: dict[str, int]
) -> dict[str, tuple[float, float]]:
    by_layer: dict[int, list[str]] = {}
    for node, layer in layers.items():
        by_layer.setdefault(layer, []).append(node)

    x_spacing = 4.2
    y_spacing = 1.80  # suficiente para nodo + badge sin solapamiento
    positions: dict[str, tuple[float, float]] = {}
    for layer, nodes in by_layer.items():
        nodes_sorted = sorted(nodes, key=lambda n: graph.nodes[n]["course"].name)
        n = len(nodes_sorted)
        for idx, node in enumerate(nodes_sorted):
            y = (n - 1) / 2.0 - idx
            positions[node] = (layer * x_spacing, y * y_spacing)
    return positions


def _compute_ports(
    graph: nx.DiGraph, positions: dict[str, tuple[float, float]]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Distribuye los puntos de conexion a lo largo del borde del nodo.

    Para cada nodo, ordena sus vecinos por posicion y y les asigna un offset
    vertical dentro del borde del nodo.  Esto separa las flechas cuando un
    nodo tiene muchas entradas o salidas y evita que se solapen.
    """
    out_ports: dict[str, dict[str, float]] = {}
    in_ports: dict[str, dict[str, float]] = {}
    spread = NODE_HH * 1.5  # rango vertical total para distribuir puertos

    for node in graph.nodes():
        succs = sorted(graph.successors(node), key=lambda n: positions[n][1])
        n = len(succs)
        out_ports[node] = {
            s: ((i / max(n - 1, 1)) - 0.5) * spread if n > 1 else 0.0
            for i, s in enumerate(succs)
        }

        preds = sorted(graph.predecessors(node), key=lambda n: positions[n][1])
        n = len(preds)
        in_ports[node] = {
            p: ((i / max(n - 1, 1)) - 0.5) * spread if n > 1 else 0.0
            for i, p in enumerate(preds)
        }

    return out_ports, in_ports


def _draw_node(
    ax,
    x: float,
    y: float,
    course: Course,
    face_color: str,
    badges: list[tuple[str, str]],
) -> None:
    box = FancyBboxPatch(
        (x - NODE_HW, y - NODE_HH),
        NODE_W,
        NODE_H,
        boxstyle="round,pad=0.07,rounding_size=0.15",
        linewidth=1.3,
        edgecolor=PALETTE["node_border"],
        facecolor=face_color,
        zorder=2,
    )
    ax.add_patch(box)

    raw_lines = textwrap.wrap(course.name, width=24)
    lines = raw_lines[:2]
    if len(raw_lines) > 2:
        lines[1] = lines[1][:20] + "…"

    line_h = 0.22
    start_y = y + (line_h / 2 if len(lines) > 1 else 0.0)
    for j, line in enumerate(lines):
        ax.text(
            x,
            start_y - j * line_h,
            line,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=PALETTE["text"],
            zorder=3,
        )

    if not badges:
        return

    badge_w = 0.58
    badge_h = 0.28
    gap = 0.07
    total_w = len(badges) * badge_w + (len(badges) - 1) * gap
    start_x = x - total_w / 2
    badge_y = y - NODE_HH - badge_h / 2 - 0.04

    for i, (role_id, kind) in enumerate(badges):
        style = ROLE_STYLE.get(role_id)
        if style is None:
            continue
        fill = style[f"{kind}_fill"]
        text_color = style[f"text_{kind}"]
        bx = start_x + i * (badge_w + gap)
        badge = FancyBboxPatch(
            (bx, badge_y - badge_h / 2),
            badge_w,
            badge_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0.8,
            edgecolor=PALETTE["node_border"],
            facecolor=fill,
            zorder=4,
        )
        ax.add_patch(badge)
        marker = "✓" if kind == "required" else "○"
        ax.text(
            bx + badge_w / 2,
            badge_y,
            f"{style['label']} {marker}",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color=text_color,
            zorder=5,
        )


def _draw_edges(
    ax,
    graph: nx.DiGraph,
    positions: dict[str, tuple[float, float]],
    out_ports: dict[str, dict[str, float]],
    in_ports: dict[str, dict[str, float]],
    node_colors: dict[str, str],
) -> None:
    for src, dst in graph.edges():
        x1, y1 = positions[src]
        x2, y2 = positions[dst]

        oy_src = out_ports[src].get(dst, 0.0)
        oy_dst = in_ports[dst].get(src, 0.0)

        start = (x1 + NODE_HW, y1 + oy_src)
        end = (x2 - NODE_HW, y2 + oy_dst)

        dy = y2 - y1
        dx = max(x2 - x1, 0.01)
        ratio = abs(dy / dx)
        if ratio < 0.08:
            rad = 0.0
        else:
            # Curvatura proporcional a la pendiente, maxima 0.35
            rad = min(0.20 * ratio, 0.35) * (1 if dy > 0 else -1)

        arrow = mpatches.FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=2.0,
            color=node_colors[src],
            alpha=0.70,
            zorder=1,
        )
        ax.add_patch(arrow)


def _draw_legend(ax, dataset: Dataset) -> None:
    handles = []
    for role_id, role in dataset.roles.items():
        style = ROLE_STYLE.get(role_id)
        if style is None:
            continue
        handles.append(
            mpatches.Patch(
                facecolor=style["required_fill"],
                edgecolor=PALETTE["node_border"],
                label=f"{role.name} (requerido ✓)",
            )
        )
        handles.append(
            mpatches.Patch(
                facecolor=style["recommended_fill"],
                edgecolor=PALETTE["node_border"],
                label=f"{role.name} (recomendado ○)",
            )
        )

    if handles:
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=min(5, len(handles)),
            fontsize=8,
            frameon=False,
        )


def _component_layout(sg: nx.DiGraph) -> dict:
    """Calcula layers, positions y ports para un subgrafo (componente)."""
    layers = _assign_layers(sg)
    positions = _layered_positions(sg, layers)
    out_ports, in_ports = _compute_ports(sg, positions)
    by_layer: dict[int, int] = {}
    for layer in layers.values():
        by_layer[layer] = by_layer.get(layer, 0) + 1
    return dict(
        sg=sg,
        positions=positions,
        out_ports=out_ports,
        in_ports=in_ports,
        max_layer=max(layers.values()) if layers else 0,
        max_per_layer=max(by_layer.values()) if by_layer else 1,
    )


def build_course_graph_figure(data_dir: str = "data") -> Figure:
    """Construye y devuelve la figura del grafo de precedencia de cursos.

    El grafo se divide en componentes conexas debiles (de mayor a menor);
    cada componente se dibuja en su propio subplot para mayor claridad.
    Recarga el dataset desde disco en cada llamada.
    """
    dataset = load_dataset(data_dir)
    graph = _build_graph(dataset)
    node_colors = _assign_node_colors(graph)

    # Componentes de mayor a menor
    components = sorted(
        nx.weakly_connected_components(graph), key=len, reverse=True
    )
    comp_data = [_component_layout(graph.subgraph(c).copy()) for c in components]

    fig_width = max(10.0, max(3.5 + d["max_layer"] * 4.0 for d in comp_data))
    row_heights = [max(3.0, 2.0 + d["max_per_layer"] * 1.80) for d in comp_data]
    total_height = sum(row_heights) + len(comp_data) * 0.6

    fig = plt.figure(figsize=(fig_width, total_height), layout="constrained")
    fig.patch.set_facecolor(PALETTE["bg"])

    gs = fig.add_gridspec(
        len(comp_data), 1, height_ratios=row_heights, hspace=0.35
    )

    for i, d in enumerate(comp_data):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor(PALETTE["bg"])
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")

        _draw_edges(ax, d["sg"], d["positions"], d["out_ports"], d["in_ports"], node_colors)
        for node, (x, y) in d["positions"].items():
            course = d["sg"].nodes[node]["course"]
            badges = _role_badges_for(course, dataset.roles)
            _draw_node(ax, x, y, course, node_colors[node], badges)

        xs = [p[0] for p in d["positions"].values()]
        ys = [p[1] for p in d["positions"].values()]
        ax.set_xlim(min(xs) - 2.5, max(xs) + 2.5)
        ax.set_ylim(min(ys) - 1.8, max(ys) + 1.8)

        if i == 0:
            ax.set_title(
                "Mapa de precedencia de cursos",
                fontsize=13,
                fontweight="bold",
                color=PALETTE["text"],
                pad=14,
            )

    _draw_legend(fig.axes[-1], dataset)
    return fig


def build_course_table_markdown(data_dir: str = "data") -> str:
    """Tabla markdown con todos los cursos, prerrequisitos y roles asociados."""
    dataset = load_dataset(data_dir)
    skill_name = {sid: s.name for sid, s in dataset.skills.items()}

    lines = [
        "| Curso | Prerrequisitos | Skill resultante | Semanas | h/sem | Roles |",
        "|:---|:---|:---|:---:|:---:|:---|",
    ]
    for course in sorted(dataset.courses.values(), key=lambda c: c.name):
        prereqs = ", ".join(skill_name.get(s, s) for s in course.prerequisites) or "—"
        outcomes = ", ".join(skill_name.get(s, s) for s in course.outcomes) or "—"
        badges = _role_badges_for(course, dataset.roles)
        role_chips = []
        for role_id, kind in badges:
            style = ROLE_STYLE.get(role_id)
            role_name = dataset.roles[role_id].name
            marker = "✓ requerido" if kind == "required" else "○ recomendado"
            label = style["label"] if style else role_name
            role_chips.append(f"`{label}` {marker}")
        role_str = " · ".join(role_chips) or "—"
        lines.append(
            f"| **{course.name}** | {prereqs} | {outcomes} | "
            f"{course.duration_weeks} | {course.weekly_hours} | {role_str} |"
        )
    return "\n".join(lines)
