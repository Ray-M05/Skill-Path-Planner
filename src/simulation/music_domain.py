"""Plantilla del dominio Musica para el generador de instancias.

Define una ontologia curada y coherente: pistas (tracks) con niveles (tiers) cuyas
aristas de prerrequisito tienen sentido real en el dominio. El generador instancia
estos datos y muestrea con sembrado solo los params de cada curso
(duracion, horas, dificultad, probabilidad de aprobar); la estructura del grafo no
es aleatoria, por lo que el grafo resultante es honesto y entendible.

Invariante clave: toda skill es producible por algun curso. Las skills fundamentales
(tier 0) se aprenden desde cero (prerequisites vacios), de modo que cualquier rol es
alcanzable partiendo sin conocimientos previos.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillSpec:
    id: str
    name: str
    aliases: tuple[str, ...]
    track: str
    tier: int
    prerequisites: tuple[str, ...]


@dataclass(frozen=True)
class RoleSpec:
    id: str
    name: str
    required: tuple[str, ...]
    recommended: tuple[str, ...]


@dataclass(frozen=True)
class ProfileSpec:
    id: str
    initial: tuple[str, ...]
    max_weeks: int
    max_weekly_hours: int
    risk_tolerance: float


SKILLS: tuple[SkillSpec, ...] = (
    # Pista: Teoria musical
    SkillSpec(
        "teoria_1", "Teoria musical I",
        ("teoria musical", "teoria musical basica", "solfeo", "lectura de partitura", "leer musica"),
        "Teoria musical", 0, (),
    ),
    SkillSpec(
        "teoria_2", "Teoria musical II",
        ("teoria musical intermedia", "intervalos y escalas", "tonalidades", "escalas"),
        "Teoria musical", 1, ("teoria_1",),
    ),
    SkillSpec(
        "armonia", "Armonia",
        ("armonia", "acordes", "progresiones de acordes", "encadenamiento de acordes"),
        "Teoria musical", 2, ("teoria_2",),
    ),
    SkillSpec(
        "contrapunto", "Contrapunto",
        ("contrapunto", "polifonia", "fuga"),
        "Teoria musical", 3, ("armonia",),
    ),
    # Pista: Instrumento
    SkillSpec(
        "instrumento_1", "Instrumento I",
        ("tocar un instrumento", "instrumento basico", "piano basico", "guitarra basica"),
        "Instrumento", 0, (),
    ),
    SkillSpec(
        "instrumento_2", "Instrumento II",
        ("instrumento intermedio", "tecnica instrumental", "repertorio intermedio"),
        "Instrumento", 1, ("instrumento_1", "teoria_1"),
    ),
    SkillSpec(
        "instrumento_3", "Instrumento III",
        ("instrumento avanzado", "virtuosismo", "repertorio avanzado"),
        "Instrumento", 2, ("instrumento_2", "teoria_2"),
    ),
    # Pista: Composicion
    SkillSpec(
        "composicion_1", "Composicion I",
        ("componer", "composicion basica", "melodia y forma", "escribir musica"),
        "Composicion", 2, ("armonia", "teoria_2"),
    ),
    SkillSpec(
        "composicion_2", "Composicion II",
        ("arreglos", "orquestacion", "composicion intermedia"),
        "Composicion", 3, ("composicion_1", "instrumento_2"),
    ),
    SkillSpec(
        "composicion_avanzada", "Composicion avanzada",
        ("composicion avanzada", "composicion contemporanea"),
        "Composicion", 4, ("composicion_2", "contrapunto"),
    ),
    # Pista: Produccion musical
    SkillSpec(
        "produccion_1", "Produccion musical I",
        ("produccion musical", "daw", "home studio", "grabacion basica", "producir musica"),
        "Produccion musical", 0, (),
    ),
    SkillSpec(
        "mezcla", "Mezcla",
        ("mezcla", "mixing", "mezclar audio"),
        "Produccion musical", 1, ("produccion_1",),
    ),
    SkillSpec(
        "masterizacion", "Masterizacion",
        ("masterizacion", "mastering"),
        "Produccion musical", 2, ("mezcla",),
    ),
    SkillSpec(
        "sound_design", "Diseno sonoro",
        ("diseno sonoro", "sound design", "sintesis", "sintetizadores"),
        "Produccion musical", 3, ("masterizacion", "armonia"),
    ),
    # Pista: Canto
    SkillSpec(
        "canto_1", "Canto I",
        ("canto basico", "tecnica vocal", "aprender a cantar"),
        "Canto", 0, (),
    ),
    SkillSpec(
        "canto_2", "Canto II",
        ("canto intermedio", "afinacion", "repertorio vocal"),
        "Canto", 1, ("canto_1", "teoria_1"),
    ),
)


ROLES: tuple[RoleSpec, ...] = (
    RoleSpec(
        "role_interprete", "Interprete musical",
        ("instrumento_3", "teoria_2"), ("contrapunto", "canto_2"),
    ),
    RoleSpec(
        "role_compositor", "Compositor",
        ("composicion_2", "armonia", "teoria_2"), ("contrapunto", "instrumento_2"),
    ),
    RoleSpec(
        "role_productor", "Productor musical",
        ("masterizacion", "mezcla", "produccion_1", "armonia"), ("composicion_1", "sound_design"),
    ),
    RoleSpec(
        "role_profesor", "Profesor de musica",
        ("teoria_2", "instrumento_2", "armonia"), ("composicion_1", "canto_2"),
    ),
    RoleSpec(
        "role_cantante", "Cantante",
        ("canto_2", "teoria_1"), ("instrumento_1",),
    ),
)


PROFILES: tuple[ProfileSpec, ...] = (
    ProfileSpec("profile_principiante", (), 120, 8, 0.4),
    ProfileSpec("profile_conservatorio", ("teoria_1", "teoria_2", "instrumento_1", "instrumento_2"), 60, 10, 0.6),
    ProfileSpec("profile_autodidacta_produccion", ("produccion_1", "teoria_1"), 80, 6, 0.5),
    ProfileSpec("profile_teorico", ("teoria_1", "teoria_2", "armonia"), 70, 8, 0.6),
    ProfileSpec("profile_cantante_aficionado", ("canto_1",), 90, 6, 0.4),
    ProfileSpec("profile_con_prisa", ("teoria_1",), 24, 12, 0.3),
)


# Frases naturales para los textos de objetivo de las instancias generadas.
GOAL_TEMPLATES: tuple[str, ...] = (
    "Quiero ser {role_name}.",
    "Me gustaria dedicarme a ser {role_name}.",
    "Mi meta es convertirme en {role_name} y puedo estudiar {hours} horas semanales.",
)
