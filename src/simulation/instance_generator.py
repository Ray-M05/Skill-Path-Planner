# Generador de instancias del dominio Musica mediante simulacion. 
from __future__ import annotations

import argparse
import heapq
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import Course, Role, SearchState, Skill, StudentProfile
from ..planners.common import (
    apply_course,
    is_applicable,
    is_goal,
    violates_profile_limits,
)
from ..planners.greedy import greedy_plan
from .music_domain import GOAL_TEMPLATES, PROFILES, ROLES, SKILLS, SkillSpec

_ORACLE_MAX_NODES = 3000


@dataclass
class GenerationConfig:
    seed: int = 42
    n_instances: int = 15


@dataclass
class GeneratedDataset:
    skills: dict[str, Skill]
    courses: dict[str, Course]
    roles: dict[str, Role]
    profiles: dict[str, StudentProfile]
    instances: list[dict]


# Generadores de cada entidad
def generate_skills() -> dict[str, Skill]:
    """Skills con nombres y alias reales tomados de la plantilla del dominio."""
    return {
        spec.id: Skill(
            id=spec.id,
            name=spec.name,
            aliases=list(spec.aliases),
            category=spec.track,
        )
        for spec in SKILLS
    }


def _make_course(spec: SkillSpec, rng: np.random.Generator) -> Course:
    """Un curso que produce la skill `spec`, con parametros muestreados y escalados
    por nivel (mas avanzado => mas largo, mas dificil, menor probabilidad de aprobar)."""
    tier = spec.tier
    duration = int(rng.integers(3, 6)) + tier  # tier0: 3-5 ... tier4: 7-9
    weekly_hours = int(rng.integers(3, 7))  # 3..6 (no excede los perfiles)
    difficulty = round(min(0.9, 0.2 + 0.12 * tier + float(rng.uniform(0.0, 0.1))), 2)
    pass_prob = round(max(0.6, 0.93 - 0.04 * tier - float(rng.uniform(0.0, 0.05))), 2)
    return Course(
        id=f"curso_{spec.id}",
        name=f"Curso de {spec.name}",
        description=f"Curso del area de {spec.track} que ensena {spec.name}.",
        prerequisites=frozenset(spec.prerequisites),
        outcomes=frozenset({spec.id}),
        duration_weeks=duration,
        weekly_hours=weekly_hours,
        difficulty=difficulty,
        pass_base_probability=pass_prob,
    )


def generate_courses(rng: np.random.Generator) -> dict[str, Course]:
    """Un curso por skill; las skills fundamentales (sin prereqs) se aprenden desde
    cero, garantizando que todo objetivo sea alcanzable sin conocimientos previos."""
    return {f"curso_{spec.id}": _make_course(spec, rng) for spec in SKILLS}


def generate_roles() -> dict[str, Role]:
    return {
        spec.id: Role(
            id=spec.id,
            name=spec.name,
            required_skills=frozenset(spec.required),
            recommended_skills=frozenset(spec.recommended),
        )
        for spec in ROLES
    }


def generate_profiles() -> dict[str, StudentProfile]:
    return {
        spec.id: StudentProfile(
            id=spec.id,
            initial_skills=frozenset(spec.initial),
            max_weeks=spec.max_weeks,
            max_weekly_hours=spec.max_weekly_hours,
            risk_tolerance=spec.risk_tolerance,
        )
        for spec in PROFILES
    }


def _reachable_skills(
    initial: set[str], courses: dict[str, Course], max_weekly_hours: int
) -> set[str]:
    """Cierre de skills alcanzables (prereqs + horas semanales), ignorando el
    presupuesto de semanas. Pre-chequeo polinomial y barato de factibilidad."""
    skills = set(initial)
    changed = True
    while changed:
        changed = False
        for course in courses.values():
            if (
                course.weekly_hours <= max_weekly_hours
                and course.prerequisites <= skills
                and not course.outcomes <= skills
            ):
                skills |= course.outcomes
                changed = True
    return skills


def _feasible(
    initial: set[str],
    target: set[str],
    courses: dict[str, Course],
    profile: StudentProfile,
    max_nodes: int = _ORACLE_MAX_NODES,
) -> bool:
    """Oraculo de factibilidad: existe un plan valido dentro de las restricciones
    del perfil (prereqs, horas semanales, semanas maximas y cobertura del objetivo).
    """
    start = SearchState(skills=frozenset(initial), taken_courses=(), weeks_used=0, difficulty_sum=0.0)
    if is_goal(start, target):
        return True

    counter = itertools.count()
    frontier: list[tuple[float, int, SearchState]] = [(0.0, next(counter), start)]
    best_cost: dict[frozenset[str], float] = {}
    expanded = 0

    while frontier and expanded < max_nodes:
        cost, _, state = heapq.heappop(frontier)
        key = state.skills
        if key in best_cost and best_cost[key] <= cost:
            continue
        best_cost[key] = cost
        expanded += 1

        if is_goal(state, target):
            return True

        for course in courses.values():
            if not is_applicable(course, state):
                continue
            if violates_profile_limits(course, state, profile):
                continue
            nxt = apply_course(course, state)
            nxt_cost = nxt.weeks_used + 2.0 * nxt.difficulty_sum + 0.5 * len(nxt.taken_courses)
            heapq.heappush(frontier, (nxt_cost, next(counter), nxt))

    return False


def generate_instances(
    cfg: GenerationConfig,
    courses: dict[str, Course],
    roles: dict[str, Role],
    profiles: dict[str, StudentProfile],
    rng: np.random.Generator,
) -> list[dict]:
    """Muestrea (perfil, rol) y conserva solo combinaciones resolubles.
    """
    instances: list[dict] = []
    role_ids = list(roles)
    profile_ids = list(profiles)
    if not role_ids or not profile_ids:
        return instances

    feasible: list[tuple[str, str]] = []
    for pid in profile_ids:
        profile = profiles[pid]
        initial = set(profile.initial_skills)
        for rid in role_ids:
            target = set(roles[rid].required_skills)
            # Fase 1: cierre barato; descarta objetivos no alcanzables sin busqueda.
            if not target <= _reachable_skills(initial, courses, profile.max_weekly_hours):
                continue
            # Fase 2: greedy confirma factibilidad rapido en el caso comun.
            if greedy_plan(initial, target, courses, profile).valid:
                feasible.append((pid, rid))
                continue
            # Fase 3: si greedy falla, UCS acotado como respaldo.
            if _feasible(initial, target, courses, profile):
                feasible.append((pid, rid))

    if not feasible:
        return instances

    # Recorrer las combinaciones viables en round-robin para cubrirlas
    # todas antes de repetir; el texto del objetivo varia segun la plantilla.
    order = rng.permutation(len(feasible))
    for i in range(cfg.n_instances):
        pid, rid = feasible[int(order[i % len(feasible)])]
        profile = profiles[pid]
        role = roles[rid]
        template = GOAL_TEMPLATES[int(rng.integers(0, len(GOAL_TEMPLATES)))]
        goal_text = template.format(role_name=role.name, hours=profile.max_weekly_hours)
        instances.append(
            {
                "id": f"gen_inst_{len(instances):03d}",
                "profile_id": pid,
                "goal_text": goal_text,
                "expected_role_id": rid,
            }
        )
    return instances


def generate_dataset(cfg: GenerationConfig) -> GeneratedDataset:
    """Orquesta la generacion completa con un unico sembrado (reproducible)."""
    rng = np.random.default_rng(cfg.seed)
    skills = generate_skills()
    courses = generate_courses(rng)
    roles = generate_roles()
    profiles = generate_profiles()
    instances = generate_instances(cfg, courses, roles, profiles, rng)
    return GeneratedDataset(skills, courses, roles, profiles, instances)


# Serializacion
def _skill_to_dict(s: Skill) -> dict:
    return {"id": s.id, "name": s.name, "aliases": list(s.aliases), "category": s.category}


def _course_to_dict(c: Course) -> dict:
    return {
        "id": c.id,
        "name": c.name,
        "description": c.description,
        "prerequisites": sorted(c.prerequisites),
        "outcomes": sorted(c.outcomes),
        "duration_weeks": c.duration_weeks,
        "weekly_hours": c.weekly_hours,
        "difficulty": c.difficulty,
        "pass_base_probability": c.pass_base_probability,
    }


def _role_to_dict(r: Role) -> dict:
    return {
        "id": r.id,
        "name": r.name,
        "required_skills": sorted(r.required_skills),
        "recommended_skills": sorted(r.recommended_skills),
    }


def _profile_to_dict(p: StudentProfile) -> dict:
    return {
        "id": p.id,
        "initial_skills": sorted(p.initial_skills),
        "max_weeks": p.max_weeks,
        "max_weekly_hours": p.max_weekly_hours,
        "risk_tolerance": p.risk_tolerance,
    }


def _dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataset(ds: GeneratedDataset, out_dir: str | Path) -> Path:
    """Escribe el dataset con nombres canonicos (compatibles con load_dataset)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "skills.json", [_skill_to_dict(s) for s in ds.skills.values()])
    _dump(out / "courses.json", [_course_to_dict(c) for c in ds.courses.values()])
    _dump(out / "roles.json", [_role_to_dict(r) for r in ds.roles.values()])
    _dump(out / "student_profiles.json", [_profile_to_dict(p) for p in ds.profiles.values()])
    _dump(out / "instances.json", ds.instances)
    return out


# CLI
def _config_from_args(args: argparse.Namespace) -> GenerationConfig:
    cfg_dict: dict = {}
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text(encoding="utf-8"))

    defaults = GenerationConfig()
    if args.seed is not None:
        cfg_dict["seed"] = args.seed
    if args.n_instances is not None:
        cfg_dict["n_instances"] = args.n_instances

    return GenerationConfig(
        seed=cfg_dict.get("seed", defaults.seed),
        n_instances=cfg_dict.get("n_instances", defaults.n_instances),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera un dataset del dominio Musica para el Skill Path Planner."
    )
    parser.add_argument("--config", default=None, help="JSON con GenerationConfig.")
    parser.add_argument("--n-instances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="data/generated", help="Directorio de salida.")
    args = parser.parse_args(argv)

    cfg = _config_from_args(args)
    ds = generate_dataset(cfg)
    out = write_dataset(ds, args.out)

    print(f"Dataset generado en: {out.resolve()}")
    print(
        f"  skills={len(ds.skills)}  courses={len(ds.courses)}  roles={len(ds.roles)}  "
        f"profiles={len(ds.profiles)}  instances={len(ds.instances)}"
    )
    if len(ds.instances) < cfg.n_instances:
        print(
            f"  Aviso: solo se generaron {len(ds.instances)}/{cfg.n_instances} "
            "instancias resolubles (ajusta seed)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
