from src.dataset.loader import load_dataset
from src.dataset.resolver import RoleResolver, SkillResolver, normalize_text


def test_normalize_text_removes_accents_punctuation_and_extra_spaces() -> None:
    assert normalize_text("  Python Basico!!!  ") == "python basico"
    assert normalize_text("Estadistica basica") == "estadistica basica"
    assert normalize_text("Programacion en Python, basica") == "programacion en python basica"


def test_skill_resolver_matches_id_name_and_alias() -> None:
    dataset = load_dataset("data")
    resolver = SkillResolver(dataset.skills)

    assert resolver.resolve_skill_text("skill_python_basic") == "skill_python_basic"
    assert resolver.resolve_skill_text("Python basico") == "skill_python_basic"
    assert resolver.resolve_skill_text("programacion python basica") == "skill_python_basic"


def test_skill_resolver_matches_simple_fuzzy_text() -> None:
    dataset = load_dataset("data")
    resolver = SkillResolver(dataset.skills)

    assert resolver.resolve_skill_text("pyton basico") == "skill_python_basic"


def test_skill_resolver_returns_none_for_unknown_text() -> None:
    dataset = load_dataset("data")
    resolver = SkillResolver(dataset.skills)

    assert resolver.resolve_skill_text("habilidad inexistente") is None


def test_allowed_skill_catalog_is_prompt_ready() -> None:
    dataset = load_dataset("data")
    catalog = SkillResolver(dataset.skills).get_allowed_skill_catalog_for_prompt()

    assert catalog[0]["id"].startswith("skill_")
    assert {"id", "name", "aliases", "category"} == set(catalog[0])


def test_role_resolver_matches_id_name_and_fuzzy_text() -> None:
    dataset = load_dataset("data")
    resolver = RoleResolver(dataset.roles)

    assert resolver.resolve_role_text("role_data_analyst") == "role_data_analyst"
    assert resolver.resolve_role_text("Analista de datos") == "role_data_analyst"
    assert resolver.resolve_role_text("analista datos") == "role_data_analyst"


def test_allowed_role_catalog_is_prompt_ready() -> None:
    dataset = load_dataset("data")
    catalog = RoleResolver(dataset.roles).get_allowed_role_catalog_for_prompt()

    assert catalog[0]["id"].startswith("role_")
    assert {"id", "name", "required_skills", "recommended_skills"} == set(catalog[0])
