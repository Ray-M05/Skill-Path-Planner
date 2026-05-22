from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any

from ..models import Role, Skill


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SkillResolver:
    def __init__(self, skills: dict[str, Skill], fuzzy_cutoff: float = 0.88) -> None:
        self.skills = skills
        self.fuzzy_cutoff = fuzzy_cutoff
        self.alias_to_id = self._build_alias_index(skills)

    def _build_alias_index(self, skills: dict[str, Skill]) -> dict[str, str]:
        index: dict[str, str] = {}
        for skill in skills.values():
            keys = [skill.id, skill.name, *skill.aliases]
            for key in keys:
                normalized = normalize_text(key)
                if normalized:
                    index.setdefault(normalized, skill.id)
        return index

    def resolve_skill_text(self, text: str) -> str | None:
        if text in self.skills:
            return text

        normalized = normalize_text(text)
        if normalized in self.alias_to_id:
            return self.alias_to_id[normalized]

        matches = difflib.get_close_matches(
            normalized,
            self.alias_to_id.keys(),
            n=1,
            cutoff=self.fuzzy_cutoff,
        )
        if not matches:
            return None
        return self.alias_to_id[matches[0]]

    def get_allowed_skill_catalog_for_prompt(self) -> list[dict[str, Any]]:
        return [
            {
                "id": skill.id,
                "name": skill.name,
                "aliases": list(skill.aliases),
                "category": skill.category,
            }
            for skill in sorted(self.skills.values(), key=lambda item: item.id)
        ]


class RoleResolver:
    def __init__(self, roles: dict[str, Role], fuzzy_cutoff: float = 0.88) -> None:
        self.roles = roles
        self.fuzzy_cutoff = fuzzy_cutoff
        self.name_to_id = self._build_name_index(roles)

    def _build_name_index(self, roles: dict[str, Role]) -> dict[str, str]:
        index: dict[str, str] = {}
        for role in roles.values():
            for key in (role.id, role.name):
                normalized = normalize_text(key)
                if normalized:
                    index.setdefault(normalized, role.id)
        return index

    def resolve_role_text(self, text: str) -> str | None:
        if text in self.roles:
            return text

        normalized = normalize_text(text)
        if normalized in self.name_to_id:
            return self.name_to_id[normalized]

        matches = difflib.get_close_matches(
            normalized,
            self.name_to_id.keys(),
            n=1,
            cutoff=self.fuzzy_cutoff,
        )
        if not matches:
            return None
        return self.name_to_id[matches[0]]

    def get_allowed_role_catalog_for_prompt(self) -> list[dict[str, Any]]:
        return [
            {
                "id": role.id,
                "name": role.name,
                "required_skills": sorted(role.required_skills),
                "recommended_skills": sorted(role.recommended_skills),
            }
            for role in sorted(self.roles.values(), key=lambda item: item.id)
        ]
