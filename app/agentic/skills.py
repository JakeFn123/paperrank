from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.config import BASE_DIR


@dataclass
class Skill:
    name: str
    description: str
    tags: list[str]
    body: str
    path: str


class SkillRegistry:
    """Load markdown skills from skills/<name>/SKILL.md with YAML-like frontmatter."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else BASE_DIR / "skills"
        self._skills: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        self._skills.clear()
        if not self.root.exists():
            return
        for file in sorted(self.root.rglob("SKILL.md")):
            text = file.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            name = (meta.get("name") or file.parent.name).strip()
            if not name:
                continue
            description = (meta.get("description") or "No description").strip()
            raw_tags = (meta.get("tags") or "").strip()
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            self._skills[name] = Skill(
                name=name,
                description=description,
                tags=tags,
                body=body.strip(),
                path=str(file),
            )

    def _parse_frontmatter(self, text: str) -> tuple[dict[str, str], str]:
        match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
        if not match:
            return {}, text

        meta: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
        return meta, match.group(2)

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def descriptions_text(self) -> str:
        if not self._skills:
            return "(no skills available)"
        lines = []
        for skill in self._skills.values():
            suffix = f" [{', '.join(skill.tags)}]" if skill.tags else ""
            lines.append(f"- {skill.name}: {skill.description}{suffix}")
        return "\n".join(lines)

    def render_skill(self, name: str) -> str:
        skill = self.get(name)
        if not skill:
            available = ", ".join(sorted(self._skills.keys())) or "none"
            return f"Error: unknown skill '{name}'. Available: {available}"
        return f"<skill name=\"{skill.name}\">\n{skill.body}\n</skill>"
