from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator as _ContractValidator
except Exception:
    from jsonschema import Draft7Validator as _ContractValidator

from app.config import BASE_DIR


@dataclass
class Skill:
    name: str
    description: str
    tags: list[str]
    body: str
    path: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


class SkillContractError(ValueError):
    pass


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
            input_schema, output_schema = self._load_contract_schemas(file.parent)
            self._skills[name] = Skill(
                name=name,
                description=description,
                tags=tags,
                body=body.strip(),
                path=str(file),
                input_schema=input_schema,
                output_schema=output_schema,
            )

    def _load_contract_schemas(self, skill_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        contract_file = skill_dir / "CONTRACT.json"
        if not contract_file.exists():
            return None, None
        try:
            payload = json.loads(contract_file.read_text(encoding="utf-8"))
        except Exception:
            return None, None
        if not isinstance(payload, dict):
            return None, None
        input_schema = payload.get("input_schema")
        output_schema = payload.get("output_schema")
        if not isinstance(input_schema, dict):
            input_schema = None
        if not isinstance(output_schema, dict):
            output_schema = None
        return input_schema, output_schema

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

    def validate_input(self, name: str, payload: dict[str, Any]) -> None:
        self._validate(name=name, payload=payload, phase="input")

    def validate_output(self, name: str, payload: dict[str, Any]) -> None:
        self._validate(name=name, payload=payload, phase="output")

    def _validate(self, name: str, payload: dict[str, Any], phase: str) -> None:
        skill = self.get(name)
        if not skill:
            return
        schema = skill.input_schema if phase == "input" else skill.output_schema
        if not schema:
            return
        validator = _ContractValidator(schema)
        errors = list(validator.iter_errors(payload))
        if not errors:
            return
        lines = []
        for err in errors[:6]:
            path = "/".join([str(p) for p in err.absolute_path]) or "<root>"
            lines.append(f"{path}: {err.message}")
        raise SkillContractError(f"Skill '{name}' {phase} contract validation failed: " + "; ".join(lines))
