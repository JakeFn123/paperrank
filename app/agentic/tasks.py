from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import BASE_DIR


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskItem:
    id: int
    title: str
    assignee: str
    status: str = "pending"
    depends_on: list[int] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    error: str = ""
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)


class TaskBoard:
    """Persistent task system inspired by learn-claude-code s07.

    Tasks are stored as JSON files under `.paperrank/tasks/` so workflow state
    can survive long runs and context resets.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else BASE_DIR / ".paperrank" / "tasks"
        self.root.mkdir(parents=True, exist_ok=True)
        self._next_id = self._max_id() + 1

    def _task_path(self, task_id: int) -> Path:
        return self.root / f"task_{task_id}.json"

    def _max_id(self) -> int:
        ids = []
        for file in self.root.glob("task_*.json"):
            try:
                ids.append(int(file.stem.split("_")[1]))
            except Exception:
                continue
        return max(ids) if ids else 0

    def clear(self) -> None:
        for file in self.root.glob("task_*.json"):
            file.unlink(missing_ok=True)
        self._next_id = 1

    def create(self, title: str, assignee: str, depends_on: list[int] | None = None, payload: dict | None = None) -> TaskItem:
        item = TaskItem(
            id=self._next_id,
            title=title,
            assignee=assignee,
            depends_on=list(depends_on or []),
            payload=dict(payload or {}),
        )
        self._save(item)
        self._next_id += 1
        return item

    def _save(self, item: TaskItem) -> None:
        item.updated_at = _now_iso()
        self._task_path(item.id).write_text(json.dumps(asdict(item), ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, task_id: int) -> TaskItem:
        path = self._task_path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        data = json.loads(path.read_text(encoding="utf-8"))
        return TaskItem(**data)

    def update(
        self,
        task_id: int,
        *,
        status: str | None = None,
        result_summary: str | None = None,
        error: str | None = None,
    ) -> TaskItem:
        item = self.get(task_id)
        if status is not None:
            if status not in {"pending", "in_progress", "completed", "failed"}:
                raise ValueError(f"Invalid task status: {status}")
            item.status = status
        if result_summary is not None:
            item.result_summary = result_summary
        if error is not None:
            item.error = error
        self._save(item)
        return item

    def list(self) -> list[TaskItem]:
        items: list[TaskItem] = []
        for file in sorted(self.root.glob("task_*.json"), key=lambda p: p.name):
            data = json.loads(file.read_text(encoding="utf-8"))
            items.append(TaskItem(**data))
        return items

    def snapshot(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.list()]

    def all_dependencies_completed(self, task_id: int) -> bool:
        item = self.get(task_id)
        for dep_id in item.depends_on:
            dep = self.get(dep_id)
            if dep.status != "completed":
                return False
        return True
