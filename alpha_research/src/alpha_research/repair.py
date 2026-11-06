"""Repair playbook lookup (spec section 12).

This module only looks levers up. ``repair_agent.py`` applies them (mechanical rewrites, or
the model for levers without one). One lever per repair: ``next_lever`` never returns more
than one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from .config import ConfigError


@dataclass(frozen=True)
class Lever:
    name: str
    step: dict = field(default_factory=dict)


class Playbook:
    def __init__(self, entries: dict[str, list[Lever]]):
        self.entries = entries

    @classmethod
    def from_dict(cls, raw) -> "Playbook":
        if not isinstance(raw, dict):
            raise ConfigError("repair playbook root must be a mapping of failure class to levers")
        entries: dict[str, list[Lever]] = {}
        for failure_class, levers in raw.items():
            if not isinstance(levers, list) or not levers:
                raise ConfigError(f"playbook entry {failure_class!r} must be a non-empty list")
            parsed = []
            for i, item in enumerate(levers):
                if not isinstance(item, dict) or not isinstance(item.get("lever"), str):
                    raise ConfigError(f"playbook {failure_class}[{i}] needs a 'lever' name")
                step = item.get("step") or {}
                if not isinstance(step, dict):
                    raise ConfigError(f"playbook {failure_class}[{i}].step must be a mapping")
                parsed.append(Lever(item["lever"], dict(step)))
            entries[str(failure_class)] = parsed
        return cls(entries)

    @classmethod
    def load(cls, path: str | Path) -> "Playbook":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"playbook file not found: {path}")
        return cls.from_dict(yaml.safe_load(path.read_text()))

    def step_for(self, lever: str) -> dict | None:
        """The step of a lever's first listing in the playbook (any class), else None."""
        for levers in self.entries.values():
            for lv in levers:
                if lv.name == lever:
                    return dict(lv.step)
        return None

    def has_entry(self, failure_class: str) -> bool:
        return bool(self.entries.get(failure_class))

    def next_lever(self, failure_class: str, lineage_tried: Iterable[str]) -> Lever | None:
        """First lever for this class not already tried along the parent chain, else None."""
        tried = set(lineage_tried)
        for lever in self.entries.get(failure_class, []):
            if lever.name not in tried:
                return lever
        return None


def next_lever(playbook: Playbook, failure_class: str, lineage_tried: Iterable[str]) -> Lever | None:
    return playbook.next_lever(failure_class, lineage_tried)
