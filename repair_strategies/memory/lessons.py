"""Verified repair lessons: guidance that previously led to a verified recovery.

A lesson is written only after the verifier passed a repaired run. It is shown
again only for the same fault category and overlapping tools or agents, and a
lesson whose reuse fails loses score until it is no longer retrieved.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from .store import read_json, write_json_atomic


@dataclass
class Lesson:
    lesson_id: str
    category: str
    text: str
    strategy_id: str
    tools: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    verified: int = 1
    failed: int = 0

    @property
    def score(self) -> int:
        return self.verified - self.failed


def lesson_id(category: str, text: str) -> str:
    return hashlib.sha256(f'{category}\n{text.strip()}'.encode()).hexdigest()[:12]


class LessonMemory:
    def __init__(self, path=None):
        self.path = path
        self.lessons: dict[str, Lesson] = {}
        if path is not None:
            for item in read_json(path, {}).get('lessons', []):
                item['tools'], item['agents'] = tuple(item.get('tools', ())), tuple(item.get('agents', ()))
                self.lessons[item['lesson_id']] = Lesson(**item)

    def add_verified(self, category: str, text: str, strategy_id: str, tools=(), agents=()) -> Lesson:
        if not text.strip():
            raise ValueError('A lesson needs text')
        key = lesson_id(category, text)
        lesson = self.lessons.get(key)
        if lesson is None:
            lesson = Lesson(key, category, text.strip(), strategy_id, tuple(sorted(set(tools))),
                            tuple(sorted(set(agents))))
            self.lessons[key] = lesson
        else:
            lesson.verified += 1
            lesson.tools = tuple(sorted(set(lesson.tools) | set(tools)))
            lesson.agents = tuple(sorted(set(lesson.agents) | set(agents)))
        return lesson

    def record_success(self, key: str) -> None:
        if key in self.lessons:
            self.lessons[key].verified += 1

    def record_failure(self, key: str) -> None:
        if key in self.lessons:
            self.lessons[key].failed += 1

    def retrieve(self, category: str, tools=(), agents=(), k: int = 2) -> list[Lesson]:
        tools, agents = set(tools), set(agents)
        ranked = []
        for lesson in self.lessons.values():
            if lesson.category != category or lesson.score <= 0:
                continue
            overlap = len(tools & set(lesson.tools)) + len(agents & set(lesson.agents))
            if overlap or not (lesson.tools or lesson.agents):
                ranked.append((overlap, lesson.score, lesson.lesson_id, lesson))
        ranked.sort(key=lambda r: (-r[0], -r[1], r[2]))
        return [r[3] for r in ranked[:k]]

    def to_dict(self) -> dict:
        return {'lessons': [asdict(lesson) for _, lesson in sorted(self.lessons.items())]}

    def save(self, path=None) -> None:
        target = path or self.path
        if target is None:
            raise ValueError('No path to save the lesson memory to')
        write_json_atomic(target, self.to_dict())
