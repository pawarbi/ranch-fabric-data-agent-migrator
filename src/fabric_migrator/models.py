from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Confidence = Literal["high", "medium", "manual"]
Status = Literal[
    "migrated",
    "migrated_with_warnings",
    "partial_migration",
    "no_migration_needed",
]


@dataclass(slots=True)
class Finding:
    rule_id: str
    title: str
    cell_index: int
    confidence: Confidence
    applied: bool
    message: str
    action: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    original_excerpt: str | None = None
    replacement_excerpt: str | None = None


@dataclass(slots=True)
class Score:
    overall: int
    automatic_coverage: int
    transformation_confidence: int
    preservation_confidence: int
    runtime_correctness: str = "Not assessed"
    label: str = "Migration confidence, not runtime correctness"


@dataclass(slots=True)
class MigrationReport:
    source_notebook: str
    tool_version: str
    status: Status
    score: Score
    summary: dict[str, int]
    changes: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    manual_actions: list[Finding] = field(default_factory=list)
    user_test_checklist: list[str] = field(default_factory=list)
    sdk_mapping: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MigrationResult:
    notebook_bytes: bytes
    report: MigrationReport
    output_filename: str
    report_filename: str
