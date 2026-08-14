from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from nbformat import NotebookNode

from . import __version__
from .mapping import SDK_MAPPING
from .models import Finding, MigrationReport, MigrationResult, Score
from .notebook_io import NotebookLimits, load_notebook, write_notebook

OLD_SUFFIXES = {
    ".beta.assistants.create",
    ".beta.threads.create",
    ".beta.threads.messages.create",
    ".beta.threads.runs.create",
    ".beta.threads.runs.retrieve",
    ".beta.threads.messages.list",
    ".beta.threads.runs.steps.list",
    ".beta.threads.delete",
}
TERMINAL_OUTPUT_TYPES = (
    '"function_call", "function_call_output", "code_interpreter_call"'
)
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret|password)"
    r"\s*[:=]\s*([\"'])(.*?)\2"
)
DYNAMIC_PATTERN = re.compile(r"\b(getattr|setattr|exec|eval)\s*\(")
PERSISTED_THREAD_PATTERN = re.compile(
    r"(?is)(?:thread[_-]?id.{0,160}(?:json\.dump|json\.dumps|"
    r"pickle\.dump|pickle\.dumps|open\s*\(|write_text\s*\(|os\.environ)|"
    r"(?:json\.dump|json\.dumps|pickle\.dump|pickle\.dumps|open\s*\(|"
    r"write_text\s*\(|os\.environ).{0,160}thread[_-]?id)"
)


@dataclass(slots=True)
class Patch:
    start: int
    end: int
    replacement: str


@dataclass(slots=True)
class ParsedCell:
    index: int
    source: str
    tree: ast.Module | None
    lines: list[str]
    line_offsets: list[int]
    parents: dict[ast.AST, ast.AST] = field(default_factory=dict)
    parse_error: SyntaxError | None = None
    patches: list[Patch] = field(default_factory=list)

    def span(self, node: ast.AST) -> tuple[int, int]:
        start_line = self.lines[node.lineno - 1]
        end_line = self.lines[node.end_lineno - 1]
        start_col = _char_col(start_line, node.col_offset)
        end_col = _char_col(end_line, node.end_col_offset)
        return (
            self.line_offsets[node.lineno - 1] + start_col,
            self.line_offsets[node.end_lineno - 1] + end_col,
        )

    def text(self, node: ast.AST) -> str:
        start, end = self.span(node)
        return self.source[start:end]


@dataclass(slots=True)
class FlowCall:
    cell: ParsedCell
    statement: ast.stmt
    call: ast.Call
    client: str
    thread_expr: str | None = None
    target: str | None = None
    content: str | None = None
    assistant_expr: str | None = None

    @property
    def order(self) -> tuple[int, int]:
        return (self.cell.index, self.statement.lineno)


def _char_col(line: str, byte_col: int) -> int:
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))


def _line_offsets(source: str) -> tuple[list[str], list[int]]:
    lines = source.splitlines(keepends=True) or [""]
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    return lines, offsets


def _sanitize_magics(source: str) -> tuple[str, bool]:
    output: list[str] = []
    has_cell_magic = False
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("%%"):
            has_cell_magic = True
        if stripped.startswith(("%", "!")):
            leading = len(line) - len(stripped)
            line = line[:leading] + "#" + line[leading + 1 :]
        output.append(line)
    return "".join(output), has_cell_magic


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _simple_target(statement: ast.stmt) -> str | None:
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return statement.targets[0].id
    return None


def _redact(text: str | None, limit: int = 260) -> str | None:
    if text is None:
        return None
    redacted = SECRET_PATTERN.sub(lambda match: f'{match.group(1)}="<redacted>"', text)
    redacted = re.sub(
        r"(?i)(authorization\s*[:=]\s*[\"']?bearer\s+)[^\s\"']+",
        r"\1<redacted>",
        redacted,
    )
    return redacted if len(redacted) <= limit else redacted[: limit - 1] + "…"


def _comment_for(source: str, message: str) -> str:
    indent = source[: len(source) - len(source.lstrip())]
    return f"{indent}# Migrated: {message}"


class MigrationEngine:
    def __init__(self, limits: NotebookLimits | None = None) -> None:
        self.limits = limits or NotebookLimits()

    def migrate(self, data: bytes, filename: str = "notebook.ipynb") -> MigrationResult:
        notebook = load_notebook(data, self.limits)
        migrated = copy.deepcopy(notebook)
        parsed = self._parse_cells(notebook)
        findings: list[Finding] = []

        relevant_cells = {
            cell.index
            for cell in parsed
            if "FabricOpenAI" in cell.source or ".beta." in cell.source
        }
        blockers = self._find_blockers(parsed, relevant_cells, findings)
        self._find_runtime_diagnostics(notebook, parsed, findings)
        inventory = self._inventory(parsed)
        blockers.extend(self._validate_inventory(inventory, parsed, findings))

        query_changes = 0
        if inventory["has_old_query"] and not blockers:
            query_changes = self._apply_query_migration(
                parsed, inventory, findings, notebook
            )
        elif inventory["has_old_query"] and blockers:
            findings.append(
                Finding(
                    rule_id="SAFE-ATOMIC-001",
                    title="Old query flow preserved",
                    cell_index=min(relevant_cells or {0}),
                    confidence="manual",
                    applied=False,
                    message=(
                        "No query-plane edits were applied because at least one "
                        "unresolved pattern could make a partial rewrite unsafe."
                    ),
                    action="Resolve the manual findings, then run the migrator again.",
                )
            )

        evaluation_changes = self._apply_evaluation_rule(parsed, findings)
        sdk_changes = (
            self._upgrade_install_cell(parsed, findings)
            if query_changes or evaluation_changes
            else 0
        )
        changed_cells = self._commit_patches(parsed, migrated)
        total_changes = query_changes + evaluation_changes + sdk_changes
        output_errors = self._validate_generated_output(migrated, changed_cells)
        if output_errors:
            for finding in findings:
                if finding.applied:
                    finding.applied = False
            findings.extend(output_errors)
            migrated = copy.deepcopy(notebook)
            changed_cells = set()
            total_changes = 0

        manual = [finding for finding in findings if finding.confidence == "manual"]
        warnings = [
            finding for finding in findings if finding.confidence == "medium"
        ]
        changes = [finding for finding in findings if finding.applied]
        preservation = self._preservation_score(notebook, migrated, changed_cells)
        status = self._status(
            inventory["has_old_query"] or evaluation_changes > 0,
            total_changes,
            manual,
            warnings,
        )
        score = self._score(findings, preservation)
        report = MigrationReport(
            source_notebook=Path(filename).name,
            tool_version=__version__,
            status=status,
            score=score,
            summary={
                "cells_scanned": len(notebook.cells),
                "cells_changed": len(changed_cells),
                "rules_applied": len(changes),
                "warnings": len(warnings),
                "manual_actions": len(manual),
            },
            changes=changes,
            warnings=warnings,
            manual_actions=manual,
            user_test_checklist=self._test_checklist(inventory, manual, warnings),
            sdk_mapping=SDK_MAPPING,
        )
        stem = Path(filename).stem or "notebook"
        return MigrationResult(
            notebook_bytes=write_notebook(migrated) if changed_cells else data,
            report=report,
            output_filename=f"{stem}-responses-api.ipynb",
            report_filename=f"{stem}-migration-report.json",
        )

    def _parse_cells(self, notebook: NotebookNode) -> list[ParsedCell]:
        parsed: list[ParsedCell] = []
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            source = cell.source
            lines, offsets = _line_offsets(source)
            sanitized, has_cell_magic = _sanitize_magics(source)
            if has_cell_magic:
                parsed.append(
                    ParsedCell(
                        index,
                        source,
                        None,
                        lines,
                        offsets,
                        parse_error=SyntaxError("Cell magic requires manual review"),
                    )
                )
                continue
            try:
                tree = ast.parse(sanitized)
            except SyntaxError as exc:
                parsed.append(
                    ParsedCell(index, source, None, lines, offsets, parse_error=exc)
                )
                continue
            parents = {
                child: parent
                for parent in ast.walk(tree)
                for child in ast.iter_child_nodes(parent)
            }
            parsed.append(ParsedCell(index, source, tree, lines, offsets, parents))
        return parsed

    def _find_blockers(
        self,
        cells: list[ParsedCell],
        relevant_cells: set[int],
        findings: list[Finding],
    ) -> list[str]:
        blockers: list[str] = []
        for cell in cells:
            if cell.index not in relevant_cells:
                continue
            if cell.parse_error:
                blockers.append(f"cell-{cell.index}-syntax")
                findings.append(
                    Finding(
                        "UNSUPPORTED-SYNTAX-001",
                        "Relevant cell is not ordinary Python",
                        cell.index,
                        "manual",
                        False,
                        str(cell.parse_error),
                        "Move magics to a separate cell or fix the syntax, then retry.",
                    )
                )
            if DYNAMIC_PATTERN.search(cell.source):
                blockers.append(f"cell-{cell.index}-dynamic")
                findings.append(
                    Finding(
                        "UNSUPPORTED-DYNAMIC-001",
                        "Dynamic Python access detected",
                        cell.index,
                        "manual",
                        False,
                        "getattr, setattr, exec, or eval can hide API behavior.",
                        "Migrate the dynamic call manually.",
                    )
                )
            if re.search(r"\b(async\s+def|await|async\s+for)\b", cell.source):
                blockers.append(f"cell-{cell.index}-async")
                findings.append(
                    Finding(
                        "UNSUPPORTED-ASYNC-001",
                        "Asynchronous orchestration detected",
                        cell.index,
                        "manual",
                        False,
                        "The deterministic MVP does not rewrite asynchronous run flows.",
                        "Map this flow to Responses async/streaming behavior manually.",
                    )
                )
            if PERSISTED_THREAD_PATTERN.search(cell.source):
                blockers.append(f"cell-{cell.index}-persisted-thread")
                findings.append(
                    Finding(
                        "UNSUPPORTED-PERSISTED-001",
                        "Persisted thread state detected",
                        cell.index,
                        "manual",
                        False,
                        "Persisted thread IDs require an explicit state migration decision.",
                        "Choose conversation IDs or previous response IDs and migrate storage.",
                    )
                )
            for match in SECRET_PATTERN.finditer(cell.source):
                findings.append(
                    Finding(
                        "WARNING-SECRET-001",
                        "Possible secret in notebook",
                        cell.index,
                        "medium",
                        False,
                        "A credential-like assignment was detected; its value is omitted.",
                        "Replace secrets with a managed credential before sharing the notebook.",
                        line_start=cell.source[: match.start()].count("\n") + 1,
                    )
                )
        return blockers

    def _find_runtime_diagnostics(
        self,
        notebook: NotebookNode,
        cells: list[ParsedCell],
        findings: list[Finding],
    ) -> None:
        install_cells = [
            cell.index
            for cell in cells
            if re.search(
                r"(?im)^\s*%pip\s+install\b[^\n]*fabric-data-agent-sdk\b",
                cell.source,
            )
        ]
        evaluation_imports = [
            cell.index
            for cell in cells
            if re.search(
                r"(?m)^\s*from\s+fabric\.dataagent\.evaluation\s+import\b",
                cell.source,
            )
        ]
        traceback_match = False
        traceback_cell = evaluation_imports[0] if evaluation_imports else 0
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            output_text = self._output_text(cell.get("outputs", []))
            if re.search(
                r"ImportError:[^\n]*cannot import name ['\"]FabricOpenAI['\"]"
                r"[^\n]*fabric\.dataagent\.client",
                output_text,
                re.IGNORECASE,
            ):
                traceback_match = True
                traceback_cell = index
                break
        if not traceback_match and not (install_cells and evaluation_imports):
            return
        if traceback_match:
            message = (
                "The saved output contains the FabricOpenAI import failure shown "
                "in the reported scenario. Published SDK 0.1.27a0 and 0.1.28a0 "
                "still export FabricOpenAI, so a stale or mixed interpreter is "
                "more likely than an intentional API removal."
            )
        else:
            message = (
                "This notebook installs the SDK and imports evaluation in the same "
                "Python session. Fabric Python %pip does not automatically restart "
                "the kernel, so an older loaded client module can cause the "
                "FabricOpenAI ImportError."
            )
        findings.append(
            Finding(
                "RUNTIME-SDK-RESTART-001",
                "Restart Python after upgrading the SDK",
                traceback_cell,
                "medium",
                False,
                message,
                (
                    "After the install cell, run "
                    "notebookutils.session.restartPython(), then execute the import "
                    "in the next cell. In the fresh session, verify the installed "
                    "SDK version is at least 0.1.28a0."
                ),
            )
        )

    def _output_text(self, outputs: Iterable[object]) -> str:
        parts: list[str] = []
        for output in outputs:
            if not isinstance(output, dict):
                continue
            text = output.get("text")
            if isinstance(text, list):
                parts.extend(str(item) for item in text)
            elif text is not None:
                parts.append(str(text))
            traceback = output.get("traceback")
            if isinstance(traceback, list):
                parts.extend(str(item) for item in traceback)
            elif traceback is not None:
                parts.append(str(traceback))
            error_name = output.get("ename")
            error_value = output.get("evalue")
            if error_name or error_value:
                parts.append(f"{error_name or ''}: {error_value or ''}")
        return "\n".join(parts)

    def _inventory(self, cells: list[ParsedCell]) -> dict[str, object]:
        old_import_locals: dict[str, tuple[ParsedCell, ast.alias]] = {}
        response_imported = False
        calls: list[tuple[ParsedCell, ast.stmt, ast.Call, str]] = []
        definitions: set[str] = set()
        functions: dict[str, tuple[ParsedCell, ast.FunctionDef]] = {}
        for cell in cells:
            if not cell.tree:
                continue
            for node in ast.walk(cell.tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        node.targets
                        if isinstance(node, ast.Assign)
                        else [node.target]
                    )
                    definitions.update(
                        target.id for target in targets if isinstance(target, ast.Name)
                    )
                elif isinstance(node, ast.FunctionDef):
                    functions[node.name] = (cell, node)
                elif isinstance(node, ast.ImportFrom) and node.module == "fabric.dataagent.client":
                    for alias in node.names:
                        if alias.name == "FabricOpenAI":
                            old_import_locals[alias.asname or alias.name] = (cell, alias)
                        if alias.name == "FabricOpenAIResponses":
                            response_imported = True
            for statement in cell.tree.body:
                for node in ast.walk(statement):
                    if isinstance(node, ast.Call):
                        path = _dotted_name(node.func)
                        if path:
                            calls.append((cell, statement, node, path))
        return {
            "old_import_locals": old_import_locals,
            "response_imported": response_imported,
            "calls": calls,
            "definitions": definitions,
            "functions": functions,
            "has_old_query": bool(old_import_locals)
            or any(".beta." in path for _, _, _, path in calls),
        }

    def _validate_inventory(
        self,
        inventory: dict[str, object],
        cells: list[ParsedCell],
        findings: list[Finding],
    ) -> list[str]:
        blockers: list[str] = []
        calls = inventory["calls"]
        old_import_locals = inventory["old_import_locals"]
        assert isinstance(calls, list)
        assert isinstance(old_import_locals, dict)
        if any(".beta." in path for _, _, _, path in calls) and not old_import_locals:
            blockers.append("unresolved-client-import")
            findings.append(
                Finding(
                    "UNSUPPORTED-CLIENT-001",
                    "Fabric client import cannot be resolved",
                    min(
                        cell.index
                        for cell, _, _, path in calls
                        if ".beta." in path
                    ),
                    "manual",
                    False,
                    "The old query client is not imported with a supported direct import.",
                    (
                        "Use 'from fabric.dataagent.client import FabricOpenAI' "
                        "or migrate the client manually."
                    ),
                )
            )
        wrapper_cells: set[int] = set()
        for cell, statement, call, path in calls:
            if ".beta." in path and not any(path.endswith(suffix) for suffix in OLD_SUFFIXES):
                blockers.append(path)
                findings.append(
                    Finding(
                        "UNSUPPORTED-BETA-001",
                        "Unknown Assistants API call",
                        cell.index,
                        "manual",
                        False,
                        f"The call shape {path} is not in the verified mapping.",
                        "Migrate this call manually or add a tested mapping rule.",
                        line_start=call.lineno,
                        line_end=call.end_lineno,
                        original_excerpt=_redact(cell.text(statement)),
                    )
                )
            if ".beta." in path and isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                blockers.append(f"wrapper-{cell.index}")
                if cell.index not in wrapper_cells:
                    wrapper_cells.add(cell.index)
                    findings.append(
                        Finding(
                            "UNSUPPORTED-WRAPPER-001",
                            "Custom API wrapper detected",
                            cell.index,
                            "manual",
                            False,
                            (
                                "Assistants calls inside a function can carry custom "
                                "state or return-value contracts."
                            ),
                            "Migrate and test this wrapper explicitly.",
                            line_start=statement.lineno,
                            line_end=statement.end_lineno,
                        )
                    )
        blockers.extend(self._validate_call_shapes(calls, findings))
        for cell, statement, call, path in calls:
            if path not in old_import_locals:
                continue
            allowed = {"artifact_name", "workspace_name", "ai_skill_stage"}
            if call.args or any(
                keyword.arg not in allowed for keyword in call.keywords
            ):
                blockers.append(f"constructor-{cell.index}-{call.lineno}")
                findings.append(
                    Finding(
                        "UNSUPPORTED-CONSTRUCTOR-001",
                        "Client constructor has unknown options",
                        cell.index,
                        "manual",
                        False,
                        (
                            "Only artifact_name, workspace_name, and ai_skill_stage "
                            "have a verified constructor mapping."
                        ),
                        "Confirm every custom client option against the Responses SDK.",
                        line_start=call.lineno,
                        line_end=call.end_lineno,
                        original_excerpt=_redact(cell.text(statement)),
                    )
                )
        blockers.extend(self._validate_symbol_references(cells, calls, inventory, findings))
        return blockers

    def _validate_call_shapes(
        self,
        calls: list[tuple[ParsedCell, ast.stmt, ast.Call, str]],
        findings: list[Finding],
    ) -> list[str]:
        blockers: list[str] = []
        for cell, statement, call, path in calls:
            reason: str | None = None
            if path.endswith(".beta.assistants.create"):
                if call.args or any(keyword.arg != "model" for keyword in call.keywords):
                    reason = "Assistant setup contains instructions, tools, or unknown options."
            elif path.endswith(".beta.threads.create"):
                if call.args or call.keywords:
                    reason = "Thread creation contains initial state or unknown options."
            elif path.endswith(".beta.threads.messages.create"):
                if call.args or any(
                    keyword.arg not in {"thread_id", "role", "content"}
                    for keyword in call.keywords
                ):
                    reason = "Message creation contains attachments or unknown options."
            elif path.endswith(".beta.threads.runs.create"):
                if call.args or any(
                    keyword.arg not in {"thread_id", "assistant_id"}
                    for keyword in call.keywords
                ):
                    reason = "Run creation contains instructions, tools, or unknown options."
            elif path.endswith(".beta.threads.runs.retrieve"):
                if call.args or {
                    keyword.arg for keyword in call.keywords
                } != {"thread_id", "run_id"}:
                    reason = "Run retrieval is not the verified thread_id/run_id shape."
                enclosing_while = self._ancestor(cell, call, ast.While)
                run_id = _keyword(call, "run_id")
                run_name = (
                    run_id.value.id
                    if isinstance(run_id, ast.Attribute)
                    and run_id.attr == "id"
                    and isinstance(run_id.value, ast.Name)
                    else None
                )
                if enclosing_while and not self._is_running_status_test(
                    enclosing_while.test, run_name
                ):
                    reason = "Polling condition is not limited to queued/in_progress."
            elif path.endswith(".beta.threads.messages.list"):
                allowed = {"thread_id", "order"}
                if call.args or any(
                    keyword.arg not in allowed for keyword in call.keywords
                ):
                    reason = "Message listing uses pagination or filtering semantics."
                order = _keyword(call, "order")
                if (
                    order is not None
                    and (not isinstance(order, ast.Constant) or order.value != "asc")
                ):
                    reason = "Message ordering is not the verified ascending shape."
                if not _simple_target(statement) and not (
                    isinstance(statement, ast.For)
                    and self._is_inline_message_loop(cell, statement)
                ):
                    reason = (
                        "Message listing must be assigned to a simple name or used "
                        "in the verified inline print loop."
                    )
            elif path.endswith(".beta.threads.runs.steps.list"):
                if call.args or {
                    keyword.arg for keyword in call.keywords
                } != {"thread_id", "run_id"}:
                    reason = "Run-step listing uses an unknown shape."
                if not _simple_target(statement):
                    reason = "Run-step listing must be assigned to a simple name."
            if reason:
                blockers.append(f"shape-{cell.index}-{call.lineno}")
                findings.append(
                    Finding(
                        "UNSUPPORTED-SHAPE-001",
                        "Recognized API has unsupported options",
                        cell.index,
                        "manual",
                        False,
                        reason,
                        "Preserve these semantics in a manual Responses implementation.",
                        line_start=call.lineno,
                        line_end=call.end_lineno,
                        original_excerpt=_redact(cell.text(statement)),
                    )
                )
        return blockers

    def _is_running_status_test(
        self, node: ast.AST, run_name: str | None
    ) -> bool:
        if not run_name:
            return False
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            return all(
                self._is_running_status_test(value, run_name)
                for value in node.values
            )
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            return False
        left = node.left
        if not (
            isinstance(left, ast.Attribute)
            and left.attr == "status"
            and isinstance(left.value, ast.Name)
            and left.value.id == run_name
        ):
            return False
        comparator = node.comparators[0]
        if isinstance(node.ops[0], ast.Eq) and isinstance(comparator, ast.Constant):
            return comparator.value in {"queued", "in_progress"}
        if isinstance(node.ops[0], ast.In) and isinstance(
            comparator, (ast.Tuple, ast.List, ast.Set)
        ):
            values = {
                item.value
                for item in comparator.elts
                if isinstance(item, ast.Constant)
            }
            return bool(values) and values <= {"queued", "in_progress"}
        return False

    def _validate_symbol_references(
        self,
        cells: list[ParsedCell],
        calls: list[tuple[ParsedCell, ast.stmt, ast.Call, str]],
        inventory: dict[str, object],
        findings: list[Finding],
    ) -> list[str]:
        blockers: list[str] = []
        setup_names: set[str] = set()
        run_names: set[str] = set()
        message_list_names: set[str] = set()
        step_list_names: set[str] = set()
        helper_names: set[str] = set()
        producer_counts: dict[tuple[str, str], int] = {}
        functions = inventory["functions"]
        assert isinstance(functions, dict)
        for name, (cell, function) in functions.items():
            if self._is_plain_text_helper(cell, function):
                helper_names.add(name)
        for _, statement, _, path in calls:
            target = _simple_target(statement)
            if path.endswith((".beta.assistants.create", ".beta.threads.create")):
                if target:
                    setup_names.add(target)
                    producer_counts[("setup", target)] = (
                        producer_counts.get(("setup", target), 0) + 1
                    )
            elif path.endswith(".beta.threads.runs.create") and target:
                run_names.add(target)
                producer_counts[("run", target)] = (
                    producer_counts.get(("run", target), 0) + 1
                )
            elif path.endswith(".beta.threads.messages.list") and target:
                message_list_names.add(target)
                producer_counts[("messages", target)] = (
                    producer_counts.get(("messages", target), 0) + 1
                )
            elif path.endswith(".beta.threads.runs.steps.list") and target:
                step_list_names.add(target)
                producer_counts[("steps", target)] = (
                    producer_counts.get(("steps", target), 0) + 1
                )

        for (category, name), count in producer_counts.items():
            if count <= 1:
                continue
            blockers.append(f"reassignment-{category}-{name}")
            findings.append(
                Finding(
                    "UNSUPPORTED-REASSIGNMENT-001",
                    "API symbol is assigned more than once",
                    0,
                    "manual",
                    False,
                    (
                        f"Variable '{name}' is produced by {count} {category} "
                        "calls, so notebook-order pairing is ambiguous."
                    ),
                    "Use a unique variable for each API object, then retry.",
                )
            )

        reported: set[tuple[int, int, str]] = set()
        for cell in cells:
            if not cell.tree:
                continue
            for node in ast.walk(cell.tree):
                if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                    continue
                category: str | None = None
                if node.id in setup_names:
                    category = "setup"
                elif node.id in run_names:
                    category = "run"
                elif node.id in message_list_names:
                    category = "messages"
                elif node.id in step_list_names:
                    category = "steps"
                if not category:
                    continue
                call = self._ancestor(cell, node, ast.Call)
                call_path = _dotted_name(call.func) if isinstance(call, ast.Call) else None
                allowed = False
                if category == "setup":
                    allowed = bool(
                        call_path == "print"
                        or (
                            call_path
                            and any(call_path.endswith(suffix) for suffix in OLD_SUFFIXES)
                        )
                    )
                elif category == "run":
                    allowed = bool(
                        call_path == "print"
                        or (
                            call_path
                            and call_path.endswith(
                                (
                                    ".beta.threads.runs.retrieve",
                                    ".beta.threads.runs.steps.list",
                                )
                            )
                        )
                        or self._ancestor(cell, node, ast.While) is not None
                    )
                elif category == "messages":
                    enclosing_for = self._ancestor(cell, node, ast.For)
                    allowed = bool(
                        call_path in helper_names
                        or (
                            isinstance(enclosing_for, ast.For)
                            and enclosing_for.iter is node
                            and self._is_inline_message_loop(cell, enclosing_for)
                        )
                    )
                elif category == "steps":
                    parent = cell.parents.get(node)
                    grandparent = cell.parents.get(parent) if parent else None
                    allowed = bool(
                        isinstance(parent, ast.Attribute)
                        and parent.attr == "data"
                        and isinstance(grandparent, ast.Expr)
                    )
                if allowed:
                    continue
                key = (cell.index, node.lineno, node.id)
                if key in reported:
                    continue
                reported.add(key)
                blockers.append(f"reference-{cell.index}-{node.lineno}-{node.id}")
                findings.append(
                    Finding(
                        "UNSUPPORTED-REFERENCE-001",
                        "API object has an unknown dependency",
                        cell.index,
                        "manual",
                        False,
                        (
                            f"Variable '{node.id}' is used outside the verified "
                            f"{category} migration pattern."
                        ),
                        "Migrate this dependent code explicitly before retrying.",
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                    )
                )
        return blockers

    def _is_inline_message_loop(self, cell: ParsedCell, loop: ast.For) -> bool:
        if len(loop.body) != 1 or loop.orelse:
            return False
        body = loop.body[0]
        if not (
            isinstance(loop.target, ast.Name)
            and isinstance(body, ast.Expr)
            and isinstance(body.value, ast.Call)
            and _dotted_name(body.value.func) == "print"
        ):
            return False
        text = cell.text(body)
        return (
            f"{loop.target.id}.content[0].text.value" in text
            and f"{loop.target.id}.role" in text
        )

    def _is_plain_text_helper(
        self, cell: ParsedCell, function: ast.FunctionDef
    ) -> bool:
        if not function.args.args:
            return False
        arg_name = function.args.args[0].arg
        loops = [statement for statement in function.body if isinstance(statement, ast.For)]
        if len(loops) != 1:
            return False
        loop = loops[0]
        if not isinstance(loop.iter, ast.Name) or loop.iter.id != arg_name:
            return False
        if len(loop.body) != 1 or not isinstance(loop.body[0], ast.Expr):
            return False
        if not (
            isinstance(loop.body[0].value, ast.Call)
            and _dotted_name(loop.body[0].value.func) == "print"
        ):
            return False
        loop_text = cell.text(loop.body[0])
        if ".content[0].text.value" not in loop_text:
            return False
        return all(
            isinstance(statement, (ast.Expr, ast.For, ast.Pass))
            for statement in function.body
        )

    def _ancestor(
        self, cell: ParsedCell, node: ast.AST, kind: type[ast.AST]
    ) -> ast.AST | None:
        current = node
        while current in cell.parents:
            current = cell.parents[current]
            if isinstance(current, kind):
                return current
            if isinstance(current, ast.stmt) and kind is ast.Call:
                return None
        return None

    def _apply_query_migration(
        self,
        cells: list[ParsedCell],
        inventory: dict[str, object],
        findings: list[Finding],
        notebook: NotebookNode,
    ) -> int:
        calls = inventory["calls"]
        old_import_locals = inventory["old_import_locals"]
        definitions = inventory["definitions"]
        functions = inventory["functions"]
        assert isinstance(calls, list)
        assert isinstance(old_import_locals, dict)
        assert isinstance(definitions, set)
        assert isinstance(functions, dict)

        messages: list[FlowCall] = []
        runs: list[FlowCall] = []
        assistant_setups: list[FlowCall] = []
        thread_setups: list[FlowCall] = []
        list_calls: list[FlowCall] = []
        step_calls: list[FlowCall] = []
        retrieve_calls: list[FlowCall] = []
        delete_calls: list[FlowCall] = []
        constructor_calls: list[tuple[ParsedCell, ast.stmt, ast.Call, str]] = []

        for cell, statement, call, path in calls:
            if path in old_import_locals:
                constructor_calls.append((cell, statement, call, path))
            elif path.endswith(".beta.assistants.create"):
                assistant_setups.append(
                    FlowCall(cell, statement, call, path.split(".beta.")[0], target=_simple_target(statement))
                )
            elif path.endswith(".beta.threads.create"):
                thread_setups.append(
                    FlowCall(cell, statement, call, path.split(".beta.")[0], target=_simple_target(statement))
                )
            elif path.endswith(".beta.threads.messages.create"):
                messages.append(
                    FlowCall(
                        cell,
                        statement,
                        call,
                        path.split(".beta.")[0],
                        thread_expr=self._arg_text(cell, call, "thread_id"),
                        content=self._arg_text(cell, call, "content"),
                    )
                )
            elif path.endswith(".beta.threads.runs.create"):
                runs.append(
                    FlowCall(
                        cell,
                        statement,
                        call,
                        path.split(".beta.")[0],
                        thread_expr=self._arg_text(cell, call, "thread_id"),
                        assistant_expr=self._arg_text(cell, call, "assistant_id"),
                        target=_simple_target(statement),
                    )
                )
            elif path.endswith(".beta.threads.runs.retrieve"):
                retrieve_calls.append(
                    FlowCall(
                        cell,
                        statement,
                        call,
                        path.split(".beta.")[0],
                        thread_expr=self._arg_text(cell, call, "thread_id"),
                        target=self._arg_text(cell, call, "run_id"),
                    )
                )
            elif path.endswith(".beta.threads.messages.list"):
                list_calls.append(
                    FlowCall(
                        cell,
                        statement,
                        call,
                        path.split(".beta.")[0],
                        thread_expr=self._arg_text(cell, call, "thread_id"),
                        target=_simple_target(statement),
                    )
                )
            elif path.endswith(".beta.threads.runs.steps.list"):
                step_calls.append(
                    FlowCall(
                        cell,
                        statement,
                        call,
                        path.split(".beta.")[0],
                        thread_expr=self._arg_text(cell, call, "thread_id"),
                        target=_simple_target(statement),
                        content=self._arg_text(cell, call, "run_id"),
                    )
                )
            elif path.endswith(".beta.threads.delete"):
                delete_calls.append(
                    FlowCall(cell, statement, call, path.split(".beta.")[0])
                )

        if not self._flow_is_supported(messages, runs, assistant_setups, thread_setups):
            findings.append(
                Finding(
                    "UNSUPPORTED-FLOW-001",
                    "Query flow is not a safe one-to-one sequence",
                    min((flow.cell.index for flow in messages + runs), default=0),
                    "manual",
                    False,
                    "Every run must pair with one user text message and simple variables.",
                    "Split custom routing into explicit sequential calls, then retry.",
                )
            )
            return 0
        unresolved_reads = [
            flow
            for flow in list_calls
            if self._nearest_preceding_run(runs, flow) is None
        ]
        unresolved_steps = [
            flow
            for flow in step_calls
            if self._name_before_id(flow.content) is None
            and self._nearest_preceding_run(runs, flow) is None
        ]
        if unresolved_reads or unresolved_steps:
            first = (unresolved_reads + unresolved_steps)[0]
            findings.append(
                Finding(
                    "UNSUPPORTED-ORDER-001",
                    "Output inspection has no preceding run",
                    first.cell.index,
                    "manual",
                    False,
                    (
                        "A message or step read cannot be associated with a unique "
                        "preceding run in notebook order."
                    ),
                    "Move the read after its run or migrate the flow manually.",
                    line_start=first.statement.lineno,
                    line_end=first.statement.end_lineno,
                )
            )
            return 0

        changes = 0
        for local, (cell, alias) in old_import_locals.items():
            old = cell.text(alias)
            new = old.replace("FabricOpenAI", "FabricOpenAIResponses", 1)
            self._patch(cell, alias, new)
            findings.append(
                self._change(
                    "QUERY-IMPORT-001",
                    "Responses client import",
                    cell,
                    alias,
                    old,
                    new,
                )
            )
            changes += 1
            if not any(keyword.arg == "ai_skill_stage" for keyword in call.keywords):
                findings.append(
                    Finding(
                        "QUERY-STAGE-001",
                        "Defaulted data agent stage",
                        cell.index,
                        "medium",
                        True,
                        'The migration selected "sandbox" because no stage was present.',
                        "Confirm sandbox (draft) versus production/published is intended.",
                        line_start=call.lineno,
                        line_end=call.end_lineno,
                        replacement_excerpt='ai_skill_stage="sandbox"',
                    )
                )

        for cell, statement, call, local in constructor_calls:
            replacement = self._constructor_replacement(
                cell, call, local, definitions
            )
            self._patch(cell, call, replacement)
            if "workspace_name" not in definitions and "workspace_id" not in definitions:
                start, _ = cell.span(statement)
                placeholder = (
                    '# TODO: Set this to the Fabric workspace name or ID.\n'
                    'workspace_name = "<REQUIRED: Fabric workspace name or ID>"\n'
                )
                cell.patches.append(Patch(start, start, placeholder))
                definitions.add("workspace_name")
                findings.append(
                    Finding(
                        "QUERY-CONFIG-001",
                        "Workspace configuration placeholder",
                        cell.index,
                        "medium",
                        True,
                        "The old client did not identify a workspace.",
                        "Replace the visible workspace placeholder before running.",
                        replacement_excerpt=_redact(placeholder),
                    )
                )
                changes += 1
            findings.append(
                self._change(
                    "QUERY-CLIENT-001",
                    "Responses client construction",
                    cell,
                    call,
                    cell.text(call),
                    replacement,
                    action='Confirm "sandbox" should target the draft agent.',
                )
            )
            changes += 1

        for setup in assistant_setups + thread_setups:
            old = setup.cell.text(setup.statement)
            label = "assistant" if setup in assistant_setups else "thread"
            replacement = _comment_for(old, f"Responses API does not require a {label}.")
            self._patch(setup.cell, setup.statement, replacement)
            findings.append(
                self._change(
                    f"QUERY-{label.upper()}-001",
                    f"Removed {label} setup",
                    setup.cell,
                    setup.statement,
                    old,
                    replacement,
                )
            )
            changes += 1

        self._remove_setup_diagnostics(cells, assistant_setups, thread_setups, findings)

        used_messages: set[int] = set()
        previous_run_by_thread: dict[tuple[str, str | None], str] = {}
        for run in sorted(runs, key=lambda flow: flow.order):
            candidates = [
                (index, message)
                for index, message in enumerate(messages)
                if index not in used_messages
                and message.order < run.order
                and message.client == run.client
                and message.thread_expr == run.thread_expr
            ]
            index, message = candidates[-1]
            used_messages.add(index)
            key = (run.client, run.thread_expr)
            previous_run = previous_run_by_thread.get(key)
            input_text = message.content or '""'
            args = f"input={input_text}"
            if previous_run:
                args += f", previous_response_id={previous_run}.id"
            replacement = f"{run.target} = {run.client}.responses.create({args})"
            old_run = run.cell.text(run.statement)
            old_message = message.cell.text(message.statement)
            self._patch(run.cell, run.statement, replacement)
            self._patch(
                message.cell,
                message.statement,
                _comment_for(old_message, "question moved to the Responses call."),
            )
            previous_run_by_thread[key] = run.target or "response"
            findings.append(
                self._change(
                    "QUERY-CREATE-001",
                    "Message and run converted to one response",
                    run.cell,
                    run.statement,
                    old_run,
                    replacement,
                    action="Compare the returned answer with the original notebook.",
                )
            )
            changes += 1

        for flow in retrieve_calls:
            run_id = flow.target or "response.id"
            replacement = f"{flow.client}.responses.retrieve({run_id})"
            old = flow.cell.text(flow.call)
            self._patch(flow.cell, flow.call, replacement)
            findings.append(
                self._change(
                    "QUERY-POLL-001",
                    "Response polling",
                    flow.cell,
                    flow.call,
                    old,
                    replacement,
                )
            )
            changes += 1

        for flow in list_calls:
            preceding_run = self._nearest_preceding_run(runs, flow)
            assert preceding_run is not None and preceding_run.target is not None
            run_name = preceding_run.target
            old = flow.cell.text(flow.statement)
            if flow.target:
                replacement = f"{flow.target} = {run_name}"
            else:
                replacement = f"print({run_name}.output_text)"
            self._patch(flow.cell, flow.statement, replacement)
            findings.append(
                self._change(
                    "QUERY-OUTPUT-001",
                    "Response output source",
                    flow.cell,
                    flow.statement,
                    old,
                    replacement,
                    action=(
                        "For table-shaped answers, also inspect response.output items; "
                        "output_text alone may not contain every useful result."
                    ),
                )
            )
            changes += 1
            if flow.target:
                changes += self._replace_inline_text_loops(
                    cells, flow.target, run_name, findings
                )

        changes += self._replace_plain_text_helpers(functions, findings)

        for flow in step_calls:
            preceding_run = self._nearest_preceding_run(runs, flow)
            run_name = self._name_before_id(flow.content) or (
                preceding_run.target if preceding_run else None
            )
            assert run_name is not None
            replacement = (
                f"{flow.target} = [item for item in {run_name}.output "
                f"if getattr(item, \"type\", \"\") in {{{TERMINAL_OUTPUT_TYPES}}}]"
            )
            old = flow.cell.text(flow.statement)
            self._patch(flow.cell, flow.statement, replacement)
            self._replace_data_access(cells, flow.target, findings)
            findings.append(
                self._change(
                    "QUERY-STEPS-001",
                    "Intermediate response items",
                    flow.cell,
                    flow.statement,
                    old,
                    replacement,
                )
            )
            changes += 1

        for flow in delete_calls:
            old = flow.cell.text(flow.statement)
            replacement = _comment_for(
                old, "No thread cleanup is required with response chaining."
            )
            self._patch(flow.cell, flow.statement, replacement)
            findings.append(
                self._change(
                    "QUERY-CLEANUP-001",
                    "Removed thread cleanup",
                    flow.cell,
                    flow.statement,
                    old,
                    replacement,
                )
            )
            changes += 1

        return changes

    def _nearest_preceding_run(
        self, runs: list[FlowCall], flow: FlowCall
    ) -> FlowCall | None:
        candidates = [
            run
            for run in runs
            if run.order < flow.order
            and run.client == flow.client
            and run.thread_expr == flow.thread_expr
        ]
        return max(candidates, key=lambda candidate: candidate.order, default=None)

    def _replace_inline_text_loops(
        self,
        cells: list[ParsedCell],
        messages_name: str,
        run_name: str,
        findings: list[Finding],
    ) -> int:
        changes = 0
        for cell in cells:
            if not cell.tree:
                continue
            for node in ast.walk(cell.tree):
                if not (
                    isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Name)
                    and node.iter.id == messages_name
                    and self._is_inline_message_loop(cell, node)
                ):
                    continue
                old = cell.text(node)
                replacement = f"print({run_name}.output_text)"
                self._patch(cell, node, replacement)
                findings.append(
                    self._change(
                        "QUERY-INLINE-TEXT-001",
                        "Inline message loop",
                        cell,
                        node,
                        old,
                        replacement,
                        action=(
                            "For table-shaped answers, inspect response.output items "
                            "in addition to output_text."
                        ),
                    )
                )
                changes += 1
        return changes

    def _flow_is_supported(
        self,
        messages: list[FlowCall],
        runs: list[FlowCall],
        assistants: list[FlowCall],
        threads: list[FlowCall],
    ) -> bool:
        if not messages or len(messages) != len(runs):
            return False
        if any(not flow.target for flow in runs + assistants + threads):
            return False
        for message in messages:
            role = _keyword(message.call, "role")
            if (
                message.content is None
                or not isinstance(role, ast.Constant)
                or role.value != "user"
            ):
                return False
            if any(
                keyword.arg not in {"thread_id", "role", "content"}
                for keyword in message.call.keywords
            ):
                return False
        for run in runs:
            if run.thread_expr is None or run.assistant_expr is None:
                return False
            if any(
                keyword.arg not in {"thread_id", "assistant_id"}
                for keyword in run.call.keywords
            ):
                return False
        return True

    def _constructor_replacement(
        self,
        cell: ParsedCell,
        call: ast.Call,
        local: str,
        definitions: set[str],
    ) -> str:
        old = cell.text(call)
        renamed = re.sub(rf"\b{re.escape(local)}\b", local, old, count=1)
        if local == "FabricOpenAI":
            renamed = re.sub(
                r"\bFabricOpenAI\b", "FabricOpenAIResponses", renamed, count=1
            )
        present = {keyword.arg for keyword in call.keywords}
        additions: list[str] = []
        if "workspace_name" not in present:
            if "workspace_id" in definitions and "workspace_name" in definitions:
                value = "workspace_id or workspace_name"
            elif "workspace_id" in definitions:
                value = "workspace_id"
            else:
                value = "workspace_name"
            additions.append(f"workspace_name={value}")
        if "ai_skill_stage" not in present:
            additions.append('ai_skill_stage="sandbox"')
        if not additions:
            return renamed
        close = renamed.rfind(")")
        before = renamed[:close]
        if "\n" in renamed:
            closing_line = renamed[renamed.rfind("\n", 0, close) + 1 : close]
            close_indent = closing_line[: len(closing_line) - len(closing_line.lstrip())]
            inner_indent = close_indent + "    "
            if before.rstrip().endswith("("):
                separator = "\n"
            elif before.rstrip().endswith(","):
                separator = "\n"
            else:
                separator = ",\n"
            insertion = separator + "".join(
                f"{inner_indent}{argument},\n" for argument in additions
            )
            return before + insertion + close_indent + renamed[close:]
        separator = "" if before.rstrip().endswith("(") else ", "
        return before + separator + ", ".join(additions) + renamed[close:]

    def _remove_setup_diagnostics(
        self,
        cells: list[ParsedCell],
        assistants: list[FlowCall],
        threads: list[FlowCall],
        findings: list[Finding],
    ) -> None:
        names = {flow.target for flow in assistants + threads if flow.target}
        for cell in cells:
            if not cell.tree:
                continue
            for statement in cell.tree.body:
                if not (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and _dotted_name(statement.value.func) == "print"
                ):
                    continue
                args = statement.value.args
                if (
                    len(args) == 1
                    and isinstance(args[0], ast.Attribute)
                    and args[0].attr == "id"
                    and isinstance(args[0].value, ast.Name)
                    and args[0].value.id in names
                ):
                    old = cell.text(statement)
                    replacement = _comment_for(
                        old, "assistant/thread ID diagnostic is no longer applicable."
                    )
                    self._patch(cell, statement, replacement)
                    findings.append(
                        self._change(
                            "QUERY-DIAGNOSTIC-001",
                            "Removed obsolete ID diagnostic",
                            cell,
                            statement,
                            old,
                            replacement,
                        )
                    )

    def _replace_plain_text_helpers(
        self,
        functions: dict[str, tuple[ParsedCell, ast.FunctionDef]],
        findings: list[Finding],
    ) -> int:
        changes = 0
        for name, (cell, function) in functions.items():
            old = cell.text(function)
            if not self._is_plain_text_helper(cell, function):
                continue
            arg = function.args.args[0].arg
            replacement = (
                f"def {name}({arg}):\n"
                '    print("# Response")\n'
                f"    print({arg}.output_text)\n"
                "    print()\n"
            )
            self._patch(cell, function, replacement)
            findings.append(
                self._change(
                    "QUERY-TEXT-001",
                    "Plain-text output helper",
                    cell,
                    function,
                    old,
                    replacement,
                )
            )
            changes += 1
        return changes

    def _replace_data_access(
        self,
        cells: list[ParsedCell],
        target: str | None,
        findings: list[Finding],
    ) -> None:
        if not target:
            return
        for cell in cells:
            if not cell.tree:
                continue
            for statement in cell.tree.body:
                if (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Attribute)
                    and statement.value.attr == "data"
                    and isinstance(statement.value.value, ast.Name)
                    and statement.value.value.id == target
                ):
                    old = cell.text(statement)
                    self._patch(cell, statement, target)
                    findings.append(
                        self._change(
                            "QUERY-STEPS-DATA-001",
                            "Step list display",
                            cell,
                            statement,
                            old,
                            target,
                        )
                    )

    def _upgrade_install_cell(
        self, cells: list[ParsedCell], findings: list[Finding]
    ) -> int:
        line_pattern = re.compile(r"(?m)^\s*%pip\s+install\b[^\n]*$")
        package_pattern = re.compile(
            r"(?<=\s)fabric-data-agent-sdk(?=\s|$)"
        )
        for cell in cells:
            for match in line_pattern.finditer(cell.source):
                old_line = match.group(0)
                if not package_pattern.search(old_line):
                    continue
                replacement = package_pattern.sub(
                    '"fabric-data-agent-sdk>=0.1.28a0"',
                    old_line,
                    count=1,
                )
                cell.patches.append(Patch(match.start(), match.end(), replacement))
                findings.append(
                    Finding(
                        "QUERY-SDK-001",
                        "SDK version floor",
                        cell.index,
                        "high",
                        True,
                        "The notebook now requests a release containing the non-streamed response fix.",
                        "Restart Python after installation, then confirm the installed SDK version.",
                        line_start=cell.source[: match.start()].count("\n") + 1,
                        original_excerpt=_redact(old_line),
                        replacement_excerpt=replacement,
                    )
                )
                return 1
        return 0

    def _apply_evaluation_rule(
        self, cells: list[ParsedCell], findings: list[Finding]
    ) -> int:
        changes = 0
        for cell in cells:
            if not cell.tree:
                continue
            response_imported = "FabricOpenAIResponses" in cell.source
            for node in ast.walk(cell.tree):
                if not isinstance(node, ast.Call):
                    continue
                if _dotted_name(node.func) != "evaluate_data_agent":
                    continue
                if any(keyword.arg == "client_class" for keyword in node.keywords):
                    continue
                old = cell.text(node)
                close = old.rfind(")")
                before = old[:close]
                if "\n" in old:
                    close_line_start = old.rfind("\n", 0, close) + 1
                    before_args = old[:close_line_start]
                    closing = old[close_line_start:close]
                    closing_indent = closing[
                        : len(closing) - len(closing.lstrip())
                    ]
                    indent = closing_indent + "    "
                    separator = "" if before_args.rstrip().endswith(",") else ",\n"
                    replacement = (
                        before_args
                        + separator
                        + f"{indent}client_class=FabricOpenAIResponses,\n"
                        + closing_indent
                        + old[close:]
                    )
                else:
                    separator = "" if before.rstrip().endswith("(") else ", "
                    replacement = (
                        before
                        + separator
                        + "client_class=FabricOpenAIResponses"
                        + old[close:]
                    )
                self._patch(cell, node, replacement)
                if not response_imported:
                    statement = next(
                        (
                            statement
                            for statement in cell.tree.body
                            if statement.lineno <= node.lineno <= statement.end_lineno
                        ),
                        cell.tree.body[0],
                    )
                    start, _ = cell.span(statement)
                    cell.patches.append(
                        Patch(
                            start,
                            start,
                            "from fabric.dataagent.client import FabricOpenAIResponses\n",
                        )
                    )
                    findings.append(
                        Finding(
                            "EVAL-IMPORT-001",
                            "Responses evaluation client class import",
                            cell.index,
                            "high",
                            True,
                            (
                                "Imported FabricOpenAIResponses so client_class "
                                "resolves when the evaluation cell runs."
                            ),
                            (
                                "Pass the class itself; do not construct a client "
                                "instance for evaluate_data_agent."
                            ),
                            line_start=statement.lineno,
                            replacement_excerpt=(
                                "from fabric.dataagent.client import "
                                "FabricOpenAIResponses"
                            ),
                        )
                    )
                    response_imported = True
                findings.append(
                    self._change(
                        "EVAL-CLIENT-001",
                        "Evaluation uses Responses client",
                        cell,
                        node,
                        old,
                        replacement,
                        action="Compare evaluation result and step tables with the baseline.",
                    )
                )
                changes += 1
        return changes

    def _commit_patches(
        self, cells: list[ParsedCell], notebook: NotebookNode
    ) -> set[int]:
        changed: set[int] = set()
        for cell in cells:
            if not cell.patches:
                continue
            patches = sorted(cell.patches, key=lambda patch: (patch.start, patch.end))
            for left, right in zip(patches, patches[1:]):
                if left.end > right.start and not (
                    left.start == right.start and left.end == right.end
                ):
                    raise RuntimeError(
                        f"Overlapping migration rules in cell {cell.index}."
                    )
            source = cell.source
            for patch in sorted(
                patches, key=lambda item: (item.start, item.end), reverse=True
            ):
                source = source[: patch.start] + patch.replacement + source[patch.end :]
            if source != cell.source:
                notebook.cells[cell.index].source = source
                changed.add(cell.index)
        return changed

    def _validate_generated_output(
        self, notebook: NotebookNode, changed_cells: set[int]
    ) -> list[Finding]:
        errors: list[Finding] = []
        for index in sorted(changed_cells):
            cell = notebook.cells[index]
            if cell.cell_type != "code":
                continue
            sanitized, has_cell_magic = _sanitize_magics(cell.source)
            if has_cell_magic:
                continue
            try:
                ast.parse(sanitized)
            except SyntaxError as exc:
                errors.append(
                    Finding(
                        "OUTPUT-SYNTAX-001",
                        "Generated code failed validation",
                        index,
                        "manual",
                        False,
                        (
                            "All automatic edits were rolled back because generated "
                            f"Python did not parse: {exc.msg}."
                        ),
                        "Leave this notebook untouched and report the rule combination.",
                        line_start=exc.lineno,
                        line_end=exc.lineno,
                    )
                )
        return errors

    def _preservation_score(
        self,
        original: NotebookNode,
        migrated: NotebookNode,
        changed_cells: set[int],
    ) -> int:
        checks = 1
        passed = int(original.metadata == migrated.metadata)
        if len(original.cells) != len(migrated.cells):
            return 0
        for index, (before, after) in enumerate(zip(original.cells, migrated.cells)):
            checks += 1
            if index not in changed_cells:
                passed += int(before == after)
                continue
            before_without_source = copy.deepcopy(before)
            after_without_source = copy.deepcopy(after)
            before_without_source.source = ""
            after_without_source.source = ""
            passed += int(before_without_source == after_without_source)
        return round(100 * passed / checks)

    def _patch(self, cell: ParsedCell, node: ast.AST, replacement: str) -> None:
        start, end = cell.span(node)
        cell.patches.append(Patch(start, end, replacement))

    def _arg_text(self, cell: ParsedCell, call: ast.Call, name: str) -> str | None:
        value = _keyword(call, name)
        return cell.text(value) if value is not None else None

    def _name_before_id(self, expression: str | None) -> str | None:
        if not expression:
            return None
        match = re.fullmatch(r"([A-Za-z_]\w*)\.id", expression.strip())
        return match.group(1) if match else None

    def _change(
        self,
        rule_id: str,
        title: str,
        cell: ParsedCell,
        node: ast.AST,
        old: str,
        new: str,
        action: str | None = None,
    ) -> Finding:
        return Finding(
            rule_id,
            title,
            cell.index,
            "high",
            True,
            "Applied a verified syntax-aware migration rule.",
            action,
            node.lineno,
            node.end_lineno,
            _redact(old),
            _redact(new),
        )

    def _status(
        self,
        relevant: bool,
        total_changes: int,
        manual: list[Finding],
        warnings: list[Finding],
    ) -> str:
        if not relevant:
            return "no_migration_needed"
        if manual:
            return "partial_migration"
        if total_changes and warnings:
            return "migrated_with_warnings"
        if total_changes:
            return "migrated"
        return "no_migration_needed"

    def _score(self, findings: list[Finding], preservation: int) -> Score:
        relevant = [
            finding
            for finding in findings
            if finding.rule_id.startswith(("QUERY-", "EVAL-", "UNSUPPORTED-", "OUTPUT-"))
            and finding.rule_id != "QUERY-DIAGNOSTIC-001"
        ]
        if not relevant:
            return Score(100, 100, 100, preservation)
        applied = [finding for finding in relevant if finding.applied]
        coverage = round(100 * len(applied) / len(relevant))
        confidence_points = 0
        for finding in relevant:
            if not finding.applied:
                continue
            confidence_points += 100 if finding.confidence == "high" else 70
        confidence = round(confidence_points / len(relevant))
        overall = min(coverage, confidence, preservation)
        return Score(overall, coverage, confidence, preservation)

    def _test_checklist(
        self,
        inventory: dict[str, object],
        manual: Iterable[Finding],
        warnings: Iterable[Finding],
    ) -> list[str]:
        checklist = [
            "Replace every <REQUIRED: ...> placeholder and confirm the workspace.",
            "Confirm sandbox (draft) versus production/published stage is intended.",
            "Run the migrated notebook in a non-production Fabric workspace.",
            "Compare representative answers with the original notebook.",
            "Test a table-shaped answer and inspect response.output, not only output_text.",
            "Test failure, incomplete, cancellation, and timeout behavior.",
            "Confirm no credentials or tokens are stored in notebook cells or outputs.",
            "Review every manual finding before switching production workloads.",
        ]
        if any(True for _ in manual):
            checklist.insert(
                0, "Do not treat the notebook as complete until all manual findings are resolved."
            )
        if any(
            finding.rule_id == "RUNTIME-SDK-RESTART-001" for finding in warnings
        ):
            checklist.insert(
                0,
                (
                    "Restart Python after the SDK install, then confirm "
                    "FabricOpenAI, FabricOpenAIResponses, and evaluate_data_agent "
                    "all import in the fresh session."
                ),
            )
        return checklist
