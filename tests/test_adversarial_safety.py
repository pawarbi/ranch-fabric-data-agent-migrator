from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest

from fabric_migrator import MigrationEngine
from fabric_migrator.notebook_io import NotebookLimits, NotebookValidationError


BASE_SETUP = """from fabric.dataagent.client import FabricOpenAI
fabric_client = FabricOpenAI(artifact_name=data_agent_name)
assistant = fabric_client.beta.assistants.create(model="gpt-4o")
thread = fabric_client.beta.threads.create()
"""

QUESTION = """fabric_client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=question,
)
run = fabric_client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
"""

EVALUATION = """from fabric.dataagent.evaluation import evaluate_data_agent
evaluation_id = evaluate_data_agent(rows, data_agent_name="sales")
"""


def notebook_bytes(*sources: str) -> bytes:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(source) for source in sources]
    return nbformat.writes(notebook).encode("utf-8")


def read_notebook(data: bytes):
    notebook = nbformat.reads(data.decode("utf-8"), as_version=4)
    nbformat.validate(notebook)
    return notebook


def finding_ids(result) -> set[str]:
    return {
        finding.rule_id
        for finding in (
            result.report.changes
            + result.report.warnings
            + result.report.manual_actions
        )
    }


@pytest.mark.parametrize(
    ("dynamic_source", "name"),
    [
        (
            'method = getattr(fabric_client.beta.threads.runs, "cancel")',
            "getattr",
        ),
        (
            'setattr(fabric_client.beta.threads, "cached_id", thread.id)',
            "setattr",
        ),
        (
            'eval("fabric_client.beta.threads.runs.cancel(run.id)")',
            "eval",
        ),
        (
            'exec("fabric_client.beta.threads.delete(thread.id)")',
            "exec",
        ),
    ],
    ids=["getattr", "setattr", "eval", "exec"],
)
def test_dynamic_python_classes_fail_closed(
    dynamic_source: str, name: str
) -> None:
    data = notebook_bytes(BASE_SETUP + "\n" + dynamic_source, QUESTION)

    result = MigrationEngine().migrate(data, f"{name}.ipynb")

    assert result.notebook_bytes == data
    assert result.report.status == "partial_migration"
    assert result.report.summary["cells_changed"] == 0
    assert result.report.changes == []
    assert {"UNSUPPORTED-DYNAMIC-001", "SAFE-ATOMIC-001"} <= finding_ids(result)
    read_notebook(result.notebook_bytes)


def _hostile_cases() -> list[object]:
    run_before_message = """run = fabric_client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
fabric_client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=question,
)
"""
    ambiguous_batching = """fabric_client.beta.threads.messages.create(
    thread_id=thread.id, role="user", content=first_question
)
fabric_client.beta.threads.messages.create(
    thread_id=thread.id, role="user", content=second_question
)
first_run = fabric_client.beta.threads.runs.create(
    thread_id=thread.id, assistant_id=assistant.id
)
second_run = fabric_client.beta.threads.runs.create(
    thread_id=thread.id, assistant_id=assistant.id
)
"""
    return [
        pytest.param(
            (
                BASE_SETUP
                + """
def submit_message():
    return fabric_client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=question
    )
"""
            ),
            QUESTION,
            None,
            "UNSUPPORTED-WRAPPER-001",
            id="custom-wrapper",
        ),
        pytest.param(
            BASE_SETUP,
            QUESTION,
            (
                "fabric_client.beta.threads.runs.cancel("
                "thread_id=thread.id, run_id=run.id)"
            ),
            "UNSUPPORTED-BETA-001",
            id="unknown-beta-call",
        ),
        pytest.param(
            BASE_SETUP,
            QUESTION.replace(
                "content=question,",
                "content=question,\n    attachments=[],",
            ),
            None,
            "UNSUPPORTED-SHAPE-001",
            id="unsupported-options",
        ),
        pytest.param(
            (
                BASE_SETUP
                + """
async def wait_for_run():
    await fabric_client.beta.threads.runs.retrieve(
        thread_id=thread.id, run_id=run.id
    )
"""
            ),
            QUESTION,
            None,
            "UNSUPPORTED-ASYNC-001",
            id="async-orchestration",
        ),
        pytest.param(
            BASE_SETUP + "\nthread = fabric_client.beta.threads.create()\n",
            QUESTION,
            None,
            "UNSUPPORTED-REASSIGNMENT-001",
            id="duplicate-producer",
        ),
        pytest.param(
            BASE_SETUP + "\nthread = restore_thread()\n",
            QUESTION,
            None,
            "UNSUPPORTED-REASSIGNMENT-001",
            id="ordinary-reassignment",
        ),
        pytest.param(
            (
                BASE_SETUP
                + """
thread_id = thread.id
Path("saved-thread.txt").write_text(thread_id)
"""
            ),
            QUESTION,
            None,
            "UNSUPPORTED-PERSISTED-001",
            id="persisted-thread-id",
        ),
        pytest.param(
            BASE_SETUP,
            run_before_message,
            None,
            "UNSUPPORTED-FLOW-001",
            id="run-before-message",
        ),
        pytest.param(
            BASE_SETUP,
            ambiguous_batching,
            None,
            "UNSUPPORTED-FLOW-001",
            id="ambiguous-message-run-batching",
        ),
        pytest.param(
            BASE_SETUP,
            (
                QUESTION
                + """
while run.status != "completed":
    run = fabric_client.beta.threads.runs.retrieve(
        thread_id=thread.id, run_id=run.id
    )
"""
            ),
            None,
            "UNSUPPORTED-SHAPE-001",
            id="nonstandard-polling",
        ),
    ]


@pytest.mark.parametrize(
    ("setup", "flow", "extra_source", "expected_rule"), _hostile_cases()
)
def test_hostile_query_shapes_preserve_every_input_byte(
    setup: str,
    flow: str,
    extra_source: str | None,
    expected_rule: str,
) -> None:
    sources = [setup, flow]
    if extra_source is not None:
        sources.append(extra_source)
    data = notebook_bytes(*sources)

    result = MigrationEngine().migrate(data, "hostile.ipynb")

    assert result.notebook_bytes == data
    assert result.report.status == "partial_migration"
    assert result.report.summary["cells_changed"] == 0
    assert expected_rule in finding_ids(result)
    read_notebook(result.notebook_bytes)


def test_blocked_query_plane_is_atomic_but_safe_evaluation_is_independent() -> None:
    notebook = nbformat.v4.new_notebook(metadata={"owner": "preserve"})
    setup = nbformat.v4.new_code_cell(BASE_SETUP, metadata={"plane": "query"})
    setup.outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="old setup output\r\n")
    ]
    question = nbformat.v4.new_code_cell(QUESTION, metadata={"plane": "query"})
    blocker = nbformat.v4.new_code_cell(
        "fabric_client.beta.threads.runs.cancel("
        "thread_id=thread.id, run_id=run.id)",
        metadata={"plane": "query"},
    )
    evaluation = nbformat.v4.new_code_cell(
        EVALUATION, metadata={"plane": "evaluation"}
    )
    notebook.cells = [setup, question, blocker, evaluation]
    data = nbformat.writes(notebook).encode("utf-8")

    result = MigrationEngine().migrate(data, "mixed.ipynb")
    migrated = read_notebook(result.notebook_bytes)

    assert migrated.metadata == notebook.metadata
    assert migrated.cells[:3] == notebook.cells[:3]
    assert "client_class=FabricOpenAIResponses" in migrated.cells[3].source
    assert "from fabric.dataagent.client import FabricOpenAIResponses" in (
        migrated.cells[3].source
    )
    assert not any(
        finding.applied and finding.rule_id.startswith("QUERY-")
        for finding in result.report.changes
    )
    assert {"UNSUPPORTED-BETA-001", "SAFE-ATOMIC-001"} <= finding_ids(result)
    assert {
        finding.rule_id for finding in result.report.changes
    } == {"EVAL-IMPORT-001", "EVAL-CLIENT-001"}


def test_noop_preserves_odd_schema_valid_notebook_byte_for_byte() -> None:
    raw = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"custom": {"keep": True}},
                "source": ["# Café\r\n", "\r\n", "Unicode: 雪 🚜\r\n"],
                "attachments": {
                    "pixel.png": {
                        "image/png": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
                    }
                },
            },
            {
                "cell_type": "code",
                "execution_count": 7,
                "metadata": {"tags": ["keep-output"]},
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {
                            "text/plain": ["preserve\r\n", "this"],
                            "application/json": {"nested": ["α", 2]},
                        },
                        "metadata": {"isolated": True},
                    }
                ],
                "source": [
                    "# comments stay\r\n",
                    "\r\n",
                    "safe_value = '" + ("x" * 20_000) + "'\r\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "custom": {"nested": [1, {"two": 2}]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    data = json.dumps(
        raw, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")

    result = MigrationEngine().migrate(data, "odd-noop.ipynb")

    assert result.notebook_bytes == data
    assert result.report.status == "no_migration_needed"
    assert result.report.summary["cells_changed"] == 0
    assert result.report.changes == []
    read_notebook(result.notebook_bytes)


def test_changed_notebook_preserves_rich_structure_and_untouched_text() -> None:
    secret = "token-value-that-must-not-enter-the-report"
    long_line = "very_long = '" + ("λ" * 20_000) + "'\r\n"
    raw = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "markdown-cell",
                "metadata": {"id": "markdown"},
                "source": [
                    "# Café\r\n",
                    "\r\n",
                    "![plot](attachment:pixel.png)\r\n",
                ],
                "attachments": {
                    "pixel.png": {
                        "image/png": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
                    }
                },
            },
            {
                "cell_type": "code",
                "id": "setup-cell",
                "execution_count": 42,
                "metadata": {"tags": ["parameters"], "custom": {"keep": "yes"}},
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": ["preserved output\r\n", "雪\r\n"],
                    },
                    {
                        "output_type": "display_data",
                        "data": {
                            "text/plain": ["attachment-like output"],
                            "application/json": {"rows": [{"value": "✓"}]},
                        },
                        "metadata": {"keep": True},
                    },
                ],
                "source": [
                    "# café 🚀 comment\r\n",
                    "\r\n",
                    long_line,
                    f'access_token = "{secret}"\r\n',
                    "from fabric.dataagent.client import FabricOpenAI\r\n",
                    "fabric_client = FabricOpenAI(artifact_name=data_agent_name)\r\n",
                    'assistant = fabric_client.beta.assistants.create(model="gpt-4o")\r\n',
                    "thread = fabric_client.beta.threads.create()\r\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "question-cell",
                "execution_count": None,
                "metadata": {"id": "question"},
                "outputs": [],
                "source": QUESTION.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "language_info": {"name": "python", "version": "3.11"},
            "custom": {"preserve": ["all", "metadata"]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    data = json.dumps(raw, ensure_ascii=False).encode("utf-8")
    original = read_notebook(data)

    result = MigrationEngine().migrate(data, "rich.ipynb")
    migrated = read_notebook(result.notebook_bytes)
    report_json = json.dumps(result.report.to_dict(), ensure_ascii=False)

    assert migrated.metadata == original.metadata
    assert migrated.cells[0] == original.cells[0]
    assert migrated.cells[1].metadata == original.cells[1].metadata
    assert migrated.cells[1].execution_count == 42
    assert migrated.cells[1].outputs == original.cells[1].outputs
    assert migrated.cells[2].metadata == original.cells[2].metadata
    assert migrated.cells[2].outputs == original.cells[2].outputs
    assert "# café 🚀 comment\r\n\r\n" in migrated.cells[1].source
    assert long_line in migrated.cells[1].source
    assert f'access_token = "{secret}"\r\n' in migrated.cells[1].source
    assert "FabricOpenAIResponses" in migrated.cells[1].source
    assert secret not in report_json
    assert "WARNING-SECRET-001" in finding_ids(result)

    serialized = json.loads(result.notebook_bytes)
    assert serialized["cells"][0]["source"] == raw["cells"][0]["source"]
    assert serialized["cells"][0]["attachments"] == raw["cells"][0]["attachments"]


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"{", id="truncated-json"),
        pytest.param(b"[]", id="non-object-root"),
        pytest.param(json.dumps({"nbformat": 4}).encode(), id="missing-cells"),
        pytest.param(
            json.dumps(
                {"cells": [], "metadata": {}, "nbformat": 3, "nbformat_minor": 0}
            ).encode(),
            id="unsupported-nbformat",
        ),
        pytest.param(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "metadata": {},
                            "source": 7,
                            "outputs": [],
                            "execution_count": None,
                        }
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            ).encode(),
            id="invalid-cell-source",
        ),
        pytest.param(b"\xff\xfe\x00\x00", id="non-utf8"),
    ],
)
def test_malformed_notebooks_fail_before_migration(data: bytes) -> None:
    with pytest.raises(NotebookValidationError):
        MigrationEngine().migrate(data, "malformed.ipynb")


def test_file_and_cell_limits_fail_without_large_fixtures() -> None:
    one_cell = notebook_bytes("x = 1")
    two_cells = notebook_bytes("x = 1", "y = 2")

    with pytest.raises(NotebookValidationError, match="exceeds"):
        MigrationEngine(
            NotebookLimits(max_bytes=len(one_cell) - 1, max_cells=10)
        ).migrate(one_cell, "too-many-bytes.ipynb")
    with pytest.raises(NotebookValidationError, match="cell limit"):
        MigrationEngine(NotebookLimits(max_bytes=10_000, max_cells=1)).migrate(
            two_cells, "too-many-cells.ipynb"
        )


@pytest.mark.parametrize(
    "prefix",
    [
        pytest.param("%%time\n", id="cell-magic"),
        pytest.param("%time\nif True print('broken')\n", id="bad-python-after-magic"),
        pytest.param("value = %time expression\n", id="embedded-line-magic"),
    ],
)
def test_invalid_code_around_magics_is_preserved(prefix: str) -> None:
    data = notebook_bytes(prefix + BASE_SETUP, QUESTION)

    result = MigrationEngine().migrate(data, "invalid-magic.ipynb")

    assert result.notebook_bytes == data
    assert result.report.status == "partial_migration"
    assert "UNSUPPORTED-SYNTAX-001" in finding_ids(result)
    read_notebook(result.notebook_bytes)


def test_line_magic_with_non_python_arguments_survives_safe_migration() -> None:
    magic = "%time invalid(\r\n"
    data = notebook_bytes(magic + BASE_SETUP.replace("\n", "\r\n"), QUESTION)

    result = MigrationEngine().migrate(data, "line-magic.ipynb")
    migrated = read_notebook(result.notebook_bytes)

    assert migrated.cells[0].source.startswith(magic)
    assert "FabricOpenAIResponses" in migrated.cells[0].source


def test_generated_output_parse_failure_rolls_back_rich_notebook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook = nbformat.v4.new_notebook(metadata={"rollback": "all"})
    setup = nbformat.v4.new_code_cell(BASE_SETUP, metadata={"keep": True})
    setup.outputs = [
        nbformat.v4.new_output("stream", name="stderr", text="preserve me\n")
    ]
    notebook.cells = [
        nbformat.v4.new_markdown_cell("Keep me."),
        setup,
        nbformat.v4.new_code_cell(QUESTION),
    ]
    data = nbformat.writes(notebook).encode("utf-8")
    engine = MigrationEngine()
    monkeypatch.setattr(
        engine,
        "_constructor_replacement",
        lambda cell, call, local, definitions: "(",
    )

    result = engine.migrate(data, "parse-rollback.ipynb")

    assert result.notebook_bytes == data
    assert result.report.summary["cells_changed"] == 0
    assert result.report.changes == []
    assert "OUTPUT-SYNTAX-001" in finding_ids(result)
    read_notebook(result.notebook_bytes)


def test_migration_never_executes_uploaded_source(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist.txt"
    source = (
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        + BASE_SETUP
    )

    result = MigrationEngine().migrate(
        notebook_bytes(source, QUESTION), "never-execute.ipynb"
    )

    assert not marker.exists()
    assert result.report.summary["cells_changed"] > 0
    read_notebook(result.notebook_bytes)


def test_successful_second_pass_is_a_strict_noop_with_no_new_findings() -> None:
    engine = MigrationEngine()
    first = engine.migrate(
        notebook_bytes(BASE_SETUP, QUESTION), "idempotent.ipynb"
    )
    second = engine.migrate(first.notebook_bytes, "idempotent.ipynb")

    assert second.notebook_bytes == first.notebook_bytes
    assert second.report.status == "no_migration_needed"
    assert second.report.summary["cells_changed"] == 0
    assert second.report.changes == []
    assert second.report.warnings == []
    assert second.report.manual_actions == []
    read_notebook(second.notebook_bytes)


def test_blocked_second_pass_repeats_findings_without_mutation_or_invention() -> None:
    data = notebook_bytes(
        BASE_SETUP,
        QUESTION,
        "fabric_client.beta.threads.runs.cancel("
        "thread_id=thread.id, run_id=run.id)",
    )
    engine = MigrationEngine()

    first = engine.migrate(data, "blocked-idempotent.ipynb")
    second = engine.migrate(first.notebook_bytes, "blocked-idempotent.ipynb")

    assert first.notebook_bytes == second.notebook_bytes == data
    assert second.report.to_dict() == first.report.to_dict()
    read_notebook(second.notebook_bytes)
