from __future__ import annotations

import copy
import json
from collections.abc import Iterable

import nbformat
import pytest

from fabric_migrator import MigrationEngine


CellSpec = tuple[str, str]


def notebook_bytes(
    cells: Iterable[CellSpec],
    *,
    source_form: str = "string",
    metadata: dict[str, object] | None = None,
) -> bytes:
    notebook = nbformat.v4.new_notebook(
        metadata=metadata or {"sample": {"owner": "test", "preserve": True}}
    )
    notebook.cells = []
    for index, (cell_type, source) in enumerate(cells):
        if cell_type == "markdown":
            cell = nbformat.v4.new_markdown_cell(source)
        elif cell_type == "raw":
            cell = nbformat.v4.new_raw_cell(source)
        else:
            cell = nbformat.v4.new_code_cell(source)
            cell.outputs = [
                nbformat.v4.new_output(
                    "stream", name="stdout", text=f"saved output {index}\n"
                )
            ]
            cell.execution_count = index
        cell.metadata["sample_index"] = index
        notebook.cells.append(cell)

    raw = json.loads(nbformat.writes(notebook))
    for cell in raw["cells"]:
        source = cell["source"]
        text = "".join(source) if isinstance(source, list) else source
        if source_form == "list":
            cell["source"] = text.splitlines(keepends=True)
        elif source_form == "string":
            cell["source"] = text
        else:
            raise ValueError(f"Unknown source form: {source_form}")
    return json.dumps(raw, ensure_ascii=False, indent=1).encode()


def read_notebook(data: bytes):
    notebook = nbformat.reads(data.decode(), as_version=4)
    nbformat.validate(notebook)
    return notebook


def code_text(notebook) -> str:
    return "\n\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )


def assert_valid_preserving_structure(before: bytes, after: bytes):
    original = read_notebook(before)
    migrated = read_notebook(after)
    assert len(migrated.cells) == len(original.cells)
    assert migrated.metadata == original.metadata
    for old_cell, new_cell in zip(original.cells, migrated.cells):
        old_without_source = copy.deepcopy(old_cell)
        new_without_source = copy.deepcopy(new_cell)
        old_without_source.source = ""
        new_without_source.source = ""
        assert new_without_source == old_without_source
    return original, migrated


def configured_setup(
    *,
    import_suffix: str,
    local_class: str,
    client_name: str,
    quote: str,
    compact: bool,
) -> str:
    import_line = (
        f"from fabric.dataagent.client import FabricOpenAI{import_suffix}"
    )
    if compact:
        return "\n".join(
            [
                import_line,
                f"workspace_name={quote}Contoso Analytics{quote}",
                f"data_agent_name={quote}Revenue Agent{quote}",
                (
                    f"{client_name} = {local_class}(artifact_name=data_agent_name, "
                    "workspace_name=workspace_name, "
                    f"ai_skill_stage={quote}production{quote})"
                ),
                (
                    f"assistant={client_name}.beta.assistants.create("
                    f"model={quote}gpt-4o{quote})"
                ),
                f"thread={client_name}.beta.threads.create()",
                "print(assistant.id)",
                "print(thread.id)",
            ]
        )
    return f"""{import_line}
workspace_name = {quote}Contoso Analytics{quote}
data_agent_name = {quote}Revenue Agent{quote}

{client_name} = {local_class}(
    artifact_name=data_agent_name,
    workspace_name=workspace_name,
    ai_skill_stage={quote}production{quote},
)
assistant = {client_name}.beta.assistants.create(
    model={quote}gpt-4o{quote},
)
thread = {client_name}.beta.threads.create()
print(assistant.id)
print(thread.id)
"""


def official_single_turn_cells(
    *,
    import_suffix: str,
    local_class: str,
    client_name: str,
    quote: str,
    compact: bool,
) -> list[CellSpec]:
    setup = configured_setup(
        import_suffix=import_suffix,
        local_class=local_class,
        client_name=client_name,
        quote=quote,
        compact=compact,
    )
    if compact:
        message = (
            f"{client_name}.beta.threads.messages.create( "
            f"thread_id = thread.id , role = {quote}user{quote} , "
            "content = question )"
        )
        run = (
            f"run={client_name}.beta.threads.runs.create("
            "thread_id=thread.id,assistant_id=assistant.id)"
        )
    else:
        message = f"""{client_name}.beta.threads.messages.create(
    thread_id=thread.id,
    role={quote}user{quote},
    content=question,
)
"""
        run = f"""run = {client_name}.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
"""
    return [
        (
            "markdown",
            "# Fabric Data Agent query\n"
            "This public-shaped sample asks a revenue question and inspects the "
            "answer and intermediate output.",
        ),
        ("code", "%pip install -q fabric-data-agent-sdk pandas"),
        ("code", "%matplotlib inline\n# Keep this line magic cell unchanged."),
        ("code", "%%sql\nSELECT TOP 5 * FROM demo.revenue"),
        (
            "code",
            "# FabricOpenAI and .beta. in comments are documentation, not API calls.\n"
            f"question = {quote}Summarize quarterly revenue by region.{quote}",
        ),
        ("code", setup),
        ("code", message),
        ("code", run),
        (
            "code",
            f"""import time
while run.status in ({quote}queued{quote}, {quote}in_progress{quote}):
    run = {client_name}.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id,
    )
    time.sleep(1)
""",
        ),
        (
            "code",
            f"""messages = {client_name}.beta.threads.messages.list(
    thread_id=thread.id,
    order={quote}asc{quote},
)
""",
        ),
        (
            "code",
            """for message in messages:
    print(f"{message.role}: {message.content[0].text.value}")
""",
        ),
        (
            "code",
            f"""steps = {client_name}.beta.threads.runs.steps.list(
    thread_id=thread.id,
    run_id=run.id,
)
steps.data
""",
        ),
        ("markdown", "The next cell cleans up the query-only thread."),
        ("code", f"{client_name}.beta.threads.delete(thread.id)"),
        ("raw", "Preserve this raw cell exactly."),
    ]


@pytest.mark.parametrize(
    (
        "import_suffix",
        "local_class",
        "client_name",
        "quote",
        "compact",
        "source_form",
    ),
    [
        ("", "FabricOpenAI", "fabric_client", '"', False, "list"),
        (" as AgentClient", "AgentClient", "agent_client", "'", True, "string"),
        (
            " as LegacyFabricClient",
            "LegacyFabricClient",
            "contoso_agent",
            '"',
            False,
            "string",
        ),
    ],
    ids=["documented-list-source", "aliased-compact", "aliased-multiline"],
)
def test_official_single_turn_notebook_matrix(
    import_suffix: str,
    local_class: str,
    client_name: str,
    quote: str,
    compact: bool,
    source_form: str,
) -> None:
    cells = official_single_turn_cells(
        import_suffix=import_suffix,
        local_class=local_class,
        client_name=client_name,
        quote=quote,
        compact=compact,
    )
    data = notebook_bytes(cells, source_form=source_form)
    raw = json.loads(data)
    expected_source_type = list if source_form == "list" else str
    assert isinstance(raw["cells"][5]["source"], expected_source_type)

    result = MigrationEngine().migrate(data, "official-query-sample.ipynb")
    before, after = assert_valid_preserving_structure(data, result.notebook_bytes)
    text = code_text(after)

    expected_import = "FabricOpenAIResponses" + import_suffix
    assert expected_import in text
    # An aliased import keeps its local name, so the construction is unchanged.
    # A bare import is renamed, so the construction has to follow it.
    expected_class = local_class if import_suffix else "FabricOpenAIResponses"
    assert f"{client_name} = {expected_class}(" in text
    assert "workspace_name=workspace_name" in text
    assert f"ai_skill_stage={quote}production{quote}" in text
    assert f"{client_name}.responses.create(input=question)" in text
    assert f"{client_name}.responses.retrieve(run.id)" in text
    assert "messages = run" in text
    assert "print(run.output_text)" in text
    assert (
        'steps = [item for item in run.output if getattr(item, "type", "")'
        in text
    )
    assert '"fabric-data-agent-sdk>=0.1.28a0"' in text
    assert ".beta." not in "\n".join(
        cell.source
        for index, cell in enumerate(after.cells)
        if cell.cell_type == "code" and index != 4
    )
    assert after.cells[0] == before.cells[0]
    assert after.cells[2] == before.cells[2]
    assert after.cells[3] == before.cells[3]
    assert after.cells[12] == before.cells[12]
    assert after.cells[14] == before.cells[14]

    rules = {finding.rule_id for finding in result.report.changes}
    assert {
        "QUERY-IMPORT-001",
        "QUERY-CLIENT-001",
        "QUERY-CREATE-001",
        "QUERY-POLL-001",
        "QUERY-OUTPUT-001",
        "QUERY-INLINE-TEXT-001",
        "QUERY-STEPS-001",
        "QUERY-CLEANUP-001",
        "QUERY-SDK-001",
    } <= rules
    create_finding = next(
        finding
        for finding in result.report.changes
        if finding.rule_id == "QUERY-CREATE-001"
    )
    assert ".beta.threads.runs.create" in create_finding.original_excerpt
    assert ".responses.create" in create_finding.replacement_excerpt
    assert result.report.status == "migrated"
    assert result.report.summary["cells_scanned"] == len(cells)
    assert result.report.summary["manual_actions"] == 0
    assert result.report.summary["warnings"] == 0
    assert result.report.score.overall == 100
    assert result.report.score.preservation_confidence == 100


def turn_cells(
    question_name: str,
    run_name: str,
    messages_name: str,
    client_name: str = "client",
) -> list[CellSpec]:
    return [
        (
            "code",
            f"""{client_name}.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content={question_name},
)
""",
        ),
        (
            "code",
            f"""{run_name} = {client_name}.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
""",
        ),
        (
            "code",
            f"""{messages_name} = {client_name}.beta.threads.messages.list(
    thread_id=thread.id,
    order="asc",
)
""",
        ),
        (
            "code",
            f"""for item in {messages_name}:
    print(f"{{item.role}}: {{item.content[0].text.value}}")
""",
        ),
    ]


def test_sequential_three_turn_notebook_uses_nearest_preceding_run() -> None:
    cells: list[CellSpec] = [
        ("markdown", "# Sequential follow-up analysis"),
        (
            "code",
            """from fabric.dataagent.client import FabricOpenAI
workspace_name = "Contoso Analytics"
data_agent_name = "Revenue Agent"
client = FabricOpenAI(
    artifact_name=data_agent_name,
    workspace_name=workspace_name,
    ai_skill_stage="production",
)
assistant = client.beta.assistants.create(model="gpt-4o")
thread = client.beta.threads.create()
""",
        ),
        (
            "code",
            """first_question = "Which region led revenue last quarter?"
follow_up_question = "How did that compare with the prior quarter?"
final_question = "Which product categories explain the change?"
""",
        ),
    ]
    cells.extend(turn_cells("first_question", "first_run", "first_messages"))
    cells.append(("markdown", "Continue the same thread with business context."))
    cells.extend(
        turn_cells("follow_up_question", "follow_up_run", "follow_up_messages")
    )
    cells.append(("code", "# A harmless calculation between turns.\nthreshold = 0.05"))
    cells.extend(turn_cells("final_question", "final_run", "final_messages"))

    data = notebook_bytes(cells, source_form="list")
    result = MigrationEngine().migrate(data, "three-turn-analysis.ipynb")
    _, migrated = assert_valid_preserving_structure(data, result.notebook_bytes)
    text = code_text(migrated)

    assert "first_run = client.responses.create(input=first_question)" in text
    assert (
        "follow_up_run = client.responses.create("
        "input=follow_up_question, previous_response_id=first_run.id)"
    ) in text
    assert (
        "final_run = client.responses.create("
        "input=final_question, previous_response_id=follow_up_run.id)"
    ) in text
    assert "first_messages = first_run" in text
    assert "follow_up_messages = follow_up_run" in text
    assert "final_messages = final_run" in text
    assert "first_messages = follow_up_run" not in text
    assert "follow_up_messages = final_run" not in text
    assert text.count("print(first_run.output_text)") == 1
    assert text.count("print(follow_up_run.output_text)") == 1
    assert text.count("print(final_run.output_text)") == 1
    assert result.report.status == "migrated"
    assert result.report.summary["manual_actions"] == 0
    assert result.report.score.overall == 100


@pytest.mark.parametrize(
    ("source_form", "responses_import", "call"),
    [
        (
            "list",
            "",
            """evaluation_id = evaluate_data_agent(
    evaluation_request_df,
    data_agent_name="Revenue Agent",
    workspace_name=workspace_name,
    critic_prompt=critic_prompt,
    table_name="evaluation_results",
    data_agent_stage="production",
)
""",
        ),
        (
            "string",
            "from fabric.dataagent.client import FabricOpenAIResponses",
            (
                "evaluation_id = evaluate_data_agent(evaluation_request_df, "
                "data_agent_name='Revenue Agent', workspace_name=workspace_name)"
            ),
        ),
    ],
    ids=["multiline-add-import", "inline-reuse-prior-import"],
)
def test_evaluation_only_notebook_matrix(
    source_form: str, responses_import: str, call: str
) -> None:
    cells: list[CellSpec] = [
        ("markdown", "# Evaluate representative Data Agent questions"),
        ("code", "%pip install -U fabric-data-agent-sdk pandas --quiet"),
    ]
    if responses_import:
        cells.append(("code", responses_import))
    cells.extend(
        [
            (
                "code",
                """import pandas as pd
from fabric.dataagent.evaluation import evaluate_data_agent

workspace_name = "Contoso Analytics"
critic_prompt = "Score factual accuracy and relevance."
evaluation_request_df = pd.DataFrame(
    {
        "question": [
            "What was quarterly revenue?",
            "Which region grew fastest?",
        ]
    }
)
""",
            ),
            ("code", call),
            ("markdown", "Review the evaluation and step tables before cutover."),
        ]
    )
    data = notebook_bytes(cells, source_form=source_form)

    result = MigrationEngine().migrate(data, "evaluation-only.ipynb")
    _, migrated = assert_valid_preserving_structure(data, result.notebook_bytes)
    text = code_text(migrated)

    assert (
        text.count(
            "from fabric.dataagent.client import FabricOpenAIResponses"
        )
        == 1
    )
    assert "client_class=FabricOpenAIResponses" in text
    assert "FabricOpenAIResponses(" not in text
    assert '"fabric-data-agent-sdk>=0.1.28a0"' in text
    assert result.report.status == "migrated_with_warnings"
    assert {finding.rule_id for finding in result.report.changes} >= {
        "EVAL-CLIENT-001",
        "QUERY-SDK-001",
    }
    assert any(
        warning.rule_id == "RUNTIME-SDK-RESTART-001"
        for warning in result.report.warnings
    )
    assert result.report.summary["manual_actions"] == 0
    assert result.report.score.overall == 100


def test_existing_evaluation_client_class_gets_missing_required_import() -> None:
    cells: list[CellSpec] = [
        ("markdown", "# Existing Responses evaluation configuration"),
        (
            "code",
            """from fabric.dataagent.evaluation import evaluate_data_agent
import pandas as pd

evaluation_request_df = pd.DataFrame(
    {"question": ["Summarize revenue by region."]}
)
""",
        ),
        (
            "code",
            """evaluation_id = evaluate_data_agent(
    evaluation_request_df,
    data_agent_name="Revenue Agent",
    workspace_name="Contoso Analytics",
    client_class=FabricOpenAIResponses,
)
""",
        ),
    ]
    data = notebook_bytes(cells)

    result = MigrationEngine().migrate(data, "missing-evaluation-import.ipynb")
    _, migrated = assert_valid_preserving_structure(data, result.notebook_bytes)
    text = code_text(migrated)

    assert (
        text.count(
            "from fabric.dataagent.client import FabricOpenAIResponses"
        )
        == 1
    )
    assert text.count("client_class=FabricOpenAIResponses") == 1
    # The fixture has no Fabric metadata, so the runtime-scope warning applies.
    assert result.report.status == "migrated_with_warnings"
    assert {finding.rule_id for finding in result.report.warnings} == {
        "RUNTIME-EVAL-SCOPE-001"
    }
    assert {finding.rule_id for finding in result.report.changes} == {
        "EVAL-IMPORT-001"
    }


@pytest.mark.parametrize("source_form", ["list", "string"])
def test_fully_migrated_evaluation_notebook_is_byte_identical(
    source_form: str,
) -> None:
    cells: list[CellSpec] = [
        ("markdown", "# Responses evaluation notebook"),
        (
            "code",
            '%pip install -U "fabric-data-agent-sdk>=0.1.28a0" pandas --quiet',
        ),
        (
            "code",
            """from fabric.dataagent.client import FabricOpenAIResponses
from fabric.dataagent.evaluation import evaluate_data_agent
import pandas as pd
""",
        ),
        (
            "code",
            """evaluation_request_df = pd.DataFrame(
    {"question": ["Summarize quarterly revenue."]}
)
evaluation_id = evaluate_data_agent(
    evaluation_request_df,
    data_agent_name="Revenue Agent",
    workspace_name="Contoso Analytics",
    client_class=FabricOpenAIResponses,
)
""",
        ),
    ]
    data = notebook_bytes(cells, source_form=source_form)

    result = MigrationEngine().migrate(data, "already-migrated-evaluation.ipynb")

    assert result.notebook_bytes == data
    read_notebook(result.notebook_bytes)
    assert result.report.status == "no_migration_needed"
    assert result.report.summary["cells_changed"] == 0
    assert result.report.summary["rules_applied"] == 0
    assert result.report.score.overall == 100


@pytest.mark.parametrize(
    ("name", "source_form", "cells"),
    [
        (
            "management-plane",
            "list",
            [
                ("markdown", "# Provision and publish a Fabric Data Agent"),
                ("code", "%pip install fabric-data-agent-sdk"),
                (
                    "code",
                    """from fabric.dataagent.client import FabricDataAgentManagement

workspace_name = "Contoso Analytics"
agent_name = "Revenue Agent"
manager = FabricDataAgentManagement(workspace_name=workspace_name)
agent = manager.create_data_agent(agent_name)
agent.add_lakehouse("Finance Lakehouse")
agent.publish()
""",
                ),
                (
                    "code",
                    "# FabricOpenAI is mentioned only in migration documentation.\n"
                    "print('Published management-plane artifact')",
                ),
            ],
        ),
        (
            "unrelated-analytics",
            "string",
            [
                ("markdown", "# Unrelated notebook\nNo Data Agent query calls."),
                ("code", "%pip install pandas"),
                (
                    "code",
                    """import pandas as pd

sales = pd.DataFrame(
    {"region": ["North", "South"], "revenue": [120, 95]}
)
summary = sales.groupby("region", as_index=False)["revenue"].sum()
summary
""",
                ),
                ("code", "%%sql\nSELECT region, SUM(revenue) FROM sales GROUP BY region"),
            ],
        ),
    ],
    ids=["management-plane-only", "unrelated-analytics"],
)
def test_no_op_notebook_matrix_is_byte_identical(
    name: str, source_form: str, cells: list[CellSpec]
) -> None:
    data = notebook_bytes(
        cells,
        source_form=source_form,
        metadata={"scenario": name, "preserve": ["metadata", "outputs"]},
    )

    result = MigrationEngine().migrate(data, f"{name}.ipynb")

    assert result.notebook_bytes == data
    read_notebook(result.notebook_bytes)
    assert result.report.source_notebook == f"{name}.ipynb"
    assert result.report.status == "no_migration_needed"
    assert result.report.summary == {
        "cells_scanned": len(cells),
        "cells_changed": 0,
        "rules_applied": 0,
        "warnings": 0,
        "manual_actions": 0,
    }
    assert result.report.score.overall == 100


def test_unrelated_cell_magics_are_preserved_during_query_migration() -> None:
    cells: list[CellSpec] = [
        ("markdown", "# Query notebook with Fabric magics"),
        ("code", "%%sql\nSELECT TOP 10 * FROM demo.revenue"),
        ("code", "%%pyspark\nprint('unrelated Spark setup')"),
        ("code", "%matplotlib inline\n# Visualization setup"),
        (
            "code",
            """from fabric.dataagent.client import FabricOpenAI
workspace_name = "Contoso Analytics"
client = FabricOpenAI(
    artifact_name="Revenue Agent",
    workspace_name=workspace_name,
    ai_skill_stage="production",
)
assistant = client.beta.assistants.create(model="gpt-4o")
thread = client.beta.threads.create()
""",
        ),
        ("code", 'question = "What was revenue last quarter?"'),
    ]
    cells.extend(turn_cells("question", "run", "messages"))
    data = notebook_bytes(cells)

    result = MigrationEngine().migrate(data, "magic-and-query.ipynb")
    before, migrated = assert_valid_preserving_structure(
        data, result.notebook_bytes
    )

    assert migrated.cells[0] == before.cells[0]
    assert migrated.cells[1] == before.cells[1]
    assert migrated.cells[2] == before.cells[2]
    assert migrated.cells[3] == before.cells[3]
    assert "run = client.responses.create(input=question)" in code_text(migrated)
    assert result.report.status == "migrated"
    assert result.report.summary["manual_actions"] == 0


def test_relevant_cell_magic_blocks_realistic_flow_atomically() -> None:
    cells: list[CellSpec] = [
        ("markdown", "# Timed legacy query flow"),
        (
            "code",
            """%%time
from fabric.dataagent.client import FabricOpenAI
client = FabricOpenAI(artifact_name="Revenue Agent")
assistant = client.beta.assistants.create(model="gpt-4o")
thread = client.beta.threads.create()
""",
        ),
        ("code", 'question = "Summarize revenue by region."'),
    ]
    cells.extend(turn_cells("question", "run", "messages"))
    data = notebook_bytes(cells, source_form="list")

    result = MigrationEngine().migrate(data, "timed-legacy-query.ipynb")

    assert result.notebook_bytes == data
    read_notebook(result.notebook_bytes)
    assert result.report.status == "partial_migration"
    assert result.report.summary["cells_changed"] == 0
    assert result.report.score.overall == 0
    assert {
        finding.rule_id for finding in result.report.manual_actions
    } >= {"UNSUPPORTED-SYNTAX-001", "SAFE-ATOMIC-001"}
