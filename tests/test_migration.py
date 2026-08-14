from __future__ import annotations

import json

import nbformat
import pytest

from fabric_migrator import MigrationEngine
from fabric_migrator.notebook_io import NotebookLimits, NotebookValidationError


def notebook_bytes(*sources: str) -> bytes:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(source) for source in sources]
    return nbformat.writes(notebook).encode()


def migrated_sources(result) -> list[str]:
    notebook = nbformat.reads(result.notebook_bytes.decode(), as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


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


def test_management_notebook_is_unchanged() -> None:
    data = notebook_bytes(
        "from fabric.dataagent.client import FabricDataAgentManagement\n"
        'agent = FabricDataAgentManagement("sales")\n'
        "agent.publish()"
    )
    result = MigrationEngine().migrate(data, "management.ipynb")
    assert result.notebook_bytes == data
    assert result.report.status == "no_migration_needed"
    assert result.report.summary["cells_changed"] == 0


def test_unrelated_notebook_content_is_byte_identical() -> None:
    notebook = nbformat.v4.new_notebook(metadata={"owner": "keep"})
    code = nbformat.v4.new_code_cell(
        "%pip install pandas\n"
        "from fabric.dataagent.client import FabricDataAgentManagement\n"
        "agent = FabricDataAgentManagement('sales')\n"
        "agent.publish()\n"
        "print('FabricOpenAI is mentioned as text only')"
    )
    code.outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="keep this output\n")
    ]
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "Documentation may mention `FabricOpenAI` without using it."
        ),
        code,
    ]
    data = nbformat.writes(notebook).encode()
    result = MigrationEngine().migrate(data, "unrelated.ipynb")
    assert result.notebook_bytes == data
    assert result.report.status == "no_migration_needed"
    assert result.report.summary["cells_changed"] == 0


def test_single_question_flow_migrates_end_to_end() -> None:
    result = MigrationEngine().migrate(
        notebook_bytes(
            "%pip install fabric-data-agent-sdk",
            BASE_SETUP,
            QUESTION,
            """import time
while run.status in ("queued", "in_progress"):
    run = fabric_client.beta.threads.runs.retrieve(
        thread_id=thread.id, run_id=run.id
    )
    time.sleep(2)
""",
            """response = fabric_client.beta.threads.messages.list(
    thread_id=thread.id, order="asc"
)
""",
            """def pretty_print(messages):
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
""",
            "pretty_print(response)",
            "fabric_client.beta.threads.delete(thread.id)",
        ),
        "single.ipynb",
    )
    text = "\n".join(migrated_sources(result))
    assert "FabricOpenAIResponses" in text
    assert "workspace_name=workspace_name" in text
    assert 'ai_skill_stage="sandbox"' in text
    assert "fabric_client.responses.create(input=question)" in text
    assert "fabric_client.responses.retrieve(run.id)" in text
    assert "print(messages.output_text)" in text
    assert ".beta." not in text
    assert "fabric-data-agent-sdk>=0.1.28a0" in text
    assert result.report.status == "migrated_with_warnings"
    assert any(item.rule_id == "QUERY-STAGE-001" for item in result.report.warnings)


def test_two_turn_flow_uses_previous_response_id() -> None:
    result = MigrationEngine().migrate(
        notebook_bytes(BASE_SETUP, QUESTION, QUESTION.replace("run =", "follow_up =")),
        "follow-up.ipynb",
    )
    text = "\n".join(migrated_sources(result))
    assert "run = fabric_client.responses.create(input=question)" in text
    assert (
        "follow_up = fabric_client.responses.create("
        "input=question, previous_response_id=run.id)"
    ) in text


def test_each_multi_turn_read_uses_nearest_preceding_run() -> None:
    second_question = QUESTION.replace("run =", "follow_up_run =")
    result = MigrationEngine().migrate(
        notebook_bytes(
            BASE_SETUP,
            QUESTION,
            (
                "first_messages = fabric_client.beta.threads.messages.list("
                'thread_id=thread.id, order="asc")'
            ),
            """for m in first_messages:
    print(f"{m.role}: {m.content[0].text.value}")
""",
            second_question,
            (
                "follow_up_messages = fabric_client.beta.threads.messages.list("
                'thread_id=thread.id, order="asc")'
            ),
            """for m in follow_up_messages:
    print(f"{m.role}: {m.content[0].text.value}")
""",
        ),
        "multi-turn-reads.ipynb",
    )
    text = "\n".join(migrated_sources(result))
    assert "first_messages = run" in text
    assert "print(run.output_text)" in text
    assert "follow_up_messages = follow_up_run" in text
    assert "print(follow_up_run.output_text)" in text
    assert "first_messages = follow_up_run" not in text


def test_aliased_import_preserves_local_name() -> None:
    setup = """from fabric.dataagent.client import FabricOpenAI as Client
fabric_client = Client(artifact_name=data_agent_name)
assistant = fabric_client.beta.assistants.create(model="gpt-4o")
thread = fabric_client.beta.threads.create()
"""
    result = MigrationEngine().migrate(notebook_bytes(setup, QUESTION), "alias.ipynb")
    text = "\n".join(migrated_sources(result))
    assert "FabricOpenAIResponses as Client" in text
    assert "fabric_client = Client(" in text


def test_unknown_beta_call_blocks_all_query_edits() -> None:
    data = notebook_bytes(
        BASE_SETUP,
        QUESTION,
        "fabric_client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)",
    )
    result = MigrationEngine().migrate(data, "unknown.ipynb")
    assert result.notebook_bytes == data
    assert result.report.status == "partial_migration"
    assert any(
        finding.rule_id == "UNSUPPORTED-BETA-001"
        for finding in result.report.manual_actions
    )


def test_unknown_thread_dependency_blocks_all_query_edits() -> None:
    data = notebook_bytes(
        BASE_SETUP,
        QUESTION,
        "audit_external_thread(thread.id)",
    )
    result = MigrationEngine().migrate(data, "thread-dependency.ipynb")
    assert result.notebook_bytes == data
    assert any(
        finding.rule_id == "UNSUPPORTED-REFERENCE-001"
        for finding in result.report.manual_actions
    )


def test_custom_message_helper_blocks_destructive_rewrite() -> None:
    data = notebook_bytes(
        BASE_SETUP,
        QUESTION,
        "response = fabric_client.beta.threads.messages.list(thread_id=thread.id)",
        """def save_messages(messages):
    for m in messages:
        database.save(m.content[0].text.value)
""",
        "save_messages(response)",
    )
    result = MigrationEngine().migrate(data, "custom-helper.ipynb")
    assert result.notebook_bytes == data
    assert result.report.status == "partial_migration"


def test_official_inline_message_loop_migrates() -> None:
    result = MigrationEngine().migrate(
        notebook_bytes(
            BASE_SETUP,
            QUESTION,
            (
                "messages = fabric_client.beta.threads.messages.list("
                'thread_id=thread.id, order="asc")'
            ),
            """for m in messages:
    print(f"{m.role}: {m.content[0].text.value}")
""",
        ),
        "inline.ipynb",
    )
    text = "\n".join(migrated_sources(result))
    assert "messages = run" in text
    assert "print(run.output_text)" in text
    assert ".content[0].text.value" not in text
    assert result.report.summary["manual_actions"] == 0


def test_direct_message_list_loop_migrates_without_invalid_assignment() -> None:
    result = MigrationEngine().migrate(
        notebook_bytes(
            BASE_SETUP,
            QUESTION,
            """for m in fabric_client.beta.threads.messages.list(
    thread_id=thread.id, order="asc"
):
    print(f"{m.role}: {m.content[0].text.value}")
""",
        ),
        "direct-inline.ipynb",
    )
    text = "\n".join(migrated_sources(result))
    assert "print(run.output_text)" in text
    assert "None =" not in text
    compile(text, "direct-inline.ipynb", "exec")


def test_unassigned_step_list_is_blocked_not_rewritten() -> None:
    data = notebook_bytes(
        BASE_SETUP,
        QUESTION,
        """for step in fabric_client.beta.threads.runs.steps.list(
    thread_id=thread.id, run_id=run.id
):
    print(step)
""",
    )
    result = MigrationEngine().migrate(data, "direct-steps.ipynb")
    assert result.notebook_bytes == data
    assert any(
        finding.rule_id == "UNSUPPORTED-SHAPE-001"
        for finding in result.report.manual_actions
    )


@pytest.mark.parametrize(
    "extra",
    [
        'assistant = fabric_client.beta.assistants.create(model="gpt-4o", tools=["x"])',
        'thread = fabric_client.beta.threads.create(messages=[{"role": "user"}])',
        (
            "fabric_client.beta.threads.messages.create("
            'thread_id=thread.id, role="user", content=question, attachments=[])'
        ),
        (
            "run = fabric_client.beta.threads.runs.create("
            "thread_id=thread.id, assistant_id=assistant.id, instructions='custom')"
        ),
    ],
)
def test_custom_api_options_block_migration(extra: str) -> None:
    source = BASE_SETUP + "\n" + extra
    result = MigrationEngine().migrate(
        notebook_bytes(source, QUESTION), "custom-options.ipynb"
    )
    assert result.report.status == "partial_migration"
    assert any(
        finding.rule_id == "UNSUPPORTED-SHAPE-001"
        for finding in result.report.manual_actions
    )


def test_custom_client_constructor_options_block_migration() -> None:
    setup = BASE_SETUP.replace(
        "FabricOpenAI(artifact_name=data_agent_name)",
        "FabricOpenAI(artifact_name=data_agent_name, timeout=30)",
    )
    data = notebook_bytes(setup, QUESTION)
    result = MigrationEngine().migrate(data, "constructor.ipynb")
    assert result.notebook_bytes == data
    assert any(
        finding.rule_id == "UNSUPPORTED-CONSTRUCTOR-001"
        for finding in result.report.manual_actions
    )


def test_nonstandard_polling_condition_blocks_migration() -> None:
    data = notebook_bytes(
        BASE_SETUP,
        QUESTION,
        """while run.status != "completed":
    run = fabric_client.beta.threads.runs.retrieve(
        thread_id=thread.id, run_id=run.id
    )
""",
    )
    result = MigrationEngine().migrate(data, "polling.ipynb")
    assert result.notebook_bytes == data
    assert any(
        finding.rule_id == "UNSUPPORTED-SHAPE-001"
        for finding in result.report.manual_actions
    )


def test_reassigned_thread_symbol_blocks_pairing() -> None:
    setup = BASE_SETUP + "\nthread = fabric_client.beta.threads.create()\n"
    data = notebook_bytes(setup, QUESTION)
    result = MigrationEngine().migrate(data, "reassigned.ipynb")
    assert result.notebook_bytes == data
    assert any(
        finding.rule_id == "UNSUPPORTED-REASSIGNMENT-001"
        for finding in result.report.manual_actions
    )


def test_module_qualified_client_import_is_flagged() -> None:
    setup = """import fabric.dataagent.client as dataagent
fabric_client = dataagent.FabricOpenAI(artifact_name=data_agent_name)
assistant = fabric_client.beta.assistants.create(model="gpt-4o")
thread = fabric_client.beta.threads.create()
"""
    data = notebook_bytes(setup, QUESTION)
    result = MigrationEngine().migrate(data, "qualified.ipynb")
    assert result.notebook_bytes == data
    assert any(
        finding.rule_id == "UNSUPPORTED-CLIENT-001"
        for finding in result.report.manual_actions
    )


@pytest.mark.parametrize(
    "source,rule",
    [
        (
            BASE_SETUP + '\nmethod = getattr(fabric_client.beta.threads, "create")',
            "UNSUPPORTED-DYNAMIC-001",
        ),
        (
            BASE_SETUP + "\nasync def ask():\n    await client.run()",
            "UNSUPPORTED-ASYNC-001",
        ),
        (
            BASE_SETUP + '\njson.dump(thread_id, open("state.json", "w"))',
            "UNSUPPORTED-PERSISTED-001",
        ),
    ],
)
def test_unsafe_patterns_are_preserved(source: str, rule: str) -> None:
    data = notebook_bytes(source, QUESTION)
    result = MigrationEngine().migrate(data, "unsafe.ipynb")
    assert result.notebook_bytes == data
    assert any(item.rule_id == rule for item in result.report.manual_actions)


def test_evaluation_only_adds_responses_client() -> None:
    result = MigrationEngine().migrate(
        notebook_bytes(
            "from fabric.dataagent.evaluation import evaluate_data_agent\n"
            "evaluation_id = evaluate_data_agent(df, data_agent_name='sales')"
        ),
        "evaluation.ipynb",
    )
    text = migrated_sources(result)[0]
    assert "from fabric.dataagent.client import FabricOpenAIResponses" in text
    assert "client_class=FabricOpenAIResponses" in text
    assert any(
        change.rule_id == "EVAL-IMPORT-001" for change in result.report.changes
    )


def test_evaluation_migration_updates_sdk_requirement_and_keeps_other_packages() -> None:
    result = MigrationEngine().migrate(
        notebook_bytes(
            "%pip install -U fabric-data-agent-sdk openai --q",
            """from fabric.dataagent.evaluation import evaluate_data_agent
evaluate_data_agent(
    evaluation_request_df,
    data_agent_name,
    workspace_name=workspace_name,
    critic_prompt=sdk_critic_prompt,
    table_name=result_table_name,
    data_agent_stage=data_agent_stage,
)
""",
        ),
        "evaluation-full.ipynb",
    )
    sources = migrated_sources(result)
    assert (
        '%pip install -U "fabric-data-agent-sdk>=0.1.28a0" openai --q'
        in sources[0]
    )
    assert "\n\n    client_class" not in sources[1]
    assert "    client_class=FabricOpenAIResponses,\n)" in sources[1]


def test_sdk_install_then_evaluation_import_warns_to_restart_python() -> None:
    data = notebook_bytes(
        "%pip install -U fabric-data-agent-sdk",
        "from fabric.dataagent.evaluation import evaluate_data_agent",
    )
    result = MigrationEngine().migrate(data, "evaluation-import.ipynb")
    assert result.notebook_bytes == data
    assert result.report.status == "no_migration_needed"
    assert any(
        warning.rule_id == "RUNTIME-SDK-RESTART-001"
        for warning in result.report.warnings
    )
    assert "Restart Python after the SDK install" in result.report.user_test_checklist[0]


def test_saved_fabric_openai_import_error_is_detected() -> None:
    notebook = nbformat.v4.new_notebook()
    install = nbformat.v4.new_code_cell("%pip install -U fabric-data-agent-sdk")
    import_cell = nbformat.v4.new_code_cell(
        "from fabric.dataagent.evaluation import evaluate_data_agent"
    )
    import_cell.outputs = [
        nbformat.v4.new_output(
            "error",
            ename="ImportError",
            evalue=(
                "cannot import name 'FabricOpenAI' from "
                "'fabric.dataagent.client' (/nfs/env/client/__init__.py)"
            ),
            traceback=[
                "ImportError: cannot import name 'FabricOpenAI' from "
                "'fabric.dataagent.client'"
            ],
        )
    ]
    notebook.cells = [install, import_cell]
    data = nbformat.writes(notebook).encode()
    result = MigrationEngine().migrate(data, "failed-import.ipynb")
    warning = next(
        item
        for item in result.report.warnings
        if item.rule_id == "RUNTIME-SDK-RESTART-001"
    )
    assert warning.cell_index == 1
    assert "saved output contains" in warning.message
    assert result.notebook_bytes == data


def test_existing_evaluation_client_is_idempotent() -> None:
    data = notebook_bytes(
        "from fabric.dataagent.client import FabricOpenAIResponses\n"
        "from fabric.dataagent.evaluation import evaluate_data_agent\n"
        "evaluate_data_agent(df, client_class=FabricOpenAIResponses)"
    )
    result = MigrationEngine().migrate(data, "done.ipynb")
    assert result.notebook_bytes == data
    assert result.report.status == "no_migration_needed"


def test_migration_is_idempotent() -> None:
    engine = MigrationEngine()
    first = engine.migrate(notebook_bytes(BASE_SETUP, QUESTION), "first.ipynb")
    second = engine.migrate(first.notebook_bytes, "second.ipynb")
    assert second.notebook_bytes == first.notebook_bytes
    assert second.report.status == "no_migration_needed"


def test_scores_are_conservative_and_preservation_is_measured() -> None:
    no_op = MigrationEngine().migrate(notebook_bytes("x = 1"), "noop.ipynb")
    migrated = MigrationEngine().migrate(
        notebook_bytes(BASE_SETUP, QUESTION), "migrated.ipynb"
    )
    blocked = MigrationEngine().migrate(
        notebook_bytes(
            BASE_SETUP,
            QUESTION,
            "fabric_client.beta.threads.runs.cancel("
            "thread_id=thread.id, run_id=run.id)",
        ),
        "blocked.ipynb",
    )
    assert no_op.report.score.overall == 100
    assert migrated.report.score.preservation_confidence == 100
    assert 0 < migrated.report.score.overall < 100
    assert blocked.report.score.overall == 0


def test_invalid_generated_code_rolls_back_all_changes(monkeypatch) -> None:
    engine = MigrationEngine()
    monkeypatch.setattr(
        engine,
        "_constructor_replacement",
        lambda cell, call, local, definitions: "(",
    )
    data = notebook_bytes(BASE_SETUP, QUESTION)
    result = engine.migrate(data, "rollback.ipynb")
    assert result.notebook_bytes == data
    assert result.report.summary["cells_changed"] == 0
    assert result.report.score.overall == 0
    assert any(
        finding.rule_id == "OUTPUT-SYNTAX-001"
        for finding in result.report.manual_actions
    )


def test_outputs_metadata_and_markdown_are_preserved() -> None:
    notebook = nbformat.v4.new_notebook(metadata={"custom": {"keep": True}})
    code = nbformat.v4.new_code_cell(BASE_SETUP)
    code.metadata["tag"] = "keep"
    code.outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="keep output\n")
    ]
    markdown = nbformat.v4.new_markdown_cell("Keep **all** markdown.")
    notebook.cells = [markdown, code, nbformat.v4.new_code_cell(QUESTION)]
    result = MigrationEngine().migrate(nbformat.writes(notebook).encode(), "meta.ipynb")
    migrated = nbformat.reads(result.notebook_bytes.decode(), as_version=4)
    assert migrated.metadata["custom"]["keep"] is True
    assert migrated.cells[0] == markdown
    assert migrated.cells[1].metadata["tag"] == "keep"
    assert migrated.cells[1].outputs == code.outputs


def test_secret_values_are_redacted_from_report() -> None:
    secret = "super-secret-value"
    source = BASE_SETUP + f'\napi_key = "{secret}"'
    result = MigrationEngine().migrate(
        notebook_bytes(source, QUESTION), "secret.ipynb"
    )
    report_text = json.dumps(result.report.to_dict())
    assert secret not in report_text
    assert "<redacted>" not in "\n".join(migrated_sources(result))


def test_cell_magic_blocks_relevant_migration() -> None:
    data = notebook_bytes("%%time\n" + BASE_SETUP, QUESTION)
    result = MigrationEngine().migrate(data, "magic.ipynb")
    assert result.notebook_bytes == data
    assert result.report.status == "partial_migration"


def test_invalid_json_and_limits_fail_cleanly() -> None:
    engine = MigrationEngine(NotebookLimits(max_bytes=30, max_cells=1))
    with pytest.raises(NotebookValidationError):
        engine.migrate(b"not json", "bad.ipynb")
    with pytest.raises(NotebookValidationError):
        engine.migrate(notebook_bytes("x = 1", "y = 2"), "large.ipynb")


def test_unicode_before_api_call_keeps_source_offsets_correct() -> None:
    setup = '# café 🚀\n' + BASE_SETUP
    result = MigrationEngine().migrate(notebook_bytes(setup, QUESTION), "unicode.ipynb")
    text = "\n".join(migrated_sources(result))
    assert "# café 🚀" in text
    assert "FabricOpenAIResponses" in text


def test_comments_and_strings_are_not_globally_replaced() -> None:
    setup = (
        '# Keep the words FabricOpenAI and beta.threads in this comment.\n'
        'example = "FabricOpenAI beta.threads"\n'
        + BASE_SETUP
    )
    result = MigrationEngine().migrate(notebook_bytes(setup, QUESTION), "strings.ipynb")
    text = "\n".join(migrated_sources(result))
    assert "# Keep the words FabricOpenAI and beta.threads in this comment." in text
    assert 'example = "FabricOpenAI beta.threads"' in text
