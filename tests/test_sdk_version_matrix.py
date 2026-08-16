from __future__ import annotations

import ast

import nbformat
import pytest

from fabric_migrator import MigrationEngine
from fabric_migrator.mapping import SDK_MAPPING


RELEASE_MATRIX = {
    "0.1.23a0": {
        "version_specific_change": "fewshot-and-evaluation-fixes",
        "documented_responses_contract": False,
        "normalized_nonstreamed_response": False,
    },
    "0.1.26a0": {
        "version_specific_change": "public-management-plane",
        "documented_responses_contract": False,
        "normalized_nonstreamed_response": False,
    },
    "0.1.27a0": {
        "version_specific_change": "responses-api-support",
        "documented_responses_contract": True,
        "normalized_nonstreamed_response": False,
    },
    "0.1.28a0": {
        "version_specific_change": "nonstreamed-response-normalization",
        "documented_responses_contract": True,
        "normalized_nonstreamed_response": True,
    },
    "0.1.29a0": {
        "version_specific_change": "aiohttp-dependency-only",
        "documented_responses_contract": True,
        "normalized_nonstreamed_response": True,
    },
}

OLD_SETUP = """from fabric.dataagent.client import FabricOpenAI
client = FabricOpenAI(
    artifact_name=data_agent_name,
    workspace_name=workspace_name,
    ai_skill_stage="sandbox",
)
assistant = client.beta.assistants.create(model="gpt-4o")
thread = client.beta.threads.create()
"""

OLD_QUERY = """client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=question,
)
response = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
"""

EVALUATION = """from fabric.dataagent.evaluation import evaluate_data_agent
result = evaluate_data_agent(
    evaluation_request_df,
    data_agent_name=data_agent_name,
    workspace_name=workspace_name,
    data_agent_stage="sandbox",
    max_workers=1,
)
"""


def notebook_bytes(*sources: str) -> bytes:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(source) for source in sources]
    return nbformat.writes(notebook).encode()


def migrated_sources(result) -> list[str]:
    notebook = nbformat.reads(result.notebook_bytes.decode(), as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def evaluation_migration(install_line: str):
    return MigrationEngine().migrate(
        notebook_bytes(install_line, EVALUATION),
        "sdk-requirement.ipynb",
    )


def find_call(source: str, dotted_name: str) -> ast.Call:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts: list[str] = []
        current = node.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        if ".".join(reversed(parts)) == dotted_name:
            return node
    raise AssertionError(f"Call {dotted_name!r} was not generated")


def test_release_matrix_records_the_verified_responses_evolution() -> None:
    assert set(SDK_MAPPING["release_evidence"]) == set(RELEASE_MATRIX)
    assert SDK_MAPPING["observed_current_release"] == "0.1.29a0"
    assert SDK_MAPPING["minimum_recommended_release"] == "0.1.28a0"
    normalized_releases = [
        version
        for version, facts in RELEASE_MATRIX.items()
        if facts["normalized_nonstreamed_response"]
    ]
    assert normalized_releases == ["0.1.28a0", "0.1.29a0"]
    assert "does not announce Responses support" in SDK_MAPPING["release_evidence"][
        "0.1.23a0"
    ]
    assert "class-based evaluation" in SDK_MAPPING["release_evidence"]["0.1.27a0"]
    assert "responses.retrieve" in SDK_MAPPING["release_evidence"]["0.1.28a0"]


@pytest.mark.parametrize(
    ("install_line", "expected"),
    [
        (
            "%pip install fabric-data-agent-sdk",
            '%pip install "fabric-data-agent-sdk>=0.1.28a0"',
        ),
        (
            "%pip install fabric-data-agent-sdk==0.1.23a0",
            '%pip install "fabric-data-agent-sdk>=0.1.28a0"',
        ),
        (
            "%pip install fabric-data-agent-sdk>=0.1.27a0,<1",
            '%pip install "fabric-data-agent-sdk>=0.1.28a0,<1"',
        ),
        (
            "%pip install 'fabric-data-agent-sdk==0.1.26a0'",
            "%pip install 'fabric-data-agent-sdk>=0.1.28a0'",
        ),
        (
            '%pip install "fabric-data-agent-sdk>=0.1.27a0,<0.2" pandas',
            '%pip install "fabric-data-agent-sdk>=0.1.28a0,<0.2" pandas',
        ),
        (
            "%pip install -U pandas fabric-data-agent-sdk==0.1.27a0 --quiet",
            '%pip install -U pandas "fabric-data-agent-sdk>=0.1.28a0" --quiet',
        ),
    ],
)
def test_insufficient_sdk_requirements_are_surgically_raised(
    install_line: str, expected: str
) -> None:
    sources = migrated_sources(evaluation_migration(install_line))
    assert sources[0] == expected
    assert sources[0].replace(
        "fabric-data-agent-sdk>=0.1.28a0", ""
    ) == expected.replace("fabric-data-agent-sdk>=0.1.28a0", "")


@pytest.mark.parametrize(
    "install_line",
    [
        "%pip install fabric-data-agent-sdk==0.1.28a0",
        '%pip install "fabric-data-agent-sdk==0.1.29a0"',
        "%pip install 'fabric-data-agent-sdk>=0.1.28a0,<1' pandas --quiet",
        "%pip install fabric-data-agent-sdk>=0.1.29a0",
    ],
)
def test_sufficient_sdk_requirements_are_left_unchanged(install_line: str) -> None:
    sources = migrated_sources(evaluation_migration(install_line))
    assert sources[0] == install_line


def test_unrelated_install_line_is_left_unchanged() -> None:
    install_line = "%pip install -U pandas openai --quiet"
    sources = migrated_sources(evaluation_migration(install_line))
    assert sources[0] == install_line


def test_official_constructor_and_responses_create_shape_is_generated() -> None:
    result = MigrationEngine().migrate(
        notebook_bytes(OLD_SETUP, OLD_QUERY),
        "official-query-shape.ipynb",
    )
    source = "\n".join(migrated_sources(result))
    constructor = find_call(source, "FabricOpenAIResponses")
    create = find_call(source, "client.responses.create")

    assert {keyword.arg for keyword in constructor.keywords} == {
        "artifact_name",
        "workspace_name",
        "ai_skill_stage",
    }
    assert len(create.args) == 0
    assert [keyword.arg for keyword in create.keywords] == ["input"]
    assert isinstance(create.keywords[0].value, ast.Name)
    assert create.keywords[0].value.id == "question"


def test_evaluation_receives_the_exported_class_not_an_instance() -> None:
    source = migrated_sources(
        MigrationEngine().migrate(
            notebook_bytes(EVALUATION),
            "official-evaluation-shape.ipynb",
        )
    )[0]
    call = find_call(source, "evaluate_data_agent")
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}

    assert {"workspace_name", "data_agent_stage", "max_workers"} <= keywords.keys()
    assert isinstance(keywords["client_class"], ast.Name)
    assert keywords["client_class"].id == "FabricOpenAIResponses"
    assert not isinstance(keywords["client_class"], ast.Call)
    assert (
        "from fabric.dataagent.client import FabricOpenAIResponses"
        in source
    )


@pytest.mark.parametrize(
    ("setup", "query", "rule"),
    [
        (
            OLD_SETUP.replace(
                'ai_skill_stage="sandbox",',
                'ai_skill_stage="sandbox",\n    timeout=30,',
            ),
            OLD_QUERY,
            "UNSUPPORTED-CONSTRUCTOR-001",
        ),
        (
            OLD_SETUP.replace(
                'assistant = client.beta.assistants.create(model="gpt-4o")',
                (
                    'assistant = client.beta.assistants.create('
                    'model="gpt-4o", tools=["code_interpreter"])'
                ),
            ),
            OLD_QUERY,
            "UNSUPPORTED-SHAPE-001",
        ),
        (
            OLD_SETUP,
            OLD_QUERY.replace("content=question,", "content=question,\n    attachments=[],"),
            "UNSUPPORTED-SHAPE-001",
        ),
        (
            OLD_SETUP,
            OLD_QUERY.replace(
                "assistant_id=assistant.id,",
                'assistant_id=assistant.id,\n    instructions="custom",',
            ),
            "UNSUPPORTED-SHAPE-001",
        ),
    ],
)
def test_unsupported_historical_options_fail_closed(
    setup: str, query: str, rule: str
) -> None:
    original = notebook_bytes(setup, query)
    result = MigrationEngine().migrate(original, "unsupported-option.ipynb")

    assert result.notebook_bytes == original
    assert any(item.rule_id == rule for item in result.report.manual_actions)
