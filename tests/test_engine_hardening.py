"""Scenario coverage for the guarantees a user relies on.

Every scenario in SCENARIOS is run through the shared invariants in
test_every_scenario_holds_the_core_guarantees, so a new notebook shape only has
to be described once to be checked for preservation, valid output, and
idempotency. The targeted tests below pin down individual rules.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re

import nbformat
import pytest

from fabric_migrator import MigrationEngine
from fabric_migrator.engine import (
    SCORED_RULE_PREFIXES,
    _line_offsets,
    _sanitize_magics,
)


SETUP = """from fabric.dataagent.client import FabricOpenAI
workspace_name = "Contoso Analytics"
fabric_client = FabricOpenAI(
    artifact_name="Revenue Agent",
    workspace_name=workspace_name,
    ai_skill_stage="production",
)
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

POLL = """import time
while run.status in ("queued", "in_progress"):
    run = fabric_client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id,
    )
    time.sleep(1)
"""

READ = """messages = fabric_client.beta.threads.messages.list(
    thread_id=thread.id,
    order="asc",
)
for message in messages:
    print(f"{message.role}: {message.content[0].text.value}")
"""

EVALUATION = """from fabric.dataagent.evaluation import evaluate_data_agent
evaluation_id = evaluate_data_agent(rows, data_agent_name="Revenue Agent")
"""


def notebook_bytes(*sources: str) -> bytes:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(source) for source in sources]
    return nbformat.writes(notebook).encode("utf-8")


def read_notebook(data: bytes):
    notebook = nbformat.reads(data.decode("utf-8"), as_version=4)
    nbformat.validate(notebook)
    return notebook


def code_text(data: bytes) -> str:
    return "\n\n".join(
        cell.source
        for cell in read_notebook(data).cells
        if cell.cell_type == "code"
    )


def finding_ids(result) -> set[str]:
    return {
        finding.rule_id
        for finding in (
            result.report.changes
            + result.report.warnings
            + result.report.manual_actions
        )
    }


SCENARIOS: list[tuple[str, str, list[str]]] = [
    ("single-turn", "migrated", [SETUP, 'question = "Revenue?"', QUESTION, POLL, READ]),
    (
        "multi-turn",
        "migrated",
        [
            SETUP,
            'first = "Revenue?"\nsecond = "Why?"',
            QUESTION.replace("content=question", "content=first"),
            QUESTION.replace("content=question", "content=second").replace(
                "run =", "second_run ="
            ),
        ],
    ),
    (
        "aliased-import",
        "migrated",
        [
            SETUP.replace(
                "import FabricOpenAI\n", "import FabricOpenAI as Agent\n"
            ).replace("= FabricOpenAI(", "= Agent("),
            'question = "Revenue?"',
            QUESTION,
        ],
    ),
    # Warns because a bare synthetic notebook carries no Fabric metadata.
    ("evaluation-only", "migrated_with_warnings", [EVALUATION]),
    (
        "install-and-query",
        "migrated",
        [
            "%pip install fabric-data-agent-sdk",
            SETUP,
            'question = "Revenue?"',
            QUESTION,
        ],
    ),
    (
        "steps-inspection",
        "migrated",
        [
            SETUP,
            'question = "Revenue?"',
            QUESTION,
            "steps = fabric_client.beta.threads.runs.steps.list(\n"
            "    thread_id=thread.id,\n    run_id=run.id,\n)\nsteps.data\n",
        ],
    ),
    (
        "cleanup-and-diagnostics",
        "migrated",
        [
            SETUP + "print(assistant.id)\nprint(thread.id)\n",
            'question = "Revenue?"',
            QUESTION,
            "fabric_client.beta.threads.delete(thread.id)",
        ],
    ),
    (
        "unicode-line-separator",
        "migrated",
        [
            'question = "first second third\x0bfourth\x85fifth"',
            SETUP,
            QUESTION,
        ],
    ),
    (
        "crlf-endings",
        "migrated",
        [SETUP.replace("\n", "\r\n"), 'question = "Revenue?"', QUESTION],
    ),
    (
        "multibyte-before-patch",
        "migrated",
        ['question = "Ünïcödé 雪 🚜 revenue?"', SETUP, QUESTION],
    ),
    ("no-migration-needed", "no_migration_needed", ["import pandas as pd\ndf = pd.DataFrame()"]),
    (
        "blocked-dynamic",
        "partial_migration",
        [SETUP + 'getattr(fabric_client.beta.threads, "delete")', QUESTION],
    ),
    (
        "blocked-cell-magic",
        "partial_migration",
        ["%%time\n" + SETUP, 'question = "Revenue?"', QUESTION],
    ),
    (
        "blocked-unknown-beta",
        "partial_migration",
        [
            SETUP,
            'question = "Revenue?"',
            QUESTION,
            "fabric_client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)",
        ],
    ),
]


@pytest.mark.parametrize(
    ("expected_status", "sources"),
    [(status, sources) for _, status, sources in SCENARIOS],
    ids=[name for name, _, _ in SCENARIOS],
)
def test_every_scenario_holds_the_core_guarantees(
    expected_status: str, sources: list[str]
) -> None:
    data = notebook_bytes(*sources)
    original = read_notebook(data)

    engine = MigrationEngine()
    result = engine.migrate(data, "scenario.ipynb")
    migrated = read_notebook(result.notebook_bytes)

    # Pinned so the invariants below cannot start passing vacuously because a
    # scenario quietly stopped migrating.
    assert result.report.status == expected_status
    if expected_status == "migrated":
        assert result.report.changes
        assert result.report.summary["cells_changed"] > 0

    # Structure outside cell source is preserved exactly.
    assert migrated.metadata == original.metadata
    assert len(migrated.cells) == len(original.cells)
    for before, after in zip(original.cells, migrated.cells):
        assert before.cell_type == after.cell_type
        assert before.metadata == after.metadata
        assert before.get("outputs") == after.get("outputs")
        assert before.get("execution_count") == after.get("execution_count")

    # Every generated code cell is still parseable Python.
    for cell in migrated.cells:
        if cell.cell_type != "code":
            continue
        sanitized, has_cell_magic = _sanitize_magics(cell.source)
        if not has_cell_magic:
            ast.parse(sanitized)

    # A cell the report does not mention is returned exactly as it arrived.
    reported = {finding.cell_index for finding in result.report.changes}
    for index, (before, after) in enumerate(zip(original.cells, migrated.cells)):
        if index not in reported:
            assert before.source == after.source

    # A blocked notebook is returned byte-identical, never partially rewritten,
    # and must never advertise a perfect score.
    if result.report.manual_actions:
        assert result.report.summary["cells_changed"] == 0
        assert result.notebook_bytes == data
        assert result.report.score.overall == 0
        assert result.report.score.automatic_coverage == 0

    # Re-running the migrator changes nothing further.
    second = engine.migrate(result.notebook_bytes, "scenario.ipynb")
    assert second.notebook_bytes == result.notebook_bytes

    # The report describes the notebook it was produced from.
    assert result.report.summary["cells_scanned"] == len(sources)
    assert result.report.summary["rules_applied"] == len(result.report.changes)
    assert result.report.summary["manual_actions"] == len(
        result.report.manual_actions
    )
    json.dumps(result.report.to_dict())


@pytest.mark.parametrize(
    "separator",
    [
        pytest.param(" ", id="line-separator"),
        pytest.param(" ", id="paragraph-separator"),
        pytest.param("\x0b", id="vertical-tab"),
        pytest.param("\x0c", id="form-feed"),
        pytest.param("\x85", id="next-line"),
        pytest.param("\x1c", id="file-separator"),
    ],
)
def test_non_python_line_breaks_do_not_shift_migration_spans(
    separator: str,
) -> None:
    """str.splitlines breaks on these; the Python tokenizer does not.

    Counting them as line breaks used to shift every span in the cell, which
    corrupted the rewrite and forced a full rollback on a notebook that is
    perfectly migratable.
    """
    literal = f'question = "before{separator}after"'
    data = notebook_bytes(literal + "\n" + SETUP, QUESTION)

    result = MigrationEngine().migrate(data, "separators.ipynb")
    text = code_text(result.notebook_bytes)

    assert result.report.status == "migrated"
    assert literal in text
    assert "fabric_client = FabricOpenAIResponses(" in text
    assert "run = fabric_client.responses.create(input=question)" in text
    assert "OUTPUT-SYNTAX-001" not in finding_ids(result)


@pytest.mark.parametrize(
    "distraction",
    [
        pytest.param("# We used to call getattr(client, 'runs') here.", id="comment-getattr"),
        pytest.param('note = "run eval(expr) by hand"', id="string-eval"),
        pytest.param('doc = """exec(setup_code) is documented above."""', id="docstring-exec"),
        pytest.param("# TODO: await the answer before printing.", id="comment-await"),
        pytest.param('label = "async def was removed"', id="string-async"),
        pytest.param("# Persist thread_id with json.dump when caching.", id="comment-persist"),
    ],
)
def test_api_words_in_comments_and_strings_do_not_block_migration(
    distraction: str,
) -> None:
    """Only real code should fail the notebook closed.

    Blocking on the raw text meant a notebook that merely mentioned eval or
    await in prose was reported as unsafe and migrated nothing.
    """
    data = notebook_bytes(SETUP + "\n" + distraction, 'question = "Revenue?"', QUESTION)

    result = MigrationEngine().migrate(data, "prose.ipynb")
    text = code_text(result.notebook_bytes)

    assert result.report.manual_actions == []
    assert result.report.status == "migrated"
    assert distraction in text
    assert "run = fabric_client.responses.create(input=question)" in text


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        pytest.param(
            'getattr(fabric_client.beta.threads, "delete")()',
            "UNSUPPORTED-DYNAMIC-001",
            id="bare-getattr",
        ),
        pytest.param(
            'import builtins\nbuiltins.getattr(fabric_client.beta.threads, "delete")()',
            "UNSUPPORTED-DYNAMIC-001",
            id="qualified-getattr",
        ),
        pytest.param(
            "async def poll():\n    await fabric_client.responses.retrieve(run.id)\n",
            "UNSUPPORTED-ASYNC-001",
            id="async-def",
        ),
        pytest.param(
            "async def poll():\n    async with fabric_client as agent:\n        pass\n",
            "UNSUPPORTED-ASYNC-001",
            id="async-with",
        ),
        pytest.param(
            'import json\njson.dump({"thread_id": thread.id}, handle)',
            "UNSUPPORTED-PERSISTED-001",
            id="persisted-thread-id",
        ),
    ],
)
def test_real_dynamic_and_async_code_still_fails_closed(
    source: str, rule: str
) -> None:
    data = notebook_bytes(SETUP + "\n" + source, QUESTION)

    result = MigrationEngine().migrate(data, "hostile.ipynb")

    assert result.notebook_bytes == data
    assert result.report.summary["cells_changed"] == 0
    assert rule in finding_ids(result)


def test_unrelated_file_access_near_a_thread_id_is_not_persisted_state() -> None:
    """The persisted-state check reads one statement at a time.

    Scanning a 160-character window across the whole cell matched an ordinary
    open() several statements away from an unrelated thread_id assignment.
    """
    data = notebook_bytes(
        SETUP,
        'question = "Revenue?"',
        QUESTION,
        'sales = open("sales.csv").read()\nthread_label = "quarterly"\n',
    )

    result = MigrationEngine().migrate(data, "unrelated-io.ipynb")

    assert "UNSUPPORTED-PERSISTED-001" not in finding_ids(result)
    assert result.report.status == "migrated"


def test_workspace_placeholder_is_inserted_above_a_nested_construction() -> None:
    data = notebook_bytes(
        "from fabric.dataagent.client import FabricOpenAI\n"
        "if use_agent:\n"
        '    fabric_client = FabricOpenAI(artifact_name="Revenue Agent")\n',
        "assistant = fabric_client.beta.assistants.create(model=\"gpt-4o\")\n"
        "thread = fabric_client.beta.threads.create()\n",
        'question = "Revenue?"',
        QUESTION,
    )

    result = MigrationEngine().migrate(data, "nested-client.ipynb")
    text = code_text(result.notebook_bytes)

    assert 'workspace_name = "<REQUIRED: Fabric workspace name or ID>"' in text
    assert "QUERY-CONFIG-001" in finding_ids(result)
    # The placeholder must not disturb the indented construction it precedes.
    assert (
        "workspace_name = \"<REQUIRED: Fabric workspace name or ID>\"\n"
        "if use_agent:\n"
        "    fabric_client = FabricOpenAIResponses("
    ) in text


def test_client_sharing_a_line_with_another_statement_fails_closed() -> None:
    """A placeholder assignment has nowhere safe to go on a compound line."""
    data = notebook_bytes(
        "from fabric.dataagent.client import FabricOpenAI\n"
        'seed = 1; fabric_client = FabricOpenAI(artifact_name="Revenue Agent")\n'
        'assistant = fabric_client.beta.assistants.create(model="gpt-4o")\n'
        "thread = fabric_client.beta.threads.create()\n",
        'question = "Revenue?"',
        QUESTION,
    )

    result = MigrationEngine().migrate(data, "compound-line.ipynb")

    assert result.notebook_bytes == data
    assert result.report.status == "partial_migration"
    assert "UNSUPPORTED-CONFIG-001" in finding_ids(result)


def test_a_defined_workspace_allows_a_compound_line_to_migrate() -> None:
    data = notebook_bytes(
        "from fabric.dataagent.client import FabricOpenAI\n"
        'workspace_name = "Contoso Analytics"\n'
        'seed = 1; fabric_client = FabricOpenAI(artifact_name="Revenue Agent")\n'
        'assistant = fabric_client.beta.assistants.create(model="gpt-4o")\n'
        "thread = fabric_client.beta.threads.create()\n",
        'question = "Revenue?"',
        QUESTION,
    )

    result = MigrationEngine().migrate(data, "compound-line-configured.ipynb")
    text = code_text(result.notebook_bytes)

    assert "UNSUPPORTED-CONFIG-001" not in finding_ids(result)
    assert "seed = 1; fabric_client = FabricOpenAIResponses(" in text


@pytest.mark.parametrize(
    ("install_line", "expected"),
    [
        pytest.param(
            "!pip install fabric-data-agent-sdk",
            '!pip install "fabric-data-agent-sdk>=0.1.28a0"',
            id="shell-escape",
        ),
        pytest.param(
            "!pip install -q fabric-data-agent-sdk==0.1.27a0 pandas",
            '!pip install -q "fabric-data-agent-sdk>=0.1.28a0" pandas',
            id="shell-escape-pinned",
        ),
    ],
)
def test_shell_escape_installs_get_the_same_version_floor(
    install_line: str, expected: str
) -> None:
    """A !pip cell installs the same SDK a %pip cell does."""
    data = notebook_bytes(install_line, EVALUATION)

    result = MigrationEngine().migrate(data, "shell-install.ipynb")
    migrated = read_notebook(result.notebook_bytes)

    assert migrated.cells[0].source == expected
    assert "QUERY-SDK-001" in finding_ids(result)
    assert "RUNTIME-SDK-RESTART-001" in finding_ids(result)


def test_removed_setup_comments_read_as_english() -> None:
    data = notebook_bytes(SETUP, 'question = "Revenue?"', QUESTION)

    text = code_text(
        MigrationEngine().migrate(data, "grammar.ipynb").notebook_bytes
    )

    assert "# Migrated: Responses API does not require an assistant." in text
    assert "# Migrated: Responses API does not require a thread." in text
    assert "require a assistant" not in text


def test_each_run_carries_the_question_it_was_paired_with() -> None:
    """The pairing that clears the flow is the pairing that gets written."""
    data = notebook_bytes(
        SETUP,
        'first = "Which region led revenue?"\n'
        'second = "How did that compare?"\n'
        'third = "Which categories explain it?"\n',
        QUESTION.replace("content=question", "content=first").replace(
            "run =", "first_run ="
        ),
        QUESTION.replace("content=question", "content=second").replace(
            "run =", "second_run ="
        ),
        QUESTION.replace("content=question", "content=third").replace(
            "run =", "third_run ="
        ),
    )

    result = MigrationEngine().migrate(data, "three-turn.ipynb")
    text = code_text(result.notebook_bytes)

    assert "first_run = fabric_client.responses.create(input=first)" in text
    assert (
        "second_run = fabric_client.responses.create("
        "input=second, previous_response_id=first_run.id)"
    ) in text
    assert (
        "third_run = fabric_client.responses.create("
        "input=third, previous_response_id=second_run.id)"
    ) in text
    assert result.report.status == "migrated"


SDK_TOKENS = (
    "FabricOpenAI",
    ".beta.",
    "evaluate_data_agent",
    "fabric-data-agent-sdk",
    "fabric.dataagent",
    "assistant",
    "thread",
    "messages",
    "run",
    "steps",
)


def touches_the_sdk(text: str) -> bool:
    return any(token in text for token in SDK_TOKENS)


UNRELATED_NOTEBOOK = [
    ("markdown", "# Quarterly revenue review\n\nOwned by the finance team."),
    ("code", "%pip install fabric-data-agent-sdk pandas"),
    (
        "code",
        "import pandas as pd\n"
        "import numpy as np\n"
        "\n"
        "# Load the finance extract. Do not change this path.\n"
        'sales = pd.read_csv("/lakehouse/default/Files/sales.csv")\n'
        'sales["margin"] = sales["revenue"] - sales["cost"]\n'
        "TAX_RATE = 0.21\n",
    ),
    ("code", "%%sql\nSELECT TOP 5 * FROM demo.revenue"),
    ("raw", "Raw cell content that must survive untouched."),
    ("code", SETUP),
    ("code", 'question = "Summarize quarterly revenue by region."'),
    ("code", QUESTION),
    ("code", POLL),
    ("code", READ),
    ("markdown", "## Charting\n\nThe next cell is unrelated to the data agent."),
    (
        "code",
        "summary = sales.groupby('region', as_index=False)['margin'].sum()\n"
        "summary.plot(kind='bar', x='region', y='margin')\n"
        "print(f'Total margin: {summary[\"margin\"].sum():,.2f}')\n",
    ),
    ("code", "fabric_client.beta.threads.delete(thread.id)"),
    ("markdown", "Done."),
]


def build_rich_notebook() -> bytes:
    notebook = nbformat.v4.new_notebook(
        metadata={"owner": "finance", "custom": {"keep": ["everything"]}}
    )
    notebook.cells = []
    for index, (cell_type, source) in enumerate(UNRELATED_NOTEBOOK):
        if cell_type == "markdown":
            cell = nbformat.v4.new_markdown_cell(source)
        elif cell_type == "raw":
            cell = nbformat.v4.new_raw_cell(source)
        else:
            cell = nbformat.v4.new_code_cell(source)
            cell.outputs = [
                nbformat.v4.new_output("stream", name="stdout", text=f"output {index}\n")
            ]
            cell.execution_count = index
        cell.metadata["origin"] = index
        notebook.cells.append(cell)
    return nbformat.writes(notebook).encode("utf-8")


def test_nothing_unrelated_to_the_sdk_is_deleted_or_rewritten() -> None:
    """Every statement that does not mention the SDK survives verbatim.

    This is the promise the tool is built on: the migrated notebook differs
    from the original only where an Assistants API pattern had to become a
    Responses API pattern.
    """
    data = build_rich_notebook()
    original = read_notebook(data)

    result = MigrationEngine().migrate(data, "rich-unrelated.ipynb")
    migrated = read_notebook(result.notebook_bytes)

    assert result.report.status == "migrated"
    assert result.report.summary["cells_changed"] > 0
    assert migrated.metadata == original.metadata
    assert len(migrated.cells) == len(original.cells)

    for index, (before, after) in enumerate(zip(original.cells, migrated.cells)):
        assert before.cell_type == after.cell_type, index
        assert before.metadata == after.metadata, index
        assert before.get("outputs") == after.get("outputs"), index
        assert before.get("execution_count") == after.get("execution_count"), index

        # Markdown and raw cells are never rewritten at all.
        if before.cell_type != "code":
            assert before.source == after.source, index
            continue

        # A cell magic makes the whole cell off limits.
        if before.source.lstrip().startswith("%%"):
            assert before.source == after.source, index
            continue

        sanitized, _ = _sanitize_magics(before.source)
        try:
            tree = ast.parse(sanitized)
        except SyntaxError:
            continue
        lines, _ = _line_offsets(before.source)
        for statement in tree.body:
            text = "".join(
                lines[statement.lineno - 1 : statement.end_lineno]
            ).strip("\n")
            if touches_the_sdk(text):
                continue
            assert text in after.source, (index, text)

    # Comments and blank lines around untouched code survive too.
    finance_cell = migrated.cells[2].source
    assert "# Load the finance extract. Do not change this path." in finance_cell
    assert "TAX_RATE = 0.21" in finance_cell
    assert finance_cell == original.cells[2].source


def test_unrelated_cells_are_absent_from_the_report() -> None:
    """The report only points at cells the tool actually changed."""
    data = build_rich_notebook()
    result = MigrationEngine().migrate(data, "rich-unrelated.ipynb")

    unrelated = {0, 2, 3, 4, 10, 11, 13}
    reported = {finding.cell_index for finding in result.report.changes}

    assert not (reported & unrelated)


MCP_DOCS_URL = (
    "https://learn.microsoft.com/fabric/data-science/data-agent-mcp-server"
)


@pytest.mark.parametrize(
    "sources",
    [
        pytest.param([EVALUATION], id="evaluation"),
        pytest.param([SETUP, 'question = "Revenue?"', QUESTION], id="query"),
        pytest.param(["import pandas as pd"], id="no-migration"),
    ],
)
def test_report_says_where_the_code_can_run(sources: list[str]) -> None:
    """Evaluation is Fabric-notebook only; external callers need the MCP server."""
    result = MigrationEngine().migrate(notebook_bytes(*sources), "scope.ipynb")
    checklist = " ".join(result.report.user_test_checklist)

    assert "evaluate_data_agent" in checklist
    assert "Fabric notebook" in checklist
    assert MCP_DOCS_URL in checklist


def test_app_shows_the_same_runtime_scope_message() -> None:
    """The banner in the app and the line in the report must not drift apart."""
    app_source = (pathlib.Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )

    assert MCP_DOCS_URL in app_source
    assert "evaluate_data_agent" in app_source
    assert "only work inside a Fabric" in app_source


EXTERNAL_CLIENT = '''from openai import OpenAI


class FabricOpenAI(OpenAI):
    """Talks to the Fabric Assistants endpoint with a service principal token."""

    def __init__(self, **kwargs):
        super().__init__(api_key="", base_url="", **kwargs)


agent = FabricOpenAI()
'''

# The endpoint is written out, the way the promptfoo CI sample hardcodes a
# fallback URL.
EXTERNAL_CLIENT_WITH_URL = EXTERNAL_CLIENT + (
    'agent.base_url = "https://api.fabric.microsoft.com/v1/workspaces/w'
    '/aiskills/a/aiassistant/openai"\n'
)

# The endpoint only exists in the environment, so the openai import is the
# only signal left.
EXTERNAL_CLIENT_FROM_ENV = EXTERNAL_CLIENT + (
    'agent.base_url = os.environ["FABRIC_DATA_AGENT_URL"]\n'
)

EXTERNAL_FLOW = """assistant = agent.beta.assistants.create(model="not used")
thread = agent.beta.threads.create()
agent.beta.threads.messages.create(
    thread_id=thread.id, role="user", content=prompt
)
run = agent.beta.threads.runs.create(
    thread_id=thread.id, assistant_id=assistant.id
)
"""


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(EXTERNAL_CLIENT_WITH_URL, id="hardcoded-endpoint-url"),
        pytest.param(EXTERNAL_CLIENT_FROM_ENV, id="endpoint-from-environment"),
    ],
)
def test_direct_endpoint_callers_are_pointed_at_the_mcp_server(setup: str) -> None:
    """Telling these users to import the Fabric SDK would be wrong advice.

    They are querying from outside Fabric on purpose, so the supported path is
    the data agent MCP server, not the Responses API.
    """
    data = notebook_bytes(setup, EXTERNAL_FLOW)

    result = MigrationEngine().migrate(data, "external-caller.ipynb")
    rules = finding_ids(result)

    assert result.notebook_bytes == data
    assert result.report.summary["cells_changed"] == 0
    assert "EXTERNAL-ENDPOINT-001" in rules
    # The old advice pointed at the SDK import, which does not apply here.
    assert "UNSUPPORTED-CLIENT-001" not in rules
    finding = next(
        item
        for item in result.report.manual_actions
        if item.rule_id == "EXTERNAL-ENDPOINT-001"
    )
    assert MCP_DOCS_URL in (finding.action or "")


def test_a_blocked_notebook_never_reports_full_confidence() -> None:
    """A score of 100 next to "0 cells changed" would be actively misleading."""
    data = notebook_bytes(EXTERNAL_CLIENT_WITH_URL, EXTERNAL_FLOW)

    result = MigrationEngine().migrate(data, "external-caller.ipynb")

    assert result.report.status == "partial_migration"
    assert result.report.summary["cells_changed"] == 0
    assert result.report.score.overall == 0


def test_every_blocking_rule_family_counts_toward_the_score() -> None:
    """A new blocker whose id sits outside SCORED_RULE_PREFIXES scores 100.

    That is how EXTERNAL-ENDPOINT-001 first shipped a fully blocked notebook
    with a perfect confidence score, so the prefixes are pinned here.
    """
    engine_source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "fabric_migrator"
        / "engine.py"
    ).read_text(encoding="utf-8")

    declared = set(re.findall(r'"((?:[A-Z]+-)+)"', engine_source)) & set(
        SCORED_RULE_PREFIXES
    )
    blocking_rules = set(re.findall(r'"((?:UNSUPPORTED|EXTERNAL|OUTPUT)-[A-Z-]+-\d+)"', engine_source))

    assert declared == set(SCORED_RULE_PREFIXES)
    for rule in blocking_rules:
        assert rule.startswith(SCORED_RULE_PREFIXES), rule


def test_sdk_notebooks_are_not_mistaken_for_external_callers() -> None:
    """A normal Fabric notebook must never see the MCP finding."""
    data = notebook_bytes(SETUP, 'question = "Revenue?"', QUESTION)

    result = MigrationEngine().migrate(data, "normal.ipynb")

    assert "EXTERNAL-ENDPOINT-001" not in finding_ids(result)
    assert result.report.status == "migrated"


def test_an_unresolved_client_without_external_evidence_keeps_the_old_advice() -> None:
    data = notebook_bytes(
        "agent = build_client()\n"
        'assistant = agent.beta.assistants.create(model="gpt-4o")\n'
        "thread = agent.beta.threads.create()\n",
        QUESTION.replace("fabric_client", "agent"),
    )

    result = MigrationEngine().migrate(data, "unknown-client.ipynb")
    rules = finding_ids(result)

    assert "UNSUPPORTED-CLIENT-001" in rules
    assert "EXTERNAL-ENDPOINT-001" not in rules


@pytest.mark.parametrize(
    "assignment",
    [
        pytest.param('api_key = ""', id="empty-string"),
        pytest.param("api_key = ''", id="empty-single-quoted"),
        pytest.param('access_token = "   "', id="whitespace-only"),
        pytest.param('client_secret = "<your-secret-here>"', id="placeholder"),
    ],
)
def test_empty_and_placeholder_credentials_are_not_reported_as_secrets(
    assignment: str,
) -> None:
    """api_key="" is how the Fabric samples construct a client."""
    data = notebook_bytes(
        assignment + "\n" + SETUP, 'question = "Revenue?"', QUESTION
    )

    result = MigrationEngine().migrate(data, "empty-secret.ipynb")

    assert "WARNING-SECRET-001" not in finding_ids(result)
    assert result.report.status == "migrated"


# A cleaned kernelspec, the way the published Microsoft samples ship.
PLAIN_KERNEL = {"kernelspec": {"name": "python3", "display_name": "Python 3"}}


def notebook_with_metadata(metadata: dict, *sources: str) -> bytes:
    notebook = nbformat.v4.new_notebook(metadata=metadata)
    notebook.cells = [nbformat.v4.new_code_cell(source) for source in sources]
    return nbformat.writes(notebook).encode("utf-8")


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({"kernelspec": {"name": "synapse_pyspark", "display_name": "Synapse PySpark"}}, id="synapse-kernel"),
        pytest.param({"a365ComputeOptions": None}, id="a365-compute"),
        pytest.param({"spark_compute": {"pool": "default"}}, id="spark-compute"),
        pytest.param({"trident": {"lakehouse": {}}}, id="trident"),
        pytest.param({"microsoft": {"language": "python"}}, id="microsoft"),
        pytest.param({"kernel_info": {"name": "synapse_pyspark"}}, id="kernel-info"),
    ],
)
def test_fabric_notebooks_evaluate_without_a_runtime_warning(
    metadata: dict,
) -> None:
    """Confirmed Fabric metadata means evaluation is already in the right place."""
    data = notebook_with_metadata(metadata, EVALUATION)

    result = MigrationEngine().migrate(data, "fabric-eval.ipynb")

    assert "RUNTIME-EVAL-SCOPE-001" not in finding_ids(result)
    assert result.report.status == "migrated"


def test_evaluation_without_fabric_metadata_is_pointed_at_the_mcp_server() -> None:
    data = notebook_with_metadata(PLAIN_KERNEL, EVALUATION)

    result = MigrationEngine().migrate(data, "unknown-runtime-eval.ipynb")
    warning = next(
        item
        for item in result.report.warnings
        if item.rule_id == "RUNTIME-EVAL-SCOPE-001"
    )

    # A warning, not a blocker: the migration still happens.
    assert result.report.status == "migrated_with_warnings"
    assert result.report.summary["cells_changed"] > 0
    assert MCP_DOCS_URL in (warning.action or "")
    assert "cannot confirm" in warning.message


def test_the_runtime_warning_never_claims_the_notebook_is_not_fabric() -> None:
    """Microsoft ships Fabric samples with no Fabric metadata.

    Asserting "this is not a Fabric notebook" would be wrong on the official
    responses-api sample, so the wording has to stay a prompt to check.
    """
    data = notebook_with_metadata(PLAIN_KERNEL, EVALUATION)

    result = MigrationEngine().migrate(data, "official-shaped.ipynb")
    warning = next(
        item
        for item in result.report.warnings
        if item.rule_id == "RUNTIME-EVAL-SCOPE-001"
    )
    text = f"{warning.message} {warning.action}".lower()

    for claim in ("is not a fabric", "will not work", "does not run in fabric"):
        assert claim not in text
    assert "if this runs in a fabric notebook, no change is needed" in text


def test_the_runtime_warning_does_not_change_the_score() -> None:
    """It is advice about where to run, not a failed migration rule."""
    data = notebook_with_metadata(PLAIN_KERNEL, EVALUATION)

    result = MigrationEngine().migrate(data, "score-check.ipynb")

    assert "RUNTIME-EVAL-SCOPE-001" in finding_ids(result)
    assert result.report.score.overall == 100


def test_query_only_notebooks_do_not_get_the_evaluation_warning() -> None:
    data = notebook_bytes(SETUP, 'question = "Revenue?"', QUESTION)

    result = MigrationEngine().migrate(data, "query-only.ipynb")

    assert "RUNTIME-EVAL-SCOPE-001" not in finding_ids(result)


CALL_LAYOUTS = [
    pytest.param(
        "evaluate_data_agent(\n"
        "    df,\n"
        "    data_agent_name,\n"
        "    workspace_name=workspace_name,\n"
        "    data_agent_stage=data_agent_stage\n"
        ")\n",
        id="multiline-no-trailing-comma",
    ),
    pytest.param(
        "evaluate_data_agent(\n"
        "    df,\n"
        "    workspace_name=workspace_name,\n"
        ")\n",
        id="multiline-trailing-comma",
    ),
    pytest.param(
        "evaluate_data_agent(df,\n    workspace_name=workspace_name)\n",
        id="close-paren-shares-last-argument-line",
    ),
    pytest.param(
        "evaluate_data_agent(df, workspace_name=workspace_name)\n",
        id="single-line",
    ),
    pytest.param(
        "evaluate_data_agent(\n"
        "    df,  # the questions\n"
        "    workspace_name=workspace_name\n"
        ")\n",
        id="comment-inside-call",
    ),
    pytest.param(
        "evaluate_data_agent(\n"
        "    df,\n"
        "    workspace_name=workspace_name,\n"
        "    critic_prompt=(\n"
        '        "score it"\n'
        "    )\n"
        ")\n",
        id="nested-parentheses",
    ),
]


@pytest.mark.parametrize("call", CALL_LAYOUTS)
def test_adding_client_class_never_drops_an_existing_argument(call: str) -> None:
    """Adding a keyword must not disturb the arguments already written.

    One layout used to delete the last argument outright. The result still
    parsed, so nothing downstream noticed.
    """
    source = "from fabric.dataagent.evaluation import evaluate_data_agent\n" + call
    data = notebook_bytes(source)

    result = MigrationEngine().migrate(data, "layout.ipynb")
    migrated = code_text(result.notebook_bytes)

    before_call = ast.parse(call).body[0].value
    after_call = next(
        node
        for node in ast.walk(ast.parse(migrated))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "evaluate_data_agent"
    )

    before_keywords = {keyword.arg for keyword in before_call.keywords}
    after_keywords = {keyword.arg for keyword in after_call.keywords}

    assert before_keywords < after_keywords
    assert after_keywords - before_keywords == {"client_class"}
    assert len(after_call.args) == len(before_call.args)
    assert "client_class=FabricOpenAIResponses" in migrated
    # No stray comma stranded on a line of its own.
    assert not any(
        line.strip() == "," for line in migrated.splitlines()
    )


@pytest.mark.parametrize(
    "constructor",
    [
        pytest.param(
            'fabric_client = FabricOpenAI(\n    artifact_name="A"\n)\n',
            id="multiline-no-trailing-comma",
        ),
        pytest.param(
            'fabric_client = FabricOpenAI(artifact_name="A",\n'
            '    ai_skill_stage="production")\n',
            id="close-paren-shares-last-argument-line",
        ),
        pytest.param(
            'fabric_client = FabricOpenAI(artifact_name="A")\n',
            id="single-line",
        ),
    ],
)
def test_client_construction_keeps_every_original_option(constructor: str) -> None:
    setup = (
        "from fabric.dataagent.client import FabricOpenAI\n"
        'workspace_name = "Contoso Analytics"\n'
        + constructor
        + 'assistant = fabric_client.beta.assistants.create(model="gpt-4o")\n'
        "thread = fabric_client.beta.threads.create()\n"
    )
    data = notebook_bytes(setup, 'question = "Revenue?"', QUESTION)

    result = MigrationEngine().migrate(data, "constructor-layout.ipynb")
    migrated = code_text(result.notebook_bytes)

    before_call = next(
        node
        for node in ast.walk(ast.parse(constructor))
        if isinstance(node, ast.Call)
    )
    after_call = next(
        node
        for node in ast.walk(ast.parse(migrated))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "FabricOpenAIResponses"
    )

    before_keywords = {
        keyword.arg: ast.dump(keyword.value) for keyword in before_call.keywords
    }
    after_keywords = {
        keyword.arg: ast.dump(keyword.value) for keyword in after_call.keywords
    }

    for name, value in before_keywords.items():
        assert after_keywords[name] == value, name
    assert set(after_keywords) == {
        "artifact_name",
        "workspace_name",
        "ai_skill_stage",
    }
    assert not any(line.strip() == "," for line in migrated.splitlines())


EVAL_CALL = (
    "from fabric.dataagent.evaluation import evaluate_data_agent\n"
    "evaluate_data_agent(df, workspace_name=workspace_name)\n"
)


def test_query_and_evaluation_share_one_responses_import() -> None:
    """The renamed client import already provides the name evaluation needs."""
    data = notebook_bytes(SETUP, 'question = "Revenue?"', QUESTION, EVAL_CALL)

    result = MigrationEngine().migrate(data, "query-and-eval.ipynb")
    migrated = code_text(result.notebook_bytes)

    assert migrated.count("import FabricOpenAIResponses") == 1
    assert "client_class=FabricOpenAIResponses" in migrated
    ast.parse(migrated.replace("%pip", "#pip"))


def test_an_aliased_client_import_still_needs_its_own_evaluation_import() -> None:
    """`import FabricOpenAIResponses as Agent` does not bind the plain name."""
    aliased = SETUP.replace(
        "import FabricOpenAI\n", "import FabricOpenAI as Agent\n"
    ).replace("= FabricOpenAI(", "= Agent(")
    data = notebook_bytes(aliased, 'question = "Revenue?"', QUESTION, EVAL_CALL)

    result = MigrationEngine().migrate(data, "aliased-and-eval.ipynb")
    migrated = code_text(result.notebook_bytes)

    assert "import FabricOpenAIResponses as Agent" in migrated
    assert (
        "from fabric.dataagent.client import FabricOpenAIResponses\n" in migrated
    )
    # Otherwise client_class=FabricOpenAIResponses would be a NameError.
    assert "client_class=FabricOpenAIResponses" in migrated


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "import fabric.dataagent.evaluation\n"
            "fabric.dataagent.evaluation.evaluate_data_agent(df, workspace_name=w)\n",
            id="module-import",
        ),
        pytest.param(
            "import fabric.dataagent.evaluation as ev\n"
            "ev.evaluate_data_agent(df, workspace_name=w)\n",
            id="module-alias",
        ),
        pytest.param(
            "from fabric.dataagent import evaluation\n"
            "evaluation.evaluate_data_agent(df, workspace_name=w)\n",
            id="from-package",
        ),
    ],
)
def test_qualified_evaluation_calls_are_migrated(source: str) -> None:
    """A module-qualified call used to report "no migration needed".

    Telling someone their notebook is already done when it is not is worse
    than refusing to touch it.
    """
    result = MigrationEngine().migrate(notebook_bytes(source), "qualified.ipynb")
    migrated = code_text(result.notebook_bytes)

    assert result.report.summary["cells_changed"] == 1
    assert "client_class=FabricOpenAIResponses" in migrated
    assert "from fabric.dataagent.client import FabricOpenAIResponses" in migrated
    assert "EVAL-CLIENT-001" in finding_ids(result)


def test_a_lookalike_function_on_another_module_is_left_alone() -> None:
    source = "import mylib\nmylib.evaluate_data_agent(df, workspace_name=w)\n"
    data = notebook_bytes(source)

    result = MigrationEngine().migrate(data, "lookalike.ipynb")

    assert result.notebook_bytes == data
    assert result.report.status == "no_migration_needed"


def test_a_call_that_already_passes_client_class_is_not_touched_again() -> None:
    """Matches a partially migrated notebook where some calls are done."""
    done = (
        "from fabric.dataagent.client import FabricOpenAIResponses\n"
        "from fabric.dataagent.evaluation import evaluate_data_agent\n"
        "first = evaluate_data_agent(\n"
        "    df,\n"
        "    workspace_name=workspace_name,\n"
        "    client_class=FabricOpenAIResponses,\n"
        ")\n"
    )
    todo = (
        "second = evaluate_data_agent(\n"
        "    df,\n"
        "    workspace_name=workspace_name\n"
        ")\n"
    )
    data = notebook_bytes(done, todo)
    original = read_notebook(data)

    result = MigrationEngine().migrate(data, "partially-migrated.ipynb")
    migrated = read_notebook(result.notebook_bytes)

    assert migrated.cells[0].source == original.cells[0].source
    assert migrated.cells[1].source.count("client_class=FabricOpenAIResponses") == 1
    assert code_text(result.notebook_bytes).count("import FabricOpenAIResponses") == 1


def test_report_never_carries_a_detected_secret_value() -> None:
    secret = "sk-live-must-never-be-reported"
    data = notebook_bytes(
        f'api_key = "{secret}"\n' + SETUP, 'question = "Revenue?"', QUESTION
    )

    result = MigrationEngine().migrate(data, "secret.ipynb")

    assert secret not in json.dumps(result.report.to_dict())
    assert "WARNING-SECRET-001" in finding_ids(result)
    # The notebook itself is left exactly as the user wrote it.
    assert f'api_key = "{secret}"' in code_text(result.notebook_bytes)
