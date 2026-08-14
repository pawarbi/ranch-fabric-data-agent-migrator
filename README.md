# RANCH (BETA)

[![Tests](https://github.com/pawarbi/ranch-fabric-data-agent-migrator/actions/workflows/tests.yml/badge.svg)](https://github.com/pawarbi/ranch-fabric-data-agent-migrator/actions/workflows/tests.yml)

**R**esponses **A**PI **N**otebook **C**onversion **H**elper for Microsoft
Fabric Data Agents.

RANCH puts old threads out to pasture without trampling the rest of your
notebook. Upload a Jupyter notebook, review a conservative Assistants-to-
Responses migration, and download the migrated copy plus a JSON report.

> [!WARNING]
> RANCH is a beta migration assistant, not proof of runtime correctness. Keep
> the original notebook, test the migrated copy in a non-production Fabric
> workspace, and compare representative results before replacing or deleting
> anything.

## What RANCH returns

- A migrated `<name>-responses-api.ipynb` notebook.
- A `<name>-migration-report.json` report.
- An overall migration-confidence score.
- Per-cell applied changes with redacted before-and-after snippets.
- Warnings, manual actions, and a test checklist.
- The SDK version mapping used for the assessment.

Uploaded code is parsed in memory and is **never executed** by RANCH.

## Run locally

RANCH requires Python 3.10 or newer.

```powershell
git clone https://github.com/pawarbi/ranch-fabric-data-agent-migrator.git
cd ranch-fabric-data-agent-migrator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`, and upload one `.ipynb` file. The default upload
limit is 10 MB and 1,000 cells.

## Deploy on Streamlit Community Cloud

The repository is deployment-ready: `app.py` is the entry point,
`requirements.txt` contains runtime dependencies, and
`.streamlit/config.toml` contains the app theme and upload limit.

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with
   the GitHub account that can access this repository.
2. Select **Create app**.
3. Choose this repository and the `main` branch.
4. Set the entry-point file to `app.py`.
5. In advanced settings, use Python 3.11 or newer.
6. Select **Deploy**.

RANCH does not require credentials or a `secrets.toml` file. Never commit
uploaded notebooks, access tokens, API keys, or Streamlit secrets.

See the official Streamlit documentation:

- [Deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Community Cloud quickstart](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/quickstart)
- [Connect a GitHub account](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account)
- [Manage a deployed app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

## Migration behavior

RANCH uses Python AST location data and surgical source-range patches. It does
not perform global text replacement and does not use an LLM to rewrite code.
Comments, formatting, Markdown, outputs, metadata, and unrelated cells are
preserved whenever a verified migration rule does not target them.

### Automatically handled patterns

| Assistants pattern | Responses result |
| --- | --- |
| `FabricOpenAI` import and construction | `FabricOpenAIResponses` |
| Assistant and thread setup used only by a recognized query flow | Removed |
| `messages.create` followed by `runs.create` | `responses.create(input=...)` |
| `runs.retrieve` polling | `responses.retrieve` |
| Sequential thread reuse | `previous_response_id` chaining |
| Recognized plain-text message extraction | Responses `output_text` |
| Direct `runs.steps.list` inspection | Filtered response output items |
| Query-only thread cleanup | Removed |
| `evaluate_data_agent(...)` | `client_class=FabricOpenAIResponses` |
| Related SDK `%pip install` lines | Minimum supported SDK requirement |

For evaluation, `client_class` receives the
`FabricOpenAIResponses` **class**. RANCH does not create an unnecessary client
instance.

### Intentionally left unchanged

- Fabric Data Agent management-plane operations such as create, configure,
  publish, and data-source management.
- Unrelated Python, Markdown, comments, outputs, and notebook metadata.
- Ambiguous or unsupported query flows.

Automatic query migration is blocked when RANCH finds dynamic attribute
access, unknown wrappers or Assistants calls, async orchestration, persisted
thread state, symbol reassignment, unsupported options, duplicate API object
producers, nonstandard polling, or ambiguous message/run ordering. The original
flow remains intact and the report explains what requires review.

Streaming and conversation-object migration are not supported in the current
beta.

## Confidence score

The score combines:

1. **Automatic coverage** — how much detected migration work matched verified
   rules.
2. **Transformation confidence** — the confidence assigned to applied rules.
3. **Preservation confidence** — measured structural preservation outside
   targeted ranges.

The overall score is the lowest of those components. It measures migration
confidence only; it does not verify credentials, Fabric workspace access,
service availability, data-agent behavior, or answer quality.

## What to test

At minimum, confirm the following in a non-production Fabric workspace:

- The Responses client initializes with the intended workspace, data agent,
  and stage.
- Single-turn questions produce equivalent answers.
- Follow-up questions preserve the intended context.
- Polling terminates and surfaces service errors.
- Text and intermediate-step parsing still matches downstream expectations.
- Evaluation jobs use `FabricOpenAIResponses`.
- Package installation is followed by a Python-session restart when required.
- Representative outputs match the original notebook before any cutover.

## SDK guidance

RANCH currently recommends
`fabric-data-agent-sdk>=0.1.28a0`. Version `0.1.27a0` introduced the complete
Responses resource, and `0.1.28a0` fixed normalization of non-streamed
Responses results. The mapping was last verified on **August 13, 2026**.

When upgrading packages inside a Fabric notebook, the running Python process
may still hold the previous SDK version. Restart the Python session after the
install cell when needed:

```python
notebookutils.session.restartPython()
```

Import the upgraded package in a later cell.

## Why migration is required

Microsoft announced the Fabric Data Agent Assistants API retirement in its
July 2026 Fabric updates. Microsoft Learn states that migration support becomes
available on **August 11, 2026** and that Assistants API retirement takes
effect on **August 26, 2026**.

Official resources:

- [Microsoft Fabric retirement announcement](https://community.fabric.microsoft.com/t5/Fabric-Updates-Blog/Prepare-your-Fabric-Data-Agent-integrations-for-Assistants-API/ba-p/5314634)
- [Fabric Data Agent SDK migration guidance](https://learn.microsoft.com/fabric/data-science/fabric-data-agent-sdk)
- [Microsoft Fabric What's New](https://learn.microsoft.com/fabric/fundamentals/whats-new#fabric-data-science)
- [Official Responses API samples](https://github.com/microsoft/fabric-samples/tree/main/docs-samples/data-science/data-agent-sdk/responses-api)
- [Fabric Data Agent SDK on PyPI](https://pypi.org/project/fabric-data-agent-sdk/)
- [OpenAI Assistants migration guide](https://platform.openai.com/docs/assistants/migration)

## Development

Install the project with test dependencies:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

The migration engine is independent of Streamlit:

```text
app.py                         Streamlit interface
src/fabric_migrator/engine.py Migration analysis and transformations
src/fabric_migrator/mapping.py SDK evidence and rule mapping
src/fabric_migrator/models.py Reports, findings, and scores
src/fabric_migrator/notebook_io.py Notebook validation and serialization
tests/test_migration.py       Regression and preservation tests
```

## Privacy and security

- Notebook cells are never executed.
- Processing occurs in memory during the Streamlit session.
- RANCH does not require Fabric, Azure, or OpenAI credentials.
- Reports redact likely credential values from displayed excerpts.
- Uploaded notebooks are excluded from this repository by default.
- Streamlit secrets, environment files, caches, and virtual environments are
  excluded from Git.

Do not upload a notebook containing information you are not authorized to
process on the machine or Streamlit deployment running RANCH.
