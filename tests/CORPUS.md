# Real-world notebook corpus

Public GitHub notebooks that use the Fabric data agent SDK, collected to check
the migrator against code nobody on this project wrote. None of them are
committed here. Licensing varies by repo and the files are large, so the list
below is a manifest you re-fetch when you want to run the sweep again.

Last swept: 2026-08-14, against 20 candidates.

## How they were found

GitHub code search, restricted to notebooks:

```bash
gh api -X GET search/code --field q='"from fabric.dataagent.client import FabricOpenAI" extension:ipynb' --field per_page=50 --jq '.items[] | "\(.repository.full_name) :: \(.path)"'
```

Repeat with `"FabricOpenAI("`, `"evaluate_data_agent"`, `"fabric-data-agent-sdk"`,
`"beta.threads.runs.create" "fabric"`, and `"aiassistant/openai"`, then dedupe.
Code search only indexes default branches of public repos, so treat the result
as a sample rather than a census.

## What the sweep found

Fetch a notebook with:

```bash
gh api repos/OWNER/REPO/contents/PATH --jq .content | base64 -d > sample.ipynb
```

### Full Assistants query flow, migrates end to end

| Repo | Path | Result |
| --- | --- | --- |
| microsoft/fabric-samples | `docs-samples/data-science/data-agent-sdk/Fabric-DataAgent-OpenAI-Client-sample.ipynb` | 9 cells changed, 2 warnings |

This is the reference case: import, client construction, message and run,
polling loop, `messages.list`, a `pretty_print` helper, run-step inspection,
and thread cleanup. The two warnings are the workspace placeholder and the
defaulted `sandbox` stage, both of which need a human answer.

### Evaluation only, gains `client_class` and the SDK floor

| Repo | Path |
| --- | --- |
| microsoft/fabric-samples | `docs-samples/data-science/data-agent-sdk/Fabric-DataAgent-Evaluation-sample.ipynb` |
| lucazav/Measure-the-Real-Robustness-of-a-Fabric-Data-Agent-in-Italian | `zava_agent_evaluation.ipynb` |
| arpitaonnet/F1race-Fabric-Data-Agent | `2_notebooks/AutomatedTesting.ipynb` |
| marclelijveld/Slide-decks | `2026/2026-03-18_FabCon - Atlanta/NB_QUALY_DataAgentTesting.ipynb` |
| antoniosql/fabric-demos | `fabric-data-agent/notebooks/01_evaluate_agent.ipynb` |
| robkerr/consulting-ontology | `fabric/notebooks/data_agent_evaluation.ipynb` |
| robkerr/nflverse-fabric-reference-architecture | `notebooks/Lakehouse Data Agent Evaluations.ipynb` |
| msft-shreyas/fabconeu-2025-dataagent-tutorial | `Lab 3/Lab3_Evaluation_V2.ipynb` |

### Blocked, and correctly so

| Repo | Path | Why |
| --- | --- | --- |
| ncheruvu-MSFT/msft-fabric-utils | `fabric-sdlc-governance/notebook/fabric-data-agent-sales-demo.ipynb` | Iterates `messages.list(...).data` with a nested loop over multiple content parts. Rewriting it to `output_text` could silently drop content. |
| slammini/MultiAgentSystem_FabConLasVegas2025 | `MultiAgentSystem.ipynb` | Calls the Assistants endpoint directly and persists thread state. |
| pablosalvador10/gbb-foundry-fabric-agenticrag | `labs/01-intro-to-maf.ipynb`, `labs/02-run-data-agent.ipynb` | Custom wrappers plus a hand-built endpoint client. |

The last three are the `EXTERNAL-ENDPOINT-001` case: they reach the Assistants
endpoint without going through the Fabric SDK, so the answer for them is the
data agent MCP server, not the Responses API.

### Correctly reported as needing nothing

| Repo | Path | Why it is a no-op |
| --- | --- | --- |
| microsoft/fabric-samples | `docs-samples/data-science/data-agent-sdk/responses-api/responses-api-notebook.ipynb` | Already on the Responses API. |
| alipouw13/insurance-multi-agent | `backend/fabric/create_data_agent.ipynb` | The whole Assistants block is commented out. |
| lucazav/... | `fabric_evaluation_source_code.ipynb` | `FabricOpenAI` appears only inside a regex string literal. |
| microsoft/fabric-architecture-review | `fabric/notebooks/05_agent.ipynb` | Management plane only. |
| alpaBuddhabhatti/Fabric- | `DataAgent_SDK_1.ipynb` | Management plane only. |
| SoomroFarhanH/SemanticModelBPforAI | `Agent-Readiness-Validator/Agent Readiness Validator.ipynb` | Management plane only. |
| amritamadhav/techconnectlabsession | `order_copilot_workflow.ipynb` | No SDK query calls. |

The commented-out and regex-literal cases are worth keeping in mind. Both would
be false positives for anything that pattern-matches on raw cell text, and both
are the reason the blockers run over the parsed tree instead.

## Hand-written pass

Public notebooks skew toward evaluation-only usage, so a second corpus of five
notebooks was written by hand to cover shapes the public sample does not: a
pandas and matplotlib review with a single agent question, a multi-turn thread,
an evaluation-only run, a nightly job doing both a query and an evaluation
behind an aliased import, and a scratchpad with shell installs and a `%%time`
cell. Roughly two thirds of each notebook is ordinary analysis code.

Old-API text was deliberately placed where it must never be rewritten:
markdown prose and fenced blocks, `#` comments, a function docstring, stdout
streams, an `execute_result` repr, and an error traceback. All of it survived
byte for byte, along with outputs, execution counts and metadata.

That pass found one real defect. `help(client.beta.threads.runs.create)` was
neither migrated nor flagged, and the notebook was reported as migrated while
still holding a reference that breaks at retirement. The engine only inspected
`.beta.` paths in call position, so a bare attribute reference was invisible.
`UNSUPPORTED-ATTRIBUTE-001` now blocks it.

The findings live in `test_engine_hardening.py` rather than as committed
notebooks, since `.gitignore` excludes `*.ipynb`.

## Gap this sweep exposed

The sample contains exactly one notebook exercising the complete Assistants
query flow. Every other Assistants-era notebook found was either blocked or a
no-op. Public code has largely moved to evaluation-only usage, so the synthetic
fixtures in `test_real_sample_matrix.py` and `test_engine_hardening.py` are
still carrying most of the query-plane coverage.
