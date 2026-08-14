# Fabric Data Agent Responses API Migration Tool

## Goal

Build a deterministic migration assistant that accepts an existing Jupyter
notebook, converts supported Fabric Data Agent Assistants API patterns to the
Responses API, and returns:

1. A downloadable migrated `.ipynb` notebook.
2. A migration report listing changes, warnings, unsupported patterns, and
   recommended manual actions.

The tool is an accelerator rather than a guaranteed converter. It should make
safe changes automatically and clearly flag anything it cannot migrate with
high confidence.

## Product principles

- No LLM is required for migration.
- Never silently change ambiguous behavior.
- Preserve notebook cell order, Markdown, outputs, metadata, and unrelated code.
- Use syntax-aware transformations instead of global string replacement.
- Make every generated change explainable and traceable to a migration rule.
- Always allow users to download a result, even when some cells require review.
- Report confidence by rule and cell rather than presenting an unsupported
  notebook as fully migrated.

## Initial supported scope

The first release will target the migration patterns documented in:

- `responses-api-migration-guide.md`
- `responses-api-notebook.ipynb`
- Fabric Data Agent SDK sample notebooks in `microsoft/fabric-samples`

### Supported transformations

| Assistants API pattern | Responses API result |
| --- | --- |
| Import `FabricOpenAI` | Import `FabricOpenAIResponses` |
| Construct `FabricOpenAI(artifact_name=...)` | Construct `FabricOpenAIResponses(...)` with required configuration placeholders |
| `beta.assistants.create(...)` | Remove when it is used only for data-agent querying |
| `beta.threads.create()` | Remove or replace with a conversation, depending on detected usage |
| `threads.messages.create(...)` followed by `threads.runs.create(...)` | `responses.create(input=...)` |
| Run polling with `runs.retrieve(...)` | Response polling with `responses.retrieve(...)` |
| `threads.messages.list(...)` and `content[0].text.value` | Response output extraction helper |
| Reusing a thread for follow-up questions | `previous_response_id` chaining by default |
| Explicit long-lived thread grouping | Responses conversation object |
| `runs.steps.list(...)` | Inspection of response output items |
| Assistants streaming | Responses streaming events |
| `threads.delete(...)` | Remove when no conversation cleanup is required |
| `evaluate_data_agent(...)` | Add `client_class=FabricOpenAIResponses` |

**Current MVP status (2026-08-13):** The implemented engine covers direct
imports/client construction, sequential synchronous message/run flows,
`previous_response_id` chaining, recognized polling, plain-text extraction,
direct step inspection, cleanup, and evaluation. Conversation-object and
streaming migration remain Phase 3 work; notebooks using those patterns are
left untouched and flagged rather than being presented as supported.

## User experience

### Main workflow

1. User opens the web application.
2. User uploads one `.ipynb` file.
3. The application validates the notebook structure and file limits.
4. The migration engine analyzes all code cells.
5. The application displays:
   - overall migration status;
   - number of changed cells;
   - supported patterns migrated;
   - warnings and unsupported patterns;
   - per-cell confidence and suggested actions.
6. User downloads:
   - `<original-name>-responses-api.ipynb`;
   - `<original-name>-migration-report.json` or `.html`.

### Result statuses

- **Migrated:** All detected Assistants API patterns were transformed.
- **Migrated with warnings:** A usable notebook was generated, but manual review
  is recommended.
- **Partial migration:** Some supported changes were applied and unsupported
  code was marked for review.
- **No migration needed:** No relevant Assistants API usage was detected.
- **Invalid notebook:** The uploaded file could not be safely parsed.

## Proposed architecture

### Application layer

- A small Python web application.
- Recommended MVP framework: Streamlit for rapid delivery.
- A later production version can expose the same engine through FastAPI and use
  a separate frontend.

### Migration engine

Keep the engine independent from the UI so it can also be used from a CLI,
tests, CI pipelines, or another service.

```text
src/
  fabric_migrator/
    engine.py
    models.py
    notebook_io.py
    mapping.py
app.py
tests/
  test_migration.py
```

### Core libraries

- `nbformat` for reading, validating, and writing notebooks.
- Python `ast` location metadata for syntax-aware, range-based source patches
  that preserve text outside verified nodes byte-for-byte.
- Python standard-library parsing and line handling for detecting
  notebook-specific syntax.
- `pytest` for rule and end-to-end tests.

IPython magics and shell commands can make an entire code cell invalid Python.
The parser must identify those lines and either preserve them while transforming
safe Python regions or mark the cell for manual review.

## Migration pipeline

1. **Validate upload**
   - Confirm the file is valid JSON and a supported notebook format.
   - Enforce configurable file-size and cell-count limits.
   - Do not execute uploaded code.

2. **Inventory notebook**
   - Record notebook version, kernel, code cells, Markdown cells, imports,
     Fabric client variables, assistant variables, thread variables, run
     variables, and evaluation calls.

3. **Build a lightweight symbol map**
   - Track common aliases and assignments across cells.
   - Detect reassignment or dynamic access that makes a symbol ambiguous.
   - Track cell execution order only as advisory information; use notebook order
     as the stable migration order.

4. **Classify patterns**
   - Exact supported pattern.
   - Supported pattern with inferred variable names.
   - Ambiguous pattern requiring configuration or user review.
   - Unsupported dynamic pattern.

5. **Apply ordered migration rules**
   - Imports and client construction.
   - Assistant and thread setup.
   - Question submission and polling.
   - Answer extraction.
   - Follow-up context.
   - Streaming and intermediate steps.
   - Cleanup and evaluation.

6. **Insert generated helpers**
   - Add helpers such as `wait_for_response` and `extract_response_text` once.
   - Place them after imports or in a clearly labeled generated code cell.
   - Avoid adding a duplicate if equivalent helpers already exist.

7. **Annotate unresolved code**
   - Preserve the original code where automatic replacement is unsafe.
   - Add a nearby `TODO: Responses API migration review required` comment or
     Markdown review cell.
   - Include the same issue in the migration report.

8. **Validate output**
   - Validate notebook JSON and metadata.
   - Parse all ordinary Python code cells after transformation.
   - Confirm removed symbols are not still referenced where deterministically
     detectable.
   - Confirm generated client calls contain required configuration or explicit
     placeholders.

9. **Generate downloads**
   - Serialize the migrated notebook.
   - Generate a machine-readable report and a human-readable summary.

## Configuration handling

The old client may provide only `artifact_name`, while the new client also
requires workspace and stage information. The tool should not invent these
values.

Default generated code:

```python
fabric_client = FabricOpenAIResponses(
    artifact_name=data_agent_name,
    workspace_name=workspace_name,
    ai_skill_stage="sandbox",
)
```

If the notebook does not define `workspace_name`, insert a configuration cell
with a visible placeholder and report it as a required user action. Detect an
existing workspace variable when possible. Preserve an existing stage value,
otherwise default to `"sandbox"` and disclose that choice.

## Rule safety model

Each rule should return:

- rule identifier and version;
- source cell index;
- whether a change was applied;
- confidence: `high`, `medium`, or `manual`;
- original and replacement code ranges;
- warnings;
- required user actions.

Only high-confidence rules may delete or replace code automatically. Medium
confidence rules should generate a proposed replacement while preserving enough
context for review. Manual cases should remain unchanged and be reported.

## Patterns that must be flagged

- `getattr`, `exec`, `eval`, or generated API method names.
- Fabric client objects passed through unknown helper functions.
- Custom wrappers around assistants, threads, runs, or messages.
- Multiple assistants or threads with nontrivial routing.
- Thread IDs persisted externally and reused across notebook sessions.
- Concurrent or asynchronous run orchestration.
- Event handlers whose semantics cannot be mapped to Responses streaming.
- Output parsing that depends on annotations, attachments, or a custom schema.
- Notebook code with unresolved imports or syntax that cannot be safely parsed.
- Cells whose behavior depends on execution order different from notebook order.

## Migration report

The report should contain:

```json
{
  "source_notebook": "example.ipynb",
  "tool_version": "0.1.0",
  "status": "migrated_with_warnings",
  "summary": {
    "cells_scanned": 20,
    "cells_changed": 6,
    "rules_applied": 9,
    "warnings": 2,
    "manual_actions": 1
  },
  "changes": [],
  "warnings": [],
  "manual_actions": []
}
```

Do not present a single percentage as proof that a notebook works. If an
overall score is shown, derive it from rule coverage and unresolved findings
and label it as migration confidence, not runtime correctness.

## Validation strategy

### Fixture notebooks

Create sanitized fixtures covering:

- the Microsoft single-question sample;
- polling and output extraction;
- multi-turn response chaining;
- conversation-based context;
- streaming;
- intermediate-step inspection;
- evaluation;
- aliased imports and renamed variables;
- notebook magics;
- unsupported wrappers and dynamic code;
- partially migrated notebooks;
- idempotency.

### Automated checks

- Input notebook remains unchanged.
- Markdown, outputs, and unrelated code are preserved.
- Every supported old pattern produces the expected new pattern.
- Unsupported patterns generate warnings without destructive edits.
- Running the migration twice produces the same notebook.
- Generated notebooks pass `nbformat` validation.
- Generated ordinary Python cells compile.
- Golden-file comparisons verify complete notebook output.
- Every changed ordinary Python cell is parsed again after transformation. If
  any generated cell fails to parse, all automatic edits are rolled back.

The current suite includes synthetic edge cases plus the official Microsoft
Assistants-era query sample. The supplied automation-library notebook is a
management-plane preservation fixture and correctly produces "No migration
needed."

### Optional runtime validation

Runtime validation should be a separate, opt-in feature because it requires
Fabric access and can execute user code. For the MVP, do not execute uploaded
notebooks. Later, provide a generated validation cell or instructions the user
can run in their own Fabric environment.

## Privacy and security

- Process notebooks in memory where possible.
- Do not execute uploaded code.
- Do not send notebook contents to an LLM or external service.
- Do not log notebook source, credentials, tokens, or outputs.
- Remove temporary files after download generation.
- Apply file-size, timeout, and resource limits.
- Escape notebook content displayed in reports.
- Warn when likely secrets are detected, but do not include secret values in
  the report.

## Delivery phases

### Phase 1: Engine foundation

- Create the Python package and tests.
- Implement notebook loading, validation, cloning, and writing.
- Define migration rule and report models.
- Add analyzer inventory and symbol tracking.

**Exit criteria:** A notebook can be loaded, analyzed, written without content
loss, and accompanied by an empty structured report.

### Phase 2: Core migration

- Implement imports and client construction.
- Implement one-question message/run conversion.
- Implement response polling and text extraction helpers.
- Remove safe assistant/thread setup and cleanup.
- Add configuration placeholders and warnings.

**Exit criteria:** Standard single-question sample notebooks migrate
end-to-end with no manual edits except missing environment-specific values.

### Phase 3: Advanced patterns

- Add previous-response chaining.
- Add conversation migration.
- Add streaming and output-item migration.
- Add intermediate-step and evaluation rules.
- Improve aliases and cross-cell symbol tracking.

**Exit criteria:** All documented migration-guide patterns have fixtures and
deterministic rules.

### Phase 4: Web application

- Add notebook upload and result summary.
- Add per-cell change and warning views.
- Add notebook and report downloads.
- Add clear privacy and non-execution messaging.

**Exit criteria:** A user can complete the full upload, review, and download
workflow locally.

### Phase 5: Hardening

- Add malformed, large, unusual, and adversarial notebook tests.
- Verify idempotency and preservation behavior.
- Add telemetry limited to non-content operational metrics if needed.
- Package the application for deployment.

**Exit criteria:** Unsupported cases fail safely and supported fixture coverage
meets the target migration success rate.

## MVP acceptance criteria

- Upload and download work for valid `.ipynb` files.
- No uploaded code is executed.
- The original notebook is never modified.
- All core patterns in the migration guide are detected.
- Supported patterns are transformed using syntax-aware rules.
- Unrelated cells, Markdown, outputs, and metadata are preserved.
- Missing workspace or stage configuration is made explicit.
- Unsupported patterns remain visible and produce actionable warnings.
- The migration report identifies every changed cell and rule.
- Migration is idempotent.
- At least 80% of a curated representative fixture set migrates without manual
  code rewriting, excluding environment-specific configuration.

## Success measurement

Track:

- percentage of detected patterns automatically migrated;
- percentage of fixture notebooks requiring no code rewrite;
- false-positive transformation rate;
- notebooks that remain structurally valid;
- unresolved warnings per notebook;
- user-reported time saved.

The most important quality threshold is a near-zero rate of unsafe automatic
changes. Coverage can improve iteratively by adding new explicit rules from
real notebooks.

## Recommended first implementation

Start with a local Streamlit application and a standalone Python migration
package. Implement only the migration guide's common synchronous flow first:
client creation, single question, polling, answer extraction, cleanup, and
evaluation. Use the Microsoft sample notebooks as source material for sanitized
fixtures, then add multi-turn, streaming, and intermediate steps after the core
pipeline is stable.
