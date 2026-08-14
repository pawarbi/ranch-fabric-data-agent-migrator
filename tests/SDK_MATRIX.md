# Fabric Data Agent SDK evidence matrix

RANCH's minimum generated requirement is `fabric-data-agent-sdk>=0.1.28a0`.
The floor is based on the first release that documents normalized OpenAI
`Response` objects for both non-streamed `responses.create()` and
`responses.retrieve()`, not merely the first appearance of the opt-in client.

| Release | Version-specific evidence relevant to RANCH |
| --- | --- |
| `0.1.23a0` | Its PyPI page records duplicate-fewshot and evaluation model/prompt fixes. It does not document a Responses contract. |
| `0.1.26a0` | Its PyPI page introduces public management-plane APIs through `FabricDataAgentManagement`; it does not document a Responses contract. |
| `0.1.27a0` | PyPI documents the `FabricOpenAIResponses` export, `responses.create()` / streaming, `previous_response_id`, and `client_class=FabricOpenAIResponses` for evaluation. |
| `0.1.28a0` | PyPI adds the non-streamed normalization guarantee: `responses.create()` and `responses.retrieve()` return OpenAI `Response` objects even when Fabric sends SSE. |
| `0.1.29a0` | The `0.1.28a0` Responses contract remains; the release-specific change is the `aiohttp>=3.10,<4` dependency. |

The official migration sample uses:

```python
from fabric.dataagent.client import FabricOpenAIResponses

client = FabricOpenAIResponses(
    artifact_name=data_agent_name,
    workspace_name=workspace_name,
    ai_skill_stage="sandbox",
)
response = client.responses.create(input=question)
follow_up = client.responses.create(
    input=follow_up_question,
    previous_response_id=response.id,
)
```

Evaluation receives the class object, not a constructed instance:
`evaluate_data_agent(..., client_class=FabricOpenAIResponses)`.

Sources:

- Version-specific PyPI JSON: `https://pypi.org/pypi/fabric-data-agent-sdk/<version>/json`
- Current package description: <https://pypi.org/project/fabric-data-agent-sdk/>
- Official migration guide: <https://github.com/microsoft/fabric-samples/blob/main/docs-samples/data-science/data-agent-sdk/responses-api/responses-api-migration-guide.md>
- Official sample notebook: <https://github.com/microsoft/fabric-samples/blob/main/docs-samples/data-science/data-agent-sdk/responses-api/responses-api-notebook.ipynb>
