from __future__ import annotations

SDK_MAPPING = {
    "last_verified_utc": "2026-08-13",
    "observed_current_release": "0.1.29a0",
    "minimum_recommended_release": "0.1.28a0",
    "minimum_reason": (
        "0.1.27a0 introduced the complete Responses resource and 0.1.28a0 "
        "fixed normalization of non-streamed Responses results."
    ),
    "release_evidence": {
        "0.1.23a0": "PyPI changelog announced the opt-in client.",
        "0.1.27a0": "Downloaded wheel contains Responses create/stream support.",
        "0.1.28a0": "Downloaded wheel adds retrieve normalization; changelog records the non-streamed result fix.",
        "0.1.29a0": "PyPI release metadata observed on 2026-08-13; changelog records an aiohttp dependency change.",
    },
    "rules": [
        {
            "before": "FabricOpenAI",
            "after": "FabricOpenAIResponses",
            "automatic": True,
        },
        {
            "before": "assistants.create + threads.create",
            "after": "No setup objects",
            "automatic": "Only when all references are recognized",
        },
        {
            "before": "messages.create + runs.create",
            "after": "responses.create(input=...)",
            "automatic": "Only for a user text message paired with one run",
        },
        {
            "before": "runs.retrieve",
            "after": "responses.retrieve",
            "automatic": "Only for a recognized migrated run",
        },
        {
            "before": "Reuse thread",
            "after": "previous_response_id",
            "automatic": "For sequential, single-thread flows",
        },
        {
            "before": "messages.list text helper",
            "after": "Response output_text",
            "automatic": "Only for recognized plain-text helpers",
        },
        {
            "before": "runs.steps.list",
            "after": "Filtered response.output items",
            "automatic": "Only for direct step inspection",
        },
        {
            "before": "evaluate_data_agent(...)",
            "after": "client_class=FabricOpenAIResponses",
            "automatic": True,
        },
    ],
    "sources": [
        (
            "https://community.fabric.microsoft.com/t5/Fabric-Updates-Blog/"
            "Prepare-your-Fabric-Data-Agent-integrations-for-Assistants-API/"
            "ba-p/5314634"
        ),
        "https://learn.microsoft.com/fabric/fundamentals/whats-new#fabric-data-science",
        "https://learn.microsoft.com/fabric/data-science/fabric-data-agent-sdk",
        (
            "https://github.com/microsoft/fabric-samples/tree/main/docs-samples/"
            "data-science/data-agent-sdk/responses-api"
        ),
        "https://pypi.org/project/fabric-data-agent-sdk/",
    ],
}
