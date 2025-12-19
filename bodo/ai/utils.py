from __future__ import annotations

import json
from collections.abc import Callable


def get_default_bedrock_request_formatter(modelId: str) -> Callable[[str], str]:
    if "amazon.nova" in modelId:
        return lambda input: json.dumps(
            {"messages": [{"role": "user", "content": [{"text": input}]}]}
        )
    elif "amazon.titan-embed" in modelId:
        return lambda input: json.dumps({"inputText": input})
    elif "anthropic.claude" in modelId:
        return lambda input: json.dumps(
            {
                "messages": [{"role": "user", "content": input}],
                "max_tokens": 4000,
                "anthropic_version": "bedrock-2023-05-31",
            }
        )
    elif "openai" in modelId:
        return lambda input: json.dumps(
            {"messages": [{"role": "user", "content": input}]}
        )

    raise ValueError(
        f"Unsupported modelId {modelId} for Bedrock request formatting. "
        "Please provide a custom request formatter."
    )


def get_default_bedrock_response_formatter(
    modelId: str,
) -> Callable[[str], str | list[float]]:
    if "amazon.nova" in modelId:
        return lambda output: json.loads(output)["output"]["message"]["content"][0][
            "text"
        ]
    elif "amazon.titan-embed" in modelId:
        return lambda output: json.loads(output)["embedding"]
    elif "anthropic.claude" in modelId:
        return lambda output: json.loads(output)["content"][0]["text"]
    elif "openai" in modelId:
        return lambda output: json.loads(output)["choices"][0]["message"]["content"]

    raise ValueError(
        f"Unsupported modelId {modelId} for Bedrock reponse formatting. "
        "Please provide a custom response formatter."
    )
