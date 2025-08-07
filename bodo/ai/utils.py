import json
from typing import Callable


def get_default_bedrock_request_formatter(modelId: str) -> Callable[[str], str]:
    if modelId.startswith("amazon.nova"):
        return lambda input: json.dumps(
            {"messages": [{"role": "user", "content": [{"text": input}]}]}
        )
    elif modelId.startswith("amazon.titan-text"):
        return lambda input: json.dumps({"inputText": input})
    elif modelId.startswith("amazon.titan-embed"):
        return lambda input: json.dumps({"inputText": input})
    elif modelId.startswith("anthropic.claude"):
        return lambda input: json.dumps({"prompt": f"\n\nHuman: {input}\n\nAssistant:"})
    elif modelId.startswith("openai.gpt"):
        return lambda input: json.dumps(
            {"messages": [{"role": "user", "content": input}]}
        )

    raise ValueError(
        f"Unsupported modelId {modelId} for Bedrock request formatting. "
        "Please provide a custom request formatter."
    )
