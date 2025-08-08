from __future__ import annotations

from collections.abc import Callable

from bodo.pandas import BodoSeries


def tokenize(
    series,
    tokenizer: Callable[[], Transformers.PreTrainedTokenizer],  # noqa: F821
) -> BodoSeries:
    return series.ai.tokenize(tokenizer)


def llm_generate(
    series,
    api_key: str,
    model: str | None = None,
    base_url: str | None = None,
    **generation_kwargs,
) -> BodoSeries:
    return series.ai.llm_generate(
        api_key=api_key, model=model, base_url=base_url, **generation_kwargs
    )


def embed(
    series,
    api_key: str,
    model: str | None = None,
    base_url: str | None = None,
    **embedding_kwargs,
) -> BodoSeries:
    return series.ai.embed(
        api_key=api_key, model=model, base_url=base_url, **embedding_kwargs
    )


def llm_generate_bedrock(
    series: BodoSeries,
    modelId: str,
    request_formatter: Callable[[str], str] | None = None,
    **generation_kwargs,
) -> BodoSeries:
    """Generates text using Amazon Bedrock LLMs, see BodoSeries.ai.llm_generate_bedrock for more details."""
    return series.ai.llm_generate_bedrock(
        modelId=modelId, request_formatter=request_formatter, **generation_kwargs
    )


def embed_bedrock(
    series: BodoSeries,
    modelId: str,
    request_formatter: Callable[[str], str] | None = None,
    **embedding_kwargs,
) -> BodoSeries:
    """Embeds text using Amazon Bedrock LLMs, see BodoSeries.ai.embed_bedrock for more details."""
    return series.ai.embed_bedrock(
        modelId=modelId, request_formatter=request_formatter, **embedding_kwargs
    )
