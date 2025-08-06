from __future__ import annotations

from collections.abc import Callable

from bodo.pandas import BodoSeries


def tokenize(
    series,
    tokenizer: Callable[[], Transformers.PreTrainedTokenizer],  # noqa: F821
) -> BodoSeries:
    return series.ai.tokenize(tokenizer)


def llm_generate(
    series, endpoint: str, api_token: str, model: str | None = None, **generation_kwargs
) -> BodoSeries:
    return series.ai.llm_generate(
        endpoint=endpoint, api_token=api_token, model=model, **generation_kwargs
    )


def embed(
    series, endpoint: str, api_token: str, model: str | None = None, **embedding_kwargs
) -> BodoSeries:
    return series.ai.embedd(
        endpoint=endpoint, api_token=api_token, model=model, **embedding_kwargs
    )
