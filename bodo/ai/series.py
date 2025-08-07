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
