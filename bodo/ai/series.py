from __future__ import annotations

import typing as pt
from collections.abc import Callable

from bodo.ai.backend import Backend

if pt.TYPE_CHECKING:
    from bodo.pandas import BodoSeries


def tokenize(
    series,
    tokenizer: Callable[[], Transformers.PreTrainedTokenizer],  # noqa: F821
) -> BodoSeries:
    return series.ai.tokenize(tokenizer)


def llm_generate(
    series,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    request_formatter: Callable[[str], str] | None = None,
    response_formatter: Callable[[str], str] | None = None,
    region: str | None = None,
    backend: Backend = Backend.OPENAI,
    **generation_kwargs,
) -> BodoSeries:
    return series.ai.llm_generate(
        api_key=api_key,
        model=model,
        base_url=base_url,
        request_formatter=request_formatter,
        response_formatter=response_formatter,
        region=region,
        backend=backend,
        **generation_kwargs,
    )


def embed(
    series,
    api_key: str,
    model: str | None = None,
    base_url: str | None = None,
    request_formatter: Callable[[str], str] | None = None,
    response_formatter: Callable[[str], list[float]] | None = None,
    region: str | None = None,
    backend: Backend = Backend.OPENAI,
    **embedding_kwargs,
) -> BodoSeries:
    return series.ai.embed(
        api_key=api_key,
        model=model,
        base_url=base_url,
        request_formatter=request_formatter,
        response_formatter=response_formatter,
        region=region,
        backend=backend,
        **embedding_kwargs,
    )
