from __future__ import annotations

from collections.abc import Callable

from bodo.pandas import BodoSeries


def tokenize(
    series,
    tokenizer: Callable[[], Transformers.PreTrainedTokenizer],  # noqa: F821
) -> BodoSeries:
    return series.ai.tokenize(tokenizer)
