"""
Script for generating documentation for Series methods and attributes using
Declarative Templates.
    usage: python -m bodo.utils.generate_docs
"""

import numba
from numba.core.target_extension import dispatcher_registry

import bodo  # noqa
from bodo.ir.declarative_templates import DeclarativeTemplate
from bodo.utils.pandas_coverage_tracking import PANDAS_URLS, get_pandas_refs_from_url
from bodo.utils.search_templates import bodo_pd_types_dict, get_overload_template


def generate_pandas_docs(module: str, types: set[str]):
    """Generate a subset of the pandas API's supported by bodo for methods and
    attributes.

    Example:
        `generate_pandas_docs("Index", {"Index", "CategoricalIndex"})`

        Generates documentation for supported methods and attributes of
        `Index` and `CategoricalIndex`.

    Args:
        module: The module to gather availible APIs from (see
            .pandas_coverage_tracking.PANDAS_URLS).
        types: The subset of base classes to generate documentation
            for.
    """
    index = PANDAS_URLS[module.upper()]
    api_refs = get_pandas_refs_from_url(index)

    disp = dispatcher_registry[numba.core.target_extension.CPU]
    typing_ctx = disp.targetdescr.typing_context
    typing_ctx.refresh()

    types_dict = bodo_pd_types_dict

    for ref in api_refs:
        path = ref.text.strip().split(".")
        if path[0] in types:
            assert path[0] in types_dict, f"Could not match {path[0]} to bodo type(s)"
            base_types = bodo_pd_types_dict[path[0]]
            template = get_overload_template(typing_ctx, base_types, path[1:])

            if isinstance(template, DeclarativeTemplate):
                href = ref.attrs["href"]
                hyperlink = index.rsplit("/", 1)[0] + "/" + href
                template.hyperlink = hyperlink
                template.document()


if __name__ == "__main__":
    # generate docs attributes/methods of Series
    generate_pandas_docs("Series", {"Series"})
