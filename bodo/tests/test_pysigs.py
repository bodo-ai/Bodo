from __future__ import annotations

from inspect import Signature

import numba  # noqa TID253
import pandas as pd  # noqa
import pytest
from numba.core import utils  # noqa TID253
from numba.core.target_extension import dispatcher_registry  # noqa TID253

import bodo  # noqa
from bodo.tests.utils import pytest_mark_one_rank


def _get_series_apis() -> list[str]:
    """Get paths of all Series and attributes"""
    from bodo.utils.pandas_coverage_tracking import (
        PANDAS_URLS,
        get_pandas_apis_from_url,
    )

    url = PANDAS_URLS["SERIES"]
    return get_pandas_apis_from_url(url)


def _get_pysig_from_path(path: list[str], package_name: str | None = "pd") -> Signature:
    """Gets the python signature from `path`"""
    attr = globals()[package_name]

    for part in path:
        attr = getattr(attr, part, None)
        assert attr, f"path not found: {path}"

    return utils.pysignature(attr)


def signatures_equal(
    pysig: Signature, overload_sig: Signature, changed_defaults: set[str]
) -> bool:
    """Compares overload signature to the signature from the
    corresponding python API.

    Extra `*args` and `**kwargs` are dropped from the python API
    signature and explicitly changed defaults get ignored.

    Args:
        pysig (Signature): The signature of the python API.
        overload_sig (Signature): The signature of the
            corresponding overload function.
        changed_defaults (pt.Set[str]): A set of arguments whose
            default value differs from the python equivalent.
            Arguments in this set are excluded from the equality
            check.

    Returns:
        bool: True if the two signatures are equivalent.
    """
    overload_params = list(overload_sig.parameters.values())
    python_params = list(pysig.parameters.values())

    # remove the *args and **kwargs from the end of pandas signature if they are present
    if python_params[-1].kind == utils.pyParameter.VAR_KEYWORD:
        python_params.pop()  # **kwargs
    if python_params[-1].kind == utils.pyParameter.VAR_POSITIONAL:
        python_params.pop()  # *args

    if len(overload_params) != len(python_params):
        return False

    for overload_param, python_param in zip(overload_params, python_params):
        # For methods that use "self", accept other names e.g. "S"
        if python_param.name != "self" and overload_param.name != python_param.name:
            return False
        if (
            overload_param.name not in changed_defaults
            and overload_param.default != python_param.default
        ):
            return False

    return True


def _skip_pysig_check(path: str) -> bool:
    """
    Indicates whether to skip the check that pysig for **path** is
    consistent with external api documentation.
    """
    # TODO (fix in pandas): Series.dt.xxx methods have generic signatures that look
    # like (self, *arg, **kwargs)
    if path[:2] == ["Series", "dt"]:
        return True
    return False


@pytest.mark.skip(reason="Pandas 3 introduced new APIs")
@pytest.mark.slow
@pytest.mark.parametrize(
    "get_apis, keys", [pytest.param(_get_series_apis, ["Series"], id="series")]
)
@pytest_mark_one_rank
def test_pandas_pysigs(get_apis, keys):
    """
    Check that the pysignature of overloaded series methods matches the pysig from
    pandas. Additionally checks all Series methods are either have supported or
    unsupported (but not both).
    """
    from bodo.ir.declarative_templates import _OverloadDeclarativeMethodTemplate
    from bodo.utils.search_templates import (
        _OverloadMissingOrIncorrect,
        bodo_pd_types_dict,
        get_overload_template,
    )

    apis = get_apis()
    disp = dispatcher_registry[numba.core.target_extension.CPU]
    typing_ctx = disp.targetdescr.typing_context

    types_dict = bodo_pd_types_dict

    diff_str = ""
    total_decl_methods = 0
    total_correct_signature = 0
    invalid_ser_attrs = set()

    for api in apis:
        path = api.split(".")
        if path[0] in keys:
            assert path[0] in types_dict, f"Could not match {path[0]} to bodo type(s)"
            base_types = bodo_pd_types_dict[path[0]]
            template = get_overload_template(typing_ctx, base_types, path[1:])

            # check that method is either supported or unsupported (but not both)
            if isinstance(template, _OverloadMissingOrIncorrect):
                invalid_ser_attrs.add(api)

            # check that the pysig matches with pandas API docs
            elif isinstance(
                template, _OverloadDeclarativeMethodTemplate
            ) and not _skip_pysig_check(path):
                pysig = _get_pysig_from_path(path)
                overload_sig = template.get_signature()
                changed_defaults = getattr(template, "changed_defaults", frozenset())
                is_equivalent_signatures = signatures_equal(
                    pysig, overload_sig, changed_defaults
                )
                # keep track of all signatures that do not match and fail at the end.
                if not is_equivalent_signatures:
                    diff_str += f"\t{api}: overloaded signature: {str(overload_sig)} != {str(pysig)}\n"
                else:
                    total_correct_signature += 1
                total_decl_methods += 1

    invalid_ser_attrs_str = ", ".join(invalid_ser_attrs)

    assert len(invalid_ser_attrs) == 0, (
        f"Found {len(invalid_ser_attrs)} Series attribute(s) that either do not have an overload or have both unsupported and supported overloads: {invalid_ser_attrs_str}"
    )
    assert total_decl_methods == total_correct_signature, (
        f"Found differences in the following method signatures:\n{diff_str}"
    )
