import pandas as pd  # noqa
import pytest

import numba
from numba.core import utils, errors
from numba.core.target_extension import dispatcher_registry

import bodo  # noqa
from bodo.utils.pandas_coverage_tracking import get_pandas_apis_from_url
from bodo.utils.search_templates import bodo_pd_types_dict, is_attr_supported
from bodo.ir.declarative_templates import _OverloadDeclarativeMethodTemplate
from bodo.utils.typing import BodoError
from bodo.tests.utils import run_rank0


class _OverloadMissingOrIncorrect:
    """
    Sentinal class to indicate is_attr_supported was unsuccessful in finding a valid
    overload template
    """

    pass


def _get_series_apis():
    url = "https://pandas.pydata.org/docs/reference/series.html"
    return get_pandas_apis_from_url(url)


def _get_method_template(typing_ctx, types, attrs):
    """
    Try to get a template of attrs from typ. If template is missing or if an supported
    and unsupported overload template are found, returns an instance of
    _OverloadMissingOrIncorrect
    """
    typ = None
    template = None

    for base_type in types:
        typ = base_type
        try:
            for attr in attrs:
                is_supported = is_attr_supported(typing_ctx, typ, attr)
                if is_supported:
                    result = typing_ctx.find_matching_getattr_template(typ, attr)
                    typ = result["return_type"]
                    template = result["template"]
                elif is_supported is None:
                    # is_attr_supported was unsuccessful
                    # No template or inconsistent templates
                    return _OverloadMissingOrIncorrect()
                else:
                    # UnsupportedTemplate case return None
                    return None
        except Exception as e:
            # catch bodo/numba errors for invalid input type and raise everything else
            if not isinstance(e, errors.TypingError) and not isinstance(e, BodoError):
                raise e

    return template


def _get_pysig_from_path(path, package_name="pd"):
    """Gets the python signature from a list of strings"""
    attr = globals()[package_name]

    for part in path:
        attr = getattr(attr, part, None)
        assert attr, f"path not found: {path}"

    return utils.pysignature(attr)


def signatures_equal(pysig, overload_sig, changed_defaults):
    """
    Compare **pysig** to **overload_sig** on all parameters not listed in
    **changed_defaults** or *args/**kwargs
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


def _skip_pysig_check(path):
    """
    Returns whether to skip the check that pysig for **path** is consistent with
    external api documentation.
    """
    # TODO (fix in pandas): Series.dt.xxx methods have generic signatures that look
    # like (self, *arg, **kwargs)
    if path[:2] == ["Series", "dt"]:
        return True
    return False


@pytest.mark.slow
@pytest.mark.parametrize(
    "get_apis, keys", [pytest.param(_get_series_apis, ["Series"], id="series")]
)
@run_rank0
def test_pandas_pysigs(get_apis, keys):
    """
    Check that the pysignature of overloaded series methods matches the pysig from
    pandas. Additionally checks all Series methods are either have supported or
    unsupported (but not both).
    """
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
            template = _get_method_template(typing_ctx, base_types, path[1:])

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

    assert (
        len(invalid_ser_attrs) == 0
    ), f"Found {len(invalid_ser_attrs)} Series attribute(s) that either do not have an overload or have both unsupported and supported overloads: {invalid_ser_attrs_str}"
    assert (
        total_decl_methods == total_correct_signature
    ), f"Found differences in the following method signatures:\n{diff_str}"
