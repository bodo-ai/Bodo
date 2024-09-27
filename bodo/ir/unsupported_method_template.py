from abc import ABCMeta, abstractmethod

import numba
from numba.core import types
from numba.core.typing.templates import (
    AttributeTemplate,
    infer_getattr,
)

from bodo.utils.typing import BodoError


class _UnsupportedTemplate(metaclass=ABCMeta):
    @abstractmethod
    def is_matching_template(self, attr):
        """Check to determine when an attribute *attr* matches this template."""


class _OverloadUnsupportedMethodTemplate(_UnsupportedTemplate, AttributeTemplate):
    """A base class of templates for overload_unsupported_method."""

    def is_matching_template(self, attr):
        return self._attr == attr

    def _resolve(self, typ, attr):
        if not self.is_matching_template(attr):
            return None

        if isinstance(typ, types.TypeRef):
            assert typ == self.key
        elif isinstance(typ, types.Callable):
            assert typ == self.key
        else:
            assert isinstance(typ, self.key)

        class UnsupportedMethodTemplate(numba.core.typing.templates.AbstractTemplate):
            key = (self.key, attr)
            path_name = self.path_name
            extra_info = self.extra_info

            def generic(self, args, kws):
                raise BodoError(
                    f"{self.path_name}(){self.extra_info} not supported yet."
                )

        return types.BoundFunction(UnsupportedMethodTemplate, typ)


class _OverloadUnsupportedAttributeTemplate(_UnsupportedTemplate, AttributeTemplate):
    """A base class of templates for overload_unsupported_attribute."""

    def is_matching_template(self, attr):
        return self._attr == attr

    def _resolve(self, typ, attr):
        if not self.is_matching_template(attr):
            return None

        raise BodoError(f"{self.path_name}{self.extra_info} not supported yet.")


def make_overload_unsupported_template(typ, base, attr, path_name, extra_info):
    """
    Make a template class for attribute/method *attr* of *typ* that is not yet supported
    by Bodo.
    """
    assert isinstance(typ, types.Type) or issubclass(typ, types.Type)
    name = "OverloadUnsupportedAttributeTemplate_%s_%s" % (typ, attr)
    # Note the implementation cache is subclass-specific
    dct = dict(
        key=typ, _attr=attr, path_name=path_name, extra_info=extra_info, metadata={}
    )
    obj = type(base)(name, (base,), dct)
    return obj


def overload_unsupported_attribute(typ, attr, path_name, extra_info=""):
    """Create an overload for attribute *attr* of *typ* which raises a BodoError"""
    base = _OverloadUnsupportedAttributeTemplate
    template = make_overload_unsupported_template(
        typ, base, attr, path_name, extra_info
    )
    infer_getattr(template)


def overload_unsupported_method(typ, attr, path_name, extra_info=""):
    """Create an overload for method *attr* of *typ* which raises a BodoError"""
    base = _OverloadUnsupportedMethodTemplate
    template = make_overload_unsupported_template(
        typ, base, attr, path_name, extra_info
    )
    infer_getattr(template)
