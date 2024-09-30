from abc import ABCMeta, abstractmethod

from numba.core import types, utils
from numba.core.cpu_options import InlineOptions
from numba.core.extending import overload
from numba.core.typing.templates import (
    AbstractTemplate,
    _OverloadMethodTemplate,
    infer_getattr,
)

from bodo.utils.typing import check_unsupported_args


def get_feature_path(feature_name):
    words = feature_name.split(".")
    if words[:2] == ["pd", "Series"]:
        return "docs/docs/api_docs/pandas/series", ".".join(words[2:])
    raise Exception(f"Unrecognized feature path {feature_name}")


def replace_package_name(path):
    """Replace abbreviated package name in path with the full name"""
    abrev_package_names = {
        "pd": "pandas",
        "np": "numpy",
    }

    path_parts = path.split(".")
    package_name = path_parts[0]

    if package_name in abrev_package_names:
        path_parts[0] = abrev_package_names[package_name]
        return ".".join(path_parts)

    return path


class DeclarativeTemplate(metaclass=ABCMeta):
    @abstractmethod
    def document(self):
        """Generate docstring corresponding to this template"""

    @abstractmethod
    def is_matching_template(self, attr):
        """Check determine when attribute *attr* matches this template."""


class _OverloadDeclarativeMethodTemplate(DeclarativeTemplate, _OverloadMethodTemplate):
    def document(self):
        title_str = f"# `{self.path_name}`"
        params = list(utils.pysignature(self._overload_func).parameters.values())
        params_str = ", ".join(map(str, params[1:]))
        pysig_str = f"`{replace_package_name(self.path_name)}({params_str})`"
        unsupported_args_str = "### Supported Arguments:"
        for param in params[1:]:
            unsupported_args_str += f"\n * `{param.name}`"
            if param.name in self.unsupported_args:
                unsupported_args_str += (
                    f": only supports default value `{param.default}`."
                )

        description = self.description
        hyperlink_str = (
            ""
            if self.hyperlink is None
            else f"[Link to external documentation]({self.hyperlink})\n\n"
        )

        # extract example from existing doc for backcompatibility
        # TODO: link examples to our testing setup to verify they are still runnable
        path, name = get_feature_path(self.path_name)
        doc_path = f"{path}/{name}.md"
        begin_example = "### Example Usage"
        with open(doc_path, "r") as f:
            doc = f.read()
            example_str = (
                "" if begin_example not in doc else doc[doc.index(begin_example) :]
            )

        # overwrite document with generated information + extracted example
        with open(doc_path, "w") as f:
            f.write(f"{title_str}\n\n")
            f.write(hyperlink_str)
            f.write(f"{pysig_str}\n\n")
            f.write(f"{unsupported_args_str}\n\n")
            f.write(f"{description}\n\n")
            f.write(example_str)

    def is_matching_template(self, attr):
        return self._attr == attr

    @classmethod
    def _check_unsupported_args(cls, kws):
        path = cls.path_name.split(".")
        assert (
            len(path) > 2
        ), "Path expected to begin with '<package_name>.<module_name>.'"
        module_name = path[1]

        # use default args from pysig (only the ones that are in cls.unsupported_args)
        parameters_dict = utils.pysignature(cls._overload_func).parameters

        # TODO: handle cases where argument does not appear in function signature
        assert all(
            k in parameters_dict for k in cls.unsupported_args
        ), "Unsupported default arguments must be found in function definition."

        args_default_dict = {
            k: parameters_dict[k].default for k in cls.unsupported_args
        }

        # insert value from kws here otherwise use default
        args_dict = {
            k: parameters_dict[k].default if k not in kws else kws[k]
            for k in cls.unsupported_args
        }

        # check unsupported defaults
        check_unsupported_args(
            cls.path_name,
            args_dict,
            args_default_dict,
            package_name="pandas",
            module_name=module_name,
        )

    def _resolve(self, typ, attr):
        if not self.is_matching_template(attr):
            return None

        if isinstance(typ, types.TypeRef):
            assert typ == self.key
        elif isinstance(typ, types.Callable):
            assert typ == self.key
        else:
            assert isinstance(typ, self.key)

        class DeclarativeMethodTemplate(AbstractTemplate):
            key = (self.key, attr)
            _inline = self._inline
            _overload_func = staticmethod(self._overload_func)
            _inline_overloads = self._inline_overloads
            prefer_literal = self.prefer_literal
            path_name = self.path_name
            unsupported_args = self.unsupported_args

            def generic(_, args, kws):
                args = (typ,) + tuple(args)
                self._check_unsupported_args(kws)
                fnty = self._get_function_type(self.context, typ)
                sig = self._get_signature(self.context, fnty, args, kws)

                sig = sig.replace(pysig=utils.pysignature(self._overload_func))
                for template in fnty.templates:
                    self._inline_overloads.update(template._inline_overloads)
                if sig is not None:
                    return sig.as_method()

        return types.BoundFunction(DeclarativeMethodTemplate, typ)


def make_overload_declarative_template(
    typ,
    attr,
    overload_func,
    path_name,
    unsupported_args,
    description,
    hyperlink=None,
    inline="never",
    prefer_literal=False,
    no_unliteral=False,
    base=_OverloadDeclarativeMethodTemplate,
    **kwargs,
):
    """
    Make a template class for method *attr* of *typ* that has autodocumenting
    functionality.
    """
    assert isinstance(typ, types.Type) or issubclass(typ, types.Type)
    name = "OverloadDeclarativeAttributeTemplate_%s_%s" % (typ, attr)
    # Note the implementation cache is subclass-specific
    dct = dict(
        key=typ,
        _attr=attr,
        path_name=path_name,
        _impl_cache={},
        _inline=staticmethod(InlineOptions(inline)),
        _inline_overloads={},
        _overload_func=staticmethod(overload_func),
        prefer_literal=prefer_literal,
        _no_unliteral=no_unliteral,
        unsupported_args=unsupported_args,
        description=description,
        hyperlink=hyperlink,
        metadata=kwargs,
    )
    obj = type(base)(name, (base,), dct)
    return obj


def overload_method_declarative(
    typ, attr, path_name, unsupported_args, description, hyperlink=None, **kwargs
):
    """Common code for overload_method and overload_classmethod"""

    def decorate(overload_func):
        copied_kwargs = kwargs.copy()
        base = _OverloadDeclarativeMethodTemplate

        # NOTE: _no_unliteral is a bodo specific attribute and is linked to changes in numba_compat.py
        template = make_overload_declarative_template(
            typ,
            attr,
            overload_func,
            path_name,
            unsupported_args,
            description,
            hyperlink=hyperlink,
            inline=copied_kwargs.pop("inline", "never"),
            prefer_literal=copied_kwargs.pop("prefer_literal", False),
            no_unliteral=copied_kwargs.pop("no_unliteral", False),
            base=base,
            **copied_kwargs,
        )
        infer_getattr(template)
        overload(overload_func, **kwargs)(overload_func)
        return overload_func

    return decorate
