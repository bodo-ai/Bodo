import inspect
import typing as pt
import warnings
from abc import ABCMeta, abstractmethod
from inspect import Parameter, Signature
from pathlib import Path

from numba.core import types, utils
from numba.core.cpu_options import InlineOptions
from numba.core.extending import overload
from numba.core.typing.templates import (
    AbstractTemplate,
    _OverloadAttributeTemplate,
    _OverloadMethodTemplate,
    infer_getattr,
)

from bodo.ir.argument_checkers import OverloadArgumentsChecker, OverloadAttributeChecker
from bodo.utils.typing import BodoError, check_unsupported_args


### Helpers for generating documentation.
def get_feature_path(feature_name: str) -> str:  # pragma: no cover
    """
    Gets the path in Series docs given a `feature_name`.
    """
    words = feature_name.split(".")
    if words[:2] == ["pd", "Series"]:
        return "docs/docs/api_docs/pandas/series", ".".join(words[2:])
    raise Exception(f"Unrecognized feature path {feature_name}")


def replace_package_name(path: str) -> tuple[str, str]:  # pragma: no cover
    """Replace abbreviated package name in `path` with the full name"""
    abrev_package_names = {
        "pd": "pandas",
        "np": "numpy",
    }

    path_parts = path.split(".")
    package_name = path_parts[0]

    if package_name in abrev_package_names:
        package_name = abrev_package_names[package_name]
        path_parts[0] = package_name
        return package_name, ".".join(path_parts)

    return package_name, path


def get_example_from_docs(doc_path: str) -> str:  # pragma: no cover
    """
    Searches for index of "### Example Usage" in a document and returns all
    text following.
    """
    example_str = ""
    if Path(doc_path).is_file():
        begin_example = "### Example Usage"
        with open(doc_path) as f:
            doc = f.read()
            if begin_example in doc:
                example_str = f"{doc[doc.index(begin_example) :].strip()}\n\n"
    return example_str


def format_argument_restrictions_str(
    params: list[Parameter],
    arg_restrictions: dict[str, str],
    unsupported_args: dict[str, pt.Any],
) -> str:  # pragma: no cover
    """Creates a bulleted list of argument restrictions for documentation.

    This list starts with a header "### Argument restrictions" and then adds a
    bullet point for each argument that looks like 'name: restrictions'. If an
    argument does not have a restriction then _restrictions_ is "None".

    Args:
        params: A list of parameters from the overloaded function.
        arg_restrictions: Mapping from parameter's name to a string descripton
            of it's restrictions.
        unsupported_args: Mapping from parameter's name to it's default value.

    Returns:
        A bulleted list of argument restrictions plus a header (or "" if there
            are no arguments)
    """
    if len(params) == 0:
        return ""

    argument_restrictions_str = "### Argument Restrictions:"
    for param in params:
        argument_restrictions_str += f"\n * `{param.name}`: "
        if param.name in unsupported_args:
            argument_restrictions_str += (
                f"only supports default value `{param.default}`."
            )
        elif param.name in arg_restrictions:
            argument_restrictions_str += f"{arg_restrictions[param.name]}."
        else:
            argument_restrictions_str += "None."

    return argument_restrictions_str + "\n\n"


class DeclarativeTemplate(metaclass=ABCMeta):
    @abstractmethod
    def document(self) -> str:
        """Generate docstring corresponding to this template"""

    @abstractmethod
    def is_matching_template(self, attr: str) -> bool:
        """Check to determine when attribute `attr` matches this template."""


class _OverloadDeclarativeMethodTemplate(DeclarativeTemplate, _OverloadMethodTemplate):
    def document(self, write_out: bool | None = True) -> str:  # pragma: no cover
        """
        Generates a documentation string for the method and writes to
        corresponding file in the documentation if `write_out`.
        """
        title_str = f"# `{self.path}`"
        params_dict = utils.pysignature(self._overload_func).parameters
        params_list = list(params_dict.values())
        params_str = ", ".join(map(str, params_list[1:]))

        # Signature taken from the overload defintion
        package_name, full_path = replace_package_name(self.path)
        pysig_str = f"`{full_path}({params_str})`"

        # Write bullet point for each argument and any restriction imposed by Bodo
        arg_restrictions = (
            {}
            if self.method_args_checker is None
            else self.method_args_checker.explain_args()
        )
        argument_restrictions_str = format_argument_restrictions_str(
            params_list[1:], arg_restrictions, self.unsupported_args
        )

        # Separate restriction on "self" argument from other arguments for clarity
        supported_types_str = ""
        if "self" in arg_restrictions:
            supported_types_str += f"!!! note\n\tInput {arg_restrictions['self']}.\n\n"
        argument_restrictions_str += supported_types_str

        # Defaults that have changed between Bodo and Package (optional)
        changed_defaults_str = ""
        for changed_arg in self.changed_defaults:
            default_value = params_dict[changed_arg].default
            changed_defaults_str += f"!!! note\n\tArgument `{changed_arg}` has default value `{default_value}` that's different than {package_name.capitalize()} default.\n\n"

        # Additional notes (manually specified in overload)
        description = "" if self.description is None else f"{self.description}\n\n"
        hyperlink_str = (
            ""
            if self.hyperlink is None
            else f"[Link to {package_name.capitalize()} documentation]({self.hyperlink})\n\n"
        )

        # Extract example from existing doc for backcompatibility
        # TODO: link examples to our testing setup to verify they are still runnable
        path, name = get_feature_path(self.path)
        doc_path = f"{path}/{name}.md"
        example_str = get_example_from_docs(doc_path)
        if example_str == "":
            warnings.warn(
                f"No example found for {self.path}: example must be manually embedded in {doc_path}."
            )

        documentation = (
            f"{title_str}\n\n"
            f"{hyperlink_str}"
            f"{pysig_str}\n\n"
            f"{argument_restrictions_str}"
            f"{changed_defaults_str}"
            f"{description}"
            f"{example_str}"
        )

        # Overwrite document with generated information + extracted example
        if write_out:
            with open(doc_path, "w") as f:
                f.write(documentation)

        return documentation

    def is_matching_template(self, attr: str) -> bool:
        return self._attr == attr

    def get_signature(self) -> Signature:
        return utils.pysignature(self._overload_func)

    @classmethod
    def _check_unsupported_args(cls, kws: dict[str, pt.Any]):
        """Checks that unsupported argument requirements are enforced on
        `kws`.

        Checks that all arguments in `kws` are either supported (at least
        partially), or if they are unsupported, checks that they are equal to
        the default value as specified by the `unsupported_args` attribute.

        Args:
            kws: The key word arguments to check.
        """
        path = cls.path.split(".")
        assert len(path) > 2, (
            "Path expected to begin with '<package_name>.<module_name>.'"
        )
        module_name = path[1]

        # use default args from pysig (only the ones that are in cls.unsupported_args)
        parameters_dict = utils.pysignature(cls._overload_func).parameters

        # TODO: handle cases where argument does not appear in function signature
        assert all(k in parameters_dict for k in cls.unsupported_args), (
            "Unsupported default arguments must be found in function definition."
        )

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
            cls.path,
            args_dict,
            args_default_dict,
            package_name="pandas",
            module_name=module_name,
        )

    @classmethod
    def _check_argument_types(cls, args: tuple, kws: dict[str, pt.Any]):
        """Checks that `args` and `kws` are valid arguments.

        Checks that `args` and `kws` are valid arguments to the method using
        `method_args_checkers` (or does nothing if a `method_args_checker`
        does not exist).

        Args:
            args: The positional arguments to check.
            kws: The keyword arguments to check.

        Raises:
            BodoError: If any of the arguments are invalid or number of
                arguments is incorrect.
        """
        if cls.method_args_checker is None:
            return

        # match every arg and kwarg with it's argument name in the signature
        overload_params = utils.pysignature(cls._overload_func).parameters
        arg_types = []
        for i, (name, param) in enumerate(overload_params.items()):
            if i < len(args):  # positional only
                name = "self" if i == 0 else name
                arg_types.append((name, args[i]))
            elif name in kws:
                arg_types.append((name, kws[name]))
            elif param.default != inspect._empty:
                # no argument supplied, still check default for consistency
                arg_types.append((name, param.default))
            else:
                raise BodoError(f"{cls.path}(): required argument {name} not supplied.")

        cls.method_args_checker.check_args(f"{cls.path}()", dict(arg_types))

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
            _no_unliteral = self._no_unliteral
            _overload_func = staticmethod(self._overload_func)
            _inline_overloads = self._inline_overloads
            prefer_literal = self.prefer_literal
            path = self.path
            unsupported_args = self.unsupported_args

            def generic(_, args, kws):
                args = (typ,) + tuple(args)
                self._check_unsupported_args(kws)
                self._check_argument_types(args, kws)
                fnty = self._get_function_type(self.context, typ)
                sig = self._get_signature(self.context, fnty, args, kws)

                sig = sig.replace(pysig=utils.pysignature(self._overload_func))
                for template in fnty.templates:
                    self._inline_overloads.update(template._inline_overloads)
                if sig is not None:
                    return sig.as_method()

        return types.BoundFunction(DeclarativeMethodTemplate, typ)


class _OverloadDeclarativeAttributeTemplate(
    DeclarativeTemplate, _OverloadAttributeTemplate
):
    def is_matching_template(self, attr: str) -> bool:
        return self._attr == attr

    def document(self, write_out: bool | None = True) -> str:  # pragma: no cover
        """
        Generates a documentation string for the method, writes to
        corresponding file in documentation if `write_out` is True.
        """
        title_str = f"# `{self.path}`\n\n"
        package_name, path_str = replace_package_name(self.path)

        hyperlink_str = (
            ""
            if self.hyperlink is None
            else f"[Link to {package_name.capitalize()} documentation]({self.hyperlink})\n\n"
        )
        path_str = f"`{path_str}`\n\n"

        supported_types = ""
        if self.arg_checker is not None:
            explain_attr = self.arg_checker.explain_args()
            supported_types += f"!!! note\n\tInput {explain_attr}.\n\n"

        description_str = (
            "" if self.description is None else f"{self.description.strip()}\n\n"
        )

        path, name = get_feature_path(self.path)
        doc_path = f"{path}/{name}.md"
        example_str = get_example_from_docs(doc_path)
        if example_str == "":
            warnings.warn(
                f"No example found for {self.path}: example must be manually embedded in {doc_path}."
            )

        documentation = "".join(
            [
                title_str,
                hyperlink_str,
                path_str,
                supported_types,
                description_str,
                example_str,
            ]
        )

        if write_out:
            with open(doc_path, "w") as f:
                f.write(documentation)

        return documentation

    @classmethod
    def _check_type(cls, typ):
        """Check that the type of an object is valid for this attribute"""
        if cls.arg_checker is not None:
            # get the argument name from the overload
            cls.arg_checker.check_args(cls.path, typ)

    def _resolve(self, typ, attr):
        if not self.is_matching_template(attr):
            return None

        self._check_type(typ)
        fnty = self._get_function_type(self.context, typ)
        sig = self._get_signature(self.context, fnty, (typ,), {})
        # There should only be one template
        for template in fnty.templates:
            self._inline_overloads.update(template._inline_overloads)
        return sig.return_type


def make_overload_declarative_method_template(
    typ,
    attr,
    overload_func,
    path,
    unsupported_args,
    method_args_checker,
    description,
    changed_defaults=frozenset(),
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
    name = f"OverloadDeclarativeAttributeTemplate_{typ}_{attr}"
    # Note the implementation cache is subclass-specific
    dct = {
        "key": typ,
        "_attr": attr,
        "path": path,
        "_impl_cache": {},
        "_inline": staticmethod(InlineOptions(inline)),
        "_inline_overloads": {},
        "_overload_func": staticmethod(overload_func),
        "prefer_literal": prefer_literal,
        "_no_unliteral": no_unliteral,
        "unsupported_args": unsupported_args,
        "changed_defaults": changed_defaults,
        "method_args_checker": method_args_checker,
        "description": description,
        "hyperlink": hyperlink,
        "metadata": kwargs,
    }
    obj = type(base)(name, (base,), dct)
    return obj


def overload_method_declarative(
    typ,
    attr: str,
    path: str,
    unsupported_args: set[str],
    description: str,
    method_args_checker: OverloadArgumentsChecker | None = None,
    changed_defaults: set[str] | None = frozenset(),
    hyperlink: str | None = None,
    **kwargs,
):
    """A decorator for creating an overload declarative template for a method.

    This decorator marks the decorated function as typing and implementing
    attribute `attr` for the given Numba type in nopython mode. The
    "declarative" aspects of this decorator allow for specifiying verifiable
    information about the attribute that can be used to generate
    documentation.

    Args:
        path: The path starting with the library that corresponds to *attr*
        unsupported_defaults (set): The set of arguments that only support
            their default value.
        description: A description of additional notes to be displayed in
            documentation.
        arg_checker: An overload checker which checks that
            `typ` and method args obey specific properties at compile time and
            raises an error. This check is translated into documentation.
            Defaults to None.
        changed_defaults: The set of arguments whose default value differs from
            the corresponding python API.
        hyperlink: Link to external documentation. Defaults to None.
        **kwargs: Additional key word arguments accepted by Bodo/Numba for
            overloaded methods (see numba_compat.py)
    """

    def decorate(overload_func: pt.Callable):
        copied_kwargs = kwargs.copy()
        base = _OverloadDeclarativeMethodTemplate
        # NOTE: _no_unliteral is a bodo specific attribute and is linked to changes in numba_compat.py
        template = make_overload_declarative_method_template(
            typ,
            attr,
            overload_func,
            path,
            unsupported_args,
            method_args_checker,
            description,
            changed_defaults=changed_defaults,
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


def make_overload_declarative_attribute_template(
    typ,
    attr,
    overload_func,
    path,
    arg_checker,
    description,
    hyperlink=None,
    inline="never",
    prefer_literal=False,
    no_unliteral=False,
    base=_OverloadDeclarativeAttributeTemplate,
    **kwargs,
):
    """
    Make a template class for method *attr* of *typ* that has autodocumenting
    functionality.
    """
    assert isinstance(typ, types.Type) or issubclass(typ, types.Type)
    name = f"OverloadDeclarativeAttributeTemplate_{typ}_{attr}"
    # Note the implementation cache is subclass-specific
    dct = {
        "key": typ,
        "_attr": attr,
        "path": path,
        "_impl_cache": {},
        "_inline": staticmethod(InlineOptions(inline)),
        "_inline_overloads": {},
        "_overload_func": staticmethod(overload_func),
        "prefer_literal": prefer_literal,
        "_no_unliteral": no_unliteral,
        "arg_checker": arg_checker,
        "description": description,
        "hyperlink": hyperlink,
        "metadata": kwargs,
    }
    obj = type(base)(name, (base,), dct)
    return obj


def overload_attribute_declarative(
    typ,
    attr: str,
    path: str,
    description: str,
    arg_checker: OverloadAttributeChecker | None = None,
    hyperlink: str | None = None,
    **kwargs,
):
    """A decorator for creating an overload declarative template for an
    attribute.

    This decorator marks the decorated function as typing and implementing
    attribute `attr` for the given Numba type in nopython mode. The
    "declarative" aspects of this decorator allow for specifiying verifiable
    information about the attribute that can be used to generate
    documentation.

    Args:
        path: The path starting with the library that corresponds to `attr`
        description: A description of additional notes to be displayed in
            documentation.
        arg_checker: An overload checker which checks that `typ` obeys
            specific properties at compile time and raises an error. This
            check is translated into documentation. Defaults to None.
        hyperlink: Link to external documentation. Defaults to None.
        **kwargs: Additional keyword arguments accepted by Bodo/Numba for
            overloaded attributes (see numba_compat.py)
    """

    def decorate(overload_func: pt.Callable):
        copied_kwargs = kwargs.copy()
        base = _OverloadDeclarativeAttributeTemplate
        # NOTE: _no_unliteral is a bodo specific attribute and is linked to changes in numba_compat.py
        template = make_overload_declarative_attribute_template(
            typ,
            attr,
            overload_func,
            path,
            arg_checker,
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
