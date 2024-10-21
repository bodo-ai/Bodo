import inspect
import warnings
from abc import ABCMeta, abstractmethod
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

from bodo.utils.typing import BodoError, check_unsupported_args


### Helpers for generating documentation.
def get_feature_path(feature_name):  # pragma: no cover
    words = feature_name.split(".")
    if words[:2] == ["pd", "Series"]:
        return "docs/docs/api_docs/pandas/series", ".".join(words[2:])
    raise Exception(f"Unrecognized feature path {feature_name}")


def replace_package_name(path):  # pragma: no cover
    """Replace abbreviated package name in path with the full name"""
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


def get_example_from_docs(doc_path):  # pragma: no cover
    """
    Searches for index of "### Example Usage" in a document and returns all text
    following.
    """
    example_str = ""
    if Path(doc_path).is_file():
        begin_example = "### Example Usage"
        with open(doc_path, "r") as f:
            doc = f.read()
            if begin_example in doc:
                example_str = f"{doc[doc.index(begin_example) :].strip()}\n\n"
    return example_str


def format_argument_restrictions_str(
    params, arg_restrictions, unsupported_args
):  # pragma: no cover
    """
    Creates a bulleted list of argument restrictions for documentation. This list
    starts with a header "### Argument restrictions" and then adds a bullet point
    for each argument that looks like 'name: restrictions'. If an argument does not
    have a restriction then _restrictions_ is "None".

    Args:
        params (List[Parameter]): _description_
        arg_restrictions (dict): Mapping from parameter's name to a string descripton
            of it's restrictions.
        unsupported_args (dict): Mapping from parameter's name to it's default value.

    Returns:
        str: A bulleted list of argument restrictions plus a header (or "" if there are
            no arguments)
    """
    if len(params) == 0:
        return ""

    argument_restrictions_str = "### Argument Restrictions: "
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
    def document(self):
        """Generate docstring corresponding to this template"""

    @abstractmethod
    def is_matching_template(self, attr):
        """Check determine when attribute *attr* matches this template."""


class _OverloadDeclarativeMethodTemplate(DeclarativeTemplate, _OverloadMethodTemplate):
    def document(self, write_out=True):  # pragma: no cover
        """
        Generates a documentation string for the method, writes to corresponding file in
        documentation if *write_out*
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
        if params_list[0].name in arg_restrictions:
            supported_types_str += (
                "!!! note\n\t" f"Input {arg_restrictions[params_list[0].name]}."
            )
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
            f"{argument_restrictions_str}\n\n"
            f"{changed_defaults_str}"
            f"{description}"
            f"{example_str}"
        )

        # Overwrite document with generated information + extracted example
        if write_out:
            with open(doc_path, "w") as f:
                f.write(documentation)

        return documentation

    def is_matching_template(self, attr):
        return self._attr == attr

    def get_signature(self):
        return utils.pysignature(self._overload_func)

    @classmethod
    def _check_unsupported_args(cls, kws):
        path = cls.path.split(".")
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
            cls.path,
            args_dict,
            args_default_dict,
            package_name="pandas",
            module_name=module_name,
        )

    @classmethod
    def _check_argument_types(cls, args, kws):
        """
        Check that args and kwargs are valid types
        """
        if cls.method_args_checker is None:
            return

        # match every arg and kwarg with it's argument name in the signature
        overload_params = utils.pysignature(cls._overload_func).parameters
        arg_types = []
        for i, (name, param) in enumerate(overload_params.items()):
            if i < len(args):  # positional only
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
    def is_matching_template(self, attr):
        return self._attr == attr

    def document(self, write_out=True):  # pragma: no cover
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
            supported_types += "!!! note\n\t" f"Input {explain_attr}.\n\n"

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
        self.document()
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
    name = "OverloadDeclarativeAttributeTemplate_%s_%s" % (typ, attr)
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
    attr,
    path,
    unsupported_args,
    description,
    method_args_checker=None,
    changed_defaults=frozenset(),
    hyperlink=None,
    **kwargs,
):
    """A decorator marking the decorated function as typing and implementing
    attribute *attr* for the given Numba type in nopython mode. The "declarative"
    aspects of this decorator allow for specifiying verifiable information about the
    attribute that can be used to generate documentation using the following arguments:

    Args:
        path (str): The path starting with the library that corresponds to *attr*
        unsupported_defaults (set): The set of arguments that only support their
            default value.
        description (str): A description of additional notes to be displayed in
            documentation.
        arg_checker (OverloadArgumentsChecker): An overload checker which checks that
            *typ* and method args obey specific properties at compile time and raises
            an error. This check is translated into documentation. Defaults to None.
        changed_defaults (set): The set of arguments whose default value differs from
            the corresponding python API.
        hyperlink (str, optional): Link to external documentation. Defaults to None.
    """

    def decorate(overload_func):
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
    name = "OverloadDeclarativeAttributeTemplate_%s_%s" % (typ, attr)
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
    attr,
    path,
    description,
    arg_checker=None,
    hyperlink=None,
    **kwargs,
):
    """A decorator marking the decorated function as typing and implementing
    attribute *attr* for the given Numba type in nopython mode. The "declarative"
    aspects of this decorator allow for specifiying verifiable information about the
    attribute that can be used to generate documentation using the following arguments:

    Args:
        path (str): The path starting with the library that corresponds to *attr*
        description (str): A description of additional notes to be displayed in
            documentation.
        arg_checker (OverloadAttributeChecker): An overload checker which checks that
            *typ* obeys specific properties at compile time and raises an error. This
            check is translated into documentation. Defaults to None.
        hyperlink (str, optional): Link to external documentation. Defaults to None.
    """

    def decorate(overload_func):
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
