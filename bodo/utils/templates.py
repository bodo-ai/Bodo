"""
Helper functions and classes to simplify Template Generation
for Bodo classes.
"""

import numba
from numba.core.typing.templates import AttributeTemplate

from bodo.ir.unsupported_method_template import _UnsupportedTemplate


class OverloadedKeyAttributeTemplate(AttributeTemplate):
    """
    Template Class for AttributeTemplates where the Key
    also has OverlateTemplates constructed for it. It
    includes some helper methods that can be used to help
    check for implemented overloads.
    """

    # Set of attribute names stored for caching.
    _attr_set = None

    def _is_existing_attr(self, attr_name):
        """
        Helper function that checks if attr_name is
        an existing attribute in an OverloadTemplate
        with lower priority than this AttributeTemplate.

        This function should be used in generic_resolve
        when attributes are dynamically dependent on
        type parameters (i.e. column names).

        All non-Overload templates aren't checked.
        """
        if self._attr_set is None:
            s = set()
            templates = list(self.context._get_attribute_templates(self.key))
            # If we reached generic_resolve of our current template, we never
            # need to check any template before or at our current position.
            idx = templates.index(self) + 1
            for i in range(idx, len(templates)):
                # All overloads are stored as _OverloadAttributeTemplate which store the name
                # in _attr.
                if isinstance(
                    templates[i], numba.core.typing.templates._OverloadAttributeTemplate
                ) or isinstance(templates[i], _UnsupportedTemplate):
                    s.add(templates[i]._attr)
            self._attr_set = s

        return attr_name in self._attr_set
