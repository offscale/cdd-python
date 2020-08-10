"""
Functions to handle default parameterisation
"""
from copy import deepcopy


def extract_default(line, emit_default_doc=True):
    """
    Extract the a tuple of (doc, default) from a doc line

    :param line: Example - "dataset. Defaults to mnist"
    :type line: ```str``

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :returns: Example - ("dataset. Defaults to mnist", "mnist") if emit_default_doc else ("dataset", "mnist")
    :rtype: Tuple[str, Optional[str]]
    """
    search_str = "defaults to "
    if line is None:
        return line, line
    doc, _, default = (
        lambda parts: parts if parts[1] else line.partition(search_str.capitalize())
    )(line.partition(search_str))
    return (
        line if emit_default_doc else doc.rstrip(";\n, "),
        default if len(default) else None,
    )


def remove_defaults_from_intermediate_repr(intermediate_repr, emit_defaults=True):
    """
    Remove "Default of" text from IR

    :param intermediate_repr: a dictionary of form
              {
                  'short_description': ...,
                  'long_description': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :type intermediate_repr: ```dict```

    :param emit_defaults: Whether to emit default property
    :type emit_defaults: ```bool```

    :returns: a dictionary of form
              {
                  'short_description': ...,
                  'long_description': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :rtype: ```dict```
    """
    ir = deepcopy(intermediate_repr)

    def handle_param(param):
        """
        :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
        :type param: ```dict```

        :returns: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
        :rtype: ```dict```
        """
        doc, default = extract_default(param["doc"], emit_default_doc=False)
        param.update({"doc": doc, "default": default})
        if default is None or not emit_defaults:
            del param["default"]
        return param

    ir["params"] = list(map(handle_param, ir["params"]))
    ir["returns"] = handle_param(ir["returns"])
    return ir


def set_default_doc(param, emit_default_doc=True):
    """
    Emit param with 'doc' set to include 'Defaults'

    :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :type param: ```dict```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :returns: Same shape as input but with Default append to doc.
    :rtype: ```dict``
    """

    has_defaults = "Defaults" in param["doc"] or "defaults" in param["doc"]

    if emit_default_doc and "default" in param and not has_defaults:
        param["doc"] = "{doc} Defaults to {default}".format(
            doc=(
                param["doc"]
                if param["doc"][-1] in frozenset((".", ","))
                else "{doc}.".format(doc=param["doc"])
            ),
            default=param["default"],
        )
    elif has_defaults:
        param["doc"] = extract_default(param["doc"], emit_default_doc=emit_default_doc)[
            0
        ]

    return param


__all__ = [
    "extract_default",
    "remove_defaults_from_intermediate_repr",
    "set_default_doc",
]
