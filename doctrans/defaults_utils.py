"""
Functions to handle default parameterisation
"""
from copy import deepcopy
from functools import partial
from itertools import takewhile
from operator import eq, contains

from doctrans.pure_utils import location_within, count_iter_items


def extract_default(line, rstrip_default=True, emit_default_doc=True):
    """
    Extract the a tuple of (doc, default) from a doc line

    :param line: Example - "dataset. Defaults to mnist"
    :type line: ```str``

    :param rstrip_default: Whether to rstrip whitespace, newlines, and '.' from the default
    :type rstrip_default: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :return: Example - ("dataset. Defaults to mnist", "mnist") if emit_default_doc else ("dataset", "mnist")
    :rtype: Tuple[str, Optional[str]]
    """
    search_str = "defaults to "
    if line is None:
        return line, line

    _start_idx, _end_idx, _found = location_within(
        line, (search_str,), cmp=lambda a, b: eq(*map(str.casefold, (a, b)))
    )
    if _start_idx == -1:
        return line, None

    default = ""
    par = {"{": 0, "[": 0, "(": 0, "}": 0, "]": 0, ")": 0}
    for ch in line[_end_idx:]:
        if ch == "." and not sum(par.values()):
            break
        elif ch in par:
            par[ch] += 1
        default += ch
    # default = "".join(takewhile(rpartial(ne, "."), line[_end_idx:]))
    rest_offset = _end_idx + len(default)
    default = default.strip("`")
    if emit_default_doc:
        return line, default
    else:
        if rstrip_default:
            offset = count_iter_items(
                takewhile(
                    partial(contains, frozenset((" ", "\t", "\n", ".", "\n"))),
                    line[rest_offset:],
                )
            )
            rest_offset += offset

        fst = line[: _start_idx - 1]
        return fst + line[rest_offset:], default


def remove_defaults_from_intermediate_repr(intermediate_repr, emit_defaults=True):
    """
    Remove "Default of" text from IR

    :param intermediate_repr: a dictionary of form
              {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :type intermediate_repr: ```dict```

    :param emit_defaults: Whether to emit default property
    :type emit_defaults: ```bool```

    :return: a dictionary of form
              {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :rtype: ```dict```
    """
    ir = deepcopy(intermediate_repr)

    remove_default_from_param = partial(
        _remove_default_from_param, emit_defaults=emit_defaults
    )
    ir["params"] = list(map(remove_default_from_param, ir["params"]))
    ir["returns"] = remove_default_from_param(ir["returns"])
    return ir


def _remove_default_from_param(param, emit_defaults=True):
    """
    :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :type param: ```dict```

    :param emit_defaults: Whether to emit default property
    :type emit_defaults: ```bool```

    :return: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :rtype: ```dict```
    """
    doc, default = extract_default(param["doc"], emit_default_doc=False)
    param.update({"doc": doc, "default": default})
    if default is None or not emit_defaults:
        del param["default"]
    return param


def set_default_doc(param, emit_default_doc=True):
    """
    Emit param with 'doc' set to include 'Defaults'

    :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :type param: ```dict```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :return: Same shape as input but with Default append to doc.
    :rtype: ```dict``
    """
    # if param is None: param = {"doc": "", "typ": "Any"}
    has_defaults = (
        ("Defaults" in param["doc"] or "defaults" in param["doc"])
        if "doc" in param
        else False
    )

    if emit_default_doc and not has_defaults and "default" in param:
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
