"""
Functions to handle default parameterisation
"""

import ast
from ast import literal_eval
from collections import OrderedDict
from contextlib import suppress
from copy import deepcopy
from functools import partial
from itertools import takewhile
from operator import contains, eq
from typing import Dict

from cdd.shared.pure_utils import (
    PY_GTE_3_8,
    PY_GTE_3_9,
    count_iter_items,
    location_within,
    none_types,
    quote,
    simple_types,
)
from cdd.shared.types import IntermediateRepr

if PY_GTE_3_8:
    from cdd.shared.pure_utils import FakeConstant as Str
else:
    from ast import Str

NoneStr = "```(None)```" if PY_GTE_3_9 else "```None```"
DEFAULTS_TO_VARIANTS = (  # could do a whole r"[dD]efault[s]?\s+[value is|to|is|:]" but this suffices for now
    "defaults to ",
    "defaults to\n",
    "Default value is ",
    "Default:",
    "defaults\n to ",
    "defaults\n to\n",
    "Default value\n is ",
    "Defaults\n            to",
)


def ast_parse_fix(s):
    """
    Hack to resolve unbalanced parentheses SyntaxError acquired from PyTorch parsing
    TODO: remove

    :param s: String to parse
    :type s: ```str```

    :return: Value
    """
    # return ast.parse(s).body[0].value
    balanced = (s.count("[") + s.count("]")) & 1 == 0
    return ast.parse(s if balanced else "{}]".format(s)).body[0].value


def needs_quoting(typ):
    """
    Figures out whether values with this type need quoting

    :param typ: The type
    :type typ: ```Optional[str]```

    :return: Whether the type needs quoting
    :rtype: ```bool```
    """
    if typ is None or typ.startswith("*"):
        return False
    elif typ == "str":
        return True
    elif typ == "Optional[str]":
        return True

    typ = typ.replace("\n", "").strip()
    parsed_typ_ast = ast_parse_fix(typ)
    if isinstance(parsed_typ_ast, ast.Name):
        return parsed_typ_ast.id == "str"

    return any(
        filter(
            lambda node: isinstance(node, Str)
            or isinstance(node, ast.Constant)
            and type(node.value).__name__ == "str"
            or isinstance(node, ast.Name)
            and node.id == "str",
            ast.walk(parsed_typ_ast),
        )
    )


def extract_default(
    line,
    rstrip_default=True,
    default_search_announce=None,
    typ=None,
    emit_default_doc=True,
):
    """
    Extract a tuple of (doc, default) from a doc line

    :param line: Example - "dataset. Defaults to mnist"
    :type line: ```str```

    :param rstrip_default: Whether to rstrip whitespace, newlines, and '.' from the default
    :type rstrip_default: ```bool```

    :param default_search_announce: Default text(s) to look for. If None, uses default specified in default_utils.
    :type default_search_announce: ```Optional[Union[str, Iterable[str]]]```

    :param typ: The type of the default value, useful to disambiguate `25` the float from  `25` the float
    :type typ: ```Optional[str]```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: Example - ("dataset. Defaults to mnist", "mnist") if emit_default_doc else ("dataset", "mnist")
    :rtype: ```tuple[str, Optional[str]]```
    """
    if line is None:
        return line, line

    default_search_announce_paren, default_search_announce = (
        lambda _default_search_announce: (
            map(partial(str.format, "({}"), _default_search_announce),
            _default_search_announce,
        )
    )(
        DEFAULTS_TO_VARIANTS
        if default_search_announce is None
        else (
            (default_search_announce,)
            if isinstance(default_search_announce, str)
            else default_search_announce
        )
    )

    _start_idx, _end_idx, default_end_offset = None, None, None
    for idx, _default_search_announce in enumerate(
        (default_search_announce_paren, default_search_announce)
    ):
        _start_idx, _end_idx, _found = location_within(
            line,
            _default_search_announce,
            cmp=lambda a, b: eq(*map(str.casefold, (a, b))),
        )

        if idx == 0:
            if _start_idx != -1:
                _start_idx += 1  # eat '('
                default_end_offset = (
                    -1 if line[-1] == ")" else -2 if line[-2:] == ")." else 0
                )  # eat ')', ').'
                break
        elif _start_idx < 0:
            return line, None

    default = ""
    par: Dict[str, int] = {"{": 0, "[": 0, "(": 0, ")": 0, "]": 0, "}": 0}
    sub_l: str = line[_end_idx:default_end_offset]
    sub_l_len: int = len(sub_l)
    for idx, ch in enumerate(sub_l):
        if (
            ch == "."
            and (idx == (sub_l_len - 1) or not (sub_l[idx + 1]).isdigit())
            and not sum(par.values())
        ):
            break
        elif ch in par:
            par[ch] += 1
        default += ch

    start_rest_offset = _end_idx + len(default)

    default = default.strip(" \t`")

    if default.count('"') & 1:
        default = default.strip('"')
    if default.count("'") & 1:
        default = default.strip("'")

    # Correct for parsing errors where a quote is captured at the start or end, but not both
    if len(default) > 0:
        if default.count('"') == 1 and default.count("'") == 0:
            if default.startswith('"'):
                default = default + '"'
            elif default.endswith('"'):
                default = '"' + default
        elif default.count("'") == 1 and default.count('"') == 0:
            if default.startswith("'"):
                default = default + "'"
            elif default.endswith("'"):
                default = "'" + default

    return _parse_out_default_and_doc(
        _start_idx,
        start_rest_offset,
        default,
        line,
        rstrip_default,
        typ,
        default_end_offset,
        emit_default_doc,
    )


def _parse_out_default_and_doc(
    _start_idx,
    start_rest_offset,
    default,
    line,
    rstrip_default,
    typ,
    default_end_offset,
    emit_default_doc,
):
    """
    Internal function to parse the default and extract out the doc iff `emit_default_doc is False`

    :param _start_idx: The start index to look from
    :type _start_idx: ```int```

    :param start_rest_offset: The start index to look from, for the rest that's appended
    :type start_rest_offset: ```int```

    :param default: The currently parsed out default, could be the end form, could parse into something more specific
    :type default: ```Any```

    :param line: Example - "dataset. Defaults to mnist"
    :type line: ```str```

    :param rstrip_default: Whether to rstrip whitespace, newlines, and '.' from the default
    :type rstrip_default: ```bool```

    :param typ: The type of the default value, useful to disambiguate `25` the float from  `25` the float
    :type typ: ```Optional[str]```

    :param default_end_offset: Set to `-1` if one parenthesis, `-2` if ')'., and `0` if none
    :type default_end_offset: ```int```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: Example - ("dataset. Defaults to mnist", "mnist") if emit_default_doc else ("dataset", "mnist")
    :rtype: Tuple[str, Optional[str]]
    """
    if typ is not None and typ in simple_types and default not in none_types:
        if typ == "str":
            from cdd.shared.pure_utils import unquote

            default = unquote(default)
        else:
            lit = (
                ast.AST()
                if any(
                    map(
                        partial(
                            contains, frozenset(("*", "^", "&", "|", "$", "@", "!"))
                        ),
                        default,
                    )
                )
                else literal_eval("({default})".format(default=default))
            )
            default = (
                "```{default}```".format(default=default)
                if isinstance(lit, ast.AST)
                else {
                    "bool": bool,
                    "int": int,
                    "float": float,
                    "complex": complex,
                }[
                    typ
                ](lit)
            )
    elif default.isdecimal():
        default = int(default)
    elif default in frozenset(("True", "False")):
        default = literal_eval(default)
    else:
        with suppress(ValueError):
            default = float(default)
    if emit_default_doc:
        return line, default
    else:
        stop_tokens = frozenset((" ", "\t", "\n", "\n", "."))
        if _start_idx == 0:
            fst = ""
            extra_offset = 0
        else:
            end = line[: _start_idx - 1]
            extra_offset = int(
                end[-1] in frozenset((" ", "\t", "\n", "\n")) if end else 0
            )
            fst = line[: _start_idx - 1 - extra_offset]

        if rstrip_default:
            offset: int = count_iter_items(
                takewhile(
                    partial(contains, stop_tokens),
                    line[start_rest_offset:],
                )
            )
            start_rest_offset += offset

        rest = line[
            start_rest_offset : (
                (-extra_offset if extra_offset > 0 else None)
                if default_end_offset is None
                else default_end_offset
            )
        ]
        return (
            fst + rest,
            default,
        )


def remove_defaults_from_intermediate_repr(intermediate_repr, emit_default_prop=True):
    """
    Remove "Default of" text from IR

    :param intermediate_repr: a dictionary consistent with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    :type intermediate_repr: ```dict```

    :param emit_default_prop: Whether to emit default property
    :type emit_default_prop: ```bool```

    :return: a dictionary consistent with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    :rtype: ```dict```
    """
    ir: IntermediateRepr = deepcopy(intermediate_repr)

    remove_default_from_param = partial(
        _remove_default_from_param, emit_default_prop=emit_default_prop
    )
    ir.update(
        {
            key: OrderedDict(map(remove_default_from_param, ir[key].items()))
            for key in ("params", "returns")
        }
    )
    return ir


def _remove_default_from_param(param, emit_default_prop=True):
    """
    Remove default from param iff emit_default_prop is False

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```tuple[str, dict]```

    :param emit_default_prop: Whether to emit default property
    :type emit_default_prop: ```bool```

    :return: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```tuple[str, dict]```
    """
    name, _param = param
    del param
    doc, default = extract_default(_param["doc"], emit_default_doc=False)
    _param.update({"doc": doc, "default": default})
    if default is None or not emit_default_prop:
        del _param["default"]
    return name, _param


def set_default_doc(param, emit_default_doc=True):
    """
    Emit param with 'doc' set to include 'Defaults'

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```tuple[str, dict]```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: Same shape as input but with Default append to doc.
    :rtype: ```dict```
    """
    name, _param = param
    del param
    # if param is None: param = {"doc": "", "typ": "Any"}
    if _param is None or "doc" not in _param:
        return name, _param
    has_defaults = "Defaults" in _param["doc"] or "defaults" in _param["doc"]

    if has_defaults and not emit_default_doc:
        # Remove the default text
        _param["doc"] = extract_default(
            _param["doc"], emit_default_doc=emit_default_doc
        )[0]
    elif "default" in _param and not has_defaults and emit_default_doc:
        # if _param["default"] == NoneStr: _param["default"] = None
        if _param["default"] is not None or not name.endswith("kwargs"):
            _param["doc"] = "{doc} Defaults to {default}".format(
                doc=(
                    _param["doc"]
                    if _param["doc"][-1] in frozenset((".", ","))
                    else "{doc}.".format(doc=_param["doc"])
                ),
                default=(
                    quote(_param["default"])
                    if (
                        needs_quoting(_param.get("typ"))
                        and (
                            len(_param["default"]) < 2
                            or not _param["default"].startswith("`")
                            or not _param["default"].endswith("`")
                        )
                        if isinstance(_param["default"], str)
                        else True
                    )
                    else _param["default"]
                ),
            )

    return name, _param


__all__ = [
    "extract_default",
    "needs_quoting",
    "remove_defaults_from_intermediate_repr",
    "set_default_doc",
    "_remove_default_from_param",
]  # type: list[str]
