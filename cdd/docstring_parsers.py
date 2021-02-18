"""
Docstring parsers.

Translates from the [ReST docstring format (Sphinx)](
  https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format)

Translates from the [numpydoc docstring format](https://numpydoc.readthedocs.io/en/latest/format.html)

Translates from [Google's docstring format](https://google.github.io/styleguide/pyguide.html)
"""
import ast
from ast import AST
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import partial
from itertools import takewhile
from operator import attrgetter, contains, eq, le
from typing import Dict, List, Tuple

from cdd.ast_utils import NoneStr, get_value
from cdd.defaults_utils import _remove_default_from_param, needs_quoting
from cdd.docstring_utils import ARG_TOKENS, RETURN_TOKENS, TOKENS
from cdd.emit import to_code
from cdd.emitter_utils import interpolate_defaults
from cdd.pure_utils import (
    code_quoted,
    count_iter_items,
    location_within,
    none_types,
    paren_wrap_code,
    rpartial,
    tab,
    unquote,
    update_d,
)


class Style(Enum):
    """
    Simple enum taken from the docstring_parser codebase
    """

    rest = 1
    google = 2
    numpydoc = 3
    auto = 255


def parse_docstring(
    docstring,
    infer_type=False,
    default_search_announce=None,
    word_wrap=True,
    emit_default_prop=True,
    emit_default_doc=False,
):
    """Parse the docstring into its components.

    :param docstring: the docstring
    :type docstring: ```Optional[str]```

    :param default_search_announce: Default text(s) to look for. If None, uses default specified in default_utils.
    :type default_search_announce: ```Optional[Union[str, Iterable[str]]]```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param emit_default_prop: Whether to include the default dictionary property.
    :type emit_default_prop: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :returns: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :rtype: ```dict```
    """

    assert isinstance(docstring, (type(None), str)), "{typ} != str".format(
        typ=type(docstring).__name__
    )
    if docstring is None or any(map(partial(contains, docstring), TOKENS.rest)):
        style = Style.rest
    elif any(map(partial(contains, docstring), TOKENS.google)):
        style = Style.google
    else:
        style = Style.numpydoc

    ir = {
        "name": None,
        "type": "static",
        # "_internal": {"_style": style.value},
        "doc": "",
        "params": OrderedDict(),
        "returns": None,
    }
    if not docstring:
        return ir

    scanned = _scan_phase(docstring, style=style)

    _parse_phase(
        ir,
        scanned,
        emit_default_doc=emit_default_doc,
        emit_default_prop=emit_default_prop,
        word_wrap=word_wrap,
        default_search_announce=default_search_announce,
        infer_type=infer_type,
        style=style,
    )

    # Apply certain functions regardless of style
    if style is Style.rest:
        ir.update(
            {
                k: OrderedDict(
                    map(
                        partial(
                            interpolate_defaults, emit_default_doc=emit_default_doc
                        ),
                        ir[k].items(),
                    )
                )
                if ir[k]
                else ir[k]
                for k in ("params", "returns")
            }
        )
    return ir


def _scan_phase(docstring, style=Style.rest):
    """
    Scanner phase. Lexical analysis; to some degree…

    :param docstring: the docstring
    :type docstring: ```str```

    :param style: the style of docstring
    :type style: ```Style```

    :returns: List with each element a tuple of (whether value is a token, value)
    :rtype: ```Union[Dict[str, str], List[Tuple[bool, str]]]```
    """
    arg_tokens, return_tokens = map(attrgetter(style.name), (ARG_TOKENS, RETURN_TOKENS))
    return (
        _scan_phase_rest
        if style is Style.rest
        else partial(_scan_phase_numpydoc_and_google, style=style)
    )(docstring, arg_tokens=arg_tokens, return_tokens=return_tokens)


def _scan_phase_numpydoc_and_google(docstring, arg_tokens, return_tokens, style):
    """
    numpydoc and google scanner phase. Lexical analysis; to some degree…

    :param docstring: the docstring
    :type docstring: ```str```

    :param arg_tokens: Valid tokens like `"Parameters\n----------"`
    :type arg_tokens: ```Tuple[str]```

    :param return_tokens: Valid tokens like `"Returns\n-------"`
    :type return_tokens: ```Tuple[str]```

    :param style: the style of docstring
    :type style: ```Style```

    :returns: List with each element a tuple of (whether value is a token, value)
    :rtype: ```Dict[str, str]```
    """
    scanned: Dict[str, List[List[str]]] = {
        token: [] for token in ("doc",) + arg_tokens + return_tokens
    }
    # ^ Dict[Union[Literal["doc"], arg_tokens, return_tokens], List[dict]]

    # First doc, if present
    _start_idx, _end_idx, _found = location_within(docstring, arg_tokens)
    if _start_idx == -1:
        # Return type no args?
        _start_idx, _end_idx, _found = location_within(docstring, return_tokens)

    if _start_idx > -1:
        namespace = _found
        scanned["doc"] = docstring[:_start_idx].strip()
        docstring = docstring[_end_idx + 1 :]  # .strip()
    else:
        scanned["doc"] = docstring.strip()
        return scanned

    # Scan all lines so that that each element in `stacker` refers to one 'unit'
    stacker, docstring_lines = [], docstring.splitlines()
    first_indent = count_iter_items(takewhile(str.isspace, docstring_lines[0]))
    for line_no, line in enumerate(docstring_lines):
        indent = count_iter_items(takewhile(str.isspace, line))

        if indent == first_indent:
            stacker.append([line])
        else:
            if indent < first_indent:
                scanned[namespace] = scanned.get(namespace, []) + deepcopy(stacker)
                stacker.clear()
                if len(docstring_lines) > line_no + 3 and any(
                    filter(rpartial(eq, docstring_lines[line_no + 1]), return_tokens)
                    # return_token = return_tokens[0].splitlines()
                    # filter(rpartial(eq, docstring_lines[line_no + 1]), return_tokens)
                ):
                    return_indent = count_iter_items(
                        takewhile(str.isspace, docstring_lines[line_no + 3])
                    )
                    next_smallest_indent = count_iter_items(
                        takewhile(
                            partial(le, return_indent),
                            map(
                                lambda l: count_iter_items(takewhile(str.isspace, l)),
                                docstring_lines[line_no + 3 :],
                            ),
                        )
                    )
                    scanned[return_tokens[0]] = docstring_lines[
                        line_no + 2 : line_no + 3 + next_smallest_indent
                    ]
                    scanned_afterward = docstring_lines[
                        line_no + 3 + next_smallest_indent :
                    ]
                else:
                    scanned_afterward = docstring_lines[line_no + 1 :]
                    if (
                        len(scanned_afterward) > 1
                        and scanned_afterward[0] == return_tokens[0]
                    ):
                        return_indent = count_iter_items(
                            takewhile(str.isspace, scanned_afterward[1])
                        )
                        next_smallest_indent = count_iter_items(
                            takewhile(
                                partial(le, return_indent),
                                map(
                                    lambda l: count_iter_items(
                                        takewhile(str.isspace, l)
                                    ),
                                    scanned_afterward[2:],
                                ),
                            )
                        )
                        scanned[return_tokens[0]] = scanned_afterward[
                            1 : next_smallest_indent + 2
                        ]
                        scanned_afterward = (
                            None
                            if next_smallest_indent == 0
                            else scanned_afterward[next_smallest_indent + 2 :]
                        )

                if scanned_afterward:
                    scanned["scanned_afterward"] = scanned_afterward
                break
            else:
                stacker[-1].append(line)

    # Split out return, if present and not already set
    if not scanned.get(return_tokens[0], False):
        stacker = _return_parse_phase_numpydoc_and_google(
            return_tokens, scanned, stacker, style
        )

    if stacker:
        scanned[namespace] = stacker

    # # Join ["a", "b"] into ["ab"] as each has been resolved to a scanned symbol now
    # scanned.update(
    #     {
    #         token: [
    #             line_list
    #             if len(line_list) < 3 or isinstance(line_list, str)
    #             else [
    #                 "{} {}".format(
    #                     line_list[0], " ".join(map(str.lstrip, line_list[1:]))
    #                 )
    #             ]
    #             for line_list in scanned[token]
    #         ]
    #         for token in arg_tokens + return_tokens
    #         if scanned.get(token)
    #     }
    # )

    return scanned


def _return_parse_phase_numpydoc_and_google(return_tokens, scanned, stacker, style):
    """
    numpydoc and google scanner phase for return. Lexical analysis; to some degree…

    :param return_tokens: Valid tokens like `"Returns\n-------"`
    :type return_tokens: ```Tuple[str]```

    :param scanned: List with each element a tuple of (whether value is a token, value)
    :type scanned: ```Union[Dict[str, str], List[Tuple[bool, str]]]```

    :param stacker: Stack of strings part that forms int scanned
    :type stacker: ```List[List[str]]```

    :param style: the style of docstring
    :type style: ```Style```

    :returns: Whatever is left of `stacker`. This function may also set the return key of the `scanned`
    :rtype: ```List[List[str]]```
    """
    rev_return_token = return_tokens[0].splitlines()[::-1]
    rng = range(len(stacker) - 1, -1, -1)
    if style is Style.numpydoc:
        for i in rng:
            if i - 1 > 0 and stacker[i] + stacker[i - 1] == rev_return_token:
                scanned[return_tokens[0]] = stacker[i + 1 :]
                stacker = stacker[: i - 1]
                break
    # # All the google docstring examples are now handled earlier in the scan phase
    # else:
    #     for i in rng:
    #         for idx, token in enumerate(stacker[i]):
    #             if token == return_tokens[0]:
    #                 scanned[return_tokens[0]] = (
    #                     stacker[i][idx + 1 :] + stacker[i + 1 :]
    #                 )[0]
    #                 stacker[i] = stacker[i][: idx - 1]
    #                 stacker = stacker[: i + 1]
    #                 break
    return stacker


def _scan_phase_rest(docstring, arg_tokens, return_tokens):
    """
    Scanner phase. Lexical analysis; to some degree…

    :param docstring: the docstring
    :type docstring: ```str```

    :param arg_tokens: Valid tokens like `":param"`
    :type arg_tokens: ```Tuple[str]```

    :param return_tokens: Valid tokens like `":rtype:"`
    :type return_tokens: ```Tuple[str]```

    :returns: List with each element a tuple of (whether value is a token, value)
    :rtype: ```List[Tuple[bool, str]]```
    """

    all_tokens = arg_tokens + return_tokens
    rev_known_tokens_t = tuple(map(tuple, map(reversed, all_tokens)))
    scanned: List[Tuple[bool, str]] = []
    stack: List[str] = []

    for ch in docstring:
        stack.append(ch)

        stack_rev = stack[::-1]

        for token in rev_known_tokens_t:
            token_len = len(token)
            if tuple(stack_rev[:token_len]) == token:
                scanned.append((bool(len(scanned)), "".join(stack[:-token_len])))
                stack = stack[len(scanned[-1][1]) :][:token_len]
                continue

    if stack:
        final = "".join(stack)
        scanned.append(
            (
                bool(scanned and scanned[-1][0])
                or any(map(final.startswith, all_tokens)),
                final,
            )
        )

    return scanned


def _parse_phase(
    intermediate_repr,
    scanned,
    default_search_announce,
    infer_type,
    word_wrap,
    emit_default_prop,
    emit_default_doc,
    style=Style.rest,
):
    """
    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param scanned: List with each element a tuple of (whether value is a token, value)
    :type scanned: ```Union[Dict[str, str], List[Tuple[bool, str]]]```

    :param default_search_announce: Default text(s) to look for. If None, uses default specified in default_utils.
    :type default_search_announce: ```Optional[Union[str, Iterable[str]]]```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param emit_default_prop: Whether to include the default dictionary property.
    :type emit_default_prop: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :param style: the style of docstring
    :type style: ```Style```
    """
    arg_tokens, return_tokens = map(attrgetter(style.name), (ARG_TOKENS, RETURN_TOKENS))
    (
        _parse_phase_rest
        if style is Style.rest
        else partial(
            _parse_phase_numpydoc_and_google, style=style, arg_tokens=arg_tokens
        )
    )(
        intermediate_repr,
        scanned,
        emit_default_doc=emit_default_doc,
        emit_default_prop=emit_default_prop,
        return_tokens=return_tokens,
        default_search_announce=default_search_announce,
        infer_type=infer_type,
        word_wrap=word_wrap,
    )


def _set_name_and_type(param, infer_type, word_wrap):
    """
    Sanitise the name and set the type (iff default and no existing type) for the param

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, dict]```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :returns: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```Tuple[str, dict]```
    """
    name, _param = param
    del param
    if name is not None and (name.endswith("kwargs") or name.startswith("**")):
        name = name.lstrip("*")
        if _param.get("typ", "dict") == "dict":
            _param["typ"] = "Optional[dict]"
        if "default" not in _param:
            _param["default"] = NoneStr
    elif "default" in _param:
        _infer_default(_param, infer_type)
    google_opt = ", optional"
    if (_param.get("typ") or "").endswith(google_opt):
        _param["typ"] = "Optional[{}]".format(_param["typ"][: -len(google_opt)])
    if "doc" in _param and not _param["doc"]:
        del _param["doc"]

    # if "doc" in _param and isinstance(_param["doc"], list):
    #     _param["doc"] = "".join(_param["doc"])
    if "doc" in _param:
        if not isinstance(_param["doc"], str):
            _param["doc"] = "".join(_param["doc"]).rstrip()
        else:
            _param["doc"] = (
                " ".join(map(str.strip, _param["doc"].split("\n")))
                if word_wrap
                else _param["doc"]
            ).rstrip()
        if (
            (
                _param["doc"].startswith("(Optional)")
                or _param["doc"].startswith("Optional")
            )
            and "typ" in _param
            and not _param["typ"].startswith("Optional[")
        ):
            _param["typ"] = "Optional[{}]".format(_param["typ"])

    return name, _param


def _infer_default(_param, infer_type):
    """
    Internal function to infer the default. Not intended for use by more than [the current] one function.

    :param _param: dict with keys: 'typ', 'doc', 'default'
    :type _param: ```dict```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```
    """
    if isinstance(
        _param["default"], (ast.Str, ast.Num, ast.Constant, ast.NameConstant)
    ):
        _param["default"] = get_value(_param["default"])
    if _param.get("default", False) in none_types:
        _param["default"] = NoneStr
    if infer_type and _param.get("typ") is None and _param["default"] not in none_types:
        _param["typ"] = type(_param["default"]).__name__
    if needs_quoting(_param.get("typ")) or isinstance(_param["default"], str):
        _param["default"] = unquote(_param["default"])
    elif isinstance(_param["default"], AST):
        try:
            _param["default"] = ast.literal_eval(_param["default"])
            # if _param.get("typ") is None or _param["typ"] == "UnaryOp":
            #    _param["typ"] = type(_param["default"]).__name__
        except ValueError:
            _param["default"] = "```{default}```".format(
                default=paren_wrap_code(to_code(_param["default"]).rstrip("\n"))
            )
    if _param.get("typ") is None and _param["default"] != NoneStr:
        _param["typ"] = type(_param["default"]).__name__
    if (
        _param["default"] != NoneStr
        and code_quoted(_param["default"])
        and "["
        not in _param.get(
            "typ", iter(())
        )  # Skip if you've actually formed a proper type
    ):
        del _param["typ"]  # Could make it `object` I suppose…


def _parse_phase_numpydoc_and_google(
    intermediate_repr,
    scanned,
    default_search_announce,
    infer_type,
    word_wrap,
    style,
    arg_tokens,
    return_tokens,
    emit_default_prop,
    emit_default_doc,
):
    """
    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param scanned: List with each element a tuple of (whether value is a token, value)
    :type scanned: ```Dict[str, str]```

    :param default_search_announce: Default text(s) to look for. If None, uses default specified in default_utils.
    :type default_search_announce: ```Optional[Union[str, Iterable[str]]]```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param style: the style of docstring
    :type style: ```Style```

    :param arg_tokens: Valid tokens like `"Parameters\n----------"`
    :type arg_tokens: ```Tuple[str]```

    :param return_tokens: Valid tokens like `"Returns\n-------"`
    :type return_tokens: ```Tuple[str]```

    :param emit_default_prop: Whether to include the default dictionary property.
    :type emit_default_prop: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```
    """
    if style is Style.numpydoc:

        def _parse(scan):
            """
            Parse the scanned input (numpydoc)

            :param scan: Scanned input
            :type scan: ```List[str]```

            :returns: dict with keys: 'name', 'typ', 'doc'
            :rtype: ```dict```
            """
            name, _, typ = scan[0].partition(":")
            if not name:
                return None
            cur = {"name": name.rstrip()}
            if typ:
                cur.update(
                    {"typ": typ.lstrip(), "doc": "\n".join(map(str.lstrip, scan[1:]))}
                )
            # elif name.endswith("kwargs"): cur["typ"] = "dict"
            return cur

    else:

        def _parse(scan, partitioned=None):
            """
            Parse the scanned input (Google)

            :param scan: Scanned input
            :type scan: ```List[str]```

            :param partitioned: Prep-partitioned `scan`, if given doesn't partition on `scan`, just uses this
            :type partitioned: ```Optional[Tuple[str, str, str]]```

            :returns: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
            :rtype: ```dict```
            """
            offset = next(idx for idx, ch in enumerate(scan[0]) if ch == ":")
            s = scan[0][:offset].lstrip()
            name, delim, typ = partitioned or s.partition("(")
            name = name.rstrip()
            typ = (delim + typ).rstrip()
            # if not name: return None
            cur = {"name": name}
            if typ:
                assert typ.startswith("(") and typ.endswith(
                    ")"
                ), "Expected third partition to be paren wrapped {!r}".format(s)
                cur["typ"] = typ[1:-1]
                if " or " in cur["typ"]:
                    cur["typ"] = "Union[{}]".format(", ".join(cur["typ"].split(" or ")))
                end = scan[0][offset + 1 :].lstrip()
                if len(end) > 3 and end.startswith("{") and end.endswith("}"):
                    # PyTorch invented their own syntax for this I guess?
                    cur["typ"] = "Literal{}".format(
                        list(map(rpartial(str.strip, "'"), end[1:-1].split(", ")))
                    )
                    scan[0] = ""
                # elif partitioned is None:
                #    return _parse(scan, " ".join((name, typ)).partition("="))
                # else:
            # elif name.endswith("kwargs"): cur["typ"] = "dict"
            cur["doc"] = "\n".join([scan[0][offset + 1 :].lstrip()] + scan[1:]).strip()
            return cur

    scanned_params = scanned[arg_tokens[0]]

    # Handle stuff after the Args, e.g., usage notes; doctests; references.
    afterward_idx = next(
        (idx for idx, elem in enumerate(scanned_params) if elem[0].endswith(":")), None
    )
    if afterward_idx:
        scanned_params, scanned_afterward = (
            scanned_params[:afterward_idx],
            scanned_params[afterward_idx:],
        )
        scanned["doc"] += "\n\n\n{}".format(
            "\n".join(
                map(
                    "\n".join,
                    map(
                        lambda l: map(
                            lambda s: s if s.endswith(":") else "{}{}".format(tab, s), l
                        ),
                        scanned_afterward,
                    ),
                )
            )
        )

    if "scanned_afterward" in scanned:
        scanned["doc"] += "\n\n\n{}".format("\n".join(scanned["scanned_afterward"]))

    def _interpolate_defaults_and_force_future_default(name_param):
        """
        interpolate the defaults and force future default if current default is set

        :param name_param: Name, dict with keys: 'typ', 'doc', 'default'
        :type name_param: ```Tuple[str, dict]```

        :returns: Name, dict with keys: 'typ', 'doc', 'default'
        :rtype: ```Tuple[str, dict]```
        """
        name, param = interpolate_defaults(
            name_param,
            emit_default_doc=emit_default_doc,
            require_default=_interpolate_defaults_and_force_future_default.require_default,
            default_search_announce=default_search_announce,
        )
        if (
            not _interpolate_defaults_and_force_future_default.require_default
            and param.get("default") is not None
        ):
            _interpolate_defaults_and_force_future_default.require_default = True
        return name, param

    _interpolate_defaults_and_force_future_default.require_default = False

    intermediate_repr.update(
        {
            "doc": scanned["doc"],
            "params": OrderedDict(
                map(
                    partial(
                        _set_name_and_type, infer_type=infer_type, word_wrap=word_wrap
                    ),
                    map(
                        _interpolate_defaults_and_force_future_default,
                        map(
                            lambda d: (d.pop("name"), d),
                            filter(None, map(_parse, scanned_params)),
                        ),
                    ),
                ),
            ),
            "returns": OrderedDict(
                (
                    _interpolate_defaults_and_force_future_default(
                        _set_name_and_type(
                            (
                                "return_type",
                                (
                                    {
                                        "typ": scanned[return_tokens[0]][0][
                                            :-1
                                        ].lstrip(),
                                        "doc": scanned[return_tokens[0]][1].lstrip(),
                                    }
                                    if len(scanned[return_tokens[0]]) == 2
                                    and isinstance(scanned[return_tokens[0]][1], str)
                                    else {}
                                    if isinstance(scanned[return_tokens[0]][0], str)
                                    and scanned[return_tokens[0]][0].isspace()
                                    else {
                                        "doc": scanned[return_tokens[0]][0].lstrip()
                                        if isinstance(scanned[return_tokens[0]][0], str)
                                        else scanned[return_tokens[0]][0]
                                    }
                                )
                                if style is Style.google
                                else {
                                    "typ": scanned[return_tokens[0]][0][0],
                                    "doc": scanned[return_tokens[0]][0][1].lstrip(),
                                },
                            ),
                            infer_type=infer_type,
                            word_wrap=word_wrap,
                        ),
                    ),
                ),
            )
            if scanned[return_tokens[0]]
            else None,
        }
    )


def _parse_phase_rest(
    intermediate_repr,
    scanned,
    default_search_announce,
    infer_type,
    word_wrap,
    emit_default_prop,
    emit_default_doc,
    return_tokens,
):
    """
    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param scanned: List with each element a tuple of (whether value is a token, value)
    :type scanned: ```List[Tuple[bool, str]]```

    :param default_search_announce: Default text(s) to look for. If None, uses default specified in default_utils.
    :type default_search_announce: ```Optional[Union[str, Iterable[str]]]```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param emit_default_prop: Whether to include the default dictionary property.
    :type emit_default_prop: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :param return_tokens: Valid tokens like `":rtype:"`
    :type return_tokens: ```Tuple[str]```
    """
    param = [
        None,
        {},
    ]  # First elem is name and second elem is dict with keys: 'typ', 'doc', 'default'
    for is_token, line in scanned:
        if is_token is True:
            if any(map(line.startswith, return_tokens)):
                nxt_colon = line.find(":", 1)
                val = line[nxt_colon + 1 :].strip()
                if intermediate_repr["returns"] is None:
                    intermediate_repr["returns"] = OrderedDict((("return_type", {}),))
                intermediate_repr["returns"]["return_type"].update(
                    interpolate_defaults(
                        (
                            "return_type",
                            dict((_set_param_values(line, val, return_tokens[-1]),)),
                        ),
                        emit_default_doc=emit_default_doc,
                        default_search_announce=default_search_announce,
                    )[1]
                )
            else:
                fst_space = line.find(" ")
                nxt_colon = line.find(":", fst_space)
                name = line[fst_space + 1 : nxt_colon]

                if param[0] is not None and not param[0] == name:
                    if not param[0][0] == "*":
                        intermediate_repr["params"][param[0]] = param[1]
                    param = [None, {}]

                val = line[nxt_colon + 1 :].strip()

                param = _set_name_and_type(
                    interpolate_defaults(
                        (
                            name,
                            update_d(param[1], dict((_set_param_values(line, val),))),
                        ),
                        emit_default_doc=emit_default_doc,
                    ),
                    infer_type=infer_type,
                    word_wrap=word_wrap,
                )
                if not emit_default_doc and not emit_default_prop:
                    param = _remove_default_from_param(
                        param, emit_default_prop=emit_default_doc
                    )
        elif not intermediate_repr["doc"]:
            intermediate_repr["doc"] = line.strip()
    if param != [None, {}]:
        # if param['name'] == 'return_type': intermediate_repr['returns'] = param
        name, param = _set_name_and_type(
            interpolate_defaults(param, emit_default_doc=emit_default_doc),
            infer_type=infer_type,
            word_wrap=word_wrap,
        )
        if not emit_default_doc and not emit_default_prop:
            name, param = _remove_default_from_param(
                (name, param), emit_default_prop=emit_default_doc
            )
        intermediate_repr["params"][name] = param


def _set_param_values(input_str, val, sw=":type"):
    """
    Sets the typ or doc values properly.

    :param val: The value (`sw` figures out what it means semantically)
    :type val: ```str```

    :param sw: Startswith condition
    :type sw: ```str```

    :returns: Properly derived key and [potentially modified] value
    :rtype: Tuple[Literal['doc', 'typ'], str]
    """
    return (
        ("typ", (lambda v: "dict" if v.startswith("**") else v)(val.replace("```", "")))
        if input_str.startswith(sw)
        else ("doc", val)
    )


__all__ = ["parse_docstring"]
