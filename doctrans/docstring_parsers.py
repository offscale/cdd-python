"""
Docstring parsers.

Translates from the [ReST docstring format (Sphinx)](
  https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format)

Translates from the [numpydoc docstring format](https://numpydoc.readthedocs.io/en/latest/format.html)

Translates from [Google's docstring format](https://google.github.io/styleguide/pyguide.html)
"""
from collections import namedtuple
from functools import partial
from itertools import takewhile
from operator import contains, attrgetter
from typing import Tuple, List, Dict

from docstring_parser import Style

from doctrans.emitter_utils import interpolate_defaults
from doctrans.pure_utils import location_within, lstrip_diff

Tokens = namedtuple("Tokens", ("rest", "google", "numpydoc"))

TOKENS = Tokens(
    (":param", ":cvar", ":ivar", ":var", ":type", ":return", ":rtype"),
    ("Args:", "Kwargs:", "Raises:", "Returns:"),
    ("Parameters\n----------", "Returns\n-------"),
)

ARG_TOKENS = Tokens(
    TOKENS.rest[:-2],
    (TOKENS.google[0],),
    (TOKENS.numpydoc[0],),
)

RETURN_TOKENS = Tokens(TOKENS.rest[-2:], (TOKENS.google[-1],), (TOKENS.numpydoc[-1],))


def parse_docstring(docstring, emit_default_doc=False):
    """Parse the docstring into its components.

    :param docstring: the docstring
    :type docstring: ```Optional[str]```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

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
        "doc": "",
        "params": [],
        "returns": None,
    }
    if not docstring:
        return ir

    scanned = _scan_phase(docstring, style=style)
    _parse_phase(ir, scanned, emit_default_doc, style=style)

    return ir


def _scan_phase(docstring, style=Style.rest):
    """
    Scanner phase. Lexical analysis; to some degree…

    :param docstring: the docstring
    :type docstring: ```str```

    :param style: the style of docstring
    :type style: ```Style```

    :return: List with each element a tuple of (whether value is a token, value)
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
    numpydoc scanner phase. Lexical analysis; to some degree…

    :param docstring: the docstring
    :type docstring: ```str```

    :param arg_tokens: Valid tokens like `"Parameters\n----------"`
    :type arg_tokens: ```Tuple[str]```

    :param return_tokens: Valid tokens like `"Returns\n-------"`
    :type return_tokens: ```Tuple[str]```

    :param style: the style of docstring
    :type style: ```Style```

    :return: List with each element a tuple of (whether value is a token, value)
    :rtype: ```Dict[str, str]```
    """
    scanned: Dict[str, List[dict]] = {
        token: [] for token in ("doc",) + arg_tokens + return_tokens
    }
    # ^ Dict[Union[Literal["doc"], arg_tokens + return_tokens], List[dict]]

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

    docstring_lines = docstring.splitlines()
    stacker = []
    first_indent = len(tuple(takewhile(str.isspace, docstring_lines[0])))
    for line in docstring_lines:
        indent = len(tuple(takewhile(str.isspace, line)))
        if indent == first_indent:
            stacker.append([line])
        else:
            stacker[-1].append(line)

    rev_return_token = return_tokens[0].splitlines()[::-1]
    for i in range(len(stacker) - 1, -1, -1):
        if i - 1 > 0 and stacker[i] + stacker[i - 1] == rev_return_token:
            scanned[return_tokens[0]] = stacker[i + 1 :]
            stacker = stacker[: -i - 1]
            break
    scanned[namespace] = stacker
    return scanned


def parse_return(typ, _, doc):
    """
    Internal function to parse `str.partition` output into a return param

    :param typ: the type
    :type typ: ```str```

    :param _: Ignore this. It should be a newline character.
    :type _: ```str```

    :param doc: the doc
    :type doc: ```str```

    :return: dict of shape {'name': ..., 'typ': ..., 'doc': ... }
    :rtype: ```dict``
    """
    doc, typ = doc.lstrip(), typ.rstrip("\n \t\r:")
    # if any(filter(doc.startswith, BUILTIN_TYPES)): typ, doc = doc, typ
    return {
        "name": "return_type",
        "typ": typ,
        "doc": doc,
    }


def _parse_params_from_numpydoc_and_google(docstring, namespace, scanned, style):
    """
    Internal function used by `_scan_phase_numpydoc_and_google` to extract the params into the doctrans ir

    :param docstring: The docstring in numpydoc format
    :type docstring: ```str```

    :param namespace: The namespace, i.e., the key to update on the `scanned` param.
    :type namespace: ```Literal["Parameters\n----------"]```

    :param scanned: A list of dicts in docstring IR format, but with an outermost key of numpydoc known tokens
    :type scanned: ```Dict[str, List[dict]]```

    :param style: the style of docstring
    :type style: ```Style```
    """
    stack, cur, col_on_line = [], {}, False

    if style is Style.numpydoc:
        fallback_name = "typ"

        from doctrans.pure_utils import identity as _parse_param

    else:
        fallback_name = "doc"

        def _parse_param(param):
            """
            Parse param

            :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ... }
            :type param: ```dict```

            :return: Parsed param, i.e., a maybe changed version of the input `param`
            """
            name, _, typ = param["name"].partition("(")
            if typ:
                param.update({"name": name.rstrip(), "typ": typ[:-1]})
            return param

    for idx, ch in enumerate(docstring):
        stack.append(ch)
        if ch == "\n":
            changed_len, stack_str = lstrip_diff("".join(stack[:-1]))
            if stack_str:
                if col_on_line is True:
                    col_on_line = False
                    # cur["rest"] += stack_str
                    cur[fallback_name] = stack_str
                else:
                    assert (
                        cur
                    ), "Unhandled empty `cur`, maybe try `cur = {'doc': stack_str}`"
                    cur["doc"] = stack_str
                stack.clear()
        elif ch == ":":
            if "name" in cur:
                scanned[namespace].append(cur.copy())
                cur.clear()
            changed_len, stack_str = lstrip_diff("".join(stack[:-1]))
            if stack_str:
                cur = _parse_param({"name": stack_str, "doc": ""})
                col_on_line = True
                stack.clear()
    if cur:
        changed_len, stack_str = lstrip_diff("".join(stack))
        if stack_str:
            cur["doc"] = stack_str
        stack.clear()
        scanned[namespace].append(cur)


def _scan_phase_rest(docstring, arg_tokens, return_tokens):
    """
    Scanner phase. Lexical analysis; to some degree…

    :param docstring: the docstring
    :type docstring: ```str```

    :param arg_tokens: Valid tokens like `":param"`
    :type arg_tokens: ```Tuple[str]```

    :param return_tokens: Valid tokens like `":rtype:"`
    :type return_tokens: ```Tuple[str]```

    :return: List with each element a tuple of (whether value is a token, value)
    :rtype: ```List[Tuple[bool, str]]```
    """

    rev_known_tokens_t = tuple(map(tuple, map(reversed, arg_tokens + return_tokens)))
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
                or any(map(final.startswith, arg_tokens + return_tokens)),
                final,
            )
        )

    return scanned


def _parse_phase(intermediate_repr, scanned, emit_default_doc, style=Style.rest):
    """
    :param intermediate_repr: a dictionary of form
              {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :type intermediate_repr: ```dict```

    :param scanned: List with each element a tuple of (whether value is a token, value)
    :type scanned: ```Union[Dict[str, str], List[Tuple[bool, str]]]```

    :param style: the style of docstring
    :type style: ```Style```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``
    """
    arg_tokens, return_tokens = map(attrgetter(style.name), (ARG_TOKENS, RETURN_TOKENS))
    (_parse_phase_rest if style is Style.rest else _parse_phase_numpydoc_and_google)(
        intermediate_repr,
        scanned,
        emit_default_doc=emit_default_doc,
        arg_tokens=arg_tokens,
        return_tokens=return_tokens,
    )


def _parse_phase_numpydoc_and_google(
    intermediate_repr, scanned, arg_tokens, return_tokens, emit_default_doc
):
    """
    :param intermediate_repr: a dictionary of form
              {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :type intermediate_repr: ```dict```

    :param scanned: List with each element a tuple of (whether value is a token, value)
    :type scanned: ```Dict[str, str]```

    :param arg_tokens: Valid tokens like `"Parameters\n----------"`
    :type arg_tokens: ```Tuple[str]```

    :param return_tokens: Valid tokens like `"Returns\n-------"`
    :type return_tokens: ```Tuple[str]```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``
    """

    def _parse(scan):
        name, _, typ = scan[0].partition(":")
        cur = {"name": name.rstrip()}
        if typ:
            cur.update({"typ": typ, "doc": "\n".join(map(str.lstrip, scan[1:]))})
        return cur

    _interpolate_defaults = partial(
        interpolate_defaults, emit_default_doc=emit_default_doc
    )
    intermediate_repr.update(
        {
            "doc": scanned["doc"],
            "params": list(
                map(_interpolate_defaults, map(_parse, scanned[arg_tokens[0]]))
            ),
            "returns": _interpolate_defaults(
                {
                    "name": "return_type",
                    "typ": scanned[return_tokens[0]][0][0],
                    "doc": scanned[return_tokens[0]][0][1].lstrip(),
                }
            )
            if intermediate_repr["returns"]
            else None,
        }
    )


def _parse_phase_rest(
    intermediate_repr, scanned, emit_default_doc, arg_tokens, return_tokens
):
    """
    :param intermediate_repr: a dictionary of form
              {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :type intermediate_repr: ```dict```

    :param scanned: List with each element a tuple of (whether value is a token, value)
    :type scanned: ```List[Tuple[bool, str]]```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :param arg_tokens: Valid tokens like `":param"`
    :type arg_tokens: ```Tuple[str]```

    :param return_tokens: Valid tokens like `":rtype:"`
    :type return_tokens: ```Tuple[str]```
    """
    param = {}
    for is_token, line in scanned:
        if is_token is True:
            if any(map(line.startswith, return_tokens)):
                nxt_colon = line.find(":", 1)
                val = line[nxt_colon + 1 :].strip()
                if intermediate_repr["returns"] is None:
                    intermediate_repr["returns"] = {}
                intermediate_repr["returns"].update(
                    interpolate_defaults(
                        dict(
                            (_set_param_values(line, val, return_tokens[-1]),),
                            name="return_type",
                        ),
                        emit_default_doc=emit_default_doc,
                    )
                )
            else:
                fst_space = line.find(" ")
                nxt_colon = line.find(":", fst_space)
                name = line[fst_space + 1 : nxt_colon]

                if "name" in param and not param["name"] == name:
                    if not param["name"][0] == "*":
                        intermediate_repr["params"].append(param)
                    param = {}

                val = line[nxt_colon + 1 :].strip()
                param.update(dict((_set_param_values(line, val),), name=name))
                param = interpolate_defaults(param, emit_default_doc=emit_default_doc)
        elif not intermediate_repr["doc"]:
            intermediate_repr["doc"] = line.strip()
    if param:
        # if param['name'] == 'return_type': intermediate_repr['returns'] = param
        intermediate_repr["params"].append(param)


def _set_param_values(input_str, val, sw=":type"):
    """
    Sets the typ or doc values properly.

    :param val: The value (`sw` figures out what it means semantically)
    :type val: ```str```

    :param sw: Startswith condition
    :type sw: ```str```

    :return: Properly derived key and [potentially modified] value
    :rtype: Tuple[Literal['doc', 'typ'], str]
    """
    return (
        ("typ", (lambda v: "dict" if v.startswith("**") else v)(val.replace("```", "")))
        if input_str.startswith(sw)
        else ("doc", val)
    )


__all__ = ["parse_docstring"]
