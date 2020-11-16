"""
Docstring parsers.

Translates from the [ReST docstring format (Sphinx)](
  https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format)

Translates from the [numpydoc docstring format](https://numpydoc.readthedocs.io/en/latest/format.html)

Translates from [Google's docstring format](https://google.github.io/styleguide/pyguide.html)
"""
from collections import namedtuple
from functools import partial
from operator import contains
from typing import Tuple, List, Dict

from docstring_parser import Style

from doctrans.emitter_utils import interpolate_defaults
from doctrans.pure_utils import location_within, BUILTIN_TYPES

TOKENS = namedtuple("Tokens", ("rest", "google", "numpydoc"))(
    (":param", ":cvar", ":ivar", ":var", ":type", ":return", ":rtype"),
    ("Args:", "Kwargs:", "Raises:", "Returns:"),
    ("Parameters\n----------", "Returns\n-------"),
)

RETURN_TOKENS = namedtuple("Tokens", ("rest", "google", "numpydoc"))(
    TOKENS.rest[-2:], (TOKENS.google[-1],), (TOKENS.numpydoc[-1],)
)


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
    known_tokens = getattr(TOKENS, style.name)
    return (
        _scan_phase_rest
        if style is Style.rest
        else partial(_scan_phase_numpydoc_and_google, style=style)
    )(docstring, known_tokens=known_tokens)


def _scan_phase_numpydoc_and_google(docstring, known_tokens, style):
    """
    numpydoc scanner phase. Lexical analysis; to some degree…

    :param docstring: the docstring
    :type docstring: ```str```

    :param known_tokens: Valid tokens like `"Parameters\n----------"`
    :type known_tokens: ```Tuple[str]```

    :param style: the style of docstring
    :type style: ```Style```

    :return: List with each element a tuple of (whether value is a token, value)
    :rtype: ```Dict[str, str]```
    """
    scanned: Dict[str, List[dict]] = {token: [] for token in ("doc",) + known_tokens}
    # ^ Dict[Union[Literal["doc"], known_tokens], List[dict]]

    # First doc, if present
    _start_idx, _end_idx, _found = location_within(docstring, (known_tokens[0],))
    if _start_idx == -1:
        # Return type no args?
        _start_idx, _end_idx, _found = location_within(docstring, (known_tokens[-1],))

    if _start_idx > -1:
        namespace = _found
        scanned["doc"] = docstring[:_start_idx].strip()
        docstring = docstring[_end_idx:].strip()
    else:
        scanned["doc"] = docstring.strip()
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
        if any(filter(typ.startswith, BUILTIN_TYPES)):
            typ, doc = doc, typ
        return {
            "name": "return_type",
            "typ": typ,
            "doc": doc,
        }

    if namespace == known_tokens[0]:
        _start_idx, _end_idx, _found = location_within(docstring, (known_tokens[-1],))
        if _start_idx > -1:
            ret_docstring = docstring[_end_idx:].lstrip()
            docstring = docstring[:_start_idx]

            scanned[_found] = parse_return(*ret_docstring.partition("\n"))

            # Next, separate into (namespace, name, [typ, doc, default]), updating `scanned` accordingly
            _parse_params_from_numpydoc_and_google(
                docstring, namespace, scanned, style=style
            )
    else:
        scanned[known_tokens[-1]] = parse_return(*docstring.partition("\n"))

    return scanned


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
            stack_str = "".join(stack).strip()
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
            stack_str = "".join(stack[:-1]).strip()
            if stack_str:
                cur = _parse_param({"name": stack_str, "doc": ""})
                col_on_line = True
                stack.clear()
    if cur:
        scanned[namespace].append(cur)


def _scan_phase_rest(docstring, known_tokens):
    """
    Scanner phase. Lexical analysis; to some degree…

    :param docstring: the docstring
    :type docstring: ```str```

    :param known_tokens: Valid tokens like `:param`
    :type known_tokens: ```Tuple[str]```

    :return: List with each element a tuple of (whether value is a token, value)
    :rtype: ```List[Tuple[bool, str]]```
    """

    rev_known_tokens_t = tuple(map(tuple, map(reversed, known_tokens)))
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
                or any(map(final.startswith, known_tokens)),
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
    return_tokens = getattr(RETURN_TOKENS, style.name)
    (
        _parse_phase_rest
        if style is Style.rest
        else partial(_parse_phase_numpydoc_and_google, style=style)
    )(intermediate_repr, scanned, emit_default_doc, return_tokens)


def _parse_phase_numpydoc_and_google(
    intermediate_repr, scanned, emit_default_doc, return_tokens, style
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

    :param style: the style of docstring
    :type style: ```Style```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``
    """
    known_tokens = getattr(TOKENS, style.name)
    _interpolate_defaults = partial(
        interpolate_defaults, emit_default_doc=emit_default_doc
    )
    intermediate_repr.update(
        {
            "doc": scanned["doc"],
            "params": list(
                map(_interpolate_defaults, scanned.get(known_tokens[0], []))
            ),
            "returns": _interpolate_defaults(scanned[return_tokens[0]])
            if scanned.get(return_tokens[0])
            else None,
        }
    )


def _parse_phase_rest(intermediate_repr, scanned, emit_default_doc, return_tokens):
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
