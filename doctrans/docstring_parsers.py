"""
Docstring parsers.

Translates from the [ReST docstring format (Sphinx)](
  https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format)

Translates from the [numpydoc docstring format](https://numpydoc.readthedocs.io/en/latest/format.html)
"""
from collections import namedtuple
from functools import partial
from operator import contains, eq
from typing import Tuple, List, Union, Dict

from docstring_parser import Style

from doctrans.emitter_utils import interpolate_defaults
from doctrans.pure_utils import BUILTIN_TYPES, location_within

TOKENS = namedtuple("Tokens", ("rest", "google", "numpydoc"))(
    (":param", ":cvar", ":ivar", ":var", ":type", ":return", ":rtype"),
    ("Args:", "Kwargs:", "Returns:", "Raises:"),
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

    # if style is Style.numpydoc:
    #     raise NotImplementedError()
    # elif style is not Style.rest:
    #
    #     def process_param(param):
    #         """
    #         Postprocess the param
    #
    #         :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    #         :type param: ```dict``
    #
    #         :return: Potentially changed param
    #         :rtype: ```dict```
    #         """
    #         if "type_name" in param:
    #             param["typ"] = param.pop("type_name")
    #         elif param["name"].endswith("kwargs"):
    #             param.update({"typ": "dict", "name": param["name"].lstrip("*")})
    #         if "is_optional" in param:
    #             if param["is_optional"] and "optional" not in param["typ"].lower():
    #                 param["typ"] = "Optional[{}]".format(param["typ"])
    #             del param["is_optional"]
    #         return param
    #
    #     ir = parse.docstring_parser(docstring_parser_(docstring, style=style))
    #     ir.update(
    #         {
    #             "params": list(map(process_param, ir["params"])),
    #             "type": {"self": "self", "cls": "cls"}.get(
    #                 ir["params"][0]["name"] if ir["params"] else None, "static"
    #             ),
    #         }
    #     )
    #     if ir.get("returns"):
    #         ir["returns"]["name"] = "return_type"
    #         ir["returns"]["doc"], ir["returns"]["default"] = extract_default(
    #             ir["returns"]["doc"], emit_default_doc=emit_default_doc
    #         )
    #     del ir["raises"]
    #     return ir

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
    if style is Style.rest:
        return _scan_phase_rest(docstring, known_tokens=known_tokens)
    elif style is Style.numpydoc:
        return _scan_phase_numpydoc(docstring, known_tokens=known_tokens)
    else:
        raise NotImplementedError(Style.name)


def add_to_scanned(scanned_str):
    """
    Internal function to add to the internal scanned ```OrderedDict```

    :param scanned_str: Scanned string
    :type scanned_str: ```str```
    """
    add_to_scanned.namespace = next(
        filter(partial(str.startswith, scanned_str), known_tokens),
        add_to_scanned.namespace,
    )
    if scanned_str.startswith(add_to_scanned.namespace):
        scanned_str = scanned_str[len(add_to_scanned.namespace):]
    if scanned_str:
        scanned[add_to_scanned.namespace] += "\n{}".format(scanned_str)


def is_type(s, where=eq):
    """
    Checks if it's a type

    :param s: input
    :type s: ```str```

    :param where: Where to look for type
    :type where: ```Callable[[Any, Any], bool]```

    :return: Whether it's a type
    :rtype: ```bool```
    """
    return any(filter(partial(where, s), BUILTIN_TYPES))


def where_type(s):
    """
    Finds type within str

    :param s: input
    :type s: ```str```

    :return: (Start index iff found else -1, End index iff found else -1, type iff found else None)
    :rtype: ```Tuple[int, int, Optional[str]]```
    """
    return location_within(s, BUILTIN_TYPES)


def is_name(s):
    """
    Checks if it's a name

    :param s: input
    :type s: ```str```

    :return: Whether it's a name
    :rtype: ```bool```
    """
    return s.isalpha()


def is_doc(s):
    """
    Checks if it's a doc

    :param s: input
    :type s: ```str```

    :return: Whether it's a doc
    :rtype: ```bool```
    """
    return not is_type(s) and not is_name(s)


add_to_scanned.namespace = "doc"


def _scan_phase_numpydoc(docstring, known_tokens):
    """
    numpydoc scanner phase. Lexical analysis; to some degree…

    :param docstring: the docstring
    :type docstring: ```str```

    :param known_tokens: Valid tokens like `"Parameters\n----------"`
    :type known_tokens: ```Tuple[str]```

    :return: List with each element a tuple of (whether value is a token, value)
    :rtype: ```Dict[str, str]```
    """
    scanned: Dict[str, str] = {token: [] for token in ("doc",) + known_tokens}
    # ^ Union[Literal["doc"], known_tokens]

    # First doc, if present
    _start_idx, _end_idx, _found = location_within(docstring, (known_tokens[0],))
    if _start_idx == -1:
        # Return type no args?
        _start_idx, _end_idx, _found = location_within(docstring, (known_tokens[1],))

    if _start_idx > -1:
        namespace = _found
        scanned["doc"] = docstring[:_start_idx]
        docstring = docstring[_end_idx:].strip()
    else:
        scanned["doc"] = docstring
        return scanned

    def parse_return(typ, _, doc):
        return {"name": "return_type",
                "typ": typ,
                "doc": doc.lstrip()}

    if namespace == known_tokens[0]:
        _start_idx, _end_idx, _found = location_within(docstring, (known_tokens[1],))
        if _start_idx > -1:
            ret_docstring = docstring[_end_idx:].lstrip()
            docstring = docstring[:_start_idx]

            scanned[_found] = parse_return(*ret_docstring.partition("\n"))

            # Next, separate into (namespace, name, [typ, doc, default]), updating `scanned` accordingly
            stack, lines, cur, col_on_line = [], [], {}, False
            for idx, ch in enumerate(docstring):
                stack.append(ch)
                if ch == '\n':
                    stack_str = "".join(stack).strip()
                    if col_on_line is True:
                        col_on_line = False
                        # cur["rest"] += stack_str
                        cur["typ"] = stack_str
                    else:
                        if cur:
                            cur["doc"] = stack_str
                        else:
                            cur = {"doc": stack_str}
                    stack.clear()
                elif ch == ':':
                    if "name" in cur:
                        scanned[namespace].append(cur.copy())
                        cur.clear()
                    stack_str = "".join(stack[:-1]).strip()
                    if stack_str:
                        cur = {"name": stack_str, "doc": ""}
                        col_on_line = True
                        stack.clear()
            if cur:
                scanned[namespace].append(cur)
    else:
        scanned[known_tokens[1]] = parse_return(*docstring.partition("\n"))

    return scanned


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
                stack = stack[len(scanned[-1][1]):][:token_len]
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
    if style is Style.rest:
        return _parse_phase_rest(
            intermediate_repr, scanned, emit_default_doc, return_tokens
        )
    elif style is Style.numpydoc:
        return _parse_phase_numpydoc(
            intermediate_repr, scanned, emit_default_doc, return_tokens
        )
    else:
        raise NotImplementedError(style.name)


def _parse_phase_numpydoc(intermediate_repr, scanned, emit_default_doc, return_tokens):
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
    known_tokens = getattr(TOKENS, Style.numpydoc.name)
    intermediate_repr.update(
        interpolate_defaults({
            "doc": scanned["doc"],
            "params": scanned.get(known_tokens[0], []),
            "returns": scanned.get(return_tokens[0], None)
        }, emit_default_doc=emit_default_doc)
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

    :param style: the style of docstring
    :type style: ```Style```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``
    """
    param = {}
    for is_token, line in scanned:
        if is_token is True:
            if any(map(line.startswith, return_tokens)):
                nxt_colon = line.find(":", 1)
                val = line[nxt_colon + 1:].strip()
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
                name = line[fst_space + 1: nxt_colon]

                if "name" in param and not param["name"] == name:
                    if not param["name"][0] == "*":
                        intermediate_repr["params"].append(param)
                    param = {}

                val = line[nxt_colon + 1:].strip()
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
