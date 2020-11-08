"""
ReST docstring parser.

Translates from the [ReST docstring format (Sphinx)](
  https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format)
"""
from typing import Tuple, List

from docstring_parser import parse as docstring_parser_, Style

from doctrans import parse
from doctrans.defaults_utils import extract_default
from doctrans.emitter_utils import interpolate_defaults


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

    assert isinstance(docstring, (type(None), str))
    if docstring is None or any(
        e in docstring for e in (":param", ":cvar", ":ivar", ":return")
    ):
        style = Style.rest
    elif any(e in docstring for e in ("Args:", "Kwargs:", "Returns:", "Raises:")):
        style = Style.google
    else:
        style = Style.numpydoc

    if style is not Style.rest:

        def process_param(param):
            """
            Postprocess the param

            :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
            :type param: ```dict``

            :return: Potentially changed param
            :rtype: ```dict```
            """
            if "type_name" in param:
                param["typ"] = param.pop("type_name")
            elif param["name"].endswith("kwargs"):
                param.update({"typ": "dict", "name": param["name"].lstrip("*")})
            if "is_optional" in param:
                if param["is_optional"] and "optional" not in param["typ"].lower():
                    param["typ"] = "Optional[{}]".format(param["typ"])
                del param["is_optional"]
            return param

        ir = parse.docstring_parser(docstring_parser_(docstring, style=style))
        ir.update(
            {
                "params": list(map(process_param, ir["params"])),
                "type": {"self": "self", "cls": "cls"}.get(
                    ir["params"][0]["name"] if ir["params"] else None, "static"
                ),
            }
        )
        if ir.get("returns"):
            ir["returns"]["name"] = "return_type"
            ir["returns"]["doc"], ir["returns"]["default"] = extract_default(
                ir["returns"]["doc"], emit_default_doc=emit_default_doc
            )
        del ir["raises"]
        return ir

    ir = {
        "name": None,
        "type": "static",
        "doc": "",
        "params": [],
        "returns": None,
    }
    if not docstring:
        return ir

    scanned = _scan_phase(docstring)
    _parse_phase(ir, scanned, emit_default_doc)

    return ir


def _scan_phase(docstring):
    """
    Scanner phase. Lexical analysis; to some degree…

    :param docstring: the docstring
    :type docstring: ```str```

    :return: List with each element a tuple of (whether value is a token, value)
    :rtype: ```List[Tuple[bool, str]]```
    """
    known_tokens = ":param", ":cvar", ":ivar", ":var", ":type", ":rtype", ":return"
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

    if len(stack):
        final = "".join(stack)
        scanned.append(
            (
                bool(len(scanned) and scanned[-1][0])
                or any(final.startswith(known_token) for known_token in known_tokens),
                final,
            )
        )

    return scanned


def _parse_phase(intermediate_repr, scanned, emit_default_doc):
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
            if any(map(line.startswith, (":rtype", ":return"))):
                nxt_colon = line.find(":", 1)
                val = line[nxt_colon + 1 :].strip()
                if intermediate_repr["returns"] is None:
                    intermediate_repr["returns"] = {}
                intermediate_repr["returns"].update(
                    interpolate_defaults(
                        dict(
                            (_set_param_values(line, val, ":rtype"),),
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
