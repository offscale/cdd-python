"""
ReST docstring parser.

Translates from the [ReST docstring format (Sphinx)](
  https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format)
"""

from typing import Tuple, List

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
                  'short_description': ...,
                  'long_description': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :rtype: ```dict```
    """

    assert isinstance(docstring, (type(None), str))

    ir = {
        "name": None,
        "type": "static",
        "short_description": "",
        "long_description": "",
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
        scanned.append((bool(len(scanned) and scanned[-1][0]), "".join(stack)))

    return scanned


def _parse_phase(intermediate_repr, scanned, emit_default_doc):
    """
    :param intermediate_repr: a dictionary of form
              {
                  'short_description': ...,
                  'long_description': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
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
                    intermediate_repr["params"].append(param)
                    param = {}

                val = line[nxt_colon + 1 :].strip()
                param.update(dict((_set_param_values(line, val),), name=name))
                param = interpolate_defaults(param, emit_default_doc=emit_default_doc)
        elif not intermediate_repr["short_description"]:
            intermediate_repr["short_description"] = line.strip()
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
