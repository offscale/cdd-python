"""
ReST docstring parser.

Translates from the [ReST docstring format (Sphinx)](
  https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format)
"""

from itertools import takewhile

from doctrans.emitter_utils import interpolate_defaults


def parse_docstring(docstring, emit_default_doc=False):
    """Parse the docstring into its components.

    :param docstring: the docstring
    :type docstring: ```Optional[str]```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :returns: a dictionary of form
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

    idx, param = 0, {}
    while idx < len(docstring):
        if not ir["short_description"]:
            ir["short_description"] = " {}".format(
                "".join(takewhile(lambda ch: ch != ":", docstring))
            )
            idx = len(ir["short_description"])
        else:
            first_col = "".join(takewhile(lambda ch: ch != ":", docstring[idx:]))
            key = name = first_col[first_col.find(" ") + 1 :]
            if first_col[: len("return")] in frozenset(("param ", "return")):
                key = name
                name = "doc"

            idx += len(first_col) + 1
            second_col = "".join(takewhile(lambda ch: ch != ":", docstring[idx:]))
            idx += len(second_col) + 1
            if "name" in param and "typ" in param or param.get("name", key) != key:
                interpolate_defaults(param, emit_default_doc=emit_default_doc)
                # if len(ir["params"]) and ir["params"][-1]['name'] == param['name']:ir["params"][-1].update(param)
                ir["params"].append(param)
                param = {}

            second_col = second_col.strip(" \n`")
            if name == "doc":
                param.update({"name": key, name: second_col})
            else:
                param.update(
                    {
                        "name": name,
                        "typ": "dict" if second_col.startswith("**") else second_col,
                    }
                )

    ir["short_description"] = ir["short_description"].strip()
    if param:
        param = interpolate_defaults(param, emit_default_doc=emit_default_doc)
        if param.get("name") == "rtype":
            param["name"] = "return_type"
            ir["returns"] = param
        else:
            ir["params"].append(param)
    if len(ir["params"]) and ir["params"][-1]["name"] == "return":
        del ir["params"][-1]["name"]
        returns = ir["params"].pop()
        if ir["returns"] is None:
            ir["returns"] = dict(name="return_type", **returns)
        else:
            ir["returns"].update(returns)
    return ir


__all__ = ["parse_docstring"]
