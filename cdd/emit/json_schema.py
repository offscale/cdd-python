"""
JSON schema emitter
"""

from collections import OrderedDict
from functools import partial
from operator import add

from cdd.emit.docstring import docstring
from cdd.emit.utils.json_schema_utils import param2json_schema_property
from cdd.pure_utils import deindent


def json_schema(
    intermediate_repr,
    identifier=None,
    emit_original_whitespace=False,
):
    """
    Construct a JSON schema dict

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param identifier: The `$id` of the schema
    :type identifier: ```str```

    :param emit_original_whitespace: Whether to emit original whitespace (in top-level `description`) or strip it out
    :type emit_original_whitespace: ```bool```

    :return: JSON Schema dict
    :rtype: ```dict```
    """
    assert isinstance(intermediate_repr, dict), "{typ} != FunctionDef".format(
        typ=type(intermediate_repr).__name__
    )
    if identifier is None:
        identifier = "https://offscale.io/{}.schema.json".format(
            intermediate_repr["name"]
        )
    required = []
    _param2json_schema_property = partial(param2json_schema_property, required=required)
    properties = dict(
        map(_param2json_schema_property, intermediate_repr["params"].items())
    )

    return {
        "$id": identifier,
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "description": deindent(
            add(
                *map(
                    partial(
                        docstring,
                        emit_default_doc=True,
                        emit_original_whitespace=emit_original_whitespace,
                        emit_types=True,
                    ),
                    (
                        {
                            "doc": intermediate_repr["doc"],
                            "params": OrderedDict(),
                            "returns": None,
                        },
                        {
                            "doc": "",
                            "params": OrderedDict(),
                            "returns": intermediate_repr["returns"],
                        },
                    ),
                )
            )
        ).lstrip("\n")
        or None,
        "type": "object",
        "properties": properties,
        "required": required,
    }


__all__ = ["json_schema"]
