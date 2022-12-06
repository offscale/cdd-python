"""
JSON schema parser
"""

from collections import OrderedDict
from copy import deepcopy
from functools import partial

from cdd.parse.docstring import docstring
from cdd.parse.utils.json_schema_utils import json_schema_property_to_param


def json_schema(json_schema_dict, parse_original_whitespace=False):
    """
    Parse a JSON schema into the IR

    :param json_schema_dict: A valid JSON schema as a Python dict
    :type json_schema_dict: ```dict```

    :param parse_original_whitespace: Whether to parse original whitespace or strip it out
    :type parse_original_whitespace: ```bool```

    :return: IR representation of the given JSON schema
    :rtype: ```dict```
    """
    # I suppose a JSON-schema validation routine could be executed here
    schema = deepcopy(json_schema_dict)

    required = frozenset(schema["required"]) if schema.get("required") else frozenset()
    _json_schema_property_to_param = partial(
        json_schema_property_to_param, required=required
    )

    ir = docstring(
        json_schema_dict["description"],
        emit_default_doc=False,
        parse_original_whitespace=parse_original_whitespace,
    )
    ir["params"] = OrderedDict(
        map(_json_schema_property_to_param, schema["properties"].items())
    )
    return ir


__all__ = ["json_schema"]
