"""
Utility functions for `cdd.emit.json_schema`
"""

import ast
from ast import AST, Set
from typing import Dict

import cdd.shared.ast_utils
from cdd.json_schema.utils.parse_utils import json_type2typ
from cdd.shared.pure_utils import none_types


def param2json_schema_property(param, required):
    """
    Turn a param into a JSON schema property

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```tuple[str, dict]```

    :param required: Required parameters. This function may push to the list.
    :type required: ```list[str]```

    :return: JSON schema property. Also, may push to `required`.
    :rtype: ```dict```
    """
    name, _param = param
    del param
    if _param.get("doc"):
        _param["description"] = _param.pop("doc")
    if _param.get("typ") == "datetime":
        del _param["typ"]
        _param.update({"type": "string", "format": "date-time"})
        required.append(name)
    elif _param.get("typ") in typ2json_type:
        _param["type"] = typ2json_type[_param.pop("typ")]
        required.append(name)
    elif _param.get("typ", ast) is not ast:
        _param["type"] = _param.pop("typ")
        if _param["type"].startswith("Optional["):
            _param["type"] = _param["type"][len("Optional[") : -1]
            if _param["type"] in typ2json_type:
                _param["type"] = typ2json_type[_param["type"]]
            # elif _param.get("typ") in typ2json_type:
            #    _param["type"] = typ2json_type[_param.pop("typ")]
        else:
            required.append(name)

        if _param["type"].startswith("Literal["):
            parsed_typ = cdd.shared.ast_utils.get_value(
                ast.parse(_param["type"]).body[0]
            )
            assert (
                parsed_typ.value.id == "Literal"
            ), "Only basic Literal support is implemented, not {}".format(
                parsed_typ.value.id
            )
            enum = sorted(
                map(
                    cdd.shared.ast_utils.get_value,
                    cdd.shared.ast_utils.get_value(parsed_typ.slice).elts,
                )
            )
            _param.update(
                {
                    "pattern": "|".join(enum),
                    "type": typ2json_type[type(enum[0]).__name__],
                }
            )
    if _param.get("default", False) in none_types:
        del _param["default"]  # Will be inferred as `null` from the type
    elif isinstance(_param.get("default"), AST):
        _param["default"] = cdd.shared.ast_utils.ast_type_to_python_type(
            _param["default"]
        )
    if isinstance(_param.get("choices"), Set):
        _param["pattern"] = "|".join(
            sorted(map(str, cdd.shared.ast_utils.Set_to_set(_param.pop("choices"))))
        )
    return name, _param


typ2json_type: Dict[str, str] = {v: k for k, v in json_type2typ.items()}

__all__ = ["param2json_schema_property"]
