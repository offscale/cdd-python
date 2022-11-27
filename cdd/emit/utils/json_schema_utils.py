"""
Utility functions for `cdd.emit.json_schema`
"""
import ast

from cdd.ast_utils import get_value, typ2json_type
from cdd.pure_utils import none_types


def param2json_schema_property(param, required):
    """
    Turn a param into a JSON schema property

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, dict]```

    :param required: Required parameters. This function may push to the list.
    :type required: ```List[str]```

    :return: JSON schema property. Also may push to `required`.
    :rtype: ```dict```
    """
    name, _param = param
    del param

    if _param.get("doc"):
        _param["description"] = _param.pop("doc")
    if _param.get("typ", ast) is not ast:
        _param["type"] = _param.pop("typ")
        if _param["type"].startswith("Optional["):
            _param["type"] = _param["type"][len("Optional[") : -1]
        else:
            required.append(name)

        if _param["type"].startswith("Literal["):
            parsed_typ = get_value(ast.parse(_param["type"]).body[0])
            assert (
                parsed_typ.value.id == "Literal"
            ), "Only basic Literal support is implemented, not {}".format(
                parsed_typ.value.id
            )
            _param["enum"] = list(map(get_value, get_value(parsed_typ.slice).elts))
            _param["type"] = typ2json_type[type(_param["enum"][0]).__name__]
        else:
            _param["type"] = typ2json_type[_param["type"]]
    if _param.get("default", False) in none_types:
        del _param["default"]  # Will be inferred as `null` from the type
    return name, _param


__all__ = ["param2json_schema_property"]
