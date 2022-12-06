"""
Utility functions for `cdd.parse.json_schema`
"""

from cdd.ast_utils import NoneStr, json_type2typ
from cdd.pure_utils import none_types


def json_schema_property_to_param(param, required):
    """
    Convert a JSON schema property to a param

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, dict]```

    :param required: Names of all required parameters
    :type required: ```FrozenSet[str]```

    :return: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```Tuple[str, dict]```
    """
    name, _param = param
    del param
    if name.endswith("kwargs"):
        _param["typ"] = "Optional[dict]"
    elif "enum" in _param:
        _param["typ"] = "Literal{}".format(_param.pop("enum"))
        del _param["type"]
    if "description" in _param:
        _param["doc"] = _param.pop("description")

    if _param.get("type"):
        _param["typ"] = json_type2typ[_param.pop("type")]

    if name not in required and _param.get("typ") and "Optional[" not in _param["typ"]:
        _param["typ"] = "Optional[{}]".format(_param["typ"])
        if _param.get("default") in none_types:
            _param["default"] = NoneStr

    return name, _param


__all__ = ["json_schema_property_to_param"]
