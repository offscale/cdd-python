"""
Utility functions for `cdd.parse.json_schema`
"""

from itertools import filterfalse

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
    # elif "enum" in _param:
    #     _param["typ"] = "Literal{}".format(_param.pop("enum"))
    #     del _param["type"]
    if "description" in _param:
        _param["doc"] = _param.pop("description")

    if _param.get("type"):
        _param["typ"] = json_type2typ[_param.pop("type")]

    if _param.get("pattern"):
        maybe_enum = _param["pattern"].split("|")
        if all(filter(str.isalpha, maybe_enum)):
            _param["typ"] = "Literal[{}]".format(
                ", ".join(map("'{}'".format, maybe_enum))
            )
            del _param["pattern"]

    def transform_ref_fk_set(ref, foreign_key):
        """
        Transform $ref to upper camel case and add to the foreign key

        :param ref: JSON ref
        :type ref: ```str```

        :param foreign_key: Foreign key structure (pass by reference)
        :type foreign_key: ```dict```

        :rtype: ```str```
        """
        entity = "".join(filterfalse(str.isspace, ref.rpartition("/")[2]))
        foreign_key["fk"] = entity
        return entity

    fk = {"fk": None}
    if "anyOf" in _param:
        _param["typ"] = tuple(
            map(
                lambda typ: (
                    transform_ref_fk_set(typ["$ref"], fk)
                    if "$ref" in typ
                    else typ["type"]
                )
                if isinstance(typ, dict)
                else typ,
                _param.pop("anyOf"),
            )
        )
        _param["typ"] = (
            _param["typ"][0]
            if len(_param["typ"]) == 1
            else "Union[{}]".format("|".join(_param["typ"]))
        )
    elif "$ref" in _param:
        _param["typ"] = transform_ref_fk_set(_param.pop("$ref"), fk)

    if fk["fk"] is not None:
        fk_prefix = "[FK({})]".format(fk.pop("fk"))
        _param["doc"] = (
            "[{}] {}".format(fk_prefix, _param["doc"])
            if _param.get("doc")
            else fk_prefix
        )

    if (
        name not in required
        and _param.get("typ")
        and "Optional[" not in _param["typ"]
        # Could also parse out a `Union` for `None`
        or _param.pop("nullable", False)
    ):
        _param["typ"] = "Optional[{}]".format(_param["typ"])
    if _param.get("default", False) in none_types:
        _param["default"] = NoneStr

    return name, _param


__all__ = ["json_schema_property_to_param"]
