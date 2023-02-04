"""
Utility functions for `cdd.parse.json_schema`
"""

from cdd.shared.ast_utils import NoneStr
from cdd.shared.pure_utils import namespaced_pascal_to_upper_camelcase, none_types


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

        :return: $ref without the namespace and in upper camel case
        :rtype: ```str```
        """
        entity = namespaced_pascal_to_upper_camelcase(
            ref.rpartition("/")[2].replace(".", "__")
        )
        foreign_key["fk"] = entity
        return entity

    fk = {"fk": None}
    if "anyOf" in _param:
        _param["typ"] = list(
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
        if len(_param["typ"]) > 1 and "string" in _param["typ"]:
            del _param["typ"][_param["typ"].index("string")]
        _param["typ"] = (
            _param["typ"][0]
            if len(_param["typ"]) == 1
            else "Union[{}]".format(",".join(_param["typ"]))
        )
    elif "$ref" in _param:
        _param["typ"] = transform_ref_fk_set(_param.pop("$ref"), fk)

    if fk["fk"] is not None:
        fk_val = fk.pop("fk")
        fk_prefix = fk_val if fk_val.startswith("[FK(") else "[FK({})]".format(fk_val)
        _param["doc"] = (
            "{} {}".format(fk_prefix, _param["doc"]) if _param.get("doc") else fk_prefix
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


# https://json-schema.org/draft/2019-09/json-schema-core.html#rfc.section.4.2.1
json_type2typ = {
    "boolean": "bool",
    "string": "str",
    "object": "dict",
    "array": "list",
    "int": "integer",
    "integer": "int",
    "float": "number",  # <- Actually a problem, maybe `literal_eval` on default then `type()` or just `type(default)`?
    "number": "float",
    "null": "NoneType",
}


__all__ = ["json_schema_property_to_param"]
