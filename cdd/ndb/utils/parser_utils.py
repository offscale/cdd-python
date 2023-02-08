"""
Utility functions for `cdd.ndb.parse`
"""

import ast


def property_to_param(property):
    """
    Parse property assignment `Assign` into param

    :param property: assignment in ndb model
    :type property: ```Assign```

    :return: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```Tuple[str, dict]```
    """
    name, value = property.targets[0].id, property.value.func
    _param = {
        "typ": column_type2typ[value.attr]
    }
    return name, _param


column_type2typ = {
    "IntegerProperty": "int",
    "BooleanProperty": "bool",
    "DateTimeProperty": "datetime",
    "FloatProperty": "float",
    "JsonProperty": "Optional[dict]",
    "BlobProperty": "BlobProperty",
    "StringProperty": "str",
    "TextProperty": "str",
}
