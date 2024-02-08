"""
Utility functions for `cdd.emit.sqlalchemy`
"""

import cdd.sqlalchemy.utils.emit_utils

cdd.sqlalchemy.utils.emit_utils.typ2column_type.update(
    {
        "bool": "Boolean",
        "dict": "JSON",
        "float": "Float",
        "int": "Integer",
        "str": "String",
        "string": "String",
        "int64": "BigInteger",
        "Optional[dict]": "JSON",
    }
)

__all__ = []  # type: list[str]
