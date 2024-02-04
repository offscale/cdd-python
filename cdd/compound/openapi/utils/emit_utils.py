"""
Utility functions for `cdd.emit.sqlalchemy`
"""

from cdd.sqlalchemy.utils.emit_utils import typ2column_type

typ2column_type.update(
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
