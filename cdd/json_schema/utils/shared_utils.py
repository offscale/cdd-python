"""
Shared utility functions for JSON schema
"""

from typing import Any, Dict, List

from cdd.shared.pure_utils import PY_GTE_3_8

if PY_GTE_3_8:
    from typing import Literal, TypedDict
else:
    from typing_extensions import Literal, TypedDict

JSON_property = TypedDict(
    "JSON_property",
    {"description": str, "type": str, "default": Any, "x_typ": Any},
    total=False,
)
JSON_schema = TypedDict(
    "JSON_schema",
    {
        "$id": str,
        "$schema": str,
        "description": str,
        "type": Literal["object"],
        "properties": Dict[str, JSON_property],
        "required": List[str],
    },
)

__all__ = ["JSON_schema"]  # type: list[str]
