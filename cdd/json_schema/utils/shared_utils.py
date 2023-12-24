"""
Shared utility functions for JSON schema
"""

from typing import Any, Dict, List, Literal, TypedDict

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
