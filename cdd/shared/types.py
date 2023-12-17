"""
Shared types
"""

from cdd.shared.pure_utils import PY_GTE_3_8, PY_GTE_3_9

if PY_GTE_3_8:
    if PY_GTE_3_9:
        from collections import OrderedDict
    else:
        from typing import OrderedDict
    from typing import Any, Optional, Required, TypedDict
else:
    from typing_extensions import Any, Optional, OrderedDict, Required, TypedDict


# class Parse(Protocol):
#     def add(self, a: int, b: int) -> int:
#         return a + b
#
#
# def conforms_to_parse_protocol(parse: Parse):
#     pass


ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
IntermediateRepr = TypedDict(
    "IntermediateRepr",
    {
        "name": Required[Optional[str]],
        "type": Optional[str],
        "doc": Required[Optional[str]],
        "params": Required[OrderedDict[str, ParamVal]],
        "returns": Required[
            Optional[OrderedDict[str, ParamVal]]
        ],  # OrderedDict[Literal["return_type"]
    },
    total=False,
)

__all__ = ["IntermediateRepr"]
