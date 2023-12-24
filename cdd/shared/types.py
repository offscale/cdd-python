"""
Shared types
"""

from ast import AnnAssign, Assign

from cdd.shared.pure_utils import PY_GTE_3_8, PY_GTE_3_9, PY_GTE_3_11

if PY_GTE_3_8:
    if PY_GTE_3_9:
        from collections import OrderedDict
    else:
        from typing import OrderedDict
    from typing import Any, List, Optional, TypedDict, Union

    if PY_GTE_3_11:
        from typing import Required
    else:
        from typing_extensions import Required
else:
    from typing_extensions import (
        Any,
        List,
        Optional,
        OrderedDict,
        Required,
        TypedDict,
        Union,
    )


# class Parse(Protocol):
#     def add(self, a: int, b: int) -> int:
#         return a + b
#
#
# def conforms_to_parse_protocol(parse: Parse):
#     pass


ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
Internal = TypedDict(
    "Internal",
    {
        "original_doc_str": Optional[str],
        "body": List[Union[AnnAssign, Assign]],
        "from_name": str,
        "from_type": str,
    },
    total=False,
)
IntermediateRepr = TypedDict(
    "IntermediateRepr",
    {
        "name": Required[Optional[str]],
        "type": Optional[str],
        "_internal": Internal,
        "doc": Required[Optional[str]],
        "params": Required[OrderedDict[str, ParamVal]],
        "returns": Required[
            Optional[OrderedDict[str, ParamVal]]
        ],  # OrderedDict[Literal["return_type"]
    },
    total=False,
)

__all__ = ["IntermediateRepr", "Internal"]  # type: list[str]