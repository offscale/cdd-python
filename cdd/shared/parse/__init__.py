"""
Transform from string or AST representations of input, to intermediate_repr, a dictionary consistent
   with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
"""

from ast import AnnAssign, Assign, AsyncFunctionDef, ClassDef, FunctionDef
from typing import List

PARSERS: List[str] = [
    "argparse_function",
    "class_",
    "docstring",
    "function",
    "json_schema",
    # "openapi",
    "pydantic",
    "routes",
    "sqlalchemy",
    "sqlalchemy_hybrid",
    "sqlalchemy_table",
]

kind2instance_type = {
    "argparse": (FunctionDef,),
    "argparse_function": (FunctionDef,),
    "class": (ClassDef,),
    "class_": (ClassDef,),
    "function": (FunctionDef, AsyncFunctionDef),
    "method": (FunctionDef, AsyncFunctionDef),
    "pydantic": (ClassDef,),
    "sqlalchemy_hybrid": (ClassDef,),
    "sqlalchemy_table": (Assign, AnnAssign),
    "sqlalchemy": (ClassDef,),
}

__all__ = ["PARSERS", "kind2instance_type"]  # type: list[str]
