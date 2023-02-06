"""
Transform from string or AST representations of input, to intermediate_repr, a dictionary of form:
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
"""

from ast import AnnAssign, Assign, AsyncFunctionDef, ClassDef, FunctionDef

PARSERS = [
    "argparse_function",
    "class_",
    "docstring",
    "function",
    "json_schema",
    # "openapi",
    "pydantic",
    "routes",
    "sqlalchemy",
    "sqlalchemy_table",
]

kind2instance_type = {
    "sqlalchemy_table": (Assign, AnnAssign),
    "argparse": (FunctionDef,),
    "function": (FunctionDef, AsyncFunctionDef),
    "pydantic": (FunctionDef, AsyncFunctionDef),
    "class": (ClassDef,),
    "class_": (ClassDef,),
}

__all__ = ["PARSERS", "kind2instance_type"]
