"""
Transform from string or AST representations of input, to intermediate_repr, a dictionary of form:
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
"""

PARSERS = [
    "argparse_function",
    "class_",
    "docstring",
    "function",
    "json_schema",
    "openapi",
    "pydantic",
    "sqlalchemy",
    "sqlalchemy_table",
]

__all__ = ["PARSERS"]
