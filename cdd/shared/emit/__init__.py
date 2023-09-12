"""
Transform from string or AST representations of input, to AST, file, or str input_str.
"""

# from cdd.emit.argparse_function import argparse_function
# from cdd.emit.class_ import class_
# from cdd.emit.docstring import docstring
# from cdd.emit.file import file
# from cdd.emit.function import function
# from cdd.emit.json_schema import json_schema
# from cdd.emit.sqlalchemy import sqlalchemy, sqlalchemy_table

EMITTERS = [
    "argparse_function",
    "class_",
    "docstring",
    # "file",
    "function",
    "json_schema",
    # "openapi",
    "pydantic",
    "routes",
    "sqlalchemy",
    "sqlalchemy_hybrid",
    "sqlalchemy_table",
]

__all__ = ["EMITTERS"]
