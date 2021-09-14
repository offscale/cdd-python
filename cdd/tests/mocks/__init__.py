"""
Shared by the mocks. Currently unused, but has some imports mocked for later use…
"""

from ast import parse as ast_parse

from cdd.pure_utils import PY_GTE_3_8

imports_header = """
from {package} import Literal
from typing import Optional, Tuple, Union

try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    tf = type('TensorFlow', tuple(), {{ 'data': type('Dataset', tuple(), {{ "Dataset": None }}) }} )
    np = type('numpy', tuple(), {{ 'ndarray': None, 'empty': lambda _: _ }})
""".format(
    package="typing" if PY_GTE_3_8 else "typing_extensions"
)

imports_header_ast = ast_parse(imports_header).body

__all__ = ["imports_header", "imports_header_ast"]
