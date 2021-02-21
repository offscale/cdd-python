"""
Shared by the mocks. Currently unused, but has some imports mocked for later use…
"""
from cdd.pure_utils import PY_GTE_3_8

imports_header = """
from {package} import Literal
from typing import Optional, Tuple, Union

try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    from collections import namedtuple

    tf = namedtuple('TensorFlow', ('data',))(namedtuple('data', ('Dataset',)))
    np = type('numpy', tuple(), {{'ndarray': None, 'empty': lambda _: _}})
""".format(
    package="typing" if PY_GTE_3_8 else "typing_extensions"
)

__all__ = ["imports_header"]
