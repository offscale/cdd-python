"""
Pydantic `class` parser

https://pydantic-docs.helpmanual.io/usage/schema/
"""

from ast import ClassDef, Module
from functools import partial
from typing import Callable, Optional, Union

import cdd.class_.parse
from cdd.shared.types import IntermediateRepr

pydantic: Callable[
    [Union[Module, ClassDef], Optional[str], Optional[str], bool, bool, bool],
    IntermediateRepr,
] = partial(cdd.class_.parse.class_, infer_type=True)

__all__ = ["pydantic"]
