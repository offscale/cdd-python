"""
Pydantic `class` emitter

https://pydantic-docs.helpmanual.io/usage/schema/
"""

from ast import ClassDef
from functools import partial
from typing import Callable, Iterable, List, Literal, Optional

import cdd.class_.emit
from cdd.shared.types import IntermediateRepr

pydantic: Callable[
    [
        IntermediateRepr,
        bool,
        str,
        Iterable[str],
        Optional[List[str]],
        bool,
        Literal["rest", "numpydoc", "google"],
        bool,
        bool,
    ],
    ClassDef,
] = partial(cdd.class_.emit.class_, class_bases=("BaseModel",))

__all__ = ["pydantic"]
