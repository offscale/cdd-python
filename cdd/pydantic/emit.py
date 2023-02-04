"""
Pydantic `class` emitter

https://pydantic-docs.helpmanual.io/usage/schema/
"""

from functools import partial

import cdd.class_.emit

pydantic = partial(cdd.class_.emit.class_, class_bases=("BaseModel",))

__all__ = ["pydantic"]
