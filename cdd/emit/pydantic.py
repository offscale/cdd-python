"""
Pydantic `class` emitter

https://pydantic-docs.helpmanual.io/usage/schema/
"""

from functools import partial

import cdd.emit.class_

pydantic = partial(cdd.emit.class_.class_, class_bases=("BaseModel",))

__all__ = ["pydantic"]
