"""
Pydantic `class` parser

https://pydantic-docs.helpmanual.io/usage/schema/
"""

from functools import partial

import cdd.class_.parse

pydantic = partial(cdd.class_.parse.class_, infer_type=True)

__all__ = ["pydantic"]
