"""
Pydantic `class` parser

https://pydantic-docs.helpmanual.io/usage/schema/
"""

from functools import partial

import cdd.parse.class_

pydantic = partial(cdd.parse.class_.class_, infer_type=True)

__all__ = ["pydantic"]
