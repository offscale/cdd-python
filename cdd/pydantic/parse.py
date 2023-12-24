"""
Pydantic `class` parser

https://pydantic-docs.helpmanual.io/usage/schema/
"""

from functools import partial

import cdd.class_.parse
from cdd.class_.utils.shared_utils import ClassParserProtocol

pydantic: ClassParserProtocol = partial(cdd.class_.parse.class_, infer_type=True)

__all__ = ["pydantic"]  # type: list[str]
