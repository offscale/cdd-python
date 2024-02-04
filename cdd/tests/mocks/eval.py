""" eval.py for testing with `sync_properties` """

import sys

if sys.version_info >= (3, 9):
    FrozenSet = frozenset
else:
    from typing import FrozenSet

import cdd.tests.mocks

_attr_within: FrozenSet[str] = frozenset(("mocks",))

get_modules = tuple(
    attr for attr in dir(cdd.tests) if not attr.startswith("_") and attr in _attr_within
)  # type: tuple[str, ...]

__all__ = ["get_modules"]  # type: list[str]
