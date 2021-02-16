""" eval.py for testing with `sync_properties` """

import cdd.tests.mocks

_attr_within = frozenset(("mocks",))

get_modules = tuple(
    attr for attr in dir(cdd.tests) if not attr.startswith("_") and attr in _attr_within
)

__all__ = ["get_modules"]
