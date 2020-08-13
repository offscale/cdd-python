""" eval.py for testing with `sync_properties` """

import doctrans.tests.mocks

_attr_within = frozenset(("mocks",))

get_modules = tuple(
    attr
    for attr in dir(doctrans.tests)
    if not attr.startswith("_") and attr in _attr_within
)
