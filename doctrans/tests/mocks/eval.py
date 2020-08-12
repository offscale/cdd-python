""" eval.py for testing with `sync_property` """

import doctrans.tests.mocks

get_modules = tuple(
    attr for attr in dir(doctrans.tests) if not attr.startswith("_") and attr == "mocks"
)
