"""
Utility functions for `cdd.parse.class`
"""

from inspect import getsource


def get_source(obj):
    """
    Call inspect.getsource and raise an error unless class definition could not be found

    :param obj: object to inspect
    :type obj: ```Any```

    :return: The source
    :rtype: ```Optional[str]```
    """
    try:
        return getsource(obj)
    except OSError as e:
        if e.args and e.args[0] in frozenset(
            (
                "could not find class definition",
                "source code not available",
                "could not get source code",
            )
        ):
            return None
        raise


__all__ = ["get_source"]  # type: list[str]
