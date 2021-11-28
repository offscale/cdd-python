"""
Parser utils for routes
"""

from ast import Call, FunctionDef

from cdd.ast_utils import get_value
from cdd.pure_utils import rpartial
from cdd.routes.parse import methods


def get_route_meta(mod):
    """
    Get the (func_name, app_name, route_path, http_method)s

    :param mod: Parsed AST containing routes
    :type mod: ```Module```

    :return: Iterator of tuples of (func_name, app_name, route_path, http_method)
    :rtype: ```Iterator[Tuple[str, str, str, str]]```
    """
    return map(
        lambda func: (
            func.name,
            *next(
                map(
                    lambda call: (
                        call.func.value.id,
                        get_value(call.args[0]),
                        call.func.attr,
                    ),
                    filter(
                        lambda call: call.args and call.func.attr in methods,
                        filter(rpartial(isinstance, Call), func.decorator_list),
                    ),
                )
            ),
        ),
        filter(rpartial(isinstance, FunctionDef), mod.body),
    )


__all__ = ["get_route_meta"]
