"""
Parser utils for routes
"""
from ast import FunctionDef

from cdd.ast_utils import get_value
from cdd.pure_utils import rpartial


def get_route_meta(mod):
    """
    Get the (func_name, app_name, route_path, http_method)s

    :param mod: Parsed AST containing routes
    :type mod: ```Module```

    :returns: Iterator of tuples of (func_name, app_name, route_path, http_method)
    :rtype: ```Iterator[Tuple[str, str, str, str]]```
    """
    return map(
        lambda func: (
            func.name,
            func.decorator_list[0].func.value.id,
            get_value(func.decorator_list[0].args[0]),
            func.decorator_list[0].func.attr,
        ),
        filter(rpartial(isinstance, FunctionDef), mod.body),
    )


__all__ = ["get_route_meta"]
