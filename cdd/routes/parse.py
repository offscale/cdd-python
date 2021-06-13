"""
Parsers for routes
"""

import ast
from ast import FunctionDef
from importlib import import_module
from inspect import getsource
from types import FunctionType

from cdd.ast_utils import get_value
from cdd.docstring_parsers import parse_docstring
from cdd.openapi.parse import openapi
from cdd.pure_utils import PY_GTE_3_8

Literal = getattr(
    import_module("typing" if PY_GTE_3_8 else "typing_extensions"), "Literal"
)

methods = frozenset(("patch", "post", "put", "get", "delete", "trace"))


def bottle(function_def):
    """
    Parse bottle API

    :param function_def: Function definition of a bottle route, like `@api.get("/api") def root(): return "/"`
    :type function_def: ```Union[FunctionDef, FunctionType]```

    :returns: OpenAPI representation of the given route
    :rtype: ```dict```
    """
    if isinstance(function_def, FunctionType):
        # Dynamic function, i.e., this isn't source code; and is in your memory
        function_def = ast.parse(getsource(function_def)).body[0]

    assert isinstance(function_def, FunctionDef), "{typ} != FunctionDef".format(
        typ=type(function_def).__name__
    )
    app_decorator = next(
        filter(
            lambda call: call.func.attr in methods,
            function_def.decorator_list,
        )
    )
    route: str = get_value(app_decorator.args[0])
    name: str = app_decorator.func.value.id
    method: Literal["get", "post", "put", "patch", "delete"] = app_decorator.func.attr

    route_dict = {"route": route, "name": name, "method": method}
    doc_str = ast.get_docstring(function_def)
    if doc_str is not None:
        ir = parse_docstring(doc_str)
        yml_start_str, yml_end_str = "```yml", "```"
        yml_start = ir["doc"].find(yml_start_str)
        # if yml_start < 0:
        #    return route_dict
        openapi_str = ir["doc"][
            yml_start
            + len(yml_start_str) : ir["doc"].rfind(yml_end_str)
            - len(yml_end_str)
            + 2
        ]
        return openapi(openapi_str, route_dict, ir["doc"][:yml_start].rstrip())
    # return route_dict


__all__ = ["bottle"]
