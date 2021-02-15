"""
Parsers for routes
"""
import ast
import json
from ast import FunctionDef
from inspect import getsource
from types import FunctionType

import yaml

from doctrans.ast_utils import get_value
from doctrans.docstring_parsers import parse_docstring


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
        filter(lambda call: call.func.attr == "post", function_def.decorator_list)
    )
    route = get_value(app_decorator.args[0])  # type: str
    name = app_decorator.func.value.id  # type: str
    method = (
        app_decorator.func.attr
    )  # type: Literal["get", "post", "put", "patch", "delete"]

    route_dict = {"route": route, "name": name, "method": method}
    doc_str = ast.get_docstring(function_def)
    if doc_str is not None:
        ir = parse_docstring(doc_str)
        yml_start_str, yml_end_str = "```yml", "```"
        yml_start = ir["doc"].find(yml_start_str)
        if yml_start < 0:
            return route_dict
        openapi_str = ir["doc"][
            yml_start
            + len(yml_start_str) : ir["doc"].rfind(yml_end_str)
            - len(yml_end_str)
            + 2
        ]
        return openapi(openapi_str, route_dict)
    return route_dict


def openapi(openapi_str, routes_dict):
    """
    OpenAPI parser

    :param openapi_str: The OpenAPI str
    :type openapi_str: ```str```

    :param routes_dict: Has keys ("route", "name", "method")
    :type routes_dict: ```dict```

    :returns: OpenAPI dictionary
    """
    entities, ticks, stack, eat = [], 0, [], False
    for ch in openapi_str:
        if ticks > 2:
            eat, ticks = True, 0
            if stack:
                entity = "".join(stack)
                if entity.strip():
                    entities.append(entity[1:])
                stack.clear()
            stack.append(ch)

        elif ch == "`":
            ticks += 1

        elif stack and eat:
            stack.append(ch)

    entity = "".join(stack)
    if entity.strip():
        entities[-1] = entity[1:]
    stack.clear()
    for entity in entities:
        openapi_str = openapi_str.replace(
            "$ref: ```{entity}```".format(entity=entity), entity
        )
    openapi_d = (json.loads if openapi_str.startswith("{") else yaml.safe_load)(
        openapi_str
    )
    return openapi_d


__all__ = ["bottle"]
