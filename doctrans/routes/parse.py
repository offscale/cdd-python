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
from doctrans.pure_utils import pp


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
            lambda call: call.func.attr
            in frozenset(("patch", "post", "put", "get", "delete", "trace")),
            function_def.decorator_list,
        )
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
        return openapi(openapi_str, route_dict, ir["doc"][:yml_start].rstrip())
    return route_dict


def openapi(openapi_str, routes_dict, summary):
    """
    OpenAPI parser

    :param openapi_str: The OpenAPI str
    :type openapi_str: ```str```

    :param routes_dict: Has keys ("route", "name", "method")
    :type routes_dict: ```dict```

    :param summary: summary string (used as fallback)
    :type summary: ```str```

    :returns: OpenAPI dictionary
    """
    entities, ticks, space, stack = [], 0, 0, []

    for idx, ch in enumerate(openapi_str):
        if ch.isspace():
            space += 1
        elif ticks > 2:
            eat, ticks, space = True, 0, 0
            if stack:
                entity = "".join(stack)
                if entity.strip() and entity != "````":
                    entities.append(entity[1:].strip("`"))
                stack.clear()
            stack.append(ch)

        elif ch == "`":
            ticks += 1
            if stack:
                stack.append(ch)

        elif stack and not space:
            stack.append(ch)

    non_error_entity = None

    if routes_dict["method"] == "get":
        print("openapi_str:", openapi_str, ";")
        pp({"entities": entities})

    for entity in entities:
        openapi_str = openapi_str.replace(
            "$ref: ```{entity}```".format(entity=entity),
            "{{'$ref': '#/components/schemas/{entity}'}}".format(entity=entity),
        )
        if entity != "ServerError":
            non_error_entity = entity
    openapi_d = (json.loads if openapi_str.startswith("{") else yaml.safe_load)(
        openapi_str
    )
    if non_error_entity is not None:
        openapi_d["summary"] = "A `{entity}` object.".format(entity=non_error_entity)
        if routes_dict["method"] in frozenset(("post", "patch")):
            openapi_d["requestBody"] = {
                "$ref": "#/components/requestBodies/{entity}Body".format(
                    entity=non_error_entity
                ),
                "required": True,
            }
    else:
        openapi_d["summary"] = summary
    if "responses" in openapi_d:
        openapi_d["responses"] = {k: v or {} for k, v in openapi_d["responses"].items()}
    return openapi_d


__all__ = ["bottle"]
