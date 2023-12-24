"""
Parsers for routes
"""

import ast
from ast import FunctionDef
from importlib import import_module
from inspect import getsource
from types import FunctionType
from typing import FrozenSet, Optional, cast

import cdd.compound.openapi.parse
from cdd.shared.ast_utils import get_value
from cdd.shared.docstring_parsers import parse_docstring
from cdd.shared.pure_utils import PY_GTE_3_8
from cdd.shared.types import IntermediateRepr

Literal = getattr(
    import_module("typing" if PY_GTE_3_8 else "typing_extensions"), "Literal"
)

methods_literal_type = Literal["patch", "post", "put", "get", "delete", "trace"]
methods: FrozenSet[methods_literal_type] = frozenset(
    ("patch", "post", "put", "get", "delete", "trace")
)


def bottle(function_def):
    """
    Parse bottle API

    :param function_def: Function definition of a bottle route, like `@api.get("/api") def root(): return "/"`
    :type function_def: ```Union[FunctionDef, FunctionType]```

    :return: OpenAPI representation of the given route
    :rtype: ```dict```
    """
    if isinstance(function_def, FunctionType):
        # Dynamic function, i.e., this isn't source code; and is in your memory
        function_def: FunctionDef = cast(
            FunctionDef, ast.parse(getsource(function_def)).body[0]
        )

    assert isinstance(
        function_def, FunctionDef
    ), "Expected `FunctionDef` got `{type_name}`".format(
        type_name=type(function_def).__name__
    )
    app_decorator = next(
        filter(
            lambda call: call.func.attr in methods,
            function_def.decorator_list,
        )
    )
    route: str = get_value(app_decorator.args[0])
    name: str = app_decorator.func.value.id
    method: methods_literal_type = app_decorator.func.attr

    route_dict = {"route": route, "name": name, "method": method}
    doc_str: Optional[str] = ast.get_docstring(function_def, clean=True)
    if doc_str is not None:
        ir: IntermediateRepr = parse_docstring(doc_str)
        yml_start_str, yml_end_str = "```yml", "```"
        yml_start: int = ir["doc"].find(yml_start_str)
        # if yml_start < 0:
        #    return route_dict
        openapi_str: str = ir["doc"][
            yml_start
            + len(yml_start_str) : ir["doc"].rfind(yml_end_str)
            - len(yml_end_str)
            + 2
        ]
        return cdd.compound.openapi.parse.openapi(
            openapi_str, route_dict, ir["doc"][:yml_start].rstrip()
        )
    # return route_dict


__all__ = ["bottle", "methods"]  # type: list[str]
