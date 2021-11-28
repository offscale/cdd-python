"""
All encompassing solution to generating the OpenAPI schema
"""

import ast
from ast import Call, ClassDef, FunctionDef, Module
from itertools import chain, groupby
from operator import itemgetter

from cdd import emit, parse
from cdd.ast_utils import get_value
from cdd.parser_utils import infer
from cdd.pure_utils import rpartial, update_d
from cdd.routes import parse as routes_parse
from cdd.routes.parser_utils import get_route_meta
from cdd.tests.mocks.json_schema import server_error_schema


def openapi_bulk(app_name, model_paths, routes_paths):
    """
    Generate OpenAPI from models, routes on app

    :param app_name: Variable name (Bottle App)
    :type app_name: ```str```

    :param model_paths: The path/module-resolution(s) whence the model(s) can be found
    :type model_paths: ```List[str]```

    :param routes_paths: The path/module-resolution(s) whence the route(s) can be found
    :type routes_paths: ```List[str]```
    """
    request_bodies = {}

    def parse_model(filename):
        """
        :param filename: The filename to open and parse AST out of
        :type filename: ```str```

        :return: Iterable of tuples of the found kind
        :rtype: ```Iterable[Tuple[AST, ...], ...]```
        """
        with open(filename, "rb") as f:
            parsed_ast = ast.parse(f.read())

        return filter(
            lambda node: (infer(node) or "").startswith("sqlalchemy"),
            filter(rpartial(isinstance, (Call, ClassDef)), ast.walk(parsed_ast)),
        )

    def parse_route(filename):
        """
        :param filename: The filename to open and parse AST out of
        :type filename: ```str```

        :return: Iterable of tuples of the found kind
        :rtype: ```Iterable[Tuple[AST, ...], ...]```
        """
        with open(filename, "rb") as f:
            parsed_ast = ast.parse(f.read())

        return filter(
            lambda node: next(
                get_route_meta(Module(body=[node], type_ignores=[], stmt=None))
            )[1]
            == app_name,
            filter(rpartial(isinstance, FunctionDef), parsed_ast.body),
        )

    def construct_parameters_and_request_bodies(route, path_dict):
        """
        Construct `parameters` and `requestBodies`

        :param route: Route path, like "/api/foo"
        :type route: ```str```

        :param path_dict: OpenAPI paths key
        :type path_dict: ```dict```

        :return: (route, path_dict) with `"parameters"` key potentially set
        :rtype: ```Tuple[str, dict]```
        """
        if ":" in route:
            path_dict["parameters"] = []
            object_name = path_dict.get(
                "get", path_dict.get("delete", {"summary": "`Object`"})
            )["summary"]
            fst = object_name.find("`")
            object_name = (
                object_name[fst + 1 : object_name.find("`", fst + 1)] or "Object"
            )

            route = "/".join(
                map(
                    lambda r: (
                        lambda pk: (
                            path_dict["parameters"].append(
                                {
                                    "description": "Primary key of target `{}`".format(
                                        object_name
                                    ),
                                    "in": "path",
                                    "name": pk,
                                    "required": True,
                                    "schema": {"type": "string"},
                                }
                            )
                            or "{{{}}}".format(pk)
                        )
                    )(r[1:])
                    if r.startswith(":")
                    else r,
                    route.split("/"),
                )
            )

        request_bodies.update(
            map(
                lambda body_name: (
                    body_name,
                    (
                        lambda key: {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/{key}".format(
                                            key=key
                                        )
                                    }
                                }
                            },
                            "description": "A `{key}` object.".format(key=key),
                            "required": True,
                        }
                    )(body_name.rpartition("Body")[0]),
                ),
                map(
                    lambda ref: ref.rpartition("/")[2],
                    map(
                        itemgetter("$ref"),
                        filter(
                            None,
                            map(
                                rpartial(dict.get, "requestBody"),
                                filter(rpartial(isinstance, dict), path_dict.values()),
                            ),
                        ),
                    ),
                ),
            )
        )

        return route, path_dict

    return {
        "openapi": "3.0.0",
        "info": {"version": "0.0.1", "title": "REST API"},
        # "servers": [{"url": "https://example.io/v1"}],
        "components": {
            "requestBodies": request_bodies,
            "schemas": {
                key: {k: v for k, v in val.items() if not k.startswith("$")}
                for key, val in dict(
                    map(
                        lambda table: (
                            table["name"].replace("_tbl", "", 1).title(),
                            emit.json_schema(table),
                        ),
                        map(
                            lambda node: parse.sqlalchemy_table(node)
                            if isinstance(node, Call)
                            else parse.sqlalchemy(node),
                            chain.from_iterable(map(parse_model, model_paths)),
                        ),
                    ),
                    ServerError=server_error_schema,
                ).items()
            },
        },
        "paths": dict(
            map(
                lambda k_v: construct_parameters_and_request_bodies(
                    k_v[0], update_d(*map(itemgetter(1), k_v[1]))
                ),
                groupby(
                    map(
                        lambda route: (
                            get_value(route.decorator_list[0].args[0]),
                            {
                                route.decorator_list[0].func.attr: routes_parse.bottle(
                                    route
                                )
                            },
                        ),
                        chain.from_iterable(map(parse_route, routes_paths)),
                    ),
                    key=itemgetter(0),
                ),
            )
        ),
    }


__all__ = ["openapi_bulk"]
