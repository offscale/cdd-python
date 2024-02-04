"""
All encompassing solution to generating the OpenAPI schema
"""

import ast
from ast import AnnAssign, Assign, Call, ClassDef, FunctionDef, Module
from itertools import chain, groupby
from operator import itemgetter

import cdd.argparse_function.parse
import cdd.class_.parse
import cdd.docstring.parse
import cdd.function.parse
import cdd.json_schema.emit
import cdd.routes.parse.bottle
import cdd.sqlalchemy.parse
from cdd.compound.openapi.utils.emit_openapi_utils import OpenAPI_requestBodies
from cdd.routes.parse.bottle_utils import get_route_meta
from cdd.shared.ast_utils import get_value
from cdd.shared.parse.utils.parser_utils import infer
from cdd.shared.pure_utils import rpartial, update_d
from cdd.tests.mocks.json_schema import server_error_schema


def openapi_bulk(app_name, model_paths, routes_paths):
    """
    Generate OpenAPI from models, routes on app

    :param app_name: Variable name (Bottle App)
    :type app_name: ```str```

    :param model_paths: The path/module-resolution(s) whence the model(s) can be found
    :type model_paths: ```list[str]```

    :param routes_paths: The path/module-resolution(s) whence the route(s) can be found
    :type routes_paths: ```list[str]```

    :return: OpenAPI dictionary
    :rtype: ```dict```
    """
    request_bodies: OpenAPI_requestBodies = {}

    def parse_model(filename):
        """
        :param filename: The filename to open and parse AST out of
        :type filename: ```str```

        :return: Iterable of tuples of the found kind
        :rtype: ```Iterable[tuple[AST, ...], ...]```
        """
        with open(filename, "rb") as f:
            parsed_ast: Module = ast.parse(f.read())

        return filter(
            lambda node: (infer(node) or "").startswith("sqlalchemy"),
            filter(rpartial(isinstance, (Call, ClassDef)), ast.walk(parsed_ast)),
        )

    def parse_route(filename):
        """
        :param filename: The filename to open and parse AST out of
        :type filename: ```str```

        :return: Iterable of tuples of the found kind
        :rtype: ```Iterable[tuple[AST, ...], ...]```
        """
        with open(filename, "rb") as f:
            parsed_ast: Module = ast.parse(f.read())

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
        :rtype: ```tuple[str, dict]```
        """
        if ":" in route:
            path_dict["parameters"] = []
            object_name: str = path_dict.get(
                "get", path_dict.get("delete", {"summary": "`Object`"})
            )["summary"]
            fst: int = object_name.find("`")
            object_name: str = (
                object_name[fst + 1 : object_name.find("`", fst + 1)] or "Object"
            )

            route: str = "/".join(
                map(
                    lambda r: (
                        (
                            lambda pk: (
                                path_dict["parameters"].append(
                                    {
                                        "description": (
                                            "Primary key of target `{}`".format(
                                                object_name
                                            )
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
                        else r
                    ),
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
                            cdd.json_schema.emit.json_schema(table),
                        ),
                        map(
                            lambda node: (
                                cdd.sqlalchemy.parse.sqlalchemy_table(node)
                                if isinstance(node, (AnnAssign, Assign, Call))
                                else cdd.sqlalchemy.parse.sqlalchemy(node)
                            ),
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
                                route.decorator_list[
                                    0
                                ].func.attr: cdd.routes.parse.bottle.bottle(route)
                            },
                        ),
                        chain.from_iterable(map(parse_route, routes_paths)),
                    ),
                    key=itemgetter(0),
                ),
            )
        ),
    }


__all__ = ["openapi_bulk"]  # type: list[str]
