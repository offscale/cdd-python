"""
Generate routes
"""
import ast
from ast import Attribute, Call, ClassDef, FunctionDef, Name
from itertools import chain
from operator import attrgetter, itemgetter
from os import path

from cdd import parse
from cdd.ast_utils import get_value
from cdd.pure_utils import filename_from_mod_or_filename, rpartial
from cdd.routes import emit as routes_emit
from cdd.routes.parse import methods
from cdd.source_transformer import to_code
from cdd.tests.mocks.routes import route_prelude


def gen_routes(app, model_path, model_name, crud, route):
    """
    Generate route(s)

    :param app: Variable name (Bottle App)
    :type app: ```str```

    :param model_path: The path/module-resolution whence the model is
    :type model_path: ```str```

    :param model_name: Name of the model to recover from the `model_path`
    :type model_name: ```str```

    :param crud: (C)reate (R)ead (U)pdate (D)elete, like "CRUD" for all or "CD" for "Create" and "Delete"
    :type crud: ```Union[Literal['C', 'R'], Literal['C', 'U'], Literal['C', 'D'], Literal['R', 'C'],
                         Literal['R', 'U'], Literal['R', 'D'], Literal['U', 'C'], Literal['U', 'R'],
                         Literal['U', 'D'], Literal['D', 'C'], Literal['D', 'R'], Literal['D', 'U'],
                         Literal['C', 'R', 'U'], Literal['C', 'R', 'D'], Literal['C', 'U', 'R'],
                         Literal['C', 'U', 'D'], Literal['C', 'D', 'R'], Literal['C', 'D', 'U'],
                         Literal['R', 'C', 'U'], Literal['R', 'C', 'D'], Literal['R', 'U', 'C'],
                         Literal['R', 'U', 'D'], Literal['R', 'D', 'C'], Literal['R', 'D', 'U'],
                         Literal['U', 'C', 'R'], Literal['U', 'C', 'D'], Literal['U', 'R', 'C'],
                         Literal['U', 'R', 'D'], Literal['U', 'D', 'C'], Literal['U', 'D', 'R'],
                         Literal['D', 'C', 'R'], Literal['D', 'C', 'U'], Literal['D', 'R', 'C'],
                         Literal['D', 'R', 'U'], Literal['D', 'U', 'C'], Literal['D', 'U', 'R']]```

    :param route: The path of the resource
    :type route: ```str```

    :returns: Iterator of functions representing relevant CRUD operations
    :rtype: ```Iterator[FunctionDef]```
    """
    model_path = filename_from_mod_or_filename(model_path)

    assert path.isfile(model_path)
    with open(model_path, "rt") as f:
        mod = ast.parse(f.read())

    sqlalchemy_node = next(
        filter(
            lambda node: isinstance(node, ClassDef)
            and node.name == model_name
            or isinstance(node, Name)
            and node.id == model_name,
            ast.walk(mod),
        ),
        None,
    )
    sqlalchemy_ir = parse.sqlalchemy(sqlalchemy_node)
    primary_key = next(
        map(
            itemgetter(0),
            filter(
                lambda param: param[1]["doc"].startswith("[PK]"),
                sqlalchemy_ir["params"].items(),
            ),
        ),
        next(iter(sqlalchemy_ir["params"].keys())),
    )
    _route_config = {"app": app, "name": model_name, "route": route, "variant": -1}
    routes = []
    if "C" in crud:
        routes.append(routes_emit.create(**_route_config))
    _route_config["primary_key"] = primary_key

    funcs = {"R": routes_emit.read, "U": None, "D": routes_emit.destroy}
    routes.extend(funcs[key](**_route_config) for key in funcs if key in crud)
    return (
        map(itemgetter(0), map(attrgetter("body"), map(ast.parse, routes))),
        primary_key,
    )


def upsert_routes(app, routes, routes_path, route, primary_key):
    """
    Upsert the `routes` to the `routes_path`, on merge use existing body and replace interface/prototype

    :param app: Variable name (Bottle App)
    :type app: ```str```

    :param routes: Iterator of functions representing relevant CRUD operations
    :type routes: ```Iterator[FunctionDef]```

    :param route: The path of the resource
    :type route: ```str```

    :param primary_key: The primary key or id to lookup on for the route
    :type primary_key: ```str```

    :param routes_path: The path/module-resolution whence the routes are / will be
    :type routes_path: ```str```
    """
    routes_path = filename_from_mod_or_filename(routes_path)

    if not path.isfile(routes_path):
        with open(routes_path, "wt") as f:
            f.write(
                "\n\n".join(
                    chain.from_iterable(
                        (
                            (
                                route_prelude.replace(
                                    "rest_api =", "{app} =".format(app=app)
                                ),
                            ),
                            map(to_code, routes),
                        )
                    )
                )
            )
        return

    with open(routes_path, "rt") as f:
        mod = ast.parse(f.read())

    def get_names(functions):
        """
        Derive a method_name -> FunctionDef dictionary

        :param functions: Routing functions
        :type functions: ```Iterator[FunctionDef]```

        :returns: Dict of `method_name` to `FunctionDef`
        :rtype: ```Dict[str, FunctionDef]```
        """
        return dict(
            map(
                lambda func: (
                    next(
                        map(
                            lambda call: call.func.attr,
                            filter(
                                lambda call: all(
                                    (
                                        isinstance(call.func, Attribute),
                                        call.func.attr in methods,
                                    )
                                ),
                                filter(rpartial(isinstance, Call), func.decorator_list),
                            ),
                        )
                    ),
                    func,
                ),
                functions,
            )
        )

    routes_required = get_names(routes)
    routes_existing = get_names(
        filter(
            lambda node: any(
                filter(
                    lambda call: all(
                        (
                            isinstance(call.func, Attribute),
                            call.func.attr in methods,
                            get_value(call.args[0])
                            == "{route}{rest}".format(
                                route=route,
                                rest=""
                                if call.func.attr == "post"
                                else "/:{primary_key}".format(primary_key=primary_key),
                            ),
                            call.func.value.id == app,
                        )
                    ),
                    filter(rpartial(isinstance, Call), node.decorator_list),
                )
            ),
            filter(rpartial(isinstance, FunctionDef), ast.walk(mod)),
        )
    )
    missing_routes = (
        routes_required.keys() & routes_existing.keys() ^ routes_required.keys()
    )

    if not missing_routes:
        return

    with open(routes_path, "a") as f:
        f.write(
            "\n\n".join(
                map(
                    to_code,
                    map(
                        routes_required.__getitem__,
                        sorted(
                            missing_routes,
                            key={
                                "post": 0,
                                "get": 1,
                                "update": 2,
                                "delete": 3,
                            }.__getitem__,
                        ),
                    ),
                )
            )
        )


__all__ = ["gen_routes", "upsert_routes"]
