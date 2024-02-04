"""
Utility functions for `cdd.emit.openapi`
"""

from collections import namedtuple
from typing import NamedTuple

from cdd.shared.pure_utils import PY_GTE_3_8, PY_GTE_3_9

if PY_GTE_3_8:
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

if PY_GTE_3_9:
    Type = type
else:
    from typing import Type

NameModelRouteType = NamedTuple(
    "NameModelRoute",
    [("name", str), ("model", str), ("route", str), ("id", str), ("crud", str)],
)
NameModelRouteIdCrud: Type[NameModelRouteType] = namedtuple(
    "NameModelRoute", ("name", "model", "route", "id", "crud")
)


def components_paths_from_name_model_route_id_crud(
    components, paths, name, model, route, _id, crud
):
    """
    Update `components` and `paths` from `(name, model, route, _id, crud)`

    :param components: OpenAPI components (updated by this function)
    :type components: ```dict```

    :param paths: OpenAPI paths (updated by this function)
    :type paths: ```dict```

    :param name: Name of the entity
    :type name: ```str```

    :param model: Schema of entity
    :type model: ```dict```

    :param route: API path
    :type route: ```str```

    :param _id: Primary key to access identity by id
    :type _id: ```str```

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
    """
    _request_body: bool = False
    if "C" in crud:
        paths[route] = {
            "post": {
                "summary": "A `{name}` object.".format(name=name),
                "requestBody": {
                    "required": True,
                    "$ref": "#/components/requestBodies/{name}Body".format(name=name),
                },
                "responses": {
                    "201": {
                        "description": "A `{name}` object.".format(name=name),
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/{name}".format(
                                        name=name
                                    )
                                }
                            }
                        },
                    },
                    "400": {
                        "description": "A `ServerError` object.",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ServerError"}
                            }
                        },
                    },
                },
            }
        }
        _request_body: bool = True
    if not frozenset(crud) - frozenset("CRUD"):
        _route: str = "{route}/{{{id}}}".format(route=route, id=_id)
        paths[_route] = {
            "parameters": [
                {
                    "name": _id,
                    "in": "path",
                    "description": "Primary key of target `{name}`".format(name=name),
                    "required": True,
                    "schema": {"type": "string"},
                }
            ]
        }
        if "R" in crud:
            paths[_route]["get"] = {
                "summary": "A `{name}` object.".format(name=name),
                "responses": {
                    "200": {
                        "description": "A `{name}` object.".format(name=name),
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/{name}".format(
                                        name=name
                                    )
                                }
                            }
                        },
                    },
                    "404": {
                        "description": "A `ServerError` object.",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ServerError"}
                            }
                        },
                    },
                },
            }

        # if "U" in crud:
        #     _request_body = True
        #     raise NotImplementedError(
        #         "UPDATE: https://github.com/sqlalchemy/sqlalchemy/discussions/5940"
        #     )

        if "D" in crud:
            paths[_route]["delete"] = {
                "summary": "Delete one `{name}`".format(name=name),
                "responses": {"204": {}},
            }
    components["schemas"][name] = {
        k: v for k, v in model.items() if not k.startswith("$")
    }
    if _request_body:
        components["requestBodies"]["{name}Body".format(name=name)] = {
            "description": "A `{name}` object.".format(name=name),
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/{name}".format(name=name)}
                }
            },
        }


# TODO: Finish writing these types
OpenAPI_info = TypedDict("OpenAPI_info", {"title": str, "version": str})
OpenAPI_requestBodies = dict
OpenAPI_components = TypedDict(
    "OpenAPI_components", {"requestBodies": OpenAPI_requestBodies, "schemas": dict}
)
JSON_ref = TypedDict("JSON_ref", {"$ref": str, "required": bool})
OpenAPI_paths = dict
OpenApiType = TypedDict(
    "OpenApiType",
    {
        "openapi": str,
        "info": OpenAPI_info,
        "components": OpenAPI_components,
        "paths": OpenAPI_paths,
    },
)

__all__ = [
    "NameModelRouteIdCrud",
    "OpenAPI_requestBodies",
    "OpenApiType",
    "components_paths_from_name_model_route_id_crud",
]  # type: list[str]
