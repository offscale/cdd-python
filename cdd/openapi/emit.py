"""
OpenAPI emitter function(s)
"""

from collections import deque

from cdd.openapi.emitter_utils import components_paths_from_name_model_route_id_crud
from cdd.tests.mocks.json_schema import server_error_schema


def openapi(name_model_route_id_cruds):
    """
    Emit OpenAPI dict

    :param name_model_route_id_cruds: Collection of (name, model, route, id, crud)
    :type name_model_route_id_cruds: ```Iterable[NameModelRouteIdCrud]```

    :returns: OpenAPI dict
    :rtype: ```dict```
    """
    paths, components = {}, {
        "requestBodies": {},
        "schemas": {
            "ServerError": {
                k: v for k, v in server_error_schema.items() if not k.startswith("$")
            }
        },
    }

    deque(
        map(
            lambda name_model_route_id_crud: components_paths_from_name_model_route_id_crud(
                components, paths, *name_model_route_id_crud
            ),
            name_model_route_id_cruds,
        ),
        maxlen=0,
    )

    return {
        "openapi": "3.0.0",
        "info": {"version": "0.0.1", "title": "REST API"},
        # "servers": [{"url": "https://example.io/v1"}],
        "components": components,
        "paths": paths,
    }


__all__ = ["openapi"]
