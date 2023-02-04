"""
Mock routes
"""

import cdd.routes.emit.bottle
from cdd.shared.pure_utils import tab

route_config = {
    "app": "rest_api",
    "name": "Config",
    "route": "/api/config",
    "variant": -1,
}

create_route = cdd.routes.emit.bottle.create(**route_config)

route_config["primary_key"] = "dataset_name"

read_route = cdd.routes.emit.bottle.read(**route_config)
destroy_route = cdd.routes.emit.bottle.destroy(**route_config)

route_mock_prelude = (
    'rest_api = type("App", tuple(),\n'
    "{sep}{{ method: lambda h: lambda g=None: g \n"
    '{sep}  for method in ("get", "post", "put", "delete") }})\n'.format(sep=tab * 4)
)

route_prelude = (
    "from bottle import Bottle, request, response\n\n"
    "rest_api = Bottle(catchall=False, autojson=True)\n"
)

__all__ = [
    "create_route",
    "read_route",
    "destroy_route",
    "route_config",
    "route_prelude",
    "route_mock_prelude",
]
