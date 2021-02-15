"""
Mock routes
"""
from doctrans.routes import emit

route_config = dict(app="rest_api", name="Config", route="/api/config", variant=-1)

create_route = emit.create(**route_config)

route_config["primary_key"] = "dataset_name"

read_route = emit.read(**route_config)
destroy_route = emit.destroy(**route_config)

__all__ = ["create_route", "read_route", "destroy_route", "route_config"]
