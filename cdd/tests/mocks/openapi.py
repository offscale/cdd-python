"""
OpenAPI mocks
"""

from cdd.tests.mocks.json_schema import config_schema, server_error_schema
from cdd.tests.mocks.routes import route_config

openapi_dict = {
    "openapi": "3.0.0",
    "info": {"title": "REST API", "version": "0.0.1"},
    "components": {
        "requestBodies": {
            "ConfigBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": "#/components/schemas/{name}".format(
                                name=route_config["name"]
                            )
                        }
                    }
                },
                "description": "A `{name}` object.".format(name=route_config["name"]),
                "required": True,
            }
        },
        "schemas": {
            name: {k: v for k, v in schema.items() if not k.startswith("$")}
            for name, schema in {
                route_config["name"]: config_schema,
                "ServerError": server_error_schema,
            }.items()
        },
    },
    "paths": {
        route_config["route"]: {
            "post": {
                "requestBody": {
                    "$ref": "#/components/requestBodies/{name}Body".format(
                        name=route_config["name"]
                    ),
                    "required": True,
                },
                "responses": {
                    "201": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/{name}".format(
                                        name=route_config["name"]
                                    )
                                }
                            }
                        },
                        "description": "A `{name}` object.".format(
                            name=route_config["name"]
                        ),
                    },
                    "400": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ServerError"}
                            }
                        },
                        "description": "A `ServerError` object.",
                    },
                },
                "summary": "A `{name}` object.".format(name=route_config["name"]),
            }
        },
        "{route_config[route]}/{{{route_config[primary_key]}}}".format(
            route_config=route_config
        ): {
            "delete": {
                "responses": {"204": {}},
                "summary": "Delete one `{name}`".format(name=route_config["name"]),
            },
            "get": {
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/{name}".format(
                                        name=route_config["name"]
                                    )
                                }
                            }
                        },
                        "description": "A `{name}` object.".format(
                            name=route_config["name"]
                        ),
                    },
                    "404": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ServerError"}
                            }
                        },
                        "description": "A `ServerError` object.",
                    },
                },
                "summary": "A `{name}` object.".format(name=route_config["name"]),
            },
            "parameters": [
                {
                    "description": "Primary key of target `{name}`".format(
                        name=route_config["name"]
                    ),
                    "in": "path",
                    "name": route_config["primary_key"],
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
    },
}

__all__ = ["openapi_dict"]
