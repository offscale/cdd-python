"""
OpenAPI parsers
"""

from json import loads
from typing import List, Optional

from yaml import safe_load

from cdd.compound.openapi.utils.parse_utils import extract_entities


def openapi(openapi_str, routes_dict, summary):
    """
    OpenAPI parser

    :param openapi_str: The OpenAPI str
    :type openapi_str: ```str```

    :param routes_dict: Has keys ("route", "name", "method")
    :type routes_dict: ```dict```

    :param summary: summary string (used as fallback)
    :type summary: ```str```

    :return: OpenAPI dictionary
    """
    entities: List[str] = extract_entities(openapi_str)

    non_error_entity: Optional[str] = None

    for entity in entities:
        openapi_str: str = openapi_str.replace(
            "$ref: ```{entity}```".format(entity=entity),
            "{{'$ref': '#/components/schemas/{entity}'}}".format(entity=entity),
        )
        if entity != "ServerError":
            non_error_entity: str = entity
    openapi_d: dict = (loads if openapi_str.startswith("{") else safe_load)(openapi_str)
    if non_error_entity is not None:
        openapi_d["summary"] = "{located} `{entity}` object.".format(
            located="A", entity=non_error_entity
        )
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


__all__ = ["openapi"]  # type: list[str]
