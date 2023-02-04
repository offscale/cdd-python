"""
FastAPI route parser
"""

from cdd.routes.parse.fastapi_utils import parse_fastapi_responses
from cdd.shared.ast_utils import get_value


def fastapi(fastapi_route):
    """
    Parse a single FastAPI route

    :param fastapi_route: A single FastAPI route
    :type fastapi_route: ```AsyncFunctionDef```

    :return: Pair of (str, dict) consisting of API path to a dictionary of form
        {  Literal["post","get","put","patch"]: {
             "requestBody": { "$ref": str, "required": boolean },
             "responses": { number: { "content": {string: {"schema": {"$ref": string},
                                      "description": string} } } },
             "summary": string
           }
        }
    :rtype: ```Tuple[str, dict]```
    """
    method = fastapi_route.decorator_list[0].func.attr
    route = get_value(fastapi_route.decorator_list[0].args[0])
    return route, {
        method: {
            "responses": parse_fastapi_responses(
                next(
                    filter(
                        lambda keyword: keyword.arg == "responses",
                        fastapi_route.decorator_list[0].keywords,
                    )
                )
            )
        }
    }


__all__ = ["fastapi"]
