"""
FastAPI route parser
"""

from cdd.ast_utils import get_value
from cdd.routes.parse.fastapi_utils import parse_fastapi_responses


def fastapi(fast_api_route):
    """
    Parse a single FastAPI route

    :param fast_api_route: A single FastAPI route
    :type fast_api_route: ```AsyncFunctionDef```

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
    method = fast_api_route.decorator_list[0].func.attr
    route = get_value(fast_api_route.decorator_list[0].args[0])
    return route, {
        method: {
            "responses": parse_fastapi_responses(
                next(
                    filter(
                        lambda keyword: keyword.arg == "responses",
                        fast_api_route.decorator_list[0].keywords,
                    )
                )
            )
        }
    }


__all__ = ["fastapi"]
