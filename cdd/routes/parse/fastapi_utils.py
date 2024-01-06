"""
FastAPI utils
"""

from functools import partial

from cdd.shared.ast_utils import Dict_to_dict, get_value


def model_handler(key, model_name, location, mime_type):
    """
    Create fully-qualified model name from unqualified name

    :param key: Key name
    :type key: ```str```

    :param model_name: Not fully-qualified model name or a `{"$ref": string}` dict
    :type model_name: ```str|dict```

    :param location: Full-qualified parent path
    :type location: ```str```

    :param mime_type: MIME type
    :type mime_type: ```str```

    :return: Tuple["content", JSON ref to model name, of form `{"$ref": string}`]
    :rtype: ```tuple[Union[str,"content"], dict]```
    """
    return (
        (key, model_name)
        if isinstance(model_name, dict)
        else (
            "content",
            {mime_type: {"schema": {"$ref": "{}{}".format(location, model_name)}}},
        )
    )


parse_handlers = {
    "model": partial(
        model_handler, location="#/components/schemas/", mime_type="application/json"
    )
}


def parse_fastapi_responses(responses):
    """
    Parse FastAPI "responses" key

    :param responses: `responses` keyword value from FastAPI decorator on route
    :type responses: ```Dict```

    :return: Transformed FastAPI "responses"
    :rtype: ```dict```
    """

    return {
        key: dict(
            (
                (
                    lambda _v: (
                        (parse_handlers[k](k, _v)) if k in parse_handlers else (k, _v)
                    )
                )(get_value(v))
            )
            for k, v in Dict_to_dict(val).items()
        )
        for key, val in Dict_to_dict(responses.value).items()
    }


__all__ = ["parse_fastapi_responses"]  # type: list[str]
