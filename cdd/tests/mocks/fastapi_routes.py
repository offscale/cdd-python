"""
FastAPI route mocks
"""

from ast import (
    AsyncFunctionDef,
    Attribute,
    Call,
    Compare,
    Dict,
    If,
    Is,
    Load,
    Name,
    Return,
    arguments,
    keyword,
)

from cdd.shared.ast_utils import maybe_type_comment, set_arg, set_value

fastapi_post_create_config_str = """
@app.post(
    "/api/config",
    response_model=Config,
    responses={
        201: {
            "model": Config,
            "description": "A `Config` object.",
        },
        404: {
            "model": ServerError,
            "description": "A `ServerError` object.",
        },
    },
)
async def create_config(config: Config):
    if config is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "NotFound",
                "error_code": "0004",
                "error_description": "Config not found",
            },
        )
    return JSONResponse(status_code=201, content=config)
"""

fastapi_post_create_config_async_func = AsyncFunctionDef(
    name="create_config",
    args=arguments(
        posonlyargs=[],
        args=[set_arg(arg="config", annotation=Name(id="Config", ctx=Load()))],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
        vararg=None,
        kwarg=None,
        arg=None,
    ),
    body=[
        If(
            test=Compare(
                left=Name(id="config", ctx=Load()),
                ops=[Is()],
                comparators=[set_value(None)],
            ),
            body=[
                Return(
                    value=Call(
                        func=Name(id="JSONResponse", ctx=Load()),
                        args=[],
                        keywords=[
                            keyword(
                                arg="status_code", value=set_value(404), identifier=None
                            ),
                            keyword(
                                arg="content",
                                value=Dict(
                                    keys=list(
                                        map(
                                            set_value,
                                            (
                                                "error",
                                                "error_code",
                                                "error_description",
                                            ),
                                        )
                                    ),
                                    values=list(
                                        map(
                                            set_value,
                                            ("NotFound", "0004", "Config not found"),
                                        )
                                    ),
                                ),
                                identifier=None,
                            ),
                        ],
                    )
                )
            ],
            orelse=[],
        ),
        Return(
            value=Call(
                func=Name(id="JSONResponse", ctx=Load()),
                args=[],
                keywords=[
                    keyword(arg="status_code", value=set_value(201), identifier=None),
                    keyword(
                        arg="content",
                        value=Name(id="config", ctx=Load()),
                        identifier=None,
                    ),
                ],
            )
        ),
    ],
    decorator_list=[
        Call(
            func=Attribute(value=Name(id="app", ctx=Load()), attr="post", ctx=Load()),
            args=[set_value("/api/config")],
            keywords=[
                keyword(
                    arg="response_model",
                    value=Name(id="Config", ctx=Load()),
                    identifier=None,
                ),
                keyword(
                    arg="responses",
                    value=Dict(
                        keys=list(map(set_value, (201, 404))),
                        values=[
                            Dict(
                                keys=[set_value("model"), set_value("description")],
                                values=[
                                    Name(id="Config", ctx=Load()),
                                    set_value("A `Config` object."),
                                ],
                            ),
                            Dict(
                                keys=[set_value("model"), set_value("description")],
                                values=[
                                    Name(id="ServerError", ctx=Load()),
                                    set_value("A `ServerError` object."),
                                ],
                            ),
                        ],
                    ),
                    identifier=None,
                ),
            ],
        )
    ],
    lineno=None,
    returns=None,
    **maybe_type_comment
)


__all__ = ["fastapi_post_create_config_str", "fastapi_post_create_config_async_func"]
