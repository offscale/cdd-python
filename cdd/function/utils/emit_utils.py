"""
Utility functions for `cdd.emit.function_utils`
"""

import ast
from ast import Expr, FunctionDef, Return, arguments

import cdd.shared.ast_utils
from cdd.class_.utils.emit_utils import RewriteName
from cdd.shared.docstring_utils import emit_param_str
from cdd.shared.pure_utils import (
    code_quoted,
    indent_all_but_first,
    multiline,
    none_types,
)


def make_call_meth(body, return_type, param_names, docstring_format, word_wrap):
    """
    Construct a `__call__` method from the provided `body`

    :param body: The body, probably from a `FunctionDef.body`
    :type body: ```list[AST]```

    :param return_type: The return type of the parent symbol (probably class). Used to fill in `__call__` return.
    :type return_type: ```Optional[str]```

    :param param_names: Container of AST `id`s to match for rename
    :type param_names: ```Optional[Iterator[str]]```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :return: Internal function for `__call__`
    :rtype: ```FunctionDef```
    """
    body_len: int = len(body)
    if body_len and isinstance(body, dict):
        body = list(
            filter(
                None,
                (
                    (
                        None
                        if body.get("doc") in none_types
                        else Expr(
                            cdd.shared.ast_utils.set_value(
                                emit_param_str(
                                    (
                                        "return_type",
                                        {
                                            "doc": multiline(
                                                indent_all_but_first(body["doc"])
                                            )
                                        },
                                    ),
                                    style=docstring_format,
                                    word_wrap=word_wrap,
                                    purpose="function",
                                )
                            ),
                            lineno=None,
                            col_offset=None,
                        )
                    ),
                    (
                        RewriteName(param_names).visit(
                            Return(
                                cdd.shared.ast_utils.get_value(
                                    ast.parse(return_type.strip("`")).body[0]
                                ),
                                expr=None,
                            )
                        )
                        if code_quoted(body["default"])
                        else Return(
                            cdd.shared.ast_utils.set_value(body["default"]), expr=None
                        )
                    ),
                ),
            )
        )

    return (
        ast.fix_missing_locations(
            FunctionDef(
                args=arguments(
                    args=[cdd.shared.ast_utils.set_arg("self")],
                    defaults=[],
                    kw_defaults=[],
                    kwarg=None,
                    kwonlyargs=[],
                    posonlyargs=[],
                    vararg=None,
                    arg=None,
                ),
                body=body,
                decorator_list=[],
                type_params=[],
                name="__call__",
                returns=None,
                arguments_args=None,
                identifier_name=None,
                stmt=None,
                lineno=None,
                **cdd.shared.ast_utils.maybe_type_comment
            )
        )
        if body
        else None
    )


__all__ = ["make_call_meth"]  # type: list[str]
