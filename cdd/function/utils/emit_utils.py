"""
Utility functions for `cdd.emit.function_utils`
"""

import ast
from ast import Expr, FunctionDef, Return, arguments

from cdd.class_.utils.emit_utils import RewriteName
from cdd.shared.ast_utils import get_value, maybe_type_comment, set_arg, set_value
from cdd.shared.docstring_utils import emit_param_str
from cdd.shared.pure_utils import (
    code_quoted,
    indent_all_but_first,
    multiline,
    none_types,
)


def _make_call_meth(body, return_type, param_names, docstring_format, word_wrap):
    """
    Construct a `__call__` method from the provided `body`

    :param body: The body, probably from a FunctionDef.body
    :type body: ```List[AST]```

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
    body_len = len(body)
    if body_len and isinstance(body, dict):
        body = list(
            filter(
                None,
                (
                    None
                    if body.get("doc") in none_types
                    else Expr(
                        set_value(
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
                        )
                    ),
                    RewriteName(param_names).visit(
                        Return(
                            get_value(ast.parse(return_type.strip("`")).body[0]),
                            expr=None,
                        )
                    )
                    if code_quoted(body["default"])
                    else Return(set_value(body["default"]), expr=None),
                ),
            )
        )

    return (
        ast.fix_missing_locations(
            FunctionDef(
                args=arguments(
                    args=[set_arg("self")],
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
                name="__call__",
                returns=None,
                arguments_args=None,
                identifier_name=None,
                stmt=None,
                lineno=None,
                **maybe_type_comment
            )
        )
        if body
        else None
    )


__all__ = ["_make_call_meth"]
