"""
Source transformer module. Uses astor on Python < 3.9
"""

from ast import AsyncFunctionDef, ClassDef, FunctionDef, Module, get_docstring, parse
from importlib import import_module
from sys import version_info
from typing import Optional

import cdd.shared.ast_utils
from cdd.shared.pure_utils import reindent, tab

unparse = (
    getattr(import_module("astor"), "to_source")
    if version_info[:2] < (3, 9)
    else getattr(import_module("ast"), "unparse")
)


def to_code(node):
    """
    Convert the AST input to Python source string

    :param node: AST node
    :type node: ```AST```

    :return: Python source
    :rtype: ```str```
    """
    # ^Not `to_code = getattr…` so docstring can be included^
    return unparse(node)


def ast_parse(
    source,
    filename="<unknown>",
    mode="exec",
    skip_annotate=False,
    skip_docstring_remit=False,
):
    """
    Convert the AST input to Python source string

    :param source: Python source
    :type  source: ```str```

    :param filename: Filename being parsed
    :type filename: ```str```

    :param mode: 'exec' to compile a module, 'single' to compile a single (interactive) statement,
      or 'eval' to compile an expression.
    :type mode: ```Literal['exec', 'single', 'eval']```

    :param skip_annotate: Don't run `annotate_ancestry`
    :type skip_annotate: ```bool```

    :param skip_docstring_remit: Don't parse & emit the docstring as a replacement for current docstring
    :type skip_docstring_remit: ```bool```

    :return: AST node
    :rtype: ```AST```
    """
    parsed_ast = parse(source, filename=filename, mode=mode)
    if not skip_annotate:
        cdd.shared.ast_utils.annotate_ancestry(parsed_ast, filename=filename)
        setattr(parsed_ast, "__file__", filename)
    if not skip_docstring_remit and isinstance(
        parsed_ast, (Module, ClassDef, FunctionDef, AsyncFunctionDef)
    ):
        docstring: Optional[str] = get_docstring(parsed_ast, clean=True)
        if docstring is None:
            return parsed_ast

        # Reindent docstring
        parsed_ast.body[0].value.value = "\n{tab}{docstring}\n{tab}".format(
            tab=tab, docstring=reindent(docstring)
        )
    return parsed_ast


__all__ = ["ast_parse", "to_code"]  # type: list[str]
