"""
Helper to traverse the AST of the input file, extract the docstring out, parse and format to intended style, and emit
"""
from copy import deepcopy

from cdd import emit
from cdd.ast_utils import cmp_ast
from cdd.doctrans_utils import DocTrans, has_inline_types
from cdd.source_transformer import ast_parse


def doctrans(filename, docstring_format, inline_types):
    """
    Transform the docstrings found within provided filename to intended docstring_format

    :param filename: Python file to convert docstrings within. Edited in place.
    :type filename: ```str```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param inline_types: Whether the type should be inline or in docstring
    :type inline_types: ```bool```
    """
    with open(filename, "rt") as f:
        node = ast_parse(f.read(), skip_docstring_remit=True)
    orig_node = deepcopy(node)

    node = DocTrans(
        docstring_format=docstring_format,
        inline_types=inline_types,
        existing_inline_types=has_inline_types(node),
    ).visit(node)

    if not cmp_ast(node, orig_node):
        emit.file(node, filename, mode="wt", skip_black=True)


__all__ = ["doctrans"]
