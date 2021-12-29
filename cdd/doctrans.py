"""
Helper to traverse the AST of the input file, extract the docstring out, parse and format to intended style, and emit
"""

from copy import deepcopy

from cdd import emit
from cdd.ast_utils import cmp_ast
from cdd.doctrans_utils import DocTrans, has_type_annotations
from cdd.source_transformer import ast_parse


def doctrans(filename, docstring_format, type_annotations, no_word_wrap):
    """
    Transform the docstrings found within provided filename to intended docstring_format

    :param filename: Python file to convert docstrings within. Edited in place.
    :type filename: ```str```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param type_annotations: True to have type annotations (3.6+), False to place in docstring
    :type type_annotations: ```bool```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```
    """
    with open(filename, "rt") as f:
        original_source = f.read()
    node = ast_parse(original_source, skip_docstring_remit=False)
    original_module = deepcopy(node)

    node = DocTrans(
        docstring_format=docstring_format,
        word_wrap=no_word_wrap is None,
        type_annotations=type_annotations,
        existing_type_annotations=has_type_annotations(node),
        whole_ast=original_module,
    ).visit(node)

    if not cmp_ast(node, original_module):
        # TODO: Replace `emit.file` with lineno aware solution so that comments aren't omitted
        emit.file(node, filename, mode="wt", skip_black=True)


__all__ = ["doctrans"]
