"""
Helper to traverse the AST of the input file, extract the docstring out, parse and format to intended style, and emit
"""

from ast import Module, fix_missing_locations
from copy import deepcopy
from operator import attrgetter
from typing import List, NamedTuple

from cdd.compound.doctrans_utils import DocTrans, doctransify_cst, has_type_annotations
from cdd.shared.ast_utils import cmp_ast
from cdd.shared.cst import cst_parse
from cdd.shared.source_transformer import ast_parse


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
        original_source: str = f.read()
    node: Module = ast_parse(original_source, skip_docstring_remit=False)
    original_module: Module = deepcopy(node)

    node: Module = fix_missing_locations(
        DocTrans(
            docstring_format=docstring_format,
            word_wrap=no_word_wrap is None,
            type_annotations=type_annotations,
            existing_type_annotations=has_type_annotations(node),
            whole_ast=original_module,
        ).visit(node)
    )

    if not cmp_ast(node, original_module):
        cst_list: List[NamedTuple] = list(cst_parse(original_source))

        # Carefully replace only docstrings, function return annotations, assignment and annotation assignments.
        # Maintaining all other existing whitespace, comments, &etc.
        doctransify_cst(cst_list, node)

        with open(filename, "wt") as f:
            f.write("".join(map(attrgetter("value"), cst_list)))


__all__ = ["doctrans"]  # type: list[str]
