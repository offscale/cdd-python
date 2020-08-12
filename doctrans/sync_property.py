"""
Functionality to synchronise properties
"""
import ast
from ast import Module

from doctrans import emit
from doctrans.ast_utils import find_in_ast, annotate_ancestry, RewriteAtQuery
from doctrans.pure_utils import strip_split


def sync_property(
    input_eval, input_file, input_param, output_file, output_param,
):
    """
    Sync one property, inline to a file

    :param input_eval: Whether to evaluate the `param`, or just leave it
    :type input_eval: ```bool```

    :param input_file: Filename to find `param` from
    :type input_file: ```str```

    :param input_param: Location within file of property.
       Can be top level like `a` for `a=5` or with the `.` syntax as in `output_param`.
    :type input_param: ```str```

    :param output_file: Filename that will be edited in place, the property within this file (to update)
     is selected by `output_param`
    :type output_file: ```str```

    :param output_param: Parameter to update. E.g., `A.F` for `class A: F`, `f.g` for `def f(g): pass`
    :type output_param: ```str```
    """
    with open(input_file, "rt") as f:
        parsed_ast = ast.parse(f.read())

    assert isinstance(parsed_ast, Module)
    replacement_node = find_in_ast(list(strip_split(input_param, ".")), parsed_ast)
    assert replacement_node is not None

    with open(output_file, "rt") as f:
        parsed_ast = ast.parse(f.read())
    annotate_ancestry(parsed_ast)
    print("replacement_node:", replacement_node, ";")
    rewrite_at_query = RewriteAtQuery(
        search=list(strip_split(output_param, ".")),
        replacement_node=replacement_node,
        root=parsed_ast,
    )
    gen_ast = rewrite_at_query.visit(parsed_ast)
    assert rewrite_at_query.replaced is True
    emit.file(gen_ast, output_file, mode="wt")
