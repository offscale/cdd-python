"""
Functionality to synchronise properties
"""

import ast

from doctrans import emit
from doctrans.ast_utils import (
    find_in_ast,
    annotate_ancestry,
    RewriteAtQuery,
    it2literal,
)
from doctrans.pure_utils import strip_split
from doctrans.source_transformer import to_code


def sync_properties(
    input_eval,
    input_file,
    input_params,
    output_file,
    output_params,
    output_param_wrap=None,
):
    """
    Sync one property, inline to a file

    :param input_eval: Whether to evaluate the `param`, or just leave it
    :type input_eval: ```bool```

    :param input_file: Filename to find `param` from
    :type input_file: ```str```

    :param input_params: Locations within file of properties.
       Can be top level like `['a']` for `a=5` or with the `.` syntax as in `output_params`.
    :type input_params: ```List[str]```

    :param output_file: Filename that will be edited in place, the property within this file (to update)
     is selected by `output_param`
    :type output_file: ```str```

    :param output_params: Parameters to update. E.g., `['A.F']` for `class A: F = None`, `['f.g']` for `def f(g): pass`
    :type output_params: ```List[str]```

    :param output_param_wrap: Wrap all output params with this. E.g., `Optional[Union[{output_param}, str]]`
    :param output_param_wrap: ```Optional[str]```
    """
    with open(input_file, "rt") as f:
        input_ast = ast.parse(f.read())

    with open(output_file, "rt") as f:
        output_ast = ast.parse(f.read())
    annotate_ancestry(output_ast)

    assert len(input_params) == len(output_params)
    gen_ast = None
    for (input_param, output_param) in zip(input_params, output_params):
        gen_ast = sync_property(
            input_eval,
            input_param,
            input_ast,
            input_file,
            output_param,
            output_param_wrap,
            output_ast,
        )

    emit.file(gen_ast, output_file, mode="wt")


def sync_property(
    input_eval,
    input_param,
    input_ast,
    input_file,
    output_param,
    output_param_wrap,
    output_ast,
):
    """
    Sync a single property

    :param input_eval: Whether to evaluate the `param`, or just leave it
    :type input_eval: ```bool```

    :param input_param: Location within file of property.
       Can be top level like `'a'` for `a=5` or with the `.` syntax as in `output_params`.
    :type input_param: ```List[str]```

    :param input_ast: AST of the input file
    :type input_ast: ```AST```

    :param input_file: Filename of the input (used in `eval`)
    :type input_file: ```str```

    :param output_param: Parameters to update. E.g., `'A.F'` for `class A: F = None`, `'f.g'` for `def f(g): pass`
    :type output_param: ```str```

    :param output_param_wrap: Wrap all output params with this. E.g., `Optional[Union[{output_param}, str]]`
    :param output_param_wrap: ```Optional[str]```

    :param output_ast: AST of the input file
    :type output_ast: ```AST```

    :returns: New AST derived from `output_ast`
    :rtype: ```AST```
    """
    if input_eval:
        if input_param.count(".") != 0:
            raise NotImplementedError("Anything not on the top-level of the module")

        local = {}
        output = eval(compile(input_ast, filename=input_file, mode="exec"), local)
        assert output is None
        replacement_node = it2literal(local[input_param])
    else:
        annotate_ancestry(input_ast)
        assert isinstance(input_ast, ast.Module)
        replacement_node = find_in_ast(list(strip_split(input_param, ".")), input_ast)

    assert replacement_node is not None
    if output_param_wrap is not None:
        replacement_node.annotation = (
            ast.parse(
                output_param_wrap.format(
                    output_param=to_code(replacement_node.annotation)
                )
            )
            .body[0]
            .value
        )

    rewrite_at_query = RewriteAtQuery(
        search=list(strip_split(output_param, ".")),
        replacement_node=replacement_node,
        root=output_ast,
    )
    gen_ast = rewrite_at_query.visit(output_ast)
    assert rewrite_at_query.replaced is True
    return gen_ast
