"""
Functionality to synchronise properties
"""

import ast
from os import path

from cdd import emit
from cdd.ast_utils import RewriteAtQuery, annotate_ancestry, find_in_ast, it2literal
from cdd.pure_utils import strip_split
from cdd.source_transformer import ast_parse, to_code


def sync_properties(
    input_eval,
    input_filename,
    input_params,
    output_filename,
    output_params,
    output_param_wrap=None,
):
    """
    Sync one property, inline to a file

    :param input_eval: Whether to evaluate the `param`, or just leave it
    :type input_eval: ```bool```

    :param input_filename: Filename to find `param` from
    :type input_filename: ```str```

    :param input_params: Locations within file of properties.
       Can be top level like `['a']` for `a=5` or with the `.` syntax as in `output_params`.
    :type input_params: ```List[str]```

    :param output_filename: Filename that will be edited in place, the property within this file (to update)
     is selected by `output_param`
    :type output_filename: ```str```

    :param output_params: Parameters to update. E.g., `['A.F']` for `class A: F = None`, `['f.g']` for `def f(g): pass`
    :type output_params: ```List[str]```

    :param output_param_wrap: Wrap all input_str params with this. E.g., `Optional[Union[{output_param}, str]]`
    :param output_param_wrap: ```Optional[str]```
    """
    with open(path.realpath(path.expanduser(input_filename)), "rt") as f:
        input_ast = ast_parse(f.read(), filename=input_filename)

    with open(path.realpath(path.expanduser(output_filename)), "rt") as f:
        output_ast = ast_parse(f.read(), filename=output_filename)

    assert len(input_params) == len(output_params)
    for (input_param, output_param) in zip(input_params, output_params):
        output_ast = sync_property(
            input_eval,
            input_param,
            input_ast,
            input_filename,
            output_param,
            output_param_wrap,
            output_ast,
        )

    emit.file(output_ast, output_filename, mode="wt", skip_black=False)


def sync_property(
    input_eval,
    input_param,
    input_ast,
    input_filename,
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

    :param input_filename: Filename of the input (used in `eval`)
    :type input_filename: ```str```

    :param output_param: Parameters to update. E.g., `'A.F'` for `class A: F = None`, `'f.g'` for `def f(g): pass`
    :type output_param: ```str```

    :param output_param_wrap: Wrap all input_str params with this. E.g., `Optional[Union[{output_param}, str]]`
    :param output_param_wrap: ```Optional[str]```

    :param output_ast: AST of the input file
    :type output_ast: ```AST```

    :returns: New AST derived from `output_ast`
    :rtype: ```AST```
    """
    search = list(strip_split(output_param, "."))
    if input_eval:
        if input_param.count(".") != 0:
            raise NotImplementedError("Anything not on the top-level of the module")

        local = {}
        output = eval(compile(input_ast, filename=input_filename, mode="exec"), local)
        assert output is None
        replacement_node = ast.AnnAssign(
            annotation=it2literal(local[input_param]),
            simple=1,
            target=ast.Name(
                # input_param
                search[-1],
                ast.Store(),
            ),
            value=None,
            expr=None,
            expr_annotation=None,
            expr_target=None,
        )
    else:
        assert isinstance(input_ast, ast.Module)
        annotate_ancestry(input_ast)
        replacement_node = find_in_ast(list(strip_split(input_param, ".")), input_ast)

    assert replacement_node is not None
    if output_param_wrap is not None:
        if hasattr(replacement_node, "annotation"):
            if replacement_node.annotation is not None:
                replacement_node.annotation = (
                    ast.parse(
                        output_param_wrap.format(
                            output_param=to_code(replacement_node.annotation)
                        )
                    )
                    .body[0]
                    .value
                )
        else:
            raise NotImplementedError(type(replacement_node).__name__)

    rewrite_at_query = RewriteAtQuery(
        search=search,
        replacement_node=replacement_node,
    )

    gen_ast = rewrite_at_query.visit(output_ast)

    assert rewrite_at_query.replaced is True, "Failed to update with {!r}".format(
        to_code(replacement_node)
    )
    return gen_ast


__all__ = ["sync_property", "sync_properties"]
