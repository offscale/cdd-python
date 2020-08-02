"""
Given the truth, show others the path
"""

from ast import parse, walk, FunctionDef, ClassDef, Module
from copy import deepcopy
from functools import partial

from meta.asttools import cmp_ast

from doctrans import docstring_struct
from doctrans import transformers
from doctrans.ast_utils import get_function_type
from doctrans.pure_utils import rpartial


def _get_name_from_namespace(args, fun_name):
    """
    Gets the arg from the namespace which matches the given prefix

    :param args: Namespace with the values of the CLI arguments
    :type args: ```Namespace```

    :param fun_name: Name of the start of the function
    :type fun_name: ```str```

    :returns: Argument from Namespace object
    :rtype: ```str```
    """
    return next(
        getattr(args, arg)
        for arg in args.__dict__.keys()
        if arg == "_".join((fun_name, "name"))
    )


def ground_truth(args, truth_file):
    """
    There is but one truth. Conform.

    :param args: Namespace with the values of the CLI arguments
    :type args: ```Namespace```

    :param truth_file: contains the filename of the one true source
    :type truth_file: ```str```
    """
    arg_name_to_func_typ = {
        "argparse_function": (docstring_struct.from_argparse_ast, FunctionDef),
        "class": (docstring_struct.from_class, ClassDef),
        "function": (docstring_struct.from_class_with_method, FunctionDef),
    }

    from_func, typ = arg_name_to_func_typ[args.truth]
    with open(truth_file, "rt") as f:
        true_docstring_structure = from_func(
            next(
                filter(
                    lambda fun: fun.name == _get_name_from_namespace(args, args.truth),
                    filter(rpartial(isinstance, typ), walk(parse(f.read()))),
                )
            )
        )
    for (
        fun_name
    ) in (
        arg_name_to_func_typ
    ):  # filter(lambda arg: arg != args.truth, arg_name_to_func_typ.keys()):
        from_func, typ = arg_name_to_func_typ[fun_name]

        name = _get_name_from_namespace(args, fun_name)
        if name.count(".") > 1:
            raise NotImplementedError(
                "We can only go one deep; e.g., `F.a.b` is not supported"
                " given `class F: def a(): def b(): pass; pass;`"
            )
        outer_name, _, inner_name = name.partition(".")

        unchanged, filename = True, getattr(args, fun_name)
        with open(filename, "rt") as f:
            parsed_ast = parse(f.read())
        assert isinstance(parsed_ast, Module)

        for idx, outer_node in enumerate(parsed_ast.body):
            replace_node_f = partial(
                replace_node,
                fun_name=fun_name,
                from_func=from_func,
                outer_node=outer_node,
                outer_name=outer_name,
                docstring_structure=true_docstring_structure,
                typ=typ,
            )
            if hasattr(outer_node, "name") and outer_node.name == outer_name:
                if inner_name:
                    for i, inner_node in enumerate(outer_node.body):
                        if (
                            isinstance(inner_node, typ)
                            and inner_node.name == inner_name
                        ):
                            unchanged, parsed_ast.body[idx].body[i] = replace_node_f(
                                inner_node=inner_node, inner_name=inner_name
                            )
                elif isinstance(outer_node, typ):
                    unchanged, parsed_ast.body[idx] = replace_node_f(
                        inner_node=None, inner_name=None
                    )

        print("unchanged" if unchanged else "modified", filename, sep="\t")
        if not unchanged:
            transformers.to_file(parsed_ast, filename, mode="wt")


def replace_node(
    fun_name,
    from_func,
    outer_name,
    inner_name,
    outer_node,
    inner_node,
    docstring_structure,
    typ,
):
    """
    They will not replace us. Except you will, with this function.

    :param fun_name: Name of function, e.g., argparse, class, method
    :type fun_name: ```str```

    :param from_func: One docstring_struct.from_* function
    :type from_func: ```Callable[[AST, str, ...], dict]```

    :param outer_name: Name of the outer node
    :type outer_name: ```str```

    :param inner_name: Name of the inner node. If unset then don't traverse to inner node.
    :type inner_name: ```Optional[str]```

    :param outer_node: The outer node.
    :type outer_node: ```AST```

    :param inner_node: The inner node. If unset then don't [try and] traverse down to it.
    :type inner_node: ```Optional[AST]```

    :param docstring_structure: dict of shape {
            'name': ..., 'platform': ...,
            'module': ..., 'title': ..., 'description': ...,
            'parameters': ..., 'schema': ...,'returns': ...}
    :type docstring_structure: ```dict```

    :param typ: AST instance
    :type typ: ```AST```

    :returns: Whether the created AST node is equal to the previous one, the created AST node
    :rtype: ```Tuple[bool, AST]```
    """
    name, node = (
        (outer_name, outer_node) if inner_name is None else (inner_name, inner_node)
    )
    previous = deepcopy(node)
    options = {
        "FunctionDef": lambda: {
            "function_type": get_function_type(node),
            "function_name": name,
        }
    }.get(typ.__name__, lambda: {})

    found = from_func(
        outer_node, *tuple() if fun_name == "argparse_function" else (name,)
    )

    if "_internal" in found:
        raise NotImplementedError()
    else:
        node = getattr(transformers, "to_{fun_name}".format(fun_name=fun_name),)(
            docstring_structure, **options()
        )

    return cmp_ast(previous, node), node
