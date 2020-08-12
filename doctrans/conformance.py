"""
Given the truth, show others the path
"""
import ast
from ast import FunctionDef, ClassDef, Module
from collections import OrderedDict
from copy import deepcopy
from functools import partial

from meta.asttools import cmp_ast

from doctrans import emit
from doctrans import parse
from doctrans.ast_utils import get_function_type
from doctrans.pure_utils import rpartial, pluralise, sanitise


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
        getattr(args, pluralise(arg))[0]
        for arg in vars(args).keys()
        if arg == "_".join((fun_name, "names"))
    )


def ground_truth(args, truth_file):
    """
    There is but one truth. Conform.

    :param args: Namespace with the values of the CLI arguments
    :type args: ```Namespace```

    :param truth_file: contains the filename of the one true source
    :type truth_file: ```str```

    :returns: Filenames and whether they were changed
    :rtype: ```OrderedDict```
    """
    arg_name_to_func_typ = {
        "argparse_function": (parse.argparse_ast, FunctionDef),
        "class": (parse.class_, ClassDef),
        "function": (parse.class_with_method, FunctionDef),
    }

    from_func, typ = arg_name_to_func_typ[args.truth]
    fun_name = _get_name_from_namespace(args, args.truth)

    with open(truth_file, "rt") as f:
        parsed_truth_file = ast.parse(f.read(), filename=truth_file)

    gold_ir = from_func(
        next(
            filter(
                lambda fun: fun.name == fun_name,
                filter(rpartial(isinstance, typ), parsed_truth_file.body),
            )
        )
    )

    effect = OrderedDict()
    # filter(lambda arg: arg != args.truth, arg_name_to_func_typ.keys()):
    for fun_name in arg_name_to_func_typ:
        name = _get_name_from_namespace(args, fun_name)

        if name.count(".") > 1:
            raise NotImplementedError(
                "We can only go one deep; e.g., `F.a.b` is not supported"
                " given `class F: def a(): def b(): pass; pass;`"
            )

        filenames = getattr(args, pluralise(fun_name))
        assert isinstance(
            filenames, (list, tuple)
        ), "Expected Union[list, tuple] got {!r}".format(type(filenames).__name__)

        from_func, typ = arg_name_to_func_typ[fun_name]
        outer_name, _, inner_name = name.partition(".")
        effect.update(
            map(
                partial(
                    _conform_filename,
                    fun_name=fun_name,
                    from_func=from_func,
                    outer_name=outer_name,
                    intermediate_repr=gold_ir,
                    typ=typ,
                    inner_name=inner_name,
                ),
                filenames,
            )
        )

    return effect


def _conform_filename(
    filename, fun_name, from_func, outer_name, inner_name, intermediate_repr, typ,
):
    """
    Conform the given file to the `intermediate_repr`

    :param filename: Location of file
    :type filename: ```str```

    :param fun_name: Function/Class/AST name
    :type fun_name: ```AST```

    :param from_func: One parse._* function
    :type from_func: ```Callable[[AST, str, ...], dict]```

    :param outer_name: Name of the outer node
    :type outer_name: ```str```

    :param inner_name: Name of the inner node. If unset then don't traverse to inner node.
    :type inner_name: ```Optional[str]```

    :param intermediate_repr: dict of shape {
            'name': ..., 'platform': ...,
            'module': ..., 'title': ..., 'description': ...,
            'parameters': ..., 'schema': ...,'returns': ...}
    :type intermediate_repr: ```dict```

    :param typ: AST instance
    :type typ: ```AST```

    :returns: filename, whether the file was modified
    :rtype: ```Tuple[str, bool]```
    """
    unchanged = True
    with open(filename, "rt") as f:
        parsed_ast = ast.parse(f.read())
    assert isinstance(parsed_ast, Module)

    for idx, outer_node in enumerate(parsed_ast.body):
        replace_node_f = partial(
            replace_node,
            fun_name=fun_name,
            from_func=from_func,
            outer_node=outer_node,
            outer_name=outer_name,
            intermediate_repr=intermediate_repr,
            typ=typ,
        )
        if hasattr(outer_node, "name") and outer_node.name == outer_name:
            if inner_name:
                for i, inner_node in enumerate(outer_node.body):
                    if isinstance(inner_node, typ) and inner_node.name == inner_name:
                        unchanged, parsed_ast.body[idx].body[i] = replace_node_f(
                            inner_node=inner_node, inner_name=inner_name
                        )
            elif isinstance(outer_node, typ):
                unchanged, parsed_ast.body[idx] = replace_node_f(
                    inner_node=None, inner_name=None
                )

    print("unchanged" if unchanged else "modified", filename, sep="\t")
    if not unchanged:
        emit.file(parsed_ast, filename, mode="wt")

    return filename, unchanged


def replace_node(
    fun_name,
    from_func,
    outer_name,
    inner_name,
    outer_node,
    inner_node,
    intermediate_repr,
    typ,
):
    """
    They will not replace us. Except you will, with this function.

    :param fun_name: Name of function, e.g., argparse, class, method
    :type fun_name: ```str```

    :param from_func: One parse.* function
    :type from_func: ```Callable[[AST, str, ...], dict]```

    :param outer_name: Name of the outer node
    :type outer_name: ```str```

    :param inner_name: Name of the inner node. If unset then don't traverse to inner node.
    :type inner_name: ```Optional[str]```

    :param outer_node: The outer node.
    :type outer_node: ```AST```

    :param inner_node: The inner node. If unset then don't [try and] traverse down to it.
    :type inner_node: ```Optional[AST]```

    :param intermediate_repr: dict of shape {
            'name': ..., 'platform': ...,
            'module': ..., 'title': ..., 'description': ...,
            'parameters': ..., 'schema': ...,'returns': ...}
    :type intermediate_repr: ```dict```

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
        node = getattr(emit, sanitise(fun_name),)(intermediate_repr, **options())

    return cmp_ast(previous, node), node


__all__ = ["ground_truth", "replace_node"]
