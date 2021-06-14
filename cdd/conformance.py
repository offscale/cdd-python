"""
Given the truth, show others the path
"""

from ast import ClassDef, FunctionDef, Module
from collections import OrderedDict
from os import path

from cdd import emit, parse
from cdd.ast_utils import RewriteAtQuery, cmp_ast, find_in_ast, get_function_type
from cdd.pure_utils import pluralise, strip_split
from cdd.source_transformer import ast_parse


def _default_options(node, search, type_wanted):
    """
    Conform the given file to the `intermediate_repr`

    :param node: AST node
    :type node: ```AST```

    :param search: Search query, e.g., ['node_name', 'function_name', 'arg_name']
    :type search: ```List[str]```

    :param type_wanted: AST instance
    :type type_wanted: ```AST```

    :returns: Arguments to pass to `emit.*` function
    :rtype: ```Callable[[], dict]```
    """
    return {
        "FunctionDef": lambda: {
            "function_type": node if node is None else get_function_type(node),
            "function_name": search[-1] if len(search) else "set_cli_args",
        },
        "ClassDef": lambda: {
            "class_name": search[-1] if len(search) else "ConfigClass",
        },
    }.get(type_wanted.__name__, lambda: {})


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
    arg2parse_emit_type = {
        "argparse_function": (parse.argparse_ast, emit.argparse_function, FunctionDef),
        "class": (parse.class_, emit.class_, ClassDef),
        "function": (parse.function, emit.function, FunctionDef),
    }

    parse_func, emit_func, type_wanted = arg2parse_emit_type[args.truth]
    search = _get_name_from_namespace(args, args.truth).split(".")

    with open(truth_file, "rt") as f:
        true_ast = ast_parse(f.read(), filename=truth_file)

    original_node = find_in_ast(search, true_ast)
    gold_ir = parse_func(
        original_node,
        **_default_options(node=original_node, search=search, type_wanted=type_wanted)()
    )

    effect = OrderedDict()
    # filter(lambda arg: arg != args.truth, arg2parse_emit_type.keys()):
    for fun_name, (parse_func, emit_func, type_wanted) in arg2parse_emit_type.items():
        search = list(strip_split(_get_name_from_namespace(args, fun_name), "."))

        filenames = getattr(args, pluralise(fun_name))
        assert isinstance(
            filenames, (list, tuple)
        ), "Expected Union[list, tuple] got {type_name!r}".format(
            type_name=type(filenames).__name__
        )

        effect.update(
            map(
                lambda filename: _conform_filename(
                    filename=filename,
                    search=search,
                    emit_func=emit_func,
                    replacement_node_ir=gold_ir,
                    type_wanted=type_wanted,
                ),
                filenames,
            )
        )

    return effect


def _conform_filename(
    filename,
    search,
    emit_func,
    replacement_node_ir,
    type_wanted,
):
    """
    Conform the given file to the `intermediate_repr`

    :param filename: Location of file
    :type filename: ```str```

    :param search: Search query, e.g., ['node_name', 'function_name', 'arg_name']
    :type search: ```List[str]```

    :param replacement_node_ir: Replace what is found with the contents of this param
    :type replacement_node_ir: ```dict```

    :param type_wanted: AST instance
    :type type_wanted: ```AST```

    :returns: filename, whether the file was modified
    :rtype: ```Tuple[str, bool]```
    """
    filename = path.realpath(path.expanduser(filename))

    if not path.isfile(filename):
        emit.file(
            emit_func(
                replacement_node_ir,
                emit_default_doc=False,  # emit_func.__name__ == "class_"
            ),
            filename=filename,
            mode="wt",
            skip_black=False,
        )
        return filename, True

    with open(filename, "rt") as f:
        parsed_ast = ast_parse(f.read(), filename=filename)
    assert isinstance(parsed_ast, Module)

    original_node = find_in_ast(search, parsed_ast)
    replacement_node = emit_func(
        replacement_node_ir,
        **_default_options(node=original_node, search=search, type_wanted=type_wanted)()
    )
    if original_node is None:
        emit.file(replacement_node, filename=filename, mode="a", skip_black=False)
        return filename, True
    assert len(search) > 0

    assert (
        type(replacement_node) == type_wanted
    ), "Expected {type_wanted!r} got {type_replacement_node!r}".format(
        type_wanted=type_wanted, type_replacement_node=type(replacement_node).__name__
    )

    replaced = False
    if not cmp_ast(original_node, replacement_node):
        rewrite_at_query = RewriteAtQuery(
            search=search,
            replacement_node=replacement_node,
        )
        rewrite_at_query.visit(parsed_ast)

        print(
            "modified" if rewrite_at_query.replaced else "unchanged", filename, sep="\t"
        )
        if rewrite_at_query.replaced:
            emit.file(parsed_ast, filename, mode="wt", skip_black=False)

        replaced = rewrite_at_query.replaced

    return filename, replaced


__all__ = ["ground_truth"]
