"""
Utility functions for cdd.gen
"""

import ast
from ast import (
    AnnAssign,
    Assign,
    AsyncFunctionDef,
    ClassDef,
    FunctionDef,
    Import,
    ImportFrom,
    Name,
    Store,
)
from importlib import import_module
from itertools import chain
from operator import itemgetter

import cdd.parse.utils.parser_utils
from cdd.ast_utils import infer_imports, maybe_type_comment, set_value
from cdd.parse.utils.parser_utils import infer
from cdd.pure_utils import find_module_filepath, rpartial
from cdd.source_transformer import to_code


def get_input_mapping_from_path(emit_name, module_path, symbol_name):
    """
    Given (module_path, symbol_name) acquire file path, `ast.parse` out all top-level symbols matching `emit_name`

    :param emit_name: What type to generate.
    :type emit_name: ```Literal["argparse", "class", "function", "json_schema",
                                "pydantic", "sqlalchemy", "sqlalchemy_table"]```

    :param module_path: Module path
    :type module_path: ```str```

    :param symbol_name: Symbol to import from module
    :type symbol_name: ```str```

    :return: Dictionary of (name, AST) where AST is produced by a cdd emitter matching `emit_name`
    :rtype: ```dict```
    """
    with open(find_module_filepath(module_path, symbol_name), "rt") as f:
        input_ast_mod = ast.parse(f.read())
    type_instance_must_be = {
        "sqlalchemy_table": (Assign, AnnAssign),
        "function": (FunctionDef, AsyncFunctionDef),
        "pydantic": (FunctionDef, AsyncFunctionDef),
    }.get(emit_name, (ClassDef,))
    return dict(
        map(
            lambda node_name: (
                node_name[0],
                (
                    (
                        lambda parser_name: getattr(
                            import_module(".".join(("cdd", "parse", parser_name))),
                            parser_name,
                        )
                    )(cdd.parse.utils.parser_utils.infer(node_name[1]))
                )(node_name[1]),
            ),
            map(
                lambda node: (node.name, node)
                if hasattr(node, "name")
                else (
                    (
                        node.target if isinstance(node, AnnAssign) else node.targets[0]
                    ).id,
                    node,
                ),
                filter(
                    rpartial(
                        isinstance,
                        type_instance_must_be,
                    ),
                    input_ast_mod.body,
                ),
            ),
        )
    )


def gen_file(
    name_tpl,
    input_mapping_it,
    parse_name,
    emit_name,
    output_filename,
    prepend,
    emit_call,
    emit_and_infer_imports,
    emit_default_doc,
    decorator_list,
    no_word_wrap,
    imports,
):
    """
    Generate Python file of containing `input_mapping_it`.values converted to `emit_name`

    :param name_tpl: Template for the name, e.g., `{name}Config`.
    :type name_tpl: ```str```

    :param input_mapping_it: Import location of mapping/2-tuple collection.
    :type input_mapping_it: ```Iterator[Tuple[str,AST]]```

    :param parse_name: What type to parse.
    :type parse_name: ```Literal["argparse", "class", "function", "json_schema",
                                 "pydantic", "sqlalchemy", "sqlalchemy_table"]```

    :param emit_name: What type to generate.
    :type emit_name: ```Literal["argparse", "class", "function", "json_schema",
                                "pydantic", "sqlalchemy", "sqlalchemy_table"]```

    :param output_filename: Output file to write to
    :type output_filename: ```str```

    :param prepend: Prepend file with this. Use '\n' for newlines.
    :type prepend: ```Optional[str]```

    :param emit_call: Whether to emit a `__call__` method from the `_internal` IR subdict
    :type emit_call: ```bool```

    :param emit_and_infer_imports: Whether to emit and infer imports at the top of the generated code
    :type emit_and_infer_imports: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :param decorator_list: List of decorators
    :type decorator_list: ```Optional[Union[List[Str], List[]]]```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```

    :param imports: Import to preclude in Python file
    :type imports: ```str```
    """
    global__all__ = []
    functions_and_classes = tuple(
        print("\nGenerating: {name!r}".format(name=name))
        or global__all__.append(name_tpl.format(name=name))
        or (
            getattr(import_module(".".join(("cdd", "emit", emit_name))), emit_name)(
                (
                    (
                        lambda parser_name: getattr(
                            import_module(".".join(("cdd", "parse", parser_name))),
                            parser_name,
                        )
                    )(infer(obj) if parse_name in (None, "infer") else parse_name)
                )(obj),
                emit_default_doc=emit_default_doc,
                word_wrap=no_word_wrap is None,
                **(
                    lambda _name: {
                        "argparse_function": {"function_name": _name},
                        "class_": {
                            "class_name": _name,
                            "decorator_list": decorator_list,
                            "emit_call": emit_call,
                        },
                        "function": {
                            "function_name": _name,
                        },
                        "sqlalchemy": {"table_name": _name},
                        "sqlalchemy_table": {"table_name": _name},
                    }[emit_name]
                )(name_tpl.format(name=name)),
            )
        )
        for name, obj in input_mapping_it
    )
    if emit_and_infer_imports:
        imports = (imports or "") + " ".join(
            map(to_code, map(infer_imports, functions_and_classes))
        )
    content = "{prepend}{imports}\n{functions_and_classes}\n{__all__}".format(
        prepend="" if prepend is None else prepend,
        imports=imports,  # TODO: Optimize imports programmatically (akin to `autoflake --remove-all-unused-imports`)
        functions_and_classes="\n\n".join(map(to_code, functions_and_classes)),
        __all__=to_code(
            Assign(
                targets=[Name("__all__", Store())],
                value=ast.parse(  # `TypeError: Type List cannot be instantiated; use list() instead`
                    str(
                        list(
                            map(
                                lambda s: s.rstrip("\n").strip("'").strip('"'),
                                map(to_code, map(set_value, global__all__)),
                            )
                        )
                    )
                )
                .body[0]
                .value,
                expr=None,
                lineno=None,
                **maybe_type_comment,
            )
        ),
    )
    parsed_ast = ast.parse(content)
    # TODO: Shebang line first, then docstring, then imports
    doc_str = ast.get_docstring(parsed_ast, clean=True)
    whole = tuple(
        map(
            lambda node: (node, None)
            if isinstance(node, (Import, ImportFrom))
            else (None, node),
            parsed_ast.body,
        )
    )
    parsed_ast.body = list(
        filter(
            None,
            chain.from_iterable(
                (
                    parsed_ast.body[:1] if doc_str else iter(()),
                    sorted(
                        map(itemgetter(0), whole),
                        key=lambda import_from: getattr(import_from, "module", None)
                        == "__future__",
                        reverse=True,
                    ),
                    map(itemgetter(1), whole[1:] if doc_str else whole),
                ),
            ),
        )
    )
    with open(output_filename, "a") as f:
        f.write(to_code(parsed_ast))


__all__ = ["get_input_mapping_from_path", "gen_file"]
