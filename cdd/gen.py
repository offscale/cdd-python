"""
Functionality to generate classes, functions, and/or argparse functions from the input mapping
"""

import ast
from ast import Assign, Import, ImportFrom, Module, Name, Store
from inspect import getfile
from itertools import chain
from operator import itemgetter
from os import path

from cdd import emit, parse
from cdd.ast_utils import get_at_root, maybe_type_comment, set_value
from cdd.parser_utils import infer
from cdd.pure_utils import get_module, sanitise_emit_name
from cdd.source_transformer import to_code


def gen(
    name_tpl,
    input_mapping,
    parse_name,
    emit_name,
    output_filename,
    prepend=None,
    imports_from_file=None,
    emit_call=False,
    emit_default_doc=True,
    decorator_list=None,
    no_word_wrap=None,
):
    """
    Generate classes, functions, and/or argparse functions from the input mapping

    :param name_tpl: Template for the name, e.g., `{name}Config`.
    :type name_tpl: ```str```

    :param input_mapping: Import location of dictionary/mapping/2-tuple collection.
    :type input_mapping: ```str```

    :param parse_name: What type to parse.
    :type parse_name: ```Literal["argparse", "class", "function", "sqlalchemy", "sqlalchemy_table"]```

    :param emit_name: What type to generate.
    :type emit_name: ```Literal["argparse", "class", "function", "sqlalchemy", "sqlalchemy_table"]```

    :param output_filename: Output file to write to
    :type output_filename: ```str```

    :param prepend: Prepend file with this. Use '\n' for newlines.
    :type prepend: ```Optional[str]```

    :param imports_from_file: Extract imports from file and append to `output_file`.
        If module or other symbol path given, resolve file then use it.
    :type imports_from_file: ```Optional[str]```

    :param emit_call: Whether to emit a `__call__` method from the `_internal` IR subdict
    :type emit_call: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :param decorator_list: List of decorators
    :type decorator_list: ```Optional[Union[List[Str], List[]]]```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```
    """
    extra_symbols = {}
    if imports_from_file is None:
        imports = ""
    else:
        if prepend:
            prepend_imports = get_at_root(
                ast.parse(prepend.strip()), (Import, ImportFrom)
            )

            # def rewrite_typings(node):
            #     """
            #     Python < 3.8 must use `typings_extensions` for `Literal`
            #
            #     :param node: import node
            #     :type node: ```Union[Import, ImportFrom]```
            #
            #     :return: The import potentially rewritten or None
            #     :rtype: ```Optional[Union[Import, ImportFrom]]```
            #     """
            #     if isinstance(node, ImportFrom) and node.module == "typing":
            #         len_names = len(node.names)
            #         if len_names == 1 and node.names[0].name == "Literal":
            #             rewrite_typings.found_literal = True
            #             return None
            #         else:
            #             node.names = list(
            #                 filter(
            #                     None,
            #                     map(
            #                         lambda _alias: None
            #                         if _alias.name == "Literal"
            #                         else _alias,
            #                         node.names,
            #                     ),
            #                 )
            #             )
            #             if len(node.names) != len_names:
            #                 rewrite_typings.found_literal = True
            #     return node
            #
            # rewrite_typings.found_literal = False
            # prepend_imports = list(filter(None, map(rewrite_typings, prepend_imports)))
            # if rewrite_typings.found_literal:
            #     prepend_imports.append(
            #         ImportFrom(
            #             level=0,
            #             module="typing_extensions"
            #             if sys.version_info[:2] < (3, 8)
            #             else "typing",
            #             names=[alias(asname=None, name="Literal")],
            #             lineno=None,
            #             col_offset=None,
            #         )
            #     )

            eval(
                compile(
                    to_code(
                        ast.fix_missing_locations(
                            Module(body=prepend_imports, stmt=None, type_ignores=[])
                        )
                    ),
                    filename="<string>",
                    mode="exec",
                ),
                extra_symbols,
            )
            # This leaks to the global scope
            globals().update(extra_symbols)
        with open(
            imports_from_file
            if path.isfile(imports_from_file)
            else getfile(get_module(imports_from_file, extra_symbols=extra_symbols)),
            "rt",
        ) as f:
            imports = "".join(
                map(to_code, get_at_root(ast.parse(f.read()), (Import, ImportFrom)))
            )

    module_path, _, symbol_name = input_mapping.rpartition(".")
    input_mapping = getattr(
        get_module(module_path, extra_symbols=extra_symbols), symbol_name
    )
    input_mapping_it = (
        input_mapping.items() if hasattr(input_mapping, "items") else input_mapping
    )

    global__all__ = []
    emit_name = sanitise_emit_name(emit_name)
    content = "{prepend}{imports}\n{functions_and_classes}\n{__all__}".format(
        prepend="" if prepend is None else prepend,
        imports=imports,  # TODO: Optimize imports programmatically (akin to `autoflake --remove-all-unused-imports`)
        functions_and_classes="\n\n".join(
            print("\nGenerating: {name!r}".format(name=name))
            or global__all__.append(name_tpl.format(name=name))
            or to_code(
                getattr(emit, emit_name)(
                    getattr(
                        parse,
                        infer(obj) if parse_name in (None, "infer") else parse_name,
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
        ),
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


__all__ = ["gen"]
