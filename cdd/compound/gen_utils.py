"""
Utility functions for cdd.gen
"""

import ast
from ast import Assign, ClassDef, FunctionDef, Import, ImportFrom, Name, Store
from itertools import chain
from json import load
from operator import itemgetter
from os import path

from cdd.shared.ast_utils import (
    infer_imports,
    maybe_type_comment,
    optimise_imports,
    set_value,
)
from cdd.shared.emit.utils.emitter_utils import get_emitter
from cdd.shared.parse import kind2instance_type
from cdd.shared.parse.utils.parser_utils import get_parser
from cdd.shared.pure_utils import (
    ensure_valid_identifier,
    find_module_filepath,
    pascal_to_upper_camelcase,
    rpartial,
)
from cdd.shared.source_transformer import ast_parse, to_code


def get_input_mapping_from_path(emit_name, module_path, symbol_name):
    """
    Given (module_path, symbol_name) acquire file path, `ast.parse` out all top-level symbols matching `emit_name`

    :param emit_name: Which type to generate.
    :type emit_name: ```Literal["argparse", "class", "function", "json_schema",
                                "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]```

    :param module_path: Module path
    :type module_path: ```str```

    :param symbol_name: Symbol to import from module
    :type symbol_name: ```str```

    :return: Dictionary of (name, AST) where AST is produced by a cdd emitter matching `emit_name`
    :rtype: ```dict```
    """
    module_filepath = find_module_filepath(module_path, symbol_name)
    with open(module_filepath, "rt") as f:
        input_ast_mod = ast.parse(f.read())
    type_instance_must_be = kind2instance_type.get(emit_name, (FunctionDef, ClassDef))
    input_mapping = dict(
        map(
            lambda ir: (ir["name"], ir),
            map(
                lambda node: get_parser(node, emit_name)(node),
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
    assert input_mapping, "Nothing parsed out of {!r}".format(module_filepath)
    return input_mapping


def get_emit_kwarg(decorator_list, emit_call, emit_name, name_tpl, name):
    """
    Emit keyword arguments have different requirements dependent on emitter
    Determine correct one, and always include the name.

    :param decorator_list: List of decorators
    :type decorator_list: ```Optional[Union[List[Str], List[]]]```

    :param emit_name: Which type to generate.
    :type emit_name: ```Literal["argparse", "class", "function", "json_schema",
                                "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]```

    :param emit_call: Whether to emit a `__call__` method from the `_internal` IR subdict
    :type emit_call: ```bool```

    :param name_tpl: Template for the name, e.g., `{name}Config`.
    :type name_tpl: ```str```

    :param name: Interpolates into `name_tpl`
    :type name: ```str```

    :return: Dictionary of keyword arguments targeted the specialised emit function.
    :rtype: ```dict``
    """
    return (
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
            "json_schema": {
                "identifier": _name,
            },
            "sqlalchemy": {"table_name": _name},
            "sqlalchemy_hybrid": {"table_name": _name},
            "sqlalchemy_table": {"table_name": _name},
        }[emit_name]
    )(None if name == "infer" else ensure_valid_identifier(name_tpl.format(name=name)))


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
    functions_and_classes=None,
):
    """
    Generate Python file of containing `input_mapping_it`.values converted to `emit_name`

    :param name_tpl: Template for the name, e.g., `{name}Config`.
    :type name_tpl: ```str```

    :param input_mapping_it: Import location of mapping/2-tuple collection.
    :type input_mapping_it: ```Iterator[Tuple[str,AST]]```

    :param parse_name: Which type to parse.
    :type parse_name: ```Literal["argparse", "class", "function", "json_schema",
                                 "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]```

    :param emit_name: Which type to generate.
    :type emit_name: ```Literal["argparse", "class", "function", "json_schema",
                                "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]```

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

    :param functions_and_classes: Functions and classes that have been preparsed
    :type functions_and_classes: ```Optional[Tuple[AST]]```
    """
    parsed_ast = gen_module(
        decorator_list,
        emit_and_infer_imports,
        emit_call,
        emit_default_doc,
        emit_name,
        functions_and_classes,
        imports,
        input_mapping_it,
        name_tpl,
        no_word_wrap,
        parse_name,
        prepend,
    )
    assert (
        len(parsed_ast.body) > 1
        or not isinstance(parsed_ast.body[0], Assign)
        and any(
            filter(
                lambda target: isinstance(target, Name) and target.id == "__all__",
                parsed_ast.body[0].targets,
            )
        )
    ), "Nothing will be append to {!r}".format(output_filename)
    with open(output_filename, "a") as f:
        f.write(to_code(parsed_ast))


def gen_module(
    decorator_list,
    emit_and_infer_imports,
    emit_call,
    emit_default_doc,
    emit_name,
    functions_and_classes,
    imports,
    input_mapping_it,
    name_tpl,
    no_word_wrap,
    parse_name,
    prepend,
    global__all__=None,
):
    """
    Generate Python module `input_mapping_it`.values converted to `emit_name`

    :param name_tpl: Template for the name, e.g., `{name}Config`.
    :type name_tpl: ```str```

    :param input_mapping_it: Import location of mapping/2-tuple collection.
    :type input_mapping_it: ```Iterator[Tuple[str,AST]]```

    :param parse_name: Which type to parse.
    :type parse_name: ```Literal["argparse", "class", "function", "json_schema",
                                 "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]```

    :param emit_name: Which type to generate.
    :type emit_name: ```Literal["argparse", "class", "function", "json_schema",
                                "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]```

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

    :param functions_and_classes: Functions and classes that have been preparsed
    :type functions_and_classes: ```Optional[Tuple[AST]]```

    :param global__all__: `__all__` symbols for that magic
    :type global__all__: ```List[str]```

    :return: Module with everything contained inside, e.g., all the imports, parsed out functions and classes
    :rtype: ```Module```
    """
    if global__all__ is None:
        global__all__ = []
    if functions_and_classes is None:
        functions_and_classes = get_functions_and_classes(
            decorator_list,
            emit_call,
            emit_default_doc,
            emit_name,
            global__all__,
            input_mapping_it,
            name_tpl,
            no_word_wrap,
            parse_name,
        )
    if emit_and_infer_imports:
        imports = "{}{}".format(
            imports or "",
            " ".join(
                map(
                    to_code,
                    optimise_imports(chain(*map(infer_imports, functions_and_classes))),
                )
            ),
        )

    # Too many params! - Clean things up for debugging:
    del (
        decorator_list,
        emit_call,
        emit_default_doc,
        emit_name,
        input_mapping_it,
        name_tpl,
        no_word_wrap,
        parse_name,
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
    return parsed_ast


def get_functions_and_classes(
    decorator_list,
    emit_call,
    emit_default_doc,
    emit_name,
    global__all__,
    input_mapping_it,
    name_tpl,
    no_word_wrap,
    parse_name,
):
    """
    :param decorator_list: List of decorators
    :type decorator_list: ```Optional[Union[List[Str], List[]]]```

    :param emit_call: Whether to emit a `__call__` method from the `_internal` IR subdict
    :type emit_call: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :param emit_name: Which type to generate.
    :type emit_name: ```Literal["argparse", "class", "function", "json_schema",
                                "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]```


    :param global__all__: `__all__` symbols for that magic
    :type global__all__: ```List[str]```

    :param input_mapping_it: Import location of mapping/2-tuple collection.
    :type input_mapping_it: ```Iterator[Tuple[str,AST]]```

    :param name_tpl: Template for the name, e.g., `{name}Config`.
    :type name_tpl: ```str```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```

    :param parse_name: Which type to parse.
    :type parse_name: ```Literal["argparse", "class", "function", "json_schema",
                                 "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]```

    :return: Side-effect of appending `__all__`, this returns emitted values out of `input_mapping_it`
    :rtype: ```Tuple[AST]```
    """
    emitter = get_emitter(emit_name)
    return tuple(
        print("\nGenerating: {name!r}".format(name=name))
        or global__all__.append(name_tpl.format(name=name))
        or emitter(
            get_parser(obj, parse_name)(obj),
            emit_default_doc=emit_default_doc,
            word_wrap=no_word_wrap is None,
            **get_emit_kwarg(decorator_list, emit_call, emit_name, name_tpl, name),
        )
        for name, obj in input_mapping_it
    )


def file_to_input_mapping(filepath, parse_name):
    """
    Create an `input_mapping` from a given file, i.e. Dict[str, AST]

    :param filepath: Location of JSON or Python file
    :type filepath: ```str```

    :param parse_name: Which type to parse.
    :type parse_name: ```Literal["argparse", "class", "function", "json_schema",
                                 "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid", "infer"]```

    :return: Dictionary of string (name) to AST node
    :rtype: ```dict``
    """
    if (
        parse_name == "json_schema"
        or parse_name == "infer"
        and filepath.endswith("{}json".format(path.extsep))
    ):
        with open(filepath, "rt") as f:
            json_contents = load(f)
        name = path.basename(filepath)
        if "name" not in json_contents:
            json_contents["name"] = pascal_to_upper_camelcase(name)
        input_mapping = {name: json_contents}
    else:
        with open(filepath, "rt") as f:
            mod = ast_parse(f.read())

        input_mapping = dict(
            map(
                lambda node: (node.name, node),
                filter(
                    rpartial(
                        isinstance,
                        tuple(kind2instance_type.values())
                        if parse_name == "infer"
                        else kind2instance_type[parse_name],
                    ),
                    mod.body,
                ),
            ),
        )
    return input_mapping


__all__ = [
    "file_to_input_mapping",
    "get_input_mapping_from_path",
    "get_emit_kwarg",
    "gen_file",
    "get_parser",
]
