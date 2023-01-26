"""
Functionality to generate classes, functions, and/or argparse functions from the input mapping
"""

import ast
from ast import Import, ImportFrom, Module
from inspect import getfile
from json import load
from os import path

import cdd.emit.json_schema
from cdd.ast_utils import get_at_root
from cdd.emit.utils.sqlalchemy_utils import (
    update_fk_for_file,
    update_with_imports_from_columns,
)
from cdd.gen_utils import gen_file, get_input_mapping_from_path
from cdd.pure_utils import get_module, pascal_to_upper_camelcase, sanitise_emit_name
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
    emit_and_infer_imports=False,
    decorator_list=None,
    phase=0,
    no_word_wrap=None,
):
    """
    Generate classes, functions, and/or argparse functions from the input mapping

    :param name_tpl: Template for the name, e.g., `{name}Config`.
    :type name_tpl: ```str```

    :param input_mapping: Import location of dictionary/mapping/2-tuple collection.
    :type input_mapping: ```str```

    :param parse_name: Which type to parse.
    :type parse_name: ```Literal["argparse", "class", "function", "json_schema",
                                 "pydantic", "sqlalchemy", "sqlalchemy_table"]```

    :param emit_name: Which type to generate.
    :type emit_name: ```Literal["argparse", "class", "function", "json_schema",
                                "pydantic", "sqlalchemy", "sqlalchemy_table"]```

    :param output_filename: Output file to write to
    :type output_filename: ```str```

    :param prepend: Prepend file with this. Use '\n' for newlines.
    :type prepend: ```Optional[str]```

    :param imports_from_file: Extract imports from file and append to `output_file`.
        If module or other symbol path given, resolve file then use it.
    :type imports_from_file: ```Optional[str]```

    :param emit_call: Whether to emit a `__call__` method from the `_internal` IR subdict
    :type emit_call: ```bool```

    :param emit_and_infer_imports: Whether to emit and infer imports at the top of the generated code
    :type emit_and_infer_imports: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :param decorator_list: List of decorators
    :type decorator_list: ```Optional[Union[List[Str], List[]]]```

    :param phase: Which phase to run through. E.g., SQLalchemy may require multiple phases to resolve foreign keys
    :type phase: ```int```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```
    """
    extra_symbols = {}
    if phase > 0 and emit_name in frozenset(("sqlalchemy", "sqlalchemy_table")):
        if phase == 1:
            return update_with_imports_from_columns(output_filename)
        elif phase == 2:
            return update_fk_for_file(output_filename)
        else:
            raise NotImplementedError("phase {}".format(phase))
    elif imports_from_file is None:
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

    emit_name = sanitise_emit_name(emit_name)
    if path.isfile(input_mapping) and parse_name == "json_schema":
        with open(input_mapping, "rt") as f:
            json_contents = load(f)
        name = path.basename(module_path)
        if "name" not in json_contents:
            json_contents["name"] = pascal_to_upper_camelcase(name)
        input_mapping = {name: json_contents}
    else:
        input_mod = get_module(module_path, extra_symbols=extra_symbols)
        input_mapping = (
            getattr(input_mod, symbol_name)
            if hasattr(input_mod, symbol_name)
            else get_input_mapping_from_path(emit_name, module_path, symbol_name)
        )
    input_mapping_it = (
        input_mapping.items() if hasattr(input_mapping, "items") else input_mapping
    )

    return (
        cdd.emit.json_schema.json_schema_file(input_mapping, output_filename)
        if emit_name == "json_schema"
        else gen_file(
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
        )
    )


__all__ = ["gen"]
