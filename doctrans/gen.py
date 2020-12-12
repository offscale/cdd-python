"""
Functionality to generate classes, functions, and/or argparse functions from the input mapping
"""

import ast
from ast import Import, ImportFrom, Module, FunctionDef, Assign, Name, List, Load, Store
from inspect import getfile, isfunction
from os import path

from doctrans import parse, emit
from doctrans.ast_utils import get_at_root, set_value, maybe_type_comment
from doctrans.pure_utils import get_module
from doctrans.source_transformer import to_code


def gen(
    name_tpl,
    input_mapping,
    type_,
    output_filename,
    prepend=None,
    imports_from_file=None,
    emit_call=False,
    emit_default_doc=True,
):
    """
    Generate classes, functions, and/or argparse functions from the input mapping

    :param name_tpl: Template for the name, e.g., `{name}Config`.
    :type name_tpl: ```str```

    :param input_mapping: Import location of dictionary/mapping/2-tuple collection.
    :type input_mapping: ```str```

    :param type_: What type to generate.
    :type type_: ```Literal["argparse", "class", "function"]```

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
    :type emit_default_doc: ```bool``
    """

    extra_symbols = {}
    if imports_from_file is None:
        imports = ""
    else:
        if prepend:
            prepend_imports = get_at_root(
                ast.parse(prepend.strip()), (Import, ImportFrom)
            )
            eval(
                compile(
                    Module(body=prepend_imports, stmt=None, type_ignores=[]),
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
    content = "{prepend}{imports}\n{functions_and_classes}\n{__all}".format(
        prepend="" if prepend is None else prepend,
        imports=imports,  # TODO: Optimize imports programmatically (rather than just with IDE or autoflake)
        functions_and_classes="\n\n".join(
            print("Generating: {!r}".format(name))
            or global__all__.append(name_tpl.format(name=name))
            or to_code(
                getattr(emit, type_.replace("class", "class_"))(
                    (
                        lambda is_func: getattr(
                            parse,
                            "function" if is_func else "class_",
                        )(
                            obj,
                            **{}
                            if is_func
                            else {
                                "merge_inner_function": "__init__",
                            }
                        )
                    )(
                        isinstance(obj, FunctionDef) or isfunction(obj)
                    ),  # TODO: Figure out if it's a function or argparse function
                    emit_call=emit_call,
                    emit_default_doc=emit_default_doc,
                    **{
                        "class_name"
                        if type_ == "class"
                        else "function_name": name_tpl.format(name=name)
                    }
                )
            )
            for name, obj in input_mapping_it
        ),
        __all=to_code(
            Assign(
                targets=[Name("__all__", Store())],
                value=List(
                    ctx=Load(),
                    elts=list(map(set_value, global__all__)),
                    expr=None,
                ),
                expr=None,
                lineno=None,
                **maybe_type_comment
            )
        ),
    )

    with open(output_filename, "a") as f:
        f.write(content)
