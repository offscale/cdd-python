"""
Functionality to generate classes, functions, and/or argparse functions from the input mapping
"""

import ast
from ast import Import, ImportFrom
from importlib import import_module
from inspect import getfile
from os import path

from doctrans import parse, emit
from doctrans.ast_utils import get_at_root
from doctrans.source_transformer import to_code


def gen(
    name_tpl,
    input_mapping,
    type_,
    output_filename,
    prepend=None,
    imports_from_file=None,
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
    """

    if imports_from_file is None:
        imports = ""
    else:
        with open(
            imports_from_file
            if path.isfile(imports_from_file)
            else getfile(import_module(imports_from_file)),
            "rt",
        ) as f:
            imports = "".join(
                map(to_code, get_at_root(ast.parse(f.read()), (Import, ImportFrom)))
            )

    module_path, _, symbol_name = input_mapping.rpartition(".")
    input_mapping = getattr(import_module(module_path), symbol_name)
    input_mapping_it = (
        input_mapping.items() if hasattr(input_mapping, "items") else input_mapping
    )
    content = "{}{}{}".format(
        "" if prepend is None else prepend,
        imports,  # TODO: Optimize imports programatically (rather than just with IDE)
        "\n\n".join(
            to_code(
                getattr(emit, type_.replace("class", "class_"))(
                    parse.class_(
                        obj
                    ),  # TODO: Figure out if it's a class, function, or argparse function
                    emit_call=True,
                    **{
                        "class_name"
                        if type_ == "class"
                        else "function_name": name_tpl.format(name=name)
                    }
                )
            )
            for name, obj in input_mapping_it
        ),
    )

    with open(output_filename, "a") as f:
        f.write(content)
