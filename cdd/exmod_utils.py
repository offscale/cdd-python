""" Exmod utils """

import ast
from ast import Assign, Expr, ImportFrom, List, Load, Module, Name, Store, alias
from functools import partial
from inspect import getfile, ismodule
from itertools import chain
from os import extsep, makedirs, path

from cdd import emit
from cdd.ast_utils import (
    maybe_type_comment,
    merge_assignment_lists,
    merge_modules,
    set_value,
)
from cdd.pure_utils import no_magic_dir2attr
from cdd.tests.mocks import imports_header


def get_module_contents(obj, module_root_dir, current_module=None, _result={}):
    """
    Helper function to get the recursive inner module contents

    :param obj: Something to `dir` on
    :type obj: ```Any```

    :param module_root_dir: Root of module
    :type module_root_dir: ```str```

    :param current_module: The current module
    :type current_module: ```Optional[str]```

    :param _result: The result var (used internally as accumulator)
    :type _result: ```dict```

    :returns: Values (could be modules, classes, and whatever other symbols are exposed)
    :rtype: ```Generator[Any]```
    """
    for name, symbol in no_magic_dir2attr(obj).items():
        fq = "{}.{}".format(current_module, name)
        try:
            symbol_location = getfile(symbol)
        except TypeError:
            symbol_location = None
        if symbol_location is not None and symbol_location.startswith(module_root_dir):
            if isinstance(symbol, type):
                _result[fq] = symbol
            elif (
                current_module != getattr(symbol, "__name__", current_module)
                and ismodule(symbol)
                and fq not in _result
            ):
                get_module_contents(
                    symbol,
                    module_root_dir=module_root_dir,
                    current_module=symbol.__name__,
                )
    return _result


def mkdir_and_emit_file(
    name_orig_ir,
    emit_name,
    module_name,
    new_module_name,
    filesystem_layout,
    output_directory,
):
    """
    Generate Java-package style file hierarchy from fully-qualified module name

    :param name_orig_ir: FQ module name, original filename path, IR
    :type name_orig_ir: ```Tuple[str, str, dict]```

    :param emit_name: What type(s) to generate.
    :type emit_name: ```List[Literal["argparse", "class", "function", "sqlalchemy", "sqlalchemy_table"]]```

    :param module_name: Name of [original] module
    :type module_name: ```str```

    :param new_module_name: Name of [new] module
    :type new_module_name: ```str```

    :param filesystem_layout: Hierarchy of folder and file names generated. "java" is file per package per name.
    :type filesystem_layout: ```Literal["java", "as_input"]```

    :param output_directory: Where to place the generated exposed interfaces to the given `--module`.
    :type output_directory: ```str```

    :returns: Import to generated module
    :rtype: ```ImportFrom```
    """
    mod_name, _, name = name_orig_ir[0].rpartition(".")
    original_relative_filename_path, ir = name_orig_ir[1], name_orig_ir[2]
    mod_path = path.join(
        output_directory,
        new_module_name,
        mod_name.replace(".", path.sep),
    )
    if not path.isdir(mod_path):
        makedirs(mod_path)
    open(
        path.join(path.dirname(mod_path), "__init__{extsep}py".format(extsep=extsep)),
        "a",
    ).close()
    gen_node = getattr(emit, emit_name.replace("class", "class_"))(
        ir,
        **dict(
            **{"{emit_name}_name".format(emit_name=emit_name): name},
            **{} if emit_name == "class" else {"function_type": "static"}
        )
    )
    __all___node = Assign(
        targets=[Name("__all__", Store())],
        value=List(
            ctx=Load(),
            elts=[set_value(name)],
            expr=None,
        ),
        expr=None,
        lineno=None,
        **maybe_type_comment
    )
    if not isinstance(gen_node, Module):
        gen_node = Module(
            body=list(
                chain.from_iterable(
                    (
                        (
                            Expr(
                                set_value(
                                    "\nGenerated from {module_name}.{name}\n".format(
                                        module_name=module_name,
                                        name=name_orig_ir[0],
                                    )
                                )
                            ),
                        ),
                        ast.parse(imports_header).body,
                        (gen_node, __all___node),
                    )
                )
            ),
            stmt=None,
            type_ignores=[],
        )

    emit_filename, init_filepath = (
        map(
            partial(path.join, output_directory, new_module_name),
            (
                original_relative_filename_path,
                path.join(
                    path.dirname(original_relative_filename_path),
                    "__init__{extsep}py".format(extsep=extsep),
                ),
            ),
        )
        if filesystem_layout == "as_input"
        else map(
            partial(path.join, mod_path),
            (
                "{name}{extsep}py".format(name=name, extsep=extsep),
                "__init__{extsep}py".format(extsep=extsep),
            ),
        )
    )

    if path.isfile(emit_filename):
        with open(emit_filename, "rt") as f:
            mod = ast.parse(f.read())
        gen_node = merge_modules(mod, gen_node)
        merge_assignment_lists(gen_node, "__all__")

    emit.file(gen_node, filename=emit_filename, mode="wt")
    # print("Emitted: {emit_filename!r} ;".format(emit_filename=emit_filename))
    if name != "__init__" and not path.isfile(init_filepath):
        emit.file(
            Module(
                body=[
                    Expr(set_value("\n__init__ to expose internals of this module\n")),
                    ImportFrom(
                        module=name,
                        names=[
                            alias(
                                name=name,
                                asname=None,
                                identifier=None,
                                identifier_name=None,
                            ),
                        ],
                        level=1,
                        identifier=None,
                    ),
                    __all___node,
                ],
                stmt=None,
                type_ignores=[],
            ),
            filename=init_filepath,
            mode="wt",
        )

        # print("Emitted: {init_filepath!r} ;".format(init_filepath=init_filepath))
    # print("\n", end="")

    return (
        mod_name,
        original_relative_filename_path,
        ImportFrom(
            module=name,
            names=[
                alias(
                    name=name,
                    asname=None,
                    identifier=None,
                    identifier_name=None,
                ),
            ],
            level=1,
            identifier=None,
        ),
    )


__all__ = ["get_module_contents", "mkdir_and_emit_file"]
