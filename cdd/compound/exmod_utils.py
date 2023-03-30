""" Exmod utils """

import ast
from _ast import FunctionDef
from ast import Assign, Expr, ImportFrom, List, Load, Module, Name, Store, alias
from collections import OrderedDict
from functools import partial
from inspect import getfile, ismodule
from itertools import chain
from operator import attrgetter, eq
from os import extsep, makedirs, path

import cdd.argparse_function.emit
import cdd.class_
import cdd.class_.emit
import cdd.class_.parse
import cdd.compound.openapi.emit
import cdd.docstring.emit
import cdd.function.emit
import cdd.json_schema.emit
import cdd.pydantic.emit
import cdd.shared.ast_utils
import cdd.shared.emit.file
import cdd.sqlalchemy.emit
from cdd.shared.parse.utils.parser_utils import get_parser
from cdd.shared.pkg_utils import relative_filename
from cdd.shared.pure_utils import (
    INIT_FILENAME,
    no_magic_dir2attr,
    pp,
    rpartial,
    sanitise_emit_name,
)
from cdd.tests.mocks import imports_header_ast


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

    :return: fully-qualified module name to values (could be modules, classes, and whatever other symbols are exposed)
    :rtype: ```Dict[str,Generator[Any]]```
    """
    if path.isfile(module_root_dir):
        with open(module_root_dir, "rt") as f:
            mod = ast.parse(f.read())
        return dict(
            map(
                lambda node: (
                    "{current_module}.{name}".format(
                        current_module=current_module, name=node.name
                    ),
                    node,
                ),
                filter(lambda node: hasattr(node, "name"), mod.body),
            )
        )

    for name, symbol in no_magic_dir2attr(obj).items():
        fq = "{current_module}.{name}".format(current_module=current_module, name=name)
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


def emit_file_on_hierarchy(
    name_orig_ir,
    emit_name,
    module_name,
    new_module_name,
    mock_imports,
    filesystem_layout,
    output_directory,
    no_word_wrap,
    dry_run,
):
    """
    Generate Java-package—or match input—style file hierarchy from fully-qualified module name

    :param name_orig_ir: FQ module name, original filename path, IR
    :type name_orig_ir: ```Tuple[str, str, dict]```

    :param emit_name: What type(s) to generate.
    :type emit_name: ```List[Literal["argparse", "class", "function", "json_schema",
                                     "pydantic", "sqlalchemy", "sqlalchemy_table"]]```

    :param module_name: Name of [original] module
    :type module_name: ```str```

    :param new_module_name: Name of [new] module
    :type new_module_name: ```str```

    :param mock_imports: Whether to generate mock TensorFlow imports
    :type mock_imports: ```bool```

    :param filesystem_layout: Hierarchy of folder and file names generated. "java" is file per package per name.
    :type filesystem_layout: ```Literal["java", "as_input"]```

    :param output_directory: Where to place the generated exposed interfaces to the given `--module`.
    :type output_directory: ```str```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```

    :param dry_run: Show what would be created; don't actually write to the filesystem
    :type dry_run: ```bool```

    :return: Import to generated module
    :rtype: ```ImportFrom```
    """
    mod_name, _, name = name_orig_ir[0].rpartition(".")
    original_relative_filename_path, ir = name_orig_ir[1], name_orig_ir[2]
    assert isinstance(original_relative_filename_path, str)
    pp({"ir": ir})
    assert isinstance(ir, dict)
    assert original_relative_filename_path
    if not name and ir.get("name") is not None:
        name = ir.get("name")
    mod_path = path.join(
        output_directory,
        new_module_name,
        mod_name.replace(".", path.sep),
    )
    # print("mkdir\t{mod_path!r}".format(mod_path=mod_path))
    if not path.isdir(mod_path):
        if dry_run:
            print("mkdir\t{mod_path!r}".format(mod_path=mod_path))
        else:
            makedirs(mod_path)

    init_filepath = path.join(path.dirname(mod_path), INIT_FILENAME)
    if dry_run:
        print("touch\t{init_filepath!r}".format(init_filepath=init_filepath))
    else:
        open(init_filepath, "a").close()

    emit_filename, init_filepath = (
        map(
            partial(path.join, output_directory, new_module_name),
            (
                original_relative_filename_path,
                path.join(
                    path.dirname(original_relative_filename_path),
                    INIT_FILENAME,
                ),
            ),
        )
        if filesystem_layout == "as_input"
        else map(
            partial(path.join, mod_path),
            (
                "{name}{extsep}py".format(name=name, extsep=extsep),
                INIT_FILENAME,
            ),
        )
    )
    isfile_emit_filename = symbol_in_file = path.isfile(emit_filename)
    existent_mod = None
    if isfile_emit_filename:
        with open(emit_filename, "rt") as f:
            emit_filename_contents = f.read()
        existent_mod = ast.parse(
            emit_filename_contents
        )  # Also, useful as this catches syntax errors
        symbol_in_file = any(
            filter(
                partial(eq, name),
                map(
                    attrgetter("name"),
                    filter(rpartial(hasattr, "name"), existent_mod.body),
                ),
            )
        )
    else:
        emit_filename_dir = path.dirname(emit_filename)
        if not path.isdir(emit_filename_dir):
            print(
                "mkdir\t{emit_filename_dir!r}".format(
                    emit_filename_dir=emit_filename_dir
                )
            ) if dry_run else makedirs(emit_filename_dir)

    if not symbol_in_file and (ir.get("name") or ir["params"] or ir["returns"]):
        _emit_symbol(
            name_orig_ir,
            emit_name,
            module_name,
            emit_filename,
            existent_mod,
            init_filepath,
            ir,
            isfile_emit_filename,
            name,
            mock_imports,
            no_word_wrap,
            dry_run,
        )

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


def _emit_symbol(
    name_orig_ir,
    emit_name,
    module_name,
    emit_filename,
    existent_mod,
    init_filepath,
    intermediate_repr,
    isfile_emit_filename,
    name,
    mock_imports,
    no_word_wrap,
    dry_run,
):
    """
    Emit symbol to file (or dry-run just print)

    :param name_orig_ir: FQ module name, original filename path, IR
    :type name_orig_ir: ```Tuple[str, str, dict]```

    :param emit_name: What type(s) to generate.
    :type emit_name: ```List[Literal["argparse", "class", "function", "json_schema",
                                     "pydantic", "sqlalchemy", "sqlalchemy_table"]]```

    :param module_name: Name of [original] module
    :type module_name: ```str```

    :param emit_filename: Filename to emit to
    :type emit_filename: ```str``

    :param existent_mod: The existing AST module (or None)
    :type existent_mod: ```Optional[Module]```

    :param init_filepath: The filepath of the __init__.py file
    :type init_filepath: ```str```

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param isfile_emit_filename: Whether the emit filename exists
    :type isfile_emit_filename: ```bool```

    :param name: Name of the node being generated
    :type name: ```str```

    :param mock_imports: Whether to generate mock TensorFlow imports
    :type mock_imports: ```bool```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```

    :param dry_run: Show what would be created; don't actually write to the filesystem
    :type dry_run: ```bool```

    :return: Import to generated module
    :rtype: ```ImportFrom```
    """
    emitter = (
        lambda sanitised_emit_name: getattr(
            getattr(
                getattr(
                    cdd,
                    "sqlalchemy"
                    if sanitised_emit_name == "sqlalchemy_table"
                    else sanitised_emit_name,
                ),
                "emit",
            ),
            sanitised_emit_name,
        )
    )(sanitise_emit_name(emit_name))
    gen_node = emitter(
        intermediate_repr,
        word_wrap=no_word_wrap is None,
        **dict(
            **{
                "{emit_name}_name".format(
                    emit_name="function"
                    if emit_name == "argparse"
                    else emit_name.replace("sqlalchemy_table", "table")
                ): name
            },
            **{"function_type": "static"} if emit_name == "function" else {}
        )
    )
    __all___node = Assign(
        targets=[Name("__all__", Store())],
        value=List(
            ctx=Load(),
            elts=[cdd.shared.ast_utils.set_value(name)],
            expr=None,
        ),
        expr=None,
        lineno=None,
        **cdd.shared.ast_utils.maybe_type_comment
    )
    if not isinstance(gen_node, Module):
        gen_node = Module(
            body=list(
                chain.from_iterable(
                    (
                        (
                            Expr(
                                cdd.shared.ast_utils.set_value(
                                    "\nGenerated from {module_name}.{name}\n".format(
                                        module_name=module_name,
                                        name=name_orig_ir[0],
                                    )
                                )
                            ),
                        ),
                        imports_header_ast if mock_imports else iter(()),
                        (gen_node, __all___node),
                    )
                )
            ),
            stmt=None,
            type_ignores=[],
        )
    if isfile_emit_filename:
        gen_node = cdd.shared.ast_utils.merge_modules(existent_mod, gen_node)
        cdd.shared.ast_utils.merge_assignment_lists(gen_node, "__all__")
    if dry_run:
        print("write\t{emit_filename!r}".format(emit_filename=emit_filename))
    else:
        cdd.shared.emit.file.file(gen_node, filename=emit_filename, mode="wt")
    if name != "__init__" and not path.isfile(init_filepath):
        if dry_run:
            print("write\t{emit_filename!r}".format(emit_filename=emit_filename))
        else:
            cdd.shared.emit.file.file(
                Module(
                    body=[
                        Expr(
                            cdd.shared.ast_utils.set_value(
                                "\n__init__ to expose internals of this module\n"
                            )
                        ),
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


def emit_files_from_module_and_return_imports(
    module_name,
    module_root_dir,
    new_module_name,
    emit_name,
    module,
    output_directory,
    mock_imports,
    no_word_wrap,
    dry_run,
    filesystem_layout,
):
    """
    Emit type `emit_name` of all files in `module_root_dir` into `output_directory`
    on `new_module_name` hierarchy. Then return the new imports.

    :param module_name: Name of existing module
    :type module_name: ```str```

    :param module_root_dir: Root dir of existing module
    :type module_root_dir: ```str```

    :param new_module_name: New module name
    :type new_module_name: ```str```

    :param emit_name: What type(s) to generate.
    :type emit_name: ```List[Literal["argparse", "class", "function", "json_schema",
                                     "pydantic", "sqlalchemy", "sqlalchemy_table"]]```

    :param module: Module itself
    :type module: ```Module```

    :param output_directory: Where to place the generated exposed interfaces to the given `--module`.
    :type output_directory: ```str```

    :param mock_imports: Whether to generate mock TensorFlow imports
    :type mock_imports: ```bool```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```

    :param dry_run: Show what would be created; don't actually write to the filesystem
    :type dry_run: ```bool```

    :param filesystem_layout: Hierarchy of folder and file names generated. "java" is file per package per name.
    :type filesystem_layout: ```Literal["java", "as_input"]```

    :return: List of `ImportFrom` refering to generated module
    :rtype: ```List[ImportFrom]```
    """
    _emit_file_on_hierarchy = partial(
        emit_file_on_hierarchy,
        emit_name=emit_name,
        module_name=module_name,
        new_module_name=new_module_name,
        mock_imports=mock_imports,
        filesystem_layout=filesystem_layout,
        output_directory=output_directory,
        no_word_wrap=no_word_wrap,
        dry_run=dry_run,
    )
    print("emit_files_from_module_and_return_imports::module_name:", module_name, ";")
    print("emit_files_from_module_and_return_imports::module:", module, ";")
    print(
        "emit_files_from_module_and_return_imports::module_root_dir:",
        module_root_dir,
        ";\n",
    )
    # Might need some `groupby` in case multiple files are in the one project; same for `get_module_contents`
    return list(
        map(
            _emit_file_on_hierarchy,
            map(
                lambda name_source: (
                    name_source[0],
                    module_root_dir
                    if path.isfile(module_root_dir)
                    else (
                        lambda filename: filename[len(module_name) + 1 :]
                        if filename.startswith(module_name)
                        else filename
                    )(relative_filename(getfile(name_source[1]))),
                    {"params": OrderedDict(), "returns": OrderedDict()}
                    if dry_run
                    else (
                        cdd.class_.parse.class_(
                            name_source[1], merge_inner_function="__init__"
                        )
                        if not isinstance(name_source[1], FunctionDef)
                        else (
                            print(
                                'get_parser(name_source[1], "infer"):',
                                get_parser(name_source[1], "infer"),
                                ";",
                            )
                            or get_parser(name_source[1], "infer")
                        )
                    )(name_source[1]),
                ),
                map(
                    lambda name_source: (
                        name_source[0][len(module_name) + 1 :],
                        name_source[1],
                    ),
                    get_module_contents(
                        module, module_root_dir=module_root_dir
                    ).items(),
                ),
            ),
        ),
    )


def emit_files_from_module_and_return_imports2(
    module_name,
    module_root_dir,
    new_module_name,
    emit_name,
    module,
    output_directory,
    mock_imports,
    no_word_wrap,
    dry_run,
    filesystem_layout,
):
    """
    Emit type `emit_name` of all files in `module_root_dir` into `output_directory`
    on `new_module_name` hierarchy. Then return the new imports.

    :param module_name: Name of existing module
    :type module_name: ```str```

    :param module_root_dir: Root dir of existing module
    :type module_root_dir: ```str```

    :param new_module_name: New module name
    :type new_module_name: ```str```

    :param emit_name: What type(s) to generate.
    :type emit_name: ```List[Literal["argparse", "class", "function", "json_schema",
                                     "pydantic", "sqlalchemy", "sqlalchemy_table"]]```

    :param module: Module itself
    :type module: ```Module```

    :param output_directory: Where to place the generated exposed interfaces to the given `--module`.
    :type output_directory: ```str```

    :param mock_imports: Whether to generate mock TensorFlow imports
    :type mock_imports: ```bool```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```

    :param dry_run: Show what would be created; don't actually write to the filesystem
    :type dry_run: ```bool```

    :param filesystem_layout: Hierarchy of folder and file names generated. "java" is file per package per name.
    :type filesystem_layout: ```Literal["java", "as_input"]```

    :return: List of `ImportFrom` refering to generated module
    :rtype: ```List[ImportFrom]```
    """
    _emit_file_on_hierarchy = partial(
        emit_file_on_hierarchy,
        emit_name=emit_name,
        module_name=module_name,
        new_module_name=new_module_name,
        mock_imports=mock_imports,
        filesystem_layout=filesystem_layout,
        output_directory=output_directory,
        no_word_wrap=no_word_wrap,
        dry_run=dry_run,
    )
    # Might need some `groupby` in case multiple files are in the one project; same for `get_module_contents`
    return list(
        map(
            _emit_file_on_hierarchy,
            map(
                lambda name_source: (
                    name_source[0],
                    (
                        lambda filename: filename[len(module_name) + 1 :]
                        if filename.startswith(module_name)
                        else filename
                    )(relative_filename(getfile(name_source[1]))),
                    {"params": OrderedDict(), "returns": OrderedDict()}
                    if dry_run
                    else cdd.class_.parse.class_(
                        name_source[1], merge_inner_function="__init__"
                    ),
                ),
                map(
                    lambda name_source: (
                        name_source[0][len(module_name) + 1 :],
                        name_source[1],
                    ),
                    get_module_contents(
                        module, module_root_dir=module_root_dir
                    ).items(),
                ),
            ),
        ),
    )


__all__ = [
    "get_module_contents",
    "emit_file_on_hierarchy",
    "emit_files_from_module_and_return_imports",
]
