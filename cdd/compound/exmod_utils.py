"""Exmod utils"""

import sys
from ast import AST, Assign, Expr, ImportFrom, List, Load, Module, Name, Store, alias
from ast import walk as ast_walk
from collections import OrderedDict, defaultdict, deque
from functools import partial
from inspect import getfile
from itertools import chain
from operator import attrgetter, eq
from os import environ, extsep, makedirs, path
from typing import Any, Dict, Optional, TextIO, cast

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
    read_file_to_str,
    rpartial,
    sanitise_emit_name,
)
from cdd.shared.source_transformer import ast_parse
from cdd.tests.mocks import imports_header_ast

EXMOD_OUT_STREAM: TextIO = getattr(sys, environ.get("EXMOD_OUT_STREAM", "stdout"))


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
    module_root_dir_init: str = path.join(module_root_dir, INIT_FILENAME)
    # process_module_contents = partial(
    #     _process_module_contents,
    #     _result=_result,
    #     current_module=current_module,
    #     module_root_dir=module_root_dir,
    # )
    if path.isfile(module_root_dir):
        with open(module_root_dir, "rt") as f:
            mod: Module = ast_parse(
                f.read(), filename=module_root_dir, skip_docstring_remit=True
            )

        # Bring in imported symbols that should be exposed based on `__all__`
        all_magic_var = next(
            map(
                lambda assign: frozenset(
                    map(cdd.shared.ast_utils.get_value, assign.value.elts)
                ),
                filter(
                    lambda assign: len(assign.targets) == 1
                    and isinstance(assign.targets[0], Name)
                    and assign.targets[0].id == "__all__",
                    filter(rpartial(isinstance, Assign), mod.body),
                ),
            ),
            iter(()),
        )  # type: Union[list[str], Iterator]
        mod_to_symbol: defaultdict[Any, list] = defaultdict(list)
        deque(
            (
                mod_to_symbol[import_from.module].append(name.name)
                for import_from in filter(
                    rpartial(isinstance, ImportFrom), ast_walk(mod)
                )
                for name in import_from.names
                if name.asname is None
                and name.name in all_magic_var
                or name.asname in all_magic_var
            ),
            maxlen=0,
        )
        res: Dict[str, AST] = {
            "{module_name}{submodule_name}.{node_name}".format(
                module_name="{}.".format(module_name) if module_name else "",
                submodule_name=submodule_name,
                node_name=node.name,
            ): node
            for module_name, submodule_names in mod_to_symbol.items()
            for submodule_name in submodule_names
            for node in (
                lambda module_filepath: (
                    iter(())
                    if module_filepath is None
                    else ast_parse(
                        read_file_to_str(module_filepath),
                        module_filepath,
                        skip_docstring_remit=True,
                    ).body
                )
            )(
                cdd.shared.pure_utils.find_module_filepath(
                    module_name, submodule_name, none_when_no_spec=True
                )
            )
            if hasattr(node, "name")
        }
        res.update(
            dict(
                map(
                    lambda node: (
                        (
                            node.name
                            if current_module is None
                            else "{current_module}.{name}".format(
                                current_module=current_module, name=node.name
                            )
                        ),
                        node,
                    ),
                    filter(lambda node: hasattr(node, "name"), mod.body),
                )
            )
        )
        return res
    elif path.isfile(module_root_dir_init):
        return get_module_contents(
            obj=obj,
            module_root_dir=module_root_dir_init,
            current_module=current_module,
            _result=_result,
        )
    # assert not isinstance(
    #     obj, (int, float, complex, str, bool, type(None))
    # ), "module is unexpected type: {!r}".format(type(obj).__name__)
    # for name, symbol in no_magic_or_builtin_dir2attr(obj).items():
    #     process_module_contents(name=name, symbol=symbol)
    return _result


# def _process_module_contents(_result, current_module, module_root_dir, name, symbol):
#     """
#     Internal function to get the symbol and store it with a fully-qualified name in `_result`
#
#     :param current_module: The current module
#     :type current_module: ```Optional[str]```
#
#     :param module_root_dir: Root of module
#     :type module_root_dir: ```str```
#
#     :param name: Name—first value—from `dir(module)`
#     :type name: ```str```
#
#     :param symbol: Symbol—second value—from `dir(module)`
#     :type symbol: ```type```
#     """
#     fq: str = "{current_module}.{name}".format(current_module=current_module, name=name)
#     try:
#         symbol_location: Optional[str] = getfile(symbol)
#     except TypeError:
#         symbol_location: Optional[str] = None
#     if symbol_location is not None and symbol_location.startswith(module_root_dir):
#         if isinstance(symbol, type):
#             _result[fq] = symbol
#         elif (
#             current_module != getattr(symbol, "__name__", current_module)
#             and ismodule(symbol)
#             and fq not in _result
#         ):
#             get_module_contents(
#                 symbol,
#                 module_root_dir=module_root_dir,
#                 current_module=symbol.__name__,
#             )


def emit_file_on_hierarchy(
    name_orig_ir,
    emit_name,
    module_name,
    new_module_name,
    mock_imports,
    filesystem_layout,
    extra_modules_to_all,
    output_directory,
    first_output_directory,
    no_word_wrap,
    dry_run,
):
    """
    Generate Java-package—or match input—style file hierarchy from fully-qualified module name

    :param name_orig_ir: FQ module name, original filename path, IR
    :type name_orig_ir: ```tuple[str, str, dict]```

    :param emit_name: What type(s) to generate.
    :type emit_name: ```list[Literal["argparse", "class", "function", "json_schema",
                                     "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]]```

    :param module_name: Name of [original] module
    :type module_name: ```str```

    :param new_module_name: Name of [new] module
    :type new_module_name: ```str```

    :param mock_imports: Whether to generate mock TensorFlow imports
    :type mock_imports: ```bool```

    :param filesystem_layout: Hierarchy of folder and file names generated. "java" is file per package per name.
    :type filesystem_layout: ```Literal["java", "as_input"]```

    :param extra_modules_to_all: Internal arg. Prepended to symbol resolver. E.g., `(("ast", {"List"}),)`.
    :type extra_modules_to_all: ```Optional[tuple[tuple[str, frozenset], ...]]```

    :param output_directory: Where to place the generated exposed interfaces to the given `--module`.
    :type output_directory: ```str```

    :param first_output_directory: Initial output directory (e.g., direct from `--output-directory`)
    :type first_output_directory: ```str```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```

    :param dry_run: Show what would be created; don't actually write to the filesystem
    :type dry_run: ```bool```

    :return: (mod_name or None, relative_filename_path, ImportFrom) to generated module
    :rtype: ```Optional[Tuple[Optional[str], str, ImportFrom]]```
    """
    mod_name, _, name = name_orig_ir[0].rpartition(".")
    original_relative_filename_path, ir = name_orig_ir[1], name_orig_ir[2]
    assert original_relative_filename_path

    relative_filename_path: str = original_relative_filename_path
    module_name_as_path: str = module_name.replace(".", path.sep)
    if relative_filename_path.startswith(module_name_as_path + path.sep):
        relative_filename_path: str = relative_filename_path[
            len(module_name_as_path + path.sep) :
        ]
    if not name and ir.get("name") is not None:
        name: Optional[str] = ir.get("name")

    output_dir_is_module: bool = output_directory.replace(path.sep, ".").endswith(
        new_module_name
    )
    mod_path: str = path.join(
        output_directory,
        *(
            ()
            if output_dir_is_module
            else (new_module_name, mod_name.replace(".", path.sep))
        )
    )
    # print("mkdir\t{mod_path!r}".format(mod_path=mod_path), file=EXMOD_OUT_STREAM)
    if not path.isdir(mod_path):
        if dry_run:
            print(
                "mkdir\t'{mod_path}'".format(mod_path=path.normcase(mod_path)),
                file=EXMOD_OUT_STREAM,
            )
        else:
            makedirs(mod_path)

    init_filepath: str = path.join(path.dirname(mod_path), INIT_FILENAME)
    if dry_run:
        print(
            "touch\t'{init_filepath}'".format(
                init_filepath=path.normcase(init_filepath)
            ),
            file=EXMOD_OUT_STREAM,
        )
    else:
        open(init_filepath, "a").close()

    emit_filename, init_filepath = (
        map(
            partial(
                path.join,
                output_directory,
                *() if output_dir_is_module else (new_module_name,)
            ),
            (
                relative_filename_path,
                path.join(
                    path.dirname(relative_filename_path),
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
    symbol_in_file: bool = path.isfile(emit_filename)
    isfile_emit_filename: bool = symbol_in_file
    existent_mod: Optional[Module] = None
    if isfile_emit_filename:
        with open(emit_filename, "rt") as f:
            emit_filename_contents: str = f.read()
        existent_mod: Module = ast_parse(
            emit_filename_contents, skip_docstring_remit=True, filename=emit_filename
        )  # Also, useful as this catches syntax errors
        symbol_in_file: bool = any(
            filter(
                partial(eq, name),
                map(
                    attrgetter("name"),
                    filter(rpartial(hasattr, "name"), existent_mod.body),
                ),
            )
        )
    else:
        emit_filename_dir: str = path.dirname(emit_filename)
        if not path.isdir(emit_filename_dir):
            (
                print(
                    "mkdir\t'{emit_filename_dir}'".format(
                        emit_filename_dir=path.normcase(emit_filename_dir)
                    ),
                    file=EXMOD_OUT_STREAM,
                )
                if dry_run
                else makedirs(emit_filename_dir)
            )

    if not symbol_in_file and (ir.get("name") or ir["params"] or ir["returns"]):
        _emit_symbol(
            name_orig_ir=name_orig_ir,
            emit_name=emit_name,
            module_name=module_name,
            emit_filename=emit_filename,
            existent_mod=existent_mod,
            init_filepath=init_filepath,
            intermediate_repr=ir,
            isfile_emit_filename=isfile_emit_filename,
            name=name,
            mock_imports=mock_imports,
            extra_modules_to_all=extra_modules_to_all,
            no_word_wrap=no_word_wrap,
            first_output_directory=first_output_directory,
            dry_run=dry_run,
        )

    # return (
    #     mod_name or None,
    #     relative_filename_path,
    #     ImportFrom(
    #         module=name,
    #         names=[
    #             alias(
    #                 name=name,
    #                 asname=None,
    #                 identifier=None,
    #                 identifier_name=None,
    #             ),
    #         ],
    #         level=1,
    #         identifier=None,
    #     ),
    # )


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
    extra_modules_to_all,
    no_word_wrap,
    first_output_directory,
    dry_run,
):
    """
    Emit symbol to file (or dry-run just print)

    :param name_orig_ir: FQ module name, original filename path, IR
    :type name_orig_ir: ```tuple[str, str, dict]```

    :param emit_name: What type(s) to generate.
    :type emit_name: ```list[Literal["argparse", "class", "function", "json_schema",
                                     "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]]```

    :param module_name: Name of [original] module
    :type module_name: ```str```

    :param emit_filename: Filename to emit to
    :type emit_filename: ```str``

    :param existent_mod: The existing AST module (or None)
    :type existent_mod: ```Optional[Module]```

    :param init_filepath: The filepath of the __init__.py file
    :type init_filepath: ```str```

    :param intermediate_repr: a dictionary consistent with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    :type intermediate_repr: ```dict```

    :param isfile_emit_filename: Whether the emit filename exists
    :type isfile_emit_filename: ```bool```

    :param name: Name of the node being generated
    :type name: ```str```

    :param mock_imports: Whether to generate mock TensorFlow imports
    :type mock_imports: ```bool```

    :param extra_modules_to_all: Additional module(s) to expose. Prepended to symbol resolver. `(("ast", {"List"}),)`.
    :type extra_modules_to_all: ```Optional[tuple[tuple[str, frozenset], ...]]```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```

    :param first_output_directory: Where to place the generated exposed interfaces to the given `--module`.
    :type first_output_directory: ```str```

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
                    (
                        "sqlalchemy"
                        if sanitised_emit_name
                        in frozenset(("sqlalchemy_hybrid", "sqlalchemy_table"))
                        else sanitised_emit_name
                    ),
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
                    emit_name={
                        "argparse": "function",
                        "sqlalchemy_table": "table",
                        "sqlalchemy_hybrid": "table",
                    }.get(emit_name, emit_name)
                ): name
            },
            **{"function_type": "static"} if emit_name == "function" else {}
        )
    )
    modules_to_all = (extra_modules_to_all or tuple()) + (
        cdd.shared.ast_utils.DEFAULT_MODULES_TO_ALL_SQL_FIRST
        if emit_name
        in frozenset(("sqlalchemy", "sqlalchemy_hybrid", "sqlalchemy_table"))
        else cdd.shared.ast_utils.DEFAULT_MODULES_TO_ALL
    )
    __all___node: Assign = Assign(
        targets=[Name("__all__", Store(), lineno=None, col_offset=None)],
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
        gen_node: Module = Module(
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
                                ),
                                lineno=None,
                                col_offset=None,
                            ),
                        ),
                        (
                            imports_header_ast
                            if mock_imports
                            else (
                                cdd.shared.ast_utils.infer_imports(
                                    gen_node, modules_to_all=modules_to_all
                                )
                                or iter(())
                            )
                        ),
                        (gen_node, __all___node),
                    )
                )
            ),
            stmt=None,
            type_ignores=[],
        )
    if isfile_emit_filename:
        if existent_mod is not None:
            gen_node: Module = cdd.shared.ast_utils.merge_modules(
                cast(Module, existent_mod), gen_node
            )
            inferred_imports = cdd.shared.ast_utils.infer_imports(
                gen_node, modules_to_all=modules_to_all
            )
            if inferred_imports:
                gen_node.body = list(
                    chain.from_iterable(
                        ((gen_node.body[0],), inferred_imports, gen_node.body[1:])
                    )
                )
                gen_node = cdd.shared.ast_utils.deduplicate_sorted_imports(gen_node)
        cdd.shared.ast_utils.merge_assignment_lists(gen_node, "__all__")
    if dry_run:
        print(
            "write\t{emit_filename!r}".format(emit_filename=emit_filename),
            file=EXMOD_OUT_STREAM,
        )
    else:
        cdd.shared.emit.file.file(gen_node, filename=emit_filename, mode="wt")
    if name != "__init__" and not path.isfile(init_filepath):
        if dry_run:
            print(
                "write\t{emit_filename!r}".format(emit_filename=emit_filename),
                file=EXMOD_OUT_STREAM,
            )
        else:
            module = path.splitext(
                emit_filename[len(path.dirname(first_output_directory)) + 1 :]
            )[0].replace(path.sep, ".")
            cdd.shared.emit.file.file(
                Module(
                    body=[
                        Expr(
                            cdd.shared.ast_utils.set_value(
                                "\n__init__ to expose internals of this module\n"
                            ),
                            lineno=None,
                            col_offset=None,
                        ),
                        ImportFrom(
                            module=module,
                            names=[
                                alias(
                                    name=name,
                                    asname=None,
                                    identifier=None,
                                    identifier_name=None,
                                ),
                            ],
                            level=0,
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
    first_output_directory,
    mock_imports,
    no_word_wrap,
    dry_run,
    filesystem_layout,
    extra_modules_to_all,
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
    :type emit_name: ```list[Literal["argparse", "class", "function", "json_schema",
                                     "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]]```

    :param module: Module itself
    :type module: ```Module```

    :param output_directory: Where to place the generated exposed interfaces to the given `--module`.
    :type output_directory: ```str```

    :param first_output_directory: Initial output directory (e.g., direct from `--output-directory`)
    :type first_output_directory: ```str```

    :param mock_imports: Whether to generate mock TensorFlow imports
    :type mock_imports: ```bool```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```

    :param dry_run: Show what would be created; don't actually write to the filesystem
    :type dry_run: ```bool```

    :param filesystem_layout: Hierarchy of folder and file names generated. "java" is file per package per name.
    :type filesystem_layout: ```Literal["java", "as_input"]```

    :param extra_modules_to_all: Internal arg. Prepended to symbol resolver. E.g., `(("ast", {"List"}),)`.
    :type extra_modules_to_all: ```Optional[tuple[tuple[str, frozenset], ...]]```

    :return: List of (mod_name or None, relative_filename_path, ImportFrom) to generated module(s)
    :rtype: ```list[Tuple[Optional[str], str, ImportFrom]]```
    """
    _emit_file_on_hierarchy = partial(
        emit_file_on_hierarchy,
        emit_name=emit_name,
        module_name=module_name,
        new_module_name=new_module_name,
        mock_imports=mock_imports,
        filesystem_layout=filesystem_layout,
        output_directory=output_directory,
        first_output_directory=first_output_directory,
        extra_modules_to_all=extra_modules_to_all,
        no_word_wrap=no_word_wrap,
        dry_run=dry_run,
    )

    # Might need some `groupby` in case multiple files are in the one project; same for `get_module_contents`
    return list(
        filter(
            None,
            map(
                _emit_file_on_hierarchy,
                map(
                    lambda name_source: (
                        name_source[0],
                        (
                            path.join(output_directory, path.basename(module_root_dir))
                            if path.isfile(module_root_dir)
                            else (
                                lambda filename: (
                                    filename[len(module_name) + 1 :]
                                    if filename.startswith(module_name)
                                    else filename
                                )
                            )(
                                relative_filename(
                                    name_source[1].__file__
                                    if hasattr(name_source[1], "__file__")
                                    else getfile(name_source[1])
                                )
                            )
                        ),
                        (
                            {"params": OrderedDict(), "returns": OrderedDict()}
                            if dry_run
                            else (
                                lambda parser: (
                                    partial(parser, merge_inner_function="__init__")
                                    if parser is cdd.class_.parse.class_
                                    else parser
                                )
                            )(get_parser(name_source[1], "infer"))(name_source[1])
                        ),
                    ),
                    map(
                        lambda name_source: (
                            (
                                name_source[0][len(module_name) + 1 :]
                                if name_source[0].startswith(module_name)
                                else name_source[0]
                            ),
                            name_source[1],
                        ),
                        get_module_contents(
                            module, module_root_dir=module_root_dir
                        ).items(),
                    ),
                ),
            ),
        )
    )


__all__ = [
    "_emit_symbol",
    "emit_file_on_hierarchy",
    "emit_files_from_module_and_return_imports",
    "get_module_contents",
]  # type: list[str]
