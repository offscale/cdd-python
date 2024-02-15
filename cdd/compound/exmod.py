"""
Not a dead module
"""

import typing
from ast import (
    Assign,
    Expr,
    Import,
    ImportFrom,
    List,
    Load,
    Module,
    Name,
    Store,
    alias,
    parse,
)
from collections import deque
from functools import partial, reduce
from itertools import chain, groupby
from operator import attrgetter, itemgetter
from os import makedirs, mkdir, path
from typing import Optional, Tuple, cast

from setuptools import find_packages

import cdd.class_.parse
import cdd.compound.exmod_utils
import cdd.shared.emit.file
from cdd.shared.ast_utils import (
    construct_module_with_symbols,
    deduplicate_sorted_imports,
    maybe_type_comment,
    merge_modules,
    module_to_all,
    set_value,
)
from cdd.shared.pure_utils import (
    INIT_FILENAME,
    PY_GTE_3_8,
    find_module_filepath,
    read_file_to_str,
    rpartial,
)
from cdd.shared.source_transformer import ast_parse, to_code
from cdd.sqlalchemy.utils.emit_utils import (
    generate_create_tables_mod,
    mock_engine_base_metadata_str,
)

if PY_GTE_3_8:
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


def exmod(
    emit_name,
    module,
    blacklist,
    whitelist,
    output_directory,
    target_module_name,
    mock_imports,
    emit_sqlalchemy_submodule,
    extra_modules,
    no_word_wrap,
    recursive,
    dry_run,
    filesystem_layout="as_input",
    extra_modules_to_all=None,
):
    """
    Expose module as `emit` types into `output_directory`

    :param emit_name: What type(s) to generate.
    :type emit_name: ```list[Literal["argparse", "class", "function", "json_schema",
                                     "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]]```

    :param module: Module name or path
    :type module: ```str```

    :param blacklist: Modules/FQN to omit. If unspecified will emit all (unless whitelist).
    :type blacklist: ```Union[list[str], tuple[str]]```

    :param whitelist: Modules/FQN to emit. If unspecified will emit all (minus blacklist).
    :type whitelist: ```Union[list[str], tuple[str]]```

    :param output_directory: Where to place the generated exposed interfaces to the given `--module`.
    :type output_directory: ```str```

    :param target_module_name: Target module name
    :type target_module_name: ```Optional[str]```

    :param mock_imports: Whether to generate mock TensorFlow imports
    :type mock_imports: ```bool```

    :param emit_sqlalchemy_submodule: Whether to emit submodule "sqlalchemy_mod/{__init__,connection,create_tables}.py"
    :type emit_sqlalchemy_submodule: ```bool```

    :param extra_modules: Additional module(s) to expose; specifiable multiple times. Prepended to symbol auto-importer
    :type extra_modules: ```Optional[List[str]]```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```

    :param recursive: Recursively traverse module hierarchy and recreate hierarchy with exposed interfaces
    :type recursive: ```bool```

    :param dry_run: Show what would be created; don't actually write to the filesystem
    :type dry_run: ```bool```

    :param filesystem_layout: Hierarchy of folder and file names generated. "java" is file per package per name.
    :type filesystem_layout: ```Literal["java", "as_input"]```

    :param extra_modules_to_all: Internal arg. Prepended to symbol resolver. E.g., `(("ast", {"List"}),)`.
    :type extra_modules_to_all: ```Optional[tuple[tuple[str, frozenset], ...]]```
    """
    output_directory = path.realpath(output_directory)
    extra_modules_to_all = (
        cdd.shared.ast_utils.module_to_all(extra_modules)
        if extra_modules is not None and extra_modules_to_all is None
        else tuple()
    )  # type: tuple[tuple[str, frozenset], ...]
    if not isinstance(emit_name, str):
        deque(
            map(
                partial(
                    exmod,
                    module=module,
                    blacklist=blacklist,
                    whitelist=whitelist,
                    mock_imports=mock_imports,
                    filesystem_layout=filesystem_layout,
                    emit_sqlalchemy_submodule=emit_sqlalchemy_submodule,
                    extra_modules=extra_modules,
                    no_word_wrap=no_word_wrap,
                    output_directory=output_directory,
                    target_module_name=target_module_name,
                    recursive=recursive,
                    dry_run=dry_run,
                    extra_modules_to_all=extra_modules_to_all,
                ),
                emit_name or iter(()),
            ),
            maxlen=0,
        )
    elif dry_run:
        print(
            "mkdir\t'{output_directory}'".format(
                output_directory=path.normcase(output_directory)
            ),
            file=cdd.compound.exmod_utils.EXMOD_OUT_STREAM,
        )
    elif not path.isdir(output_directory):
        makedirs(output_directory)

    emit_name: Optional[str] = (
        emit_name[0]
        if emit_name is not None
        and len(emit_name) == 1
        and isinstance(emit_name, (list, tuple))
        else emit_name
    )
    assert isinstance(
        emit_name, (str, type(None))
    ), "Expected `str` got `{emit_name_type!r}`".format(emit_name_type=type(emit_name))

    module_root, _, submodule = module.rpartition(".")
    module_name, new_module_name = (
        module,
        (
            target_module_name or "___".join((module_root, "gold"))
            if module_root
            else "gold"
        ),
    )

    sqlalchemy_mod: str = "sqlalchemy_mod"
    sqlalchemy_mod_dir_join: typing.Union[
        typing.Callable[[str], str], typing.Callable[[], str]
    ] = partial(path.join, output_directory, "sqlalchemy_mod")
    sqlalchemy_mod_dir = sqlalchemy_mod_dir_join()
    make_sqlalchemy_mod: bool = (
        emit_name in frozenset(("sqlalchemy", "sqlalchemy_hybrid", "sqlalchemy_table"))
        and emit_sqlalchemy_submodule
        and not path.isdir(sqlalchemy_mod_dir)
    )
    if make_sqlalchemy_mod:
        extra_modules_to_all = _create_sqlalchemy_mod(
            extra_modules_to_all,
            output_directory,
            sqlalchemy_mod,
            sqlalchemy_mod_dir,
            sqlalchemy_mod_dir_join,
        )
    try:
        module_root_dir: str = path.dirname(
            find_module_filepath(
                *(module_root, submodule) if module_root else (module_name, None)
            )
        )
    except AssertionError as e:
        raise ModuleNotFoundError(e)

    _exmod_single_folder = partial(
        exmod_single_folder,
        emit_name=emit_name,
        blacklist=blacklist,
        whitelist=whitelist,
        mock_imports=mock_imports,
        no_word_wrap=no_word_wrap,
        dry_run=dry_run,
        module_root=module_root,
        new_module_name=new_module_name,
        filesystem_layout=filesystem_layout,
        extra_modules_to_all=extra_modules_to_all,
        first_output_directory=output_directory,
    )
    packages: typing.List[str] = find_packages(
        module_root_dir,
        include=whitelist if whitelist else ("*",),
        exclude=blacklist if blacklist else iter(()),
    )

    _exmod_single_folder(
        module=module,
        module_name=module_name,
        module_root_dir=module_root_dir,
        output_directory=output_directory,
    )
    output_directory_basename = path.basename(output_directory)
    imports = (
        [output_directory_basename] if make_sqlalchemy_mod else None
    )  # type: Optional[list[str]]
    _exmod_single_folder_kwargs: Tuple[
        TypedDict(
            "_exmod_single_folder_kwargs",
            {
                "module": module,
                "module_name": module_name,
                "module_root_dir": module_root_dir,
                "output_directory": output_directory,
            },
        )
    ] = tuple(
        chain.from_iterable(
            (
                (
                    {
                        "module": module,
                        "module_name": module_name,
                        "module_root_dir": module_root_dir,
                        "output_directory": output_directory,
                    },
                ),
                (
                    map(
                        lambda package: (
                            lambda pkg_relative_dir: (
                                imports is not None
                                and imports.append(
                                    path.join(
                                        output_directory_basename,
                                        pkg_relative_dir,
                                    ).replace(path.sep, ".")
                                )
                                or {
                                    "module": ".".join((module, package)),
                                    "module_name": package,
                                    "module_root_dir": path.join(
                                        module_root_dir, pkg_relative_dir
                                    ),
                                    "output_directory": path.join(
                                        output_directory, pkg_relative_dir
                                    ),
                                }
                            )
                        )(package.replace(".", path.sep)),
                        packages,
                    )
                    if recursive
                    else iter(())
                ),
            )
        )
    )

    if make_sqlalchemy_mod:
        _add_imports_to_sqlalchemy_create_all(imports, sqlalchemy_mod_dir_join)

    # This could be executed in parallel for efficiency
    deque(
        map(
            lambda kwargs: _exmod_single_folder(**kwargs),
            _exmod_single_folder_kwargs,
        ),
        maxlen=0,
    )

    return


def _add_imports_to_sqlalchemy_create_all(imports, sqlalchemy_mod_dir_join):
    """
    Internal function to update the "create_all.py" file in the generated SQLalchemy module.

    :param imports: Collection of names to `import {name}`
    :type imports: ```Iterable[str]```

    :param sqlalchemy_mod_dir_join: `path.join` partial on `sqlalchemy_mod_dir`
    :type sqlalchemy_mod_dir_join: ```: typing.Union[
        typing.Callable[[str], str], typing.Callable[[], str]
    ]```
    """
    create_table_filepath = sqlalchemy_mod_dir_join(
        "create_tables{extsep}py".format(extsep=path.extsep)
    )
    with open(create_table_filepath, "rt") as f:
        create_table_mod = ast_parse(f.read(), filename=create_table_filepath)
    first_import_idx: int = next(
        idx
        for idx, node in enumerate(create_table_mod.body)
        if isinstance(node, (Import, ImportFrom))
    )
    create_table_mod.body = (
        create_table_mod.body[:first_import_idx]
        + [
            Import(
                names=[
                    alias(
                        name=name,
                        asname=None,
                        identifier=None,
                        identifier_name=None,
                    )
                ],
                alias=None,
            )
            for name in imports
        ]
        + create_table_mod.body[first_import_idx:]
    )
    with open(create_table_filepath, "wt") as f:
        f.write(to_code(create_table_mod))


def _create_sqlalchemy_mod(
    extra_modules_to_all,
    output_directory,
    sqlalchemy_mod,
    sqlalchemy_mod_dir,
    sqlalchemy_mod_dir_join,
):
    """
    Internal function to create the SQLalchemy module.

    :param extra_modules_to_all: Internal arg. Prepended to symbol resolver. E.g., `(("ast", {"List"}),)`.
    :type extra_modules_to_all: ```Optional[tuple[tuple[str, frozenset], ...]]```

    :param output_directory: Where to place the generated exposed interfaces to the given `--module`.
    :type output_directory: ```str```

    :param sqlalchemy_mod: module name
    :type sqlalchemy_mod: ```str```

    :param sqlalchemy_mod_dir: SQLalchemy module root directory
    :type sqlalchemy_mod_dir: ```str```

    :param sqlalchemy_mod_dir_join: `path.join` partial on `sqlalchemy_mod_dir`
    :type sqlalchemy_mod_dir_join: ```: typing.Union[
        typing.Callable[[str], str], typing.Callable[[], str]
    ]```

    :return: Updated `extra_modules_to_all`
    :rtype: ```Optional[tuple[tuple[str, frozenset], ...]]```
    """
    mkdir(sqlalchemy_mod_dir)
    open(
        sqlalchemy_mod_dir_join(INIT_FILENAME),
        "a",
    ).close()
    connection_py = "connection"
    connection_filepath = sqlalchemy_mod_dir_join(
        "{name}{extsep}py".format(name=connection_py, extsep=path.extsep)
    )
    with open(connection_filepath, "wt") as f:
        f.write(mock_engine_base_metadata_str)
    sqlalchemy_module_name = ".".join((path.basename(output_directory), sqlalchemy_mod))
    sqlalchemy_module_name_connection_py = ".".join(
        (sqlalchemy_module_name, connection_py)
    )
    sqlalchemy_module_name_create_table = ".".join(
        (sqlalchemy_module_name, "create_tables")
    )
    create_table_filepath = sqlalchemy_mod_dir_join(
        "create_tables{extsep}py".format(extsep=path.extsep)
    )
    with open(create_table_filepath, "wt") as f:
        f.write(
            to_code(generate_create_tables_mod(sqlalchemy_module_name_connection_py))
        )
    extra_modules_to_all = (
        (
            sqlalchemy_module_name_connection_py,
            frozenset(module_to_all(connection_filepath)),
        ),
        (
            sqlalchemy_module_name_create_table,
            frozenset(module_to_all(create_table_filepath)),
        ),
    ) + extra_modules_to_all
    return extra_modules_to_all


def exmod_single_folder(
    emit_name,
    module,
    blacklist,
    whitelist,
    output_directory,
    first_output_directory,
    mock_imports,
    no_word_wrap,
    dry_run,
    module_root_dir,
    module_root,
    module_name,
    new_module_name,
    filesystem_layout,
    extra_modules_to_all,
):
    """
    Expose module as `emit` types into `output_directory`. Single folder (non-recursive).

    :param emit_name: What type(s) to generate.
    :type emit_name: ```list[Literal["argparse", "class", "function", "json_schema",
                                     "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]]```

    :param module: Module name or path
    :type module: ```str```

    :param blacklist: Modules/FQN to omit. If unspecified will emit all (unless whitelist).
    :type blacklist: ```Union[list[str], tuple[str]]```

    :param whitelist: Modules/FQN to emit. If unspecified will emit all (minus blacklist).
    :type whitelist: ```Union[list[str], tuple[str]]```

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

    :param module_root_dir:
    :type module_root_dir: ```str```

    :param module_root:
    :type module_root: ```str```

    :param module_name:
    :type module_name: ```str```

    :param new_module_name:
    :type new_module_name: ```str```

    :param filesystem_layout: Hierarchy of folder and file names generated. "java" is file per package per name.
    :type filesystem_layout: ```Literal["java", "as_input"]```

    :param extra_modules_to_all: Internal arg. Prepended to symbol resolver. E.g., `(("ast", {"List"}),)`.
    :type extra_modules_to_all: ```Optional[tuple[tuple[str, frozenset], ...]]```
    """
    mod_path: str = (
        module_name
        if module_name.startswith(module_root + ".")
        else ".".join((module_root, module_name))
    )
    blacklist, whitelist = map(
        frozenset, (blacklist or iter(()), whitelist or iter(()))
    )
    proceed: bool = any(
        (
            sum(map(len, (blacklist, whitelist))) == 0,
            mod_path not in blacklist and (mod_path in whitelist or not whitelist),
        )
    )
    if not proceed:
        return

    _emit_files_from_module_and_return_imports = partial(
        cdd.compound.exmod_utils.emit_files_from_module_and_return_imports,
        new_module_name=new_module_name,
        emit_name=emit_name,
        output_directory=output_directory,
        first_output_directory=first_output_directory,
        mock_imports=mock_imports,
        no_word_wrap=no_word_wrap,
        dry_run=dry_run,
        filesystem_layout=filesystem_layout,
        extra_modules_to_all=extra_modules_to_all,
    )

    imports = _emit_files_from_module_and_return_imports(
        module_name=module_name, module=module, module_root_dir=module_root_dir
    )  # type: Optional[list[ImportFrom]]
    if not imports:
        # Case: no obvious folder hierarchy, so parse the `__init__` file in root
        top_level_init = path.join(module_root_dir, INIT_FILENAME)
        with open(top_level_init, "rt") as f:
            mod: Module = parse(f.read(), filename=top_level_init)

        # TODO: Optimise these imports
        imports = list(
            chain.from_iterable(
                map(
                    lambda filepath_name_module: _emit_files_from_module_and_return_imports(
                        module_root_dir=filepath_name_module[0],
                        module_name=filepath_name_module[1],
                        module=filepath_name_module[2],
                    ),
                    map(
                        lambda filepath2modname_group: (
                            filepath2modname_group[0][0],
                            filepath2modname_group[0][1],
                            reduce(
                                partial(merge_modules, deduplicate_names=True),
                                map(itemgetter(1), filepath2modname_group[1]),
                            ),
                        ),
                        groupby(
                            sorted(
                                map(
                                    lambda import_from: (
                                        (
                                            lambda module_filepath: (
                                                (module_filepath, import_from.module),
                                                construct_module_with_symbols(
                                                    parse(
                                                        read_file_to_str(
                                                            module_filepath
                                                        )
                                                    ),
                                                    map(
                                                        attrgetter("name"),
                                                        import_from.names,
                                                    ),
                                                ),
                                            )
                                        )(
                                            find_module_filepath(
                                                *import_from.module.rsplit(".", 1)
                                            )
                                        )
                                    ),
                                    filter(rpartial(isinstance, ImportFrom), mod.body),
                                ),
                                key=itemgetter(0),
                            ),
                            key=itemgetter(0),
                        ),
                    ),
                )
            )
        )  # type: list[ImportFrom]

    # assert imports, "Module contents are empty at {!r}".format(module_root_dir)
    modules_names: Tuple[str, ...] = cast(
        Tuple[str, ...],
        tuple(
            map(
                lambda name_module: (
                    name_module[0],
                    tuple(map(itemgetter(1), name_module[1])),
                ),
                groupby(
                    map(
                        lambda node_mod: (
                            node_mod[0],
                            node_mod[2].module,
                        ),
                        imports,
                    ),
                    itemgetter(0),
                ),
            )
        ),
    )

    init_filepath: str = path.join(
        output_directory,
        *(
            (INIT_FILENAME,)
            if output_directory.endswith(
                "{}{}".format(path.sep, new_module_name.replace(".", path.sep))
            )
            else (new_module_name, INIT_FILENAME)
        )
    )
    if dry_run:
        print(
            "write\t'{init_filepath}'".format(
                init_filepath=path.normcase(init_filepath)
            ),
            file=cdd.compound.exmod_utils.EXMOD_OUT_STREAM,
        )
    else:
        makedirs(path.dirname(init_filepath), exist_ok=True)
        cdd.shared.emit.file.file(
            deduplicate_sorted_imports(
                Module(
                    body=list(
                        chain.from_iterable(
                            (
                                (
                                    Expr(
                                        set_value("\nExport internal imports\n"),
                                        lineno=None,
                                        col_offset=None,
                                    ),
                                ),
                                map(
                                    lambda module_names: ImportFrom(
                                        module=module_names[0],
                                        names=list(
                                            map(
                                                lambda names: alias(
                                                    names,
                                                    None,
                                                    identifier=None,
                                                    identifier_name=None,
                                                ),
                                                module_names[1],
                                            )
                                        ),
                                        level=1,
                                        identifier=None,
                                    ),
                                    modules_names,
                                ),
                                (
                                    Assign(
                                        targets=[
                                            Name(
                                                "__all__",
                                                Store(),
                                                lineno=None,
                                                col_offset=None,
                                            )
                                        ],
                                        value=List(
                                            ctx=Load(),
                                            elts=list(
                                                map(
                                                    set_value,
                                                    sorted(
                                                        frozenset(
                                                            chain.from_iterable(
                                                                map(
                                                                    itemgetter(1),
                                                                    modules_names,
                                                                )
                                                            ),
                                                        )
                                                    ),
                                                )
                                            ),
                                            expr=None,
                                        ),
                                        expr=None,
                                        lineno=None,
                                        **maybe_type_comment
                                    ),
                                ),
                            )
                        )
                    ),
                    stmt=None,
                    type_ignores=[],
                )
            ),
            init_filepath,
            mode="wt",
        )


__all__ = ["exmod"]  # type: list[str]
