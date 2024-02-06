"""
Not a dead module
"""

import typing
from ast import Assign, Expr, ImportFrom, List, Load, Module, Name, Store, alias, parse
from collections import deque
from functools import partial, reduce
from itertools import chain, groupby
from operator import attrgetter, itemgetter
from os import makedirs, path
from typing import Iterator, Optional, Tuple, cast

from setuptools import find_packages

import cdd.class_.parse
import cdd.compound.exmod_utils
import cdd.shared.emit.file
from cdd.shared.ast_utils import (
    construct_module_with_symbols,
    maybe_type_comment,
    merge_modules,
    set_value,
)
from cdd.shared.pure_utils import (
    INIT_FILENAME,
    PY_GTE_3_8,
    find_module_filepath,
    read_file_to_str,
    rpartial,
)
from cdd.sqlalchemy.utils.emit_utils import mock_engine_base_metadata_str

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
    emit_base_engine_metadata,
    no_word_wrap,
    recursive,
    dry_run,
    filesystem_layout="as_input",
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

    :param emit_base_engine_metadata: Whether to produce a file with `Base`, `engine`, and `metadata`.
    :type emit_base_engine_metadata: ```bool```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```

    :param recursive: Recursively traverse module hierarchy and recreate hierarchy with exposed interfaces
    :type recursive: ```bool```

    :param dry_run: Show what would be created; don't actually write to the filesystem
    :type dry_run: ```bool```

    :param filesystem_layout: Hierarchy of folder and file names generated. "java" is file per package per name.
    :type filesystem_layout: ```Literal["java", "as_input"]```
    """
    output_directory = path.realpath(output_directory)
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
                    emit_base_engine_metadata=emit_base_engine_metadata,
                    no_word_wrap=no_word_wrap,
                    output_directory=output_directory,
                    target_module_name=target_module_name,
                    recursive=recursive,
                    dry_run=dry_run,
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
        emit_base_engine_metadata=emit_base_engine_metadata,
    )
    _exmod_single_folder_kwargs: Iterator[
        TypedDict(
            "_exmod_single_folder_kwargs",
            {
                "module": module,
                "module_name": module_name,
                "module_root_dir": module_root_dir,
                "output_directory": output_directory,
            },
        )
    ] = chain.from_iterable(
        (
            (
                {
                    "module": module,
                    "module_name": module_name,
                    "module_root_dir": module_root_dir,
                    "output_directory": output_directory,
                    "emit_base_engine_metadata": emit_base_engine_metadata,
                },
            ),
            (
                (
                    map(
                        lambda package: (
                            lambda pkg_relative_dir: {
                                "module": ".".join((module, package)),
                                "module_name": package,
                                "module_root_dir": path.join(
                                    module_root_dir, pkg_relative_dir
                                ),
                                "output_directory": path.join(
                                    output_directory, pkg_relative_dir
                                ),
                                "emit_base_engine_metadata": emit_base_engine_metadata,
                            }
                        )(package.replace(".", path.sep)),
                        packages,
                    )
                )
                if recursive
                else iter(())
            ),
        )
    )
    # This could be executed in parallel for efficiency
    deque(
        map(
            lambda kwargs: _exmod_single_folder(**kwargs),
            _exmod_single_folder_kwargs,
        ),
        maxlen=0,
    )

    return


def exmod_single_folder(
    emit_name,
    module,
    blacklist,
    whitelist,
    output_directory,
    mock_imports,
    emit_base_engine_metadata,
    no_word_wrap,
    dry_run,
    module_root_dir,
    module_root,
    module_name,
    new_module_name,
    filesystem_layout,
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

    :param mock_imports: Whether to generate mock TensorFlow imports
    :type mock_imports: ```bool```

    :param emit_base_engine_metadata: [sqlalchemy] Whether to produce a file with `Base`, `engine`, and `metadata`.
    :type emit_base_engine_metadata: ```bool```

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
        mock_imports=mock_imports,
        no_word_wrap=no_word_wrap,
        dry_run=dry_run,
        filesystem_layout=filesystem_layout,
    )

    if emit_base_engine_metadata:
        with open(
            path.join(module_root_dir, "sqlalchemy_engine{}py".format(path.extsep)),
            "wt",
        ) as f:
            f.write(mock_engine_base_metadata_str)

    imports = _emit_files_from_module_and_return_imports(
        module_name=module_name, module=module, module_root_dir=module_root_dir
    )  # type: Optional[list[ImportFrom]]
    if not imports:
        # Case: no obvious folder hierarchy, so parse the `__init__` file in root
        with open(
            path.join(module_root_dir, "__init__{extsep}py".format(extsep=path.extsep)),
            "rt",
        ) as f:
            mod: Module = parse(f.read())

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
            ),
            init_filepath,
            mode="wt",
        )


__all__ = ["exmod"]  # type: list[str]
