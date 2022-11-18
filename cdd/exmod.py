"""
Not a dead module
"""

from ast import Assign, Expr, ImportFrom, List, Load, Module, Name, Store, alias
from collections import OrderedDict, deque
from functools import partial
from importlib import import_module
from inspect import getfile
from itertools import chain, groupby
from operator import itemgetter
from os import makedirs, path

import cdd.emit.file
from cdd.ast_utils import maybe_type_comment, set_value
from cdd.exmod_utils import emit_file_on_hierarchy, get_module_contents
from cdd.pkg_utils import relative_filename
from cdd.pure_utils import INIT_FILENAME


def exmod(
    emit_name,
    module,
    blacklist,
    whitelist,
    output_directory,
    mock_imports,
    no_word_wrap,
    dry_run,
    filesystem_layout="as_input",
):
    """
    Expose module as `emit` types into `output_directory`

    :param emit_name: What type(s) to generate.
    :type emit_name: ```List[Literal["argparse", "class", "function", "sqlalchemy", "sqlalchemy_table"]]```

    :param module: Module name or path
    :type module: ```str```

    :param blacklist: Modules/FQN to omit. If unspecified will emit all (unless whitelist).
    :type blacklist: ```Union[List[str],Tuple[str]]```

    :param whitelist: Modules/FQN to emit. If unspecified will emit all (minus blacklist).
    :type whitelist: ```Union[List[str],Tuple[str]]```

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
    """
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
                    no_word_wrap=no_word_wrap,
                    output_directory=output_directory,
                    dry_run=dry_run,
                ),
                emit_name or iter(()),
            ),
            maxlen=0,
        )
    elif dry_run:
        print("mkdir\t{output_directory!r}".format(output_directory=output_directory))
    elif not path.isdir(output_directory):
        makedirs(output_directory)

    emit_name = (
        emit_name[0]
        if emit_name is not None and len(emit_name) == 1 and isinstance(emit_name, list)
        else emit_name
    )
    assert isinstance(
        emit_name, (str, type(None))
    ), "Expected str got {emit_name_type!r}".format(emit_name_type=type(emit_name))

    module_name, new_module_name = map(path.basename, (module, output_directory))
    module = (
        # partial(module_from_file, module_name=module_name)
        # if path.isdir(module) else
        import_module
    )(module)

    # if module.__file__ is None:
    #     raise ModuleNotFoundError(module_name)

    module_root_dir = path.dirname(module.__file__) + path.sep

    mod_path = ".".join((path.basename(module_root_dir[:-1]), module_name))
    blacklist, whitelist = map(
        frozenset, (blacklist or iter(()), whitelist or iter(()))
    )
    proceed = any(
        (
            sum(map(len, (blacklist, whitelist))) == 0,
            mod_path not in blacklist and (mod_path in whitelist or not whitelist),
        )
    )
    if not proceed:
        return

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
    imports = list(
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
                    else cdd.parse.class_.class_(
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
    assert len(imports), "Module contents are empty"
    modules_names = tuple(
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
    )

    init_filepath = path.join(output_directory, new_module_name, INIT_FILENAME)
    if dry_run:
        print("write\t{init_filepath!r}".format(init_filepath=init_filepath))
    else:
        cdd.emit.file.file(
            Module(
                body=list(
                    chain.from_iterable(
                        (
                            (Expr(set_value("\nExport internal imports\n")),),
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
                                    targets=[Name("__all__", Store())],
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


__all__ = ["exmod"]
