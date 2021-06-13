"""
Not a dead module
"""

from ast import Assign, Expr, ImportFrom, List, Load, Module, Name, Store, alias
from functools import partial
from importlib import import_module
from inspect import getfile
from itertools import chain, groupby
from operator import itemgetter
from os import extsep, makedirs, path

from cdd import emit, parse
from cdd.ast_utils import maybe_type_comment, set_value
from cdd.exmod_utils import get_module_contents, mkdir_and_emit_file
from cdd.pkg_utils import relative_filename
from cdd.tests.utils_for_tests import module_from_file


def exmod(
    module,
    emit_name,
    blacklist,
    whitelist,
    output_directory,
    filesystem_layout="as_input",
):
    """
    Expose module as `emit` types into `output_directory`

    :param module: Module name or path
    :type module: ```str```

    :param emit_name: What type(s) to generate.
    :type emit_name: ```List[Literal["argparse", "class", "function", "sqlalchemy", "sqlalchemy_table"]]```

    :param blacklist: Modules/FQN to omit. If unspecified will emit all (unless whitelist).
    :type blacklist: ```List[str]```

    :param whitelist: Modules/FQN to emit. If unspecified will emit all (minus blacklist).
    :type whitelist: ```List[str]```

    :param output_directory: Where to place the generated exposed interfaces to the given `--module`.
    :type output_directory: ```str```

    :param filesystem_layout: Hierarchy of folder and file names generated. "java" is file per package per name.
    :type filesystem_layout: ```Literal["java", "as_input"]```
    """
    if not path.isdir(output_directory):
        makedirs(output_directory)
    if blacklist:
        raise NotImplementedError("blacklist")
    elif whitelist:
        raise NotImplementedError("whitelist")

    module_name, new_module_name = map(path.basename, (module, output_directory))
    module = (
        partial(module_from_file, module_name=module_name)
        if path.isdir(module)
        else import_module
    )(module)

    module_root_dir = path.dirname(module.__file__) + path.sep

    _mkdir_and_emit_file = partial(
        mkdir_and_emit_file,
        emit_name=emit_name,
        module_name=module_name,
        new_module_name=new_module_name,
        filesystem_layout=filesystem_layout,
        output_directory=output_directory,
    )

    # Might need some `groupby` in case multiple files are in the one project; same for `get_module_contents`
    imports = list(
        map(
            _mkdir_and_emit_file,
            map(
                lambda name_source: (
                    name_source[0],
                    (
                        lambda filename: filename[len(module_name) + 1 :]
                        if filename.startswith(module_name)
                        else filename
                    )(relative_filename(getfile(name_source[1]))),
                    parse.class_(name_source[1]),
                ),
                # sorted(
                map(
                    lambda name_source: (
                        name_source[0][len(module_name) + 1 :],
                        name_source[1],
                    ),
                    get_module_contents(
                        module, module_root_dir=module_root_dir
                    ).items(),
                ),
                #    key=itemgetter(0),
                # ),
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
    init_filepath = path.join(
        output_directory, new_module_name, "__init__{extsep}py".format(extsep=extsep)
    )
    emit.file(
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
    # print("#Emitted: {init_filepath!r} ;".format(init_filepath=init_filepath))


__all__ = ["exmod"]
