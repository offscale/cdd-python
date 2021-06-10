"""
Not a dead module
"""
import ast
from ast import Assign, Expr, ImportFrom, List, Load, Module, Name, Store, alias
from functools import partial
from importlib import import_module
from inspect import getfile, ismodule
from itertools import chain, groupby
from operator import itemgetter
from os import extsep, makedirs, path

from cdd import emit, parse
from cdd.ast_utils import (
    maybe_type_comment,
    merge_assignment_lists,
    merge_modules,
    set_value,
)
from cdd.pkg_utils import relative_filename
from cdd.pure_utils import no_magic_dir2attr
from cdd.tests.mocks import imports_header
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
    module_name = path.basename(module)
    new_module_name = path.basename(output_directory)
    module = (
        partial(module_from_file, module_name=module_name)
        if path.isdir(module)
        else import_module
    )(module)

    def get_module_contents(obj, current_module=None, _result={}):
        """
        Helper function to get the recursive inner module contents

        :param obj: Something to `dir` on
        :type obj: ```Any```

        :param current_module: The current module
        :type current_module: ```Optional[str]```

        :param _result: The result var (used internally as accumulator)
        :type _result: ```dict```

        :returns: Values (could be modules, classes, and whatever other symbols are exposed)
        :rtype: ```Generator[Any]```
        """

        for name, symbol in no_magic_dir2attr(obj).items():
            fq = "{}.{}".format(current_module, name)
            if isinstance(symbol, type):
                _result[fq] = symbol
            elif (
                current_module != getattr(symbol, "__name__", current_module)
                and ismodule(symbol)
                and fq not in _result
            ):
                get_module_contents(symbol, current_module=symbol.__name__)
        return _result

    def mkdir_and_emit_file(name_orig_ir):
        """
        Generate Java-package style file hierarchy from fully-qualified module name

        :param name_orig_ir: FQ module name, original filename path, IR
        :type name_orig_ir: ```Tuple[str, str, dict]```

        :returns: Import to generated module
        :rtype: ```ImportFrom```
        """
        mod_name, _, name = name_orig_ir[0].rpartition(".")
        original_relative_filename_path = name_orig_ir[1]
        ir = name_orig_ir[2]
        mod_path = path.join(
            output_directory,
            new_module_name,
            mod_name.replace(".", path.sep),
        )
        if not path.isdir(mod_path):
            makedirs(mod_path)
        open(path.join(path.dirname(mod_path), "__init__.py"), "a").close()
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
            (
                path.join(
                    output_directory, new_module_name, original_relative_filename_path
                ),
                path.join(
                    output_directory,
                    new_module_name,
                    path.dirname(original_relative_filename_path),
                    "__init__.py",
                ),
            )
            if filesystem_layout == "as_input"
            else (
                path.join(
                    mod_path, "{name}{extsep}py".format(name=name, extsep=extsep)
                ),
                path.join(mod_path, "__init__{extsep}py".format(extsep=extsep)),
            )
        )

        if path.isfile(emit_filename):
            with open(emit_filename, "rt") as f:
                mod = ast.parse(f.read())
            gen_node = merge_modules(mod, gen_node)
            merge_assignment_lists(gen_node, "__all__")

        emit.file(
            gen_node,
            filename=emit_filename,
            mode="wt",
        )
        # print(
        #     "Emitted: {emit_filename!r}\n".format(
        #         emit_filename=emit_filename,
        #     )
        # )
        if name != "__init__" and not path.isfile(init_filepath):
            emit.file(
                Module(
                    body=[
                        Expr(set_value("__init__ to expose internals of this module")),
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

    # Might need some `groupby` in case multiple files are in the one project; same for `get_module_contents`
    imports = list(
        map(
            mkdir_and_emit_file,
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
                    get_module_contents(module).items(),
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
        path.join(output_directory, new_module_name, "__init__.py"),
        mode="wt",
    )


__all__ = ["exmod"]
