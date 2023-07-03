""" Tests for exmod subcommand """

from ast import Assign, ClassDef, ImportFrom, List, Load, Module, Name, Store, alias
from functools import partial
from io import StringIO
from itertools import chain, groupby
from operator import itemgetter
from os import environ, listdir, mkdir, path, walk
from os.path import extsep
from subprocess import run
from sys import executable, platform
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import cdd.class_.parse
from cdd.compound.exmod import exmod
from cdd.shared.ast_utils import maybe_type_comment, set_value
from cdd.shared.pkg_utils import relative_filename
from cdd.shared.pure_utils import ENCODING, INIT_FILENAME, rpartial, unquote
from cdd.shared.source_transformer import ast_parse, to_code
from cdd.tests.mocks import imports_header
from cdd.tests.mocks.classes import class_str
from cdd.tests.mocks.exmod import setup_py_mock
from cdd.tests.utils_for_tests import unittest_main


class TestExMod(TestCase):
    """Test class for exmod.py"""

    parent_name = ""
    parent_dir = ""
    child_name = ""
    child_dir = ""
    grandchild_name = ""
    grandchild_dir = ""
    module_hierarchy = ()

    @classmethod
    def setUpClass(cls) -> None:
        """
        Create module hierarchy
        """
        cls.parent_name, cls.parent_dir = "parent", "parent_dir"
        cls.child_name, cls.child_dir = "child", path.join(cls.parent_dir, "child_dir")
        cls.grandchild_name, cls.grandchild_dir = "grandchild", path.join(
            cls.child_dir, "grandchild_dir"
        )
        cls.module_hierarchy = (
            (cls.parent_name, cls.parent_dir),
            (cls.child_name, cls.child_dir),
            (cls.grandchild_name, cls.grandchild_dir),
        )

    @staticmethod
    def normalise_double_paths(*dictionaries):
        """
        On Windows the paths can come up weird, like C:\\\\foo instead of C:\\foo

        This fixes that issue, and also safe to work on non-Windows

        :param dictionaries: Dictionaries
        :type dictionaries: ```Tuple[dictionaries]```

        :return: `map` of normalised `dictionaries`
        :rtype: ```map```
        """
        return map(
            lambda d: {
                k: tuple(
                    map(repr, map(rpartial(str.replace, path.sep * 2, path.sep), v))
                )
                for k, v in d.items()
            },
            dictionaries,
        )

    def test_exmod(self) -> None:
        """Tests `exmod`"""

        try:
            with TemporaryDirectory(prefix="search_root", suffix="search_path") as root:
                _, new_module_dir = self.create_and_install_pkg(root)

                exmod(
                    module=self.module_name,
                    emit_name="class",
                    blacklist=tuple(),
                    whitelist=tuple(),
                    mock_imports=True,
                    output_directory=new_module_dir,
                    target_module_name=None,
                    no_word_wrap=None,
                    dry_run=False,
                )
                self._check_emission(new_module_dir)
        finally:
            # sys.path.remove(existent_module_dir)
            self._pip(["uninstall", "-y", self.package_root_name])

    def test_exmod_blacklist(self) -> None:
        """Tests `exmod` blacklist"""

        try:
            with TemporaryDirectory(prefix="search_root", suffix="search_path") as root:
                existent_module_dir, new_module_dir = self.create_and_install_pkg(root)
                exmod(
                    module=self.module_name,
                    emit_name="class",
                    blacklist=(".".join((existent_module_dir,) * 2),),
                    whitelist=tuple(),
                    mock_imports=True,
                    output_directory=new_module_dir,
                    target_module_name=None,
                    no_word_wrap=None,
                    dry_run=False,
                )
                self.assertListEqual(
                    *map(
                        sorted,
                        (listdir(new_module_dir), [INIT_FILENAME, self.parent_dir]),
                    ),
                )
        finally:
            self._pip(["uninstall", "-y", self.package_root_name])

    def test_exmod_whitelist(self) -> None:
        """Tests `exmod` whitelist"""

        try:
            with TemporaryDirectory(prefix="search_root", suffix="search_path") as root:
                existent_module_dir, new_module_dir = self.create_and_install_pkg(root)
                exmod(
                    module=self.module_name,
                    emit_name="class",
                    blacklist=tuple(),
                    whitelist=(".".join((self.package_root_name, "gen")),),
                    mock_imports=True,
                    output_directory=new_module_dir,
                    target_module_name=None,
                    no_word_wrap=None,
                    dry_run=False,
                )

                new_module_dir_len = len(new_module_dir + path.sep)
                gen, gold = map(
                    sorted,
                    (
                        (
                            path.join(dirpath, filename)[new_module_dir_len:]
                            for (dirpath, dirnames, filenames) in walk(new_module_dir)
                            for filename in filenames
                        ),
                        chain.from_iterable(
                            (
                                (INIT_FILENAME,),
                                chain.from_iterable(
                                    (
                                        path.join(directory, INIT_FILENAME),
                                        "{basepath}{extsep}py".format(
                                            basepath=path.join(directory, name),
                                            extsep=path.extsep,
                                        ),
                                    )
                                    for name, directory in self.module_hierarchy
                                ),
                            )
                        ),
                    ),
                )

                self.assertListEqual(gen, gold)
        finally:
            self._pip(["uninstall", "-y", self.package_root_name])

    def test_exmod_module_directory(self) -> None:
        """Tests `exmod` module whence directory"""

        with TemporaryDirectory() as tempdir, self.assertRaises(ModuleNotFoundError):
            exmod(
                module=tempdir,
                emit_name="cool_name",
                blacklist=tuple(),
                whitelist=tuple(),
                mock_imports=True,
                output_directory=path.join(tempdir, "nonexistent"),
                target_module_name=None,
                no_word_wrap=None,
                dry_run=False,
            )

    def test_exmod_no_module(self) -> None:
        """Tests that ModuleNotFound error is raised when module is not installed"""
        with TemporaryDirectory() as tempdir, self.assertRaises(ModuleNotFoundError):
            exmod(
                module="fubar",
                emit_name="uncool_name",
                blacklist=tuple(),
                whitelist=tuple(),
                mock_imports=True,
                output_directory=path.join(tempdir, "nonexistent"),
                target_module_name=None,
                no_word_wrap=None,
                dry_run=False,
            )

    def test_exmod_output_directory_nonexistent(self) -> None:
        """Tests `exmod` module whence directory does not exist"""

        with TemporaryDirectory() as tempdir, self.assertRaises(ModuleNotFoundError):
            output_directory = path.join(tempdir, "stuff")
            self.assertFalse(path.isdir(output_directory))
            exmod(
                module=output_directory,
                emit_name=None,
                blacklist=tuple(),
                whitelist=tuple(),
                mock_imports=True,
                output_directory=output_directory,
                target_module_name=None,
                no_word_wrap=None,
                dry_run=False,
            )

    def test_exmod_dry_run(self) -> None:
        """Tests `exmod` dry_run"""

        try:
            with TemporaryDirectory(prefix="search_root", suffix="search_path") as root:
                _, new_module_dir = self.create_and_install_pkg(root)

                with patch(
                    "cdd.compound.exmod_utils.EXMOD_OUT_STREAM", new_callable=StringIO
                ) as f:
                    exmod(
                        module=self.module_name,
                        emit_name="class",
                        blacklist=tuple(),
                        whitelist=tuple(),
                        mock_imports=True,
                        output_directory=new_module_dir,
                        target_module_name=None,
                        no_word_wrap=None,
                        dry_run=True,
                    )
                    r = f.getvalue()

                result = dict(
                    map(
                        lambda k_v: (
                            k_v[0],
                            tuple(
                                sorted(
                                    set(
                                        map(
                                            partial(
                                                relative_filename,
                                                remove_hints=(
                                                    (
                                                        lambda directory: "{directory}{sep}".format(
                                                            directory=unquote(
                                                                repr(directory)
                                                            ),
                                                            sep=path.sep,
                                                        )
                                                        if platform == "win32"
                                                        else directory
                                                    )(
                                                        path.join(
                                                            new_module_dir,
                                                            path.basename(
                                                                new_module_dir
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                            map(unquote, map(itemgetter(1), k_v[1])),
                                        )
                                    )
                                )
                            ),
                        ),
                        groupby(
                            map(rpartial(str.split, "\t", 2), sorted(r.splitlines())),
                            key=itemgetter(0),
                        ),
                    )
                )

                all_tests_running = len(result["write"]) == 1

                key_counts = (
                    (("mkdir", 4), ("touch", 1), ("write", 1))
                    if all_tests_running
                    else (("mkdir", 7), ("touch", 4), ("write", 4))
                )

                for key, count in key_counts:
                    self.assertEqual(count, len(result[key]), key)

                gold = dict(
                    touch=(path.join(path.dirname(self.gold_dir), INIT_FILENAME),),
                    **{
                        k: tuple(
                            map(
                                rpartial(str.rstrip, path.sep),
                                map(partial(path.join, new_module_dir), v),
                            )
                        )
                        for k, v in {
                            "mkdir": ("", *map(itemgetter(1), self.module_hierarchy)),
                            "write": (INIT_FILENAME,),
                        }.items()
                    },
                )
                self.assertDictEqual(*self.normalise_double_paths(result, gold))

                self._check_emission(new_module_dir, dry_run=True)
        finally:
            self._pip(["uninstall", "-y", self.package_root_name])

    def create_and_install_pkg(self, root):
        """
        Create and install the pacakge

        :param root: Root directory
        :type root: ```str```

        :return: existent_module_dir, new_module_dir
        :rtype: ```Tuple[str,str]```
        """
        self.package_root_name = path.basename(root)
        existent_module_dir = path.join(root, self.package_root_name, "gen")
        new_module_dir = path.join(root, self.package_root_name, "gold")
        package_root_mod_dir = path.join(root, self.package_root_name)
        mkdir(package_root_mod_dir)
        with open(
            path.join(package_root_mod_dir, "__init__{extsep}py".format(extsep=extsep)),
            "wt",
        ) as f:
            f.write(
                "{encoding}\n\n"
                "{mod}\n".format(
                    encoding=ENCODING,
                    mod=to_code(
                        Module(
                            body=[
                                ImportFrom(
                                    module=".".join((self.package_root_name, "gen")),
                                    names=[
                                        alias(
                                            "*",
                                            None,
                                            identifier=None,
                                            identifier_name=None,
                                        )
                                    ],
                                    level=0,
                                    identifier=None,
                                ),
                                Assign(
                                    targets=[Name("__author__", Store())],
                                    value=set_value(
                                        environ.get("CDD_AUTHOR", "Samuel Marks")
                                    ),
                                    expr=None,
                                    lineno=None,
                                    **maybe_type_comment,
                                ),
                                Assign(
                                    targets=[Name("__version__", Store())],
                                    value=set_value(
                                        environ.get("CDD_VERSION", "0.0.0")
                                    ),
                                    expr=None,
                                    lineno=None,
                                    **maybe_type_comment,
                                ),
                                Assign(
                                    targets=[Name("__all__", Store())],
                                    value=List(
                                        ctx=Load(),
                                        elts=list(
                                            map(
                                                set_value,
                                                chain.from_iterable(
                                                    (
                                                        (
                                                            "__author__",
                                                            "__version__",
                                                        ),
                                                        map(
                                                            itemgetter(0),
                                                            self.module_hierarchy,
                                                        ),
                                                    )
                                                ),
                                            )
                                        ),
                                        expr=None,
                                    ),
                                    expr=None,
                                    lineno=None,
                                    **maybe_type_comment,
                                ),
                            ],
                            type_ignores=[],
                            stmt=None,
                        )
                    ),
                )
            )
        mkdir(new_module_dir)
        self._create_fs(existent_module_dir)
        self._pip(["install", "."], root)
        return existent_module_dir, new_module_dir

    def _create_fs(self, module_root):
        """
        Populate filesystem from `module_root` root with module hierarchy &etc. for later exposure (exmod)

        :param module_root: Root directory for module, one directory below setup.py
        :type module_root: ```str```

        :return: module_root
        :rtype: ```str```
        """
        self.module_name, self.gold_dir = path.basename(module_root), module_root
        package_root = path.dirname(path.dirname(module_root))
        self.module_root_name = self.module_name

        self.module_name = ".".join((self.package_root_name, self.module_name))

        with open(
            path.join(package_root, "setup{extsep}py".format(extsep=extsep)), "wt"
        ) as f:
            f.write(
                setup_py_mock.format(
                    encoding=ENCODING,
                    package_name=self.package_root_name,
                    module_name=self.module_root_name,
                )
            )

        open(
            path.join(package_root, "README{extsep}md".format(extsep=extsep)), "a"
        ).close()
        mkdir(module_root)

        mod = Module(
            body=list(
                chain.from_iterable(
                    (
                        (
                            ImportFrom(
                                module=".".join(
                                    (
                                        self.package_root_name,
                                        "gen",
                                        directory.replace(path.sep, "."),
                                    )
                                ),
                                names=[
                                    alias(
                                        name,
                                        None,
                                        identifier=None,
                                        identifier_name=None,
                                    )
                                ],
                                level=0,
                                identifier=None,
                            )
                            for name, directory in self.module_hierarchy
                        ),
                        (
                            Assign(
                                targets=[Name("__all__", Store())],
                                value=List(
                                    ctx=Load(),
                                    elts=list(
                                        map(
                                            set_value,
                                            map(itemgetter(0), self.module_hierarchy),
                                        )
                                    ),
                                    expr=None,
                                ),
                                expr=None,
                                lineno=None,
                                **maybe_type_comment,
                            ),
                        ),
                    )
                )
            ),
            type_ignores=[],
            stmt=None,
        )

        with open(
            path.join(module_root, INIT_FILENAME),
            "wt",
        ) as f:
            f.write(
                "{encoding}\n\n"
                "{mod}".format(
                    encoding=ENCODING,
                    mod=to_code(mod),
                )
            )

        for name, _folder in self.module_hierarchy:
            folder = path.join(module_root, _folder)
            mkdir(folder)
            cls_name = "{name}Class".format(name=name.title())
            with open(
                path.join(folder, "__init__{extsep}py".format(extsep=extsep)), "wt"
            ) as f:
                f.write(
                    "{encoding}\n\n"
                    "from .{name} import {cls_name}\n\n"
                    "__all__ = [{cls_name!r}]\n".format(
                        encoding=ENCODING,
                        name=name,
                        cls_name=cls_name,
                    )
                )
            with open(
                path.join(folder, "{name}{extsep}py".format(name=name, extsep=extsep)),
                "wt",
            ) as f:
                f.write(
                    "{encoding}\n\n"
                    "{imports_header}\n"
                    "{class_str}\n\n"
                    "__all__ = [{cls_name!r}]\n".format(
                        encoding=ENCODING,
                        imports_header=imports_header,
                        class_str=class_str.replace("ConfigClass", cls_name),
                        cls_name=cls_name,
                    )
                )

        return module_root

    def _check_emission(self, tempdir, dry_run=False):
        """
        Confirm whether emission conforms to gen by verifying their IRs are equivalent

        :param tempdir: Temporary directory
        :type tempdir: ```str```

        :param dry_run: Show what would be created; don't actually write to the filesystem
        :type dry_run: ```bool```
        """
        new_module_name = path.basename(tempdir)

        for name, folder in self.module_hierarchy:
            gen_folder = path.join(
                tempdir,
                *(folder,)
                if tempdir.rpartition(path.sep)[2] == new_module_name
                else (new_module_name, folder),
            )
            gold_folder = path.join(
                self.gold_dir,
                *(folder,)
                if self.gold_dir.endswith(self.module_name.replace(".", path.sep))
                else (self.module_name, folder),
            )

            def _open(_folder):
                """
                :param _folder: Folder to join on
                :type _folder: ```str``

                :return: Open IO
                :rtype: ```open```
                """
                return open(
                    path.join(
                        _folder, "{name}{extsep}py".format(name=name, extsep=extsep)
                    ),
                    "rt",
                )

            self.assertTrue(
                path.isdir(gold_folder), "Expected {!r} to exist".format(gold_folder)
            )

            gen_is_dir = path.isdir(gen_folder)
            if dry_run:
                self.assertFalse(gen_is_dir)
            else:
                self.assertTrue(gen_is_dir, gen_folder)

                with _open(gen_folder) as gen, _open(gold_folder) as gold:
                    gen_ir, gold_ir = map(
                        lambda ir: ir["_internal"].__delitem__("original_doc_str")
                        or ir,
                        map(
                            lambda node: cdd.class_.parse.class_(
                                next(
                                    filter(
                                        rpartial(isinstance, ClassDef),
                                        (lambda node_read: ast_parse(node_read).body)(
                                            node.read()
                                        ),
                                    )
                                )
                            ),
                            (gen, gold),
                        ),
                    )
                    self.assertDictEqual(gold_ir, gen_ir)

    def _pip(self, pip_args, cwd=None):
        """
        Run `pip` with given args (and assert success).
        [Not using InstallCommand from pip anymore as its been deprecated (and now removed).]

        :param pip_args: Arguments to give pip
        :type pip_args: ```List[str]```

        :param cwd: Current working directory to run the command from. Defaults to current dir.
        :type cwd: ```Optional[str]```
        """
        self.assertEqual(
            run(
                [executable, "-m", "pip"] + pip_args,
                cwd=cwd,  # stdout=DEVNULL
            ).returncode,
            0,
            "EXIT_SUCCESS not reached",
        )


unittest_main()
