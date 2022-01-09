""" Tests for exmod subcommand """

from ast import ClassDef
from functools import partial
from io import StringIO
from itertools import chain, groupby
from operator import add, itemgetter
from os import environ, listdir, mkdir, path, walk
from os.path import extsep
from subprocess import DEVNULL, call
from sys import executable, platform
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cdd import parse
from cdd.exmod import exmod
from cdd.pkg_utils import relative_filename
from cdd.pure_utils import ENCODING, INIT_FILENAME, rpartial, unquote
from cdd.source_transformer import ast_parse
from cdd.tests.mocks import imports_header
from cdd.tests.mocks.classes import class_str
from cdd.tests.mocks.exmod import setup_py_mock
from cdd.tests.utils_for_tests import unittest_main


class TestExMod(TestCase):
    """Test class for exmod.py"""

    def test_exmod(self) -> None:
        """Tests `exmod`"""

        try:
            with TemporaryDirectory(
                prefix="gold", suffix="gold"
            ) as existent_module_dir, TemporaryDirectory(
                prefix="gen", suffix="gen"
            ) as new_module_dir:
                self._create_fs(existent_module_dir)
                # sys.path.insert(0, existent_module_dir)
                self._pip(["install", "."], existent_module_dir)

                exmod(
                    module=self.module_name,
                    emit_name="class",
                    blacklist=tuple(),
                    whitelist=tuple(),
                    mock_imports=True,
                    output_directory=new_module_dir,
                    no_word_wrap=None,
                    dry_run=False,
                )
                self._check_emission(new_module_dir)
        finally:
            # sys.path.remove(existent_module_dir)
            self._pip(["uninstall", "-y", self.module_name])

    def test_exmod_blacklist(self) -> None:
        """Tests `exmod` blacklist"""

        try:
            with TemporaryDirectory(
                prefix="gold", suffix="gold"
            ) as existent_module_dir, TemporaryDirectory(
                prefix="gen", suffix="gen"
            ) as new_module_dir:
                self._create_fs(existent_module_dir)
                self._pip(["install", "."], existent_module_dir)
                exmod(
                    module=self.module_name,
                    emit_name="class",
                    blacklist=(".".join((self.module_name,) * 2),),
                    whitelist=tuple(),
                    mock_imports=True,
                    output_directory=new_module_dir,
                    no_word_wrap=None,
                    dry_run=False,
                )
                self.assertListEqual(listdir(new_module_dir), [])
        finally:
            self._pip(["uninstall", "-y", self.module_name])

    def test_exmod_whitelist(self) -> None:
        """Tests `exmod` whitelist"""

        try:
            with TemporaryDirectory(
                prefix="gold", suffix="gold"
            ) as existent_module_dir, TemporaryDirectory(
                prefix="gen", suffix="gen"
            ) as new_module_dir:
                self._create_fs(existent_module_dir)
                self._pip(["install", "."], existent_module_dir)
                existent_module_name = path.basename(existent_module_dir)
                exmod(
                    module=self.module_name,
                    emit_name="class",
                    blacklist=tuple(),
                    whitelist=(".".join((existent_module_name,) * 2),),
                    mock_imports=True,
                    output_directory=new_module_dir,
                    no_word_wrap=None,
                    dry_run=False,
                )

                gen, gold = map(
                    sorted,
                    (
                        map(
                            lambda p: (
                                lambda _p: path.join("gold", _p.partition(path.sep)[2])
                                if _p.startswith("gold") and _p.count("gold") == 2
                                else _p
                            )(p.partition(path.sep)[2]),
                            (
                                path.join(dirpath, filename)[
                                    len(new_module_dir + path.sep) :
                                ]
                                for (dirpath, dirnames, filenames) in walk(
                                    new_module_dir
                                )
                                for filename in filenames
                            ),
                        ),
                        (
                            INIT_FILENAME,
                            *chain.from_iterable(
                                map(
                                    lambda i: map(
                                        partial(
                                            path.join,
                                            self.module_hierarchy[i][1],
                                        ),
                                        (
                                            "{name}{sep}py".format(
                                                name=self.module_hierarchy[i][0],
                                                sep=extsep,
                                            ),
                                            INIT_FILENAME,
                                        ),
                                    ),
                                    range(len(self.module_hierarchy)),
                                ),
                            ),
                        ),
                    ),
                )
                all_tests_running = len(gold) == 7
                if all_tests_running:
                    gold = list(
                        chain.from_iterable(
                            (
                                gold[:1],
                                chain.from_iterable(
                                    map(
                                        lambda p: (p, p),
                                        map(
                                            partial(path.join, "gold", "parent_dir"),
                                            (
                                                INIT_FILENAME,
                                                *map(
                                                    partial(path.join, "child_dir"),
                                                    (
                                                        INIT_FILENAME,
                                                        "child{sep}py".format(
                                                            sep=extsep
                                                        ),
                                                        path.join(
                                                            "grandchild_dir",
                                                            INIT_FILENAME,
                                                        ),
                                                        path.join(
                                                            "grandchild_dir",
                                                            "grandchild{sep}py".format(
                                                                sep=extsep
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                                "parent{sep}py".format(sep=extsep),
                                            ),
                                        ),
                                    )
                                ),
                                gold[1:],
                            )
                        )
                    )

                self.assertListEqual(gen, gold)
        finally:
            self._pip(["uninstall", "-y", self.module_name])

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
                no_word_wrap=None,
                dry_run=False,
            )

    def test_exmod_dry_run(self) -> None:
        """Tests `exmod` dry_run"""

        try:
            with TemporaryDirectory(
                prefix="gold", suffix="gold"
            ) as existent_module_dir, TemporaryDirectory(
                prefix="gen", suffix="gen"
            ) as new_module_dir:
                self._create_fs(existent_module_dir)
                self._pip(["install", "."], existent_module_dir)

                with patch("sys.stdout", new_callable=StringIO) as f:
                    exmod(
                        module=self.module_name,
                        emit_name="class",
                        blacklist=tuple(),
                        whitelist=tuple(),
                        mock_imports=True,
                        output_directory=new_module_dir,
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
                    (("mkdir", 10), ("touch", 4), ("write", 1))
                    if all_tests_running
                    else (("mkdir", 7), ("touch", 4), ("write", 4))
                )

                for key, count in key_counts:
                    self.assertEqual(count, len(result[key]), key)

                gold_module_name = next(
                    map(
                        lambda p: p.partition(path.sep)[0],
                        filter(rpartial(str.startswith, "gold"), result["mkdir"]),
                    ),
                    path.basename(self.gold_dir),
                )
                gold = {
                    k: tuple(map(unquote, map(repr, v)))
                    for k, v in {
                        "mkdir": chain.from_iterable(
                            (
                                (new_module_dir,),
                                map(
                                    partial(path.join, gold_module_name),
                                    (
                                        self.module_hierarchy[0][1],
                                        self.module_hierarchy[1][1],
                                        self.module_hierarchy[2][1],
                                    ),
                                )
                                if all_tests_running
                                else iter(()),
                                (
                                    self.module_hierarchy[0][1],
                                    self.module_hierarchy[1][1],
                                    path.join(
                                        self.module_hierarchy[1][1],
                                        self.module_hierarchy[1][0],
                                    ),
                                    self.module_hierarchy[2][1],
                                    path.join(
                                        self.module_hierarchy[2][1],
                                        self.module_hierarchy[2][0],
                                    ),
                                    path.join(
                                        self.module_hierarchy[0][1],
                                        self.module_hierarchy[0][0],
                                    ),
                                ),
                            )
                        ),
                        "touch": (
                            INIT_FILENAME,
                            *map(
                                rpartial(path.join, INIT_FILENAME),
                                (
                                    self.module_hierarchy[0][1],
                                    self.module_hierarchy[1][1],
                                    self.module_hierarchy[2][1],
                                ),
                            ),
                        ),
                        "write": (
                            lambda write_block: tuple(write_block[:1])
                            if all_tests_running
                            else write_block
                        )(
                            (
                                INIT_FILENAME,
                                path.join(
                                    self.module_hierarchy[1][1],
                                    "{name}{extsep}py".format(
                                        name=self.module_hierarchy[1][0],
                                        extsep=extsep,
                                    ),
                                ),
                                path.join(
                                    self.module_hierarchy[2][1],
                                    "{name}{extsep}py".format(
                                        name=self.module_hierarchy[2][0],
                                        extsep=extsep,
                                    ),
                                ),
                                path.join(
                                    self.module_hierarchy[0][1],
                                    "{name}{extsep}py".format(
                                        name=self.module_hierarchy[0][0],
                                        extsep=extsep,
                                    ),
                                ),
                            )
                        ),
                    }.items()
                }

                self.assertDictEqual(result, gold)

                self._check_emission(new_module_dir, dry_run=True)
        finally:
            self._pip(["uninstall", "-y", self.module_name])

    def _create_fs(self, tempdir):
        """
        Populate filesystem from `tempdir` root with module hierarchy &etc. for later exposure (exmod)

        :param tempdir: Temporary directory
        :type tempdir: ```str```

        :return: tempdir
        :rtype: ```str```
        """
        self.module_name, self.gold_dir = path.basename(tempdir), tempdir
        self.parent_name, self.parent_dir = "parent", "parent_dir"
        self.child_name, self.child_dir = "child", path.join(
            self.parent_dir, "child_dir"
        )
        self.grandchild_name, self.grandchild_dir = "grandchild", path.join(
            self.child_dir, "grandchild_dir"
        )
        self.module_hierarchy = (
            (self.parent_name, self.parent_dir),
            (self.child_name, self.child_dir),
            (self.grandchild_name, self.grandchild_dir),
        )

        with open(
            path.join(tempdir, "setup{extsep}py".format(extsep=extsep)), "wt"
        ) as f:
            f.write(
                setup_py_mock.format(encoding=ENCODING, package_name=self.module_name)
            )

        open(path.join(tempdir, "README{extsep}md".format(extsep=extsep)), "a").close()
        mkdir(path.join(tempdir, self.module_name))
        with open(
            path.join(tempdir, self.module_name, INIT_FILENAME),
            "wt",
        ) as f:
            f.write(
                "{encoding}\n\n"
                "{imports}\n"
                "__author__ = {author!r}\n"
                "__version__ = {version!r}\n\n"
                "{all__}\n".format(
                    encoding=ENCODING,
                    imports="\n".join(
                        (
                            "import {module_name}.{other_imports}\n".format(
                                module_name=self.module_name,
                                other_imports="\nimport {module_name}.".format(
                                    module_name=self.module_name
                                ).join(
                                    map(
                                        rpartial(str.replace, path.sep, "."),
                                        map(itemgetter(1), self.module_hierarchy),
                                    )
                                ),
                            ),
                        )
                    ),
                    # module_name=self.module_name,
                    # parent_name=self.parent_name,
                    # cls_name="{name}Class".format(name=self.parent_name.title()),
                    author=environ.get("CDD_AUTHOR", "Samuel Marks"),
                    version=environ.get("CDD_VERSION", "0.0.0"),
                    all__="__all__ = {__all__!r}".format(
                        __all__=list(
                            map(
                                rpartial(add, "_dir"),
                                (
                                    self.parent_name,
                                    self.child_name,
                                    self.grandchild_name,
                                ),
                            )
                        )
                    ),
                )
            )

        for name, _folder in self.module_hierarchy:
            folder = path.join(tempdir, self.module_name, _folder)
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

        return tempdir

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
            gen_folder = path.join(tempdir, new_module_name, folder)
            gold_folder = path.join(self.gold_dir, self.module_name, folder)

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

            self.assertTrue(path.isdir(gold_folder))

            gen_is_dir = path.isdir(gen_folder)
            if dry_run:
                self.assertFalse(gen_is_dir)
            else:
                self.assertTrue(gen_is_dir)

                with _open(gen_folder) as gen, _open(gold_folder) as gold:
                    gen_ir, gold_ir = map(
                        lambda ir: ir["_internal"].__delitem__("original_doc_str")
                        or ir,
                        map(
                            lambda node: parse.class_(
                                next(
                                    filter(
                                        rpartial(isinstance, ClassDef),
                                        ast_parse(node.read()).body,
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
            call(
                [executable, "-m", "pip"] + pip_args,
                cwd=cwd,
                stdout=DEVNULL,
                stderr=DEVNULL,
            ),
            0,
            "EXIT_SUCCESS not reached",
        )


unittest_main()
