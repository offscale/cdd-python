""" Tests for exmod subcommand """

from ast import ClassDef
from operator import add, itemgetter
from os import environ, mkdir, path
from os.path import extsep
from subprocess import DEVNULL, call
from sys import executable
from tempfile import TemporaryDirectory
from unittest import TestCase

from cdd import parse
from cdd.exmod import exmod
from cdd.pure_utils import ENCODING, rpartial
from cdd.source_transformer import ast_parse
from cdd.tests.mocks import imports_header
from cdd.tests.mocks.classes import class_str
from cdd.tests.mocks.exmod import setup_py_mock
from cdd.tests.utils_for_tests import unittest_main


class TestExMod(TestCase):
    """Test class for exmod.py"""

    def test_exmod_blacklist(self) -> None:
        """Tests `exmod` blacklist"""

        with TemporaryDirectory() as tempdir, self.assertRaises(NotImplementedError):
            exmod(
                module="unittest",
                emit_name=None,
                blacklist=("unittest.TestCase",),
                whitelist=tuple(),
                output_directory=tempdir,
            )

    def test_exmod_whitelist(self) -> None:
        """Tests `exmod` whitelist"""

        with TemporaryDirectory() as tempdir, self.assertRaises(NotImplementedError):
            exmod(
                module="unittest",
                emit_name=None,
                blacklist=tuple(),
                whitelist=("unittest.TestCase",),
                output_directory=tempdir,
            )

    def test_exmod_module_directory(self) -> None:
        """Tests `exmod` module whence directory"""

        with TemporaryDirectory() as tempdir, self.assertRaises(NotImplementedError):
            exmod(
                module=tempdir,
                emit_name=None,
                blacklist=["foo", "bar"],
                whitelist=tuple(),
                output_directory=tempdir,
            )

    def test_exmod_output_directory_nonexistent(self) -> None:
        """Tests `exmod` module whence directory does not exist"""

        with TemporaryDirectory() as tempdir, self.assertRaises(AssertionError):
            output_directory = path.join(tempdir, "stuff")
            self.assertFalse(path.isdir(output_directory))
            exmod(
                module=output_directory,
                emit_name=None,
                blacklist=tuple(),
                whitelist=tuple(),
                output_directory=output_directory,
            )

    def create_fs(self, tempdir):
        """
        Populate filesystem from `tempdir` root with module hierarchy &etc. for later exposure (exmod)

        :param tempdir: Temporary directory
        :type tempdir: ```str```

        :returns: tempdir
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
            path.join(
                tempdir, self.module_name, "__init__{extsep}py".format(extsep=extsep)
            ),
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
                    all__="__all__ = {!r}".format(
                        list(
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

    def check_emission(self, tempdir):
        """
        Confirm whether emission conforms to gen by verifying their IRs are equivalent

        :param tempdir: Temporary directory
        :type tempdir: ```str```
        """
        new_module_name = path.basename(tempdir)

        for name, folder in self.module_hierarchy:
            gen_folder = path.join(tempdir, new_module_name, folder)
            gold_folder = path.join(self.gold_dir, self.module_name, folder)

            def _open(folder):
                """
                :param folder: Folder to join on
                :type folder: ```str``

                :returns: Open IO
                :rtype: ```open```
                """
                return open(
                    path.join(
                        folder, "{name}{extsep}py".format(name=name, extsep=extsep)
                    ),
                    "rt",
                )

            self.assertTrue(path.isdir(gold_folder))
            self.assertTrue(path.isdir(gen_folder))

            with _open(gen_folder) as gen, _open(gold_folder) as gold:
                gen_ir, gold_ir = map(
                    lambda node: parse.class_(
                        next(
                            filter(
                                rpartial(isinstance, ClassDef),
                                ast_parse(node.read()).body,
                            )
                        )
                    ),
                    (gen, gold),
                )
                self.assertDictEqual(gold_ir, gen_ir)

    def test_exmod(self) -> None:
        """Tests `exmod`"""

        def _pip(pip_args, cwd=None):
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

        try:
            with TemporaryDirectory(
                prefix="gold", suffix="gold"
            ) as existent_module_dir, TemporaryDirectory(
                prefix="gen", suffix="gen"
            ) as new_module_dir:
                self.create_fs(existent_module_dir)
                _pip(["install", "."], existent_module_dir)
                exmod(
                    module=self.module_name,
                    emit_name="class",
                    blacklist=tuple(),
                    whitelist=tuple(),
                    output_directory=new_module_dir,
                )
                self.check_emission(new_module_dir)
        finally:
            _pip(["uninstall", "-y", self.module_name])


unittest_main()
