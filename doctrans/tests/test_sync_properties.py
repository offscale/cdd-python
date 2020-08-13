""" Tests for sync_properties """
import ast
import os
from sys import modules
from tempfile import TemporaryDirectory
from unittest import TestCase

from pkg_resources import resource_filename

from doctrans.pure_utils import tab
from doctrans.sync_properties import sync_properties
from doctrans.tests.mocks.eval import get_modules
from doctrans.tests.utils_for_tests import unittest_main, run_ast_test


class TestSyncProperties(TestCase):
    """ Test class for sync_properties.py """

    method_py_str = (
        "from typing import Literal\n\n"
        "def f(h: Literal['b']):"
        "{tab}{tab}pass".format(tab=tab)
    )

    def test_sync_properties(self) -> None:
        """ Tests `sync_properties` with `call=False` """

        with TemporaryDirectory() as tempdir:
            class_py, class_py_str, method_py = self.populate_files(tempdir)

            self.assertIsNone(
                sync_properties(
                    input_file=class_py,
                    input_params=("Foo.g.f",),
                    input_eval=False,
                    output_file=method_py,
                    output_params=("f.h",),
                )
            )

            self.run_sync_properties_test(
                class_py,
                class_py_str,
                method_py,
                ast.parse(
                    self.method_py_str.replace("h: Literal['b']", "f: Literal['a']")
                ),
            )

    def test_sync_properties_eval(self) -> None:
        """ Tests `sync_properties` with `call=True` """

        eval_mock_py = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    resource_filename(modules[__name__].__name__, "__init__.py")
                )
            ),
            "tests",
            "mocks",
            "eval.py",
        )

        with TemporaryDirectory() as tempdir:
            input_file = os.path.join(tempdir, "input_.py")
            output_file = os.path.join(tempdir, "output.py")

            with open(eval_mock_py, "rt") as f:
                eval_mock_str = f.read()

            with open(input_file, "wt") as f:
                f.write(eval_mock_str)
            with open(output_file, "wt") as f:
                f.write(self.method_py_str)

            self.assertIsNone(
                sync_properties(
                    input_file=input_file,
                    input_params=("get_modules",),
                    input_eval=True,
                    output_file=output_file,
                    output_params=("f.h",),
                )
            )

            self.run_sync_properties_test(
                input_filename=input_file,
                output_filename=output_file,
                output=eval_mock_str,
                gold=ast.parse(
                    self.method_py_str.replace("h: Literal['b']", "h: Literal['mocks']")
                ),
            )

    def test_sync_properties_output_param_wrap(self) -> None:
        """ Tests `sync_properties` with `output_param_wrap` set """

        with TemporaryDirectory() as tempdir:
            input_file, output, output_file = self.populate_files(tempdir)

            self.assertIsNone(
                sync_properties(
                    input_file=input_file,
                    input_params=("Foo.g.f",),
                    input_eval=False,
                    output_file=output_file,
                    output_params=("f.h",),
                    output_param_wrap="Optional[List[Union[{output_param}, str]]]",
                )
            )

            self.run_sync_properties_test(
                input_filename=input_file,
                output_filename=output_file,
                output=output,
                gold=ast.parse(
                    self.method_py_str.replace(
                        "h: Literal['b']", "f: Optional[List[Union[Literal['a'], str]]]"
                    )
                ),
            )

    def populate_files(self, tempdir):
        """
        Populate files in the tempdir

        :param tempdir: Temporary directory
        :type tempdir: ```str```

        :returns: input filename, output str, output filename
        :rtype: ```Tuple[str, str, str]```
        """
        class_py = os.path.join(tempdir, "class_.py")
        method_py = os.path.join(tempdir, "method.py")
        class_py_str = (
            "from typing import Literal\n\n"
            "class Foo(object):\n"
            "{tab}def g(f: Literal['a']):\n"
            "{tab}{tab}pass".format(tab=tab)
        )
        with open(class_py, "wt") as f:
            f.write(class_py_str)
        with open(method_py, "wt") as f:
            f.write(self.method_py_str)
        return class_py, class_py_str, method_py

    def test_sync_properties_eval_fails(self) -> None:
        """ Tests `sync_properties` fails with `call=True` and dots """
        with TemporaryDirectory() as tempdir:
            input_file = os.path.join(tempdir, "input_.py")
            output_file = os.path.join(tempdir, "output.py")

            open(input_file, "wt").close()
            open(output_file, "wt").close()

            self.assertRaises(
                NotImplementedError,
                lambda: sync_properties(
                    input_file=input_file,
                    input_params=("foo.bar",),
                    input_eval=True,
                    output_file=output_file,
                    output_params=("f.h",),
                ),
            )

    def test_eval(self) -> None:
        """ Ensure mock returns the right result """
        self.assertTupleEqual(get_modules, ("mocks",))

    def run_sync_properties_test(self, input_filename, output, output_filename, gold):
        """
        Common test for the suite

        :param input_filename: Filename of input
        :type input_filename: ```str```

        :param output: Python source
        :type output: ```str```

        :param output_filename: Filename of output
        :type output_filename: ```str```

        :param gold: Gold standard AST
        :type gold: ```Union[ast.Module, ast.ClassDef, ast.FunctionDef]```
        """
        # Confirm that the class is unedited
        with open(input_filename, "rt") as f:
            run_ast_test(self, gen_ast=ast.parse(output), gold=ast.parse(f.read()))
        # Confirm that the method is edited correctly
        with open(output_filename, "rt") as f:
            run_ast_test(
                self, gen_ast=ast.parse(f.read()), gold=gold,
            )


unittest_main()
