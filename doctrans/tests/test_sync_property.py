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

    def test_sync_properties(self) -> None:
        """ Tests `sync_properties` with `call=False` """

        with TemporaryDirectory() as tempdir:
            class_py = os.path.join(tempdir, "class_.py")
            method_py = os.path.join(tempdir, "method.py")

            class_py_str = (
                "from typing import Literal\n\n"
                "class Foo(object):\n"
                "{tab}def g(f: Literal['a']):\n"
                "{tab}{tab}pass".format(tab=tab)
            )
            method_py_str = (
                "from typing import Literal\n\n"
                "def f(h: Literal['b']):"
                "{tab}{tab}pass".format(tab=tab)
            )

            with open(class_py, "wt") as f:
                f.write(class_py_str)
            with open(method_py, "wt") as f:
                f.write(method_py_str)

            self.assertIsNone(
                sync_properties(
                    input_file=class_py,
                    input_params=("Foo.g.f",),
                    input_eval=False,
                    output_file=method_py,
                    output_params=("f.h",),
                )
            )

            # Confirm that the class is unedited
            with open(class_py, "rt") as f:
                run_ast_test(
                    self, gen_ast=ast.parse(class_py_str), gold=ast.parse(f.read())
                )

            # Confirm that the method is edited correctly
            with open(method_py, "rt") as f:
                run_ast_test(
                    self,
                    gen_ast=ast.parse(f.read()),
                    gold=ast.parse(
                        method_py_str.replace("h: Literal['b']", "f: Literal['a']")
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

            method_py_str = (
                "from typing import Literal\n\n"
                "def f(h: Literal['b']):"
                "{tab}{tab}pass".format(tab=tab)
            )
            with open(input_file, "wt") as f:
                f.write(eval_mock_str)
            with open(output_file, "wt") as f:
                f.write(method_py_str)

            self.assertIsNone(
                sync_properties(
                    input_file=input_file,
                    input_params=("get_modules",),
                    input_eval=True,
                    output_file=output_file,
                    output_params=("f.h",),
                )
            )

            # Confirm that the class is unedited
            with open(input_file, "rt") as f:
                run_ast_test(
                    self, gen_ast=ast.parse(eval_mock_str), gold=ast.parse(f.read())
                )

            # Confirm that the method is edited correctly
            with open(output_file, "rt") as f:
                run_ast_test(
                    self,
                    gen_ast=ast.parse(f.read()),
                    gold=ast.parse(
                        method_py_str.replace("h: Literal['b']", "h: Literal['mocks']")
                    ),
                )

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


unittest_main()
