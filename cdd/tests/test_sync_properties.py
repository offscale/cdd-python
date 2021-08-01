""" Tests for sync_properties """

import ast
import os
from os.path import extsep
from sys import modules
from tempfile import TemporaryDirectory
from unittest import TestCase

from pkg_resources import resource_filename

from cdd.pure_utils import INIT_FILENAME, PY_GTE_3_8, tab
from cdd.sync_properties import sync_properties
from cdd.tests.mocks.eval import get_modules
from cdd.tests.utils_for_tests import run_ast_test, unittest_main


def populate_files(tempdir, input_str=None, output_str=None):
    """
    Populate files in the tempdir

    :param tempdir: Temporary directory
    :type tempdir: ```str```

    :param input_str: Input string to write to the input_filename
    :type input_str: ```Optional[str]```

    :param output_str: Output string to write to the output_filename
    :type output_str: ```Optional[str]```

    :returns: input filename, input str, input_str filename
    :rtype: ```Tuple[str, str, str]```
    """
    input_filename = os.path.join(tempdir, "class_{extsep}py".format(extsep=extsep))
    output_filename = os.path.join(tempdir, "method{extsep}py".format(extsep=extsep))
    input_str = input_str or (
        "from {package} import Literal\n\n"
        "class Foo(object):\n"
        "{tab}def g(f: Literal['a']):\n"
        "{tab}{tab}pass".format(
            package="typing" if PY_GTE_3_8 else "typing_extensions", tab=tab
        )
    )
    output_str = output_str or (
        "from {package} import Literal\n\n"
        "def f(h: Literal['b']):"
        "{tab}{tab}pass".format(
            package="typing" if PY_GTE_3_8 else "typing_extensions", tab=tab
        )
    )
    with open(input_filename, "wt") as f:
        f.write(input_str)
    with open(output_filename, "wt") as f:
        f.write(output_str)
    return input_filename, input_str, output_filename, output_str


class TestSyncProperties(TestCase):
    """Test class for sync_properties.py"""

    def test_sync_properties(self) -> None:
        """Tests `sync_properties` with `call=False`"""

        with TemporaryDirectory() as tempdir:
            (
                input_filename,
                input_str,
                output_filename,
                output_str,
            ) = populate_files(tempdir)

            self.assertIsNone(
                sync_properties(
                    input_filename=input_filename,
                    input_params=("Foo.g.f",),
                    input_eval=False,
                    output_filename=output_filename,
                    output_params=("f.h",),
                )
            )

            self.run_sync_properties_test(
                input_filename,
                input_str,
                output_filename,
                ast.parse(output_str.replace("h: Literal['b']", "f: Literal['a']")),
            )

    def test_sync_properties_eval(self) -> None:
        """Tests `sync_properties` with `call=True`"""

        with open(
            os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        resource_filename(
                            modules[__name__].__name__,
                            INIT_FILENAME,
                        )
                    )
                ),
                "tests",
                "mocks",
                "eval{extsep}py".format(extsep=extsep),
            ),
            "rt",
        ) as f:
            eval_mock_str = f.read()

        with TemporaryDirectory() as tempdir:
            (
                input_filename,
                input_str,
                output_filename,
                output_str,
            ) = populate_files(tempdir, input_str=eval_mock_str)

            self.assertIsNone(
                sync_properties(
                    input_filename=input_filename,
                    input_params=("get_modules",),
                    input_eval=True,
                    output_filename=output_filename,
                    output_params=("f.h",),
                )
            )

            self.run_sync_properties_test(
                input_filename=input_filename,
                output_filename=output_filename,
                input_str=eval_mock_str,
                gold=ast.parse(
                    output_str.replace("h: Literal['b']", "h: Literal['mocks']")
                ),
            )

    def test_sync_properties_output_param_wrap(self) -> None:
        """Tests `sync_properties` with `output_param_wrap` set"""

        with TemporaryDirectory() as tempdir:
            (
                input_filename,
                input_str,
                output_filename,
                output_str,
            ) = populate_files(tempdir)

            self.assertIsNone(
                sync_properties(
                    input_filename=input_filename,
                    input_params=("Foo.g.f",),
                    input_eval=False,
                    output_filename=output_filename,
                    output_params=("f.h",),
                    output_param_wrap="Optional[List[Union[{output_param}, str]]]",
                )
            )

            self.run_sync_properties_test(
                input_filename=input_filename,
                output_filename=output_filename,
                input_str=input_str,
                gold=ast.parse(
                    output_str.replace(
                        "h: Literal['b']", "f: Optional[List[Union[Literal['a'], str]]]"
                    )
                ),
            )

    def test_sync_properties_output_param_wrap_no_annotation(self) -> None:
        """Tests `sync_properties` with `output_param_wrap` set when type annotation isn't being replaced"""

        with TemporaryDirectory() as tempdir:
            (input_filename, input_str, output_filename, output_str,) = populate_files(
                tempdir,
                input_str=(
                    "from {package} import Literal\n\n"
                    "class Foo(object):\n"
                    "{tab}def g(f):\n"
                    "{tab}{tab}pass".format(
                        package="typing" if PY_GTE_3_8 else "typing_extensions", tab=tab
                    )
                ),
            )

            self.assertIsNone(
                sync_properties(
                    input_filename=input_filename,
                    input_params=("Foo.g.f",),
                    input_eval=False,
                    output_filename=output_filename,
                    output_params=("f.h",),
                    output_param_wrap="Optional[List[Union[{output_param}, str]]]",
                )
            )

            self.run_sync_properties_test(
                input_filename=input_filename,
                output_filename=output_filename,
                input_str=input_str,
                gold=ast.parse(output_str.replace("h: Literal['b']", "f")),
            )

    def test_sync_properties_output_param_wrap_no_type(self) -> None:
        """Tests `sync_properties` with `output_param_wrap` set when replacement_node has no type"""

        with TemporaryDirectory() as tempdir:
            (input_filename, input_str, output_filename, output_str,) = populate_files(
                tempdir,
                output_str=(
                    "from {package} import Literal\n\n"
                    "class Foo(object):\n"
                    "{tab}def f(h):\n"
                    "{tab}{tab}pass".format(
                        package="typing" if PY_GTE_3_8 else "typing_extensions", tab=tab
                    )
                ),
            )

            self.assertIsNone(
                sync_properties(
                    input_filename=input_filename,
                    input_params=("Foo.g.f",),
                    input_eval=False,
                    output_filename=output_filename,
                    output_params=("Foo.f.h",),
                    output_param_wrap="Optional[List[Union[{output_param}, str]]]",
                )
            )

            self.run_sync_properties_test(
                input_filename=input_filename,
                output_filename=output_filename,
                input_str=input_str,
                gold=ast.parse(
                    output_str.replace(
                        "(h)", "(f: Optional[List[Union[Literal['a'], str]]])"
                    )
                ),
            )

    def test_sync_properties_output_param_wrap_subscript(self) -> None:
        """Tests `sync_properties` with `output_param_wrap` set when replacement_node is subscript and !input_eval"""

        with TemporaryDirectory() as tempdir:
            (input_filename, input_str, output_filename, output_str,) = populate_files(
                tempdir,
                input_str="a = tuple(range(5))",
                output_str="def j(k):\n" "{tab}pass\n".format(tab=tab),
            )

            self.assertIsNone(
                sync_properties(
                    input_filename=input_filename,
                    input_params=("a",),
                    input_eval=False,
                    output_filename=output_filename,
                    output_params=("j.k",),
                    output_param_wrap=None,
                )
            )

            with open(output_filename, "rt") as f:
                # Technically this produces an invalid AST, but we don't care… `input_eval is False`
                self.assertEqual(
                    f.read().rstrip(),
                    output_str.replace("(k)", "(a: tuple(range(5)))").rstrip(),
                )

    def test_sync_properties_output_param_wrap_subscript_eval0(self) -> None:
        """Tests `sync_properties` with `output_param_wrap` set when replacement_node is subscript"""

        with TemporaryDirectory() as tempdir:
            (input_filename, input_str, output_filename, output_str,) = populate_files(
                tempdir, input_str="a = tuple(range(5))", output_str="def j(k): pass"
            )

            self.assertIsNone(
                sync_properties(
                    input_filename=input_filename,
                    input_params=("a",),
                    input_eval=True,
                    output_filename=output_filename,
                    output_params=("j.k",),
                    output_param_wrap=None,
                )
            )

            self.run_sync_properties_test(
                input_filename=input_filename,
                output_filename=output_filename,
                input_str=input_str,
                gold=ast.parse(
                    output_str.replace("(k)", "(k: Literal[0, 1, 2, 3, 4])")
                ),
            )

    def test_sync_properties_output_param_wrap_subscript_eval1(self) -> None:
        """Tests `sync_properties` with `output_param_wrap` set when replacement_node is subscript"""

        with TemporaryDirectory() as tempdir:
            (input_filename, input_str, output_filename, output_str,) = populate_files(
                tempdir,
                input_str="import pip\n"
                "c = { attr: getattr(pip, attr)"
                "      for attr in dir(pip)"
                "      if attr in frozenset(('__version__', '__name__')) }\n"
                "a = tuple(sorted(c.keys()))\n",
                output_str="class C(object):\n"
                "{tab}def j(k: str):\n"
                "{tab}{tab}pass\n".format(tab=tab),
            )

            self.assertIsNone(
                sync_properties(
                    input_filename=input_filename,
                    input_params=("a",),
                    input_eval=True,
                    output_filename=output_filename,
                    output_params=("C.j.k",),
                    output_param_wrap=None,
                )
            )

            self.run_sync_properties_test(
                input_filename=input_filename,
                output_filename=output_filename,
                input_str=input_str,
                gold=ast.parse(
                    output_str.replace(
                        "(k: str)", "(k: Literal['__name__', '__version__'])"
                    )
                ),
            )

    def test_sync_properties_output_param_wrap_fails(self) -> None:
        """Tests `sync_properties` fails with `output_param_wrap` set when replacement_node is unknown"""

        with TemporaryDirectory() as tempdir:
            (
                input_filename,
                input_str,
                output_filename,
                output_str,
            ) = populate_files(tempdir, input_str="local = locals()")

            self.assertRaises(
                NotImplementedError,
                lambda: sync_properties(
                    input_filename=input_filename,
                    input_params=("local",),
                    input_eval=False,
                    output_filename=output_filename,
                    output_params=("f.h",),
                    output_param_wrap="Optional[List[Union[{output_param}, str]]]",
                ),
            )

    def test_sync_properties_eval_fails(self) -> None:
        """Tests `sync_properties` fails with `call=True` and dots"""
        with TemporaryDirectory() as tempdir:
            input_filename = os.path.join(
                tempdir, "input_{extsep}py".format(extsep=extsep)
            )
            output_filename = os.path.join(
                tempdir, "input_str{extsep}py".format(extsep=extsep)
            )

            open(input_filename, "wt").close()
            open(output_filename, "wt").close()

            self.assertRaises(
                NotImplementedError,
                lambda: sync_properties(
                    input_filename=input_filename,
                    input_params=("foo.bar",),
                    input_eval=True,
                    output_filename=output_filename,
                    output_params=("f.h",),
                ),
            )

    def test_eval(self) -> None:
        """Ensure mock returns the right result"""
        self.assertTupleEqual(get_modules, ("mocks",))

    def run_sync_properties_test(
        self, input_filename, input_str, output_filename, gold
    ):
        """
        Common test for the suite

        :param input_filename: Filename of input
        :type input_filename: ```str```

        :param input_str: Python source
        :type input_str: ```str```

        :param output_filename: Filename of output
        :type output_filename: ```str```

        :param gold: Gold standard AST
        :type gold: ```Union[ast.Module, ast.ClassDef, ast.FunctionDef]```
        """

        # Confirm that the input_filename is unedited
        with open(input_filename, "rt") as f:
            run_ast_test(self, gen_ast=ast.parse(input_str), gold=ast.parse(f.read()))

        # Confirm that the output_filename is edited correctly
        with open(output_filename, "rt") as f:
            run_ast_test(
                self,
                gen_ast=ast.parse(f.read()),
                gold=gold,
            )


unittest_main()
