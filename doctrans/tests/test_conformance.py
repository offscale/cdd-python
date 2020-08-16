"""
Tests for reeducation
"""
import os
from argparse import Namespace
from copy import deepcopy
from functools import partial
from io import StringIO
from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from doctrans import emit
from doctrans.conformance import _get_name_from_namespace, ground_truth
from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import intermediate_repr
from doctrans.tests.mocks.methods import (
    class_with_method_types_ast,
    class_with_method_ast,
)
from doctrans.tests.utils_for_tests import unittest_main

"""
# type: Final[bool]
"""
modified = False


class TestConformance(TestCase):
    """
    Tests must comply. They shall be assimilated.
    """

    def test_ground_truth(self) -> None:
        """ Straight from the ministry. Absolutely. """

        with TemporaryDirectory() as tempdir:
            self.assertTupleEqual(
                tuple(
                    map(
                        lambda filename_unmodified: (
                            os.path.basename(filename_unmodified[0]),
                            filename_unmodified[1],
                        ),
                        self.ground_truth_tester(tempdir=tempdir,)[0].items(),
                    )
                ),
                (("argparse.py", False), ("classes.py", False), ("methods.py", True)),
            )

    def test_ground_truths(self) -> None:
        """ My truth is being tested. """

        with TemporaryDirectory() as tempdir:
            tempdir_join = partial(path.join, tempdir)

            argparse_functions = [
                (
                    lambda argparse_function: emit.file(
                        argparse_func_ast, argparse_function, mode="wt"
                    )
                    or argparse_function
                )(tempdir_join("argparse{i}.py".format(i=i)))
                for i in range(10)
            ]

            class_ = tempdir_join("classes.py")
            emit.file(class_ast, class_, mode="wt")

            function = tempdir_join("methods.py")
            emit.file(class_with_method_ast, function, mode="wt")

            args = Namespace(
                **{
                    "argparse_functions": argparse_functions,
                    "argparse_function_names": ("set_cli_args",),
                    "classes": (class_,),
                    "class_names": ("ConfigClass",),
                    "functions": (function,),
                    "function_names": ("C.method_name",),
                    "truth": "argparse_function",
                }
            )
            with patch("sys.stdout", new_callable=StringIO), patch(
                "sys.stderr", new_callable=StringIO
            ):
                res = ground_truth(args, argparse_functions[0],)

            self.assertTupleEqual(
                tuple(
                    map(
                        lambda filename_unmodified: (
                            os.path.basename(filename_unmodified[0]),
                            filename_unmodified[1],
                        ),
                        res.items(),
                    )
                ),
                (
                    ("argparse0.py", False),
                    ("argparse1.py", True),
                    ("argparse2.py", True),
                    ("argparse3.py", True),
                    ("argparse4.py", True),
                    ("argparse5.py", True),
                    ("argparse6.py", True),
                    ("argparse7.py", True),
                    ("argparse8.py", True),
                    ("argparse9.py", True),
                    ("classes.py", False),
                    ("methods.py", True),
                ),
            )

    def test_ground_truth_fails(self) -> None:
        """ Straight from the fake news ministry. """

        with TemporaryDirectory() as tempdir:
            args = self.ground_truth_tester(tempdir=tempdir,)[1]

            with patch("sys.stdout", new_callable=StringIO), patch(
                "sys.stderr", new_callable=StringIO
            ):
                self.assertRaises(
                    NotImplementedError,
                    lambda: ground_truth(
                        Namespace(
                            **{
                                "argparse_functions": args.argparse_functions,
                                "argparse_function_names": ("set_cli_args",),
                                "classes": args.classes,
                                "class_names": ("ConfigClass",),
                                "functions": args.functions,
                                "function_names": ("C.method_name.A",),
                                "truth": "argparse_function",
                            }
                        ),
                        args.argparse_functions[0],
                    ),
                )

    def test_ground_truth_changes(self) -> None:
        """ Time for a new master. """

        ir = deepcopy(intermediate_repr)
        ir["returns"]["typ"] = "Tuple[np.ndarray, np.ndarray]"

        with TemporaryDirectory() as tempdir:
            self.assertTupleEqual(
                tuple(
                    map(
                        lambda filename_unmodified: (
                            os.path.basename(filename_unmodified[0]),
                            filename_unmodified[1],
                        ),
                        self.ground_truth_tester(
                            tempdir=tempdir, _class_ast=emit.class_(ir),
                        )[0].items(),
                    )
                ),
                (("argparse.py", False), ("classes.py", True), ("methods.py", True)),
            )

    @staticmethod
    def ground_truth_tester(
        tempdir,
        _argparse_func_ast=argparse_func_ast,
        _class_ast=class_ast,
        _class_with_method_ast=class_with_method_types_ast,
    ):
        """
        Helper for ground_truth tests

        :param tempdir: temporary directory
        :type tempdir: ```str```

        :param _argparse_func_ast: AST node
        :type _argparse_func_ast: ```FunctionDef```

        :param _class_ast: AST node
        :type _class_ast: ```ClassDef```

        :param _class_with_method_ast: AST node
        :type _class_with_method_ast: ```ClassDef```

        :returns: OrderedDict of filenames and whether they were changed, Args
        :rtype: ```Tuple[OrderedDict, Namespace]```
        """
        argparse_function = os.path.join(tempdir, "argparse.py")
        emit.file(_argparse_func_ast, argparse_function, mode="wt")

        class_ = os.path.join(tempdir, "classes.py")
        emit.file(_class_ast, class_, mode="wt")

        function = os.path.join(tempdir, "methods.py")
        emit.file(_class_with_method_ast, function, mode="wt")

        args = Namespace(
            **{
                "argparse_functions": (argparse_function,),
                "argparse_function_names": ("set_cli_args",),
                "classes": (class_,),
                "class_names": ("ConfigClass",),
                "functions": (function,),
                "function_names": ("C.method_name",),
                "truth": "argparse_function",
            }
        )

        with patch("sys.stdout", new_callable=StringIO), patch(
            "sys.stderr", new_callable=StringIO
        ):
            return ground_truth(args, argparse_function,), args

    def test__get_name_from_namespace(self) -> None:
        """ Test `_get_name_from_namespace` """
        args = Namespace(foo_names=("bar",))
        self.assertEqual(_get_name_from_namespace(args, "foo"), args.foo_names[0])


unittest_main()
