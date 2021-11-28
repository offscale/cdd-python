"""
Tests for reeducation
"""

import os
from argparse import Namespace
from ast import FunctionDef
from copy import deepcopy
from functools import partial
from io import StringIO
from os import path
from os.path import extsep
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cdd import emit
from cdd.conformance import _conform_filename, _get_name_from_namespace, ground_truth
from cdd.tests.mocks.argparse import argparse_func_ast
from cdd.tests.mocks.classes import class_ast_no_default_doc
from cdd.tests.mocks.ir import intermediate_repr
from cdd.tests.mocks.methods import class_with_method_ast, class_with_method_types_ast
from cdd.tests.utils_for_tests import unittest_main

"""
# type: Final[bool]
"""
modified = False


class TestConformance(TestCase):
    """
    Tests must comply. They shall be assimilated.
    """

    def test_ground_truth(self) -> None:
        """Straight from the ministry. Absolutely."""

        with TemporaryDirectory() as tempdir:
            self.assertTupleEqual(
                tuple(
                    map(
                        lambda filename_unmodified: (
                            os.path.basename(filename_unmodified[0]),
                            filename_unmodified[1],
                        ),
                        self.ground_truth_tester(
                            tempdir=tempdir,
                        )[0].items(),
                    )
                ),
                (
                    ("argparse{extsep}py".format(extsep=extsep), False),
                    ("classes{extsep}py".format(extsep=extsep), False),
                    ("methods{extsep}py".format(extsep=extsep), False),
                ),
            )

    def test_ground_truths(self) -> None:
        """My truth is being tested."""

        with TemporaryDirectory() as tempdir:
            tempdir_join = partial(path.join, tempdir)

            argparse_functions = list(
                map(
                    lambda i: (
                        lambda argparse_function: emit.file(
                            argparse_func_ast, argparse_function, mode="wt"
                        )
                        or argparse_function
                    )(tempdir_join("argparse{i}{extsep}py".format(i=i, extsep=extsep))),
                    range(10),
                )
            )
            # Test if can create missing file
            argparse_functions.append(
                tempdir_join("argparse_missing{extsep}py".format(extsep=extsep))
            )

            # Test if can fill in empty file
            argparse_functions.append(
                tempdir_join("argparse_empty{extsep}py".format(extsep=extsep))
            )

            class_ = tempdir_join("classes{extsep}py".format(extsep=extsep))
            emit.file(class_ast_no_default_doc, class_, mode="wt")

            function = tempdir_join("methods{extsep}py".format(extsep=extsep))
            emit.file(class_with_method_ast, function, mode="wt")

            args = Namespace(
                **{
                    "argparse_functions": argparse_functions,
                    "argparse_function_names": ("set_cli_args",),
                    "classes": (class_,),
                    "class_names": ("ConfigClass",),
                    "functions": (function,),
                    "function_names": ("C.function_name",),
                    "truth": "argparse_function",
                    "no_word_wrap": None,
                }
            )
            with patch("sys.stdout", new_callable=StringIO), patch(
                "sys.stderr", new_callable=StringIO
            ):
                res = ground_truth(
                    args,
                    argparse_functions[0],
                )

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
                    ("argparse0{extsep}py".format(extsep=extsep), False),
                    ("argparse1{extsep}py".format(extsep=extsep), False),
                    ("argparse2{extsep}py".format(extsep=extsep), False),
                    ("argparse3{extsep}py".format(extsep=extsep), False),
                    ("argparse4{extsep}py".format(extsep=extsep), False),
                    ("argparse5{extsep}py".format(extsep=extsep), False),
                    ("argparse6{extsep}py".format(extsep=extsep), False),
                    ("argparse7{extsep}py".format(extsep=extsep), False),
                    ("argparse8{extsep}py".format(extsep=extsep), False),
                    ("argparse9{extsep}py".format(extsep=extsep), False),
                    ("argparse_missing{extsep}py".format(extsep=extsep), True),
                    ("argparse_empty{extsep}py".format(extsep=extsep), True),
                    ("classes{extsep}py".format(extsep=extsep), False),
                    ("methods{extsep}py".format(extsep=extsep), False),
                ),
            )

    def test_ground_truth_changes(self) -> None:
        """Time for a new master."""

        ir = deepcopy(intermediate_repr)
        ir["returns"]["return_type"]["typ"] = "Tuple[np.ndarray, np.ndarray]"

        with TemporaryDirectory() as tempdir:
            self.assertTupleEqual(
                tuple(
                    map(
                        lambda filename_unmodified: (
                            os.path.basename(filename_unmodified[0]),
                            filename_unmodified[1],
                        ),
                        self.ground_truth_tester(
                            tempdir=tempdir,
                            _class_ast=emit.class_(ir, emit_default_doc=False),
                        )[0].items(),
                    )
                ),
                (
                    ("argparse{extsep}py".format(extsep=extsep), False),
                    ("classes{extsep}py".format(extsep=extsep), True),
                    ("methods{extsep}py".format(extsep=extsep), False),
                ),
            )

    @staticmethod
    def ground_truth_tester(
        tempdir,
        _argparse_func_ast=argparse_func_ast,
        _class_ast=class_ast_no_default_doc,
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

        :return: OrderedDict of filenames and whether they were changed, Args
        :rtype: ```Tuple[OrderedDict, Namespace]```
        """
        argparse_function = os.path.join(
            tempdir, "argparse{extsep}py".format(extsep=extsep)
        )
        emit.file(_argparse_func_ast, argparse_function, mode="wt")

        class_ = os.path.join(tempdir, "classes{extsep}py".format(extsep=extsep))
        emit.file(_class_ast, class_, mode="wt")

        function = os.path.join(tempdir, "methods{extsep}py".format(extsep=extsep))
        emit.file(_class_with_method_ast, function, mode="wt")

        args = Namespace(
            **{
                "argparse_functions": (argparse_function,),
                "argparse_function_names": ("set_cli_args",),
                "classes": (class_,),
                "class_names": ("ConfigClass",),
                "functions": (function,),
                "function_names": ("C.function_name",),
                "truth": "argparse_function",
                "no_word_wrap": None,
            }
        )

        with patch("sys.stdout", new_callable=StringIO), patch(
            "sys.stderr", new_callable=StringIO
        ):
            return (
                ground_truth(
                    args,
                    argparse_function,
                ),
                args,
            )

    def test__get_name_from_namespace(self) -> None:
        """Test `_get_name_from_namespace`"""
        args = Namespace(foo_names=("bar",))
        self.assertEqual(_get_name_from_namespace(args, "foo"), args.foo_names[0])

    def test__conform_filename_nonexistent(self) -> None:
        """Tests that _conform_filename returns the right result"""

        with TemporaryDirectory() as tempdir:
            argparse_function_filename = os.path.realpath(
                os.path.join(tempdir, "no_file_here{extsep}py".format(extsep=extsep))
            )

            self.assertTupleEqual(
                _conform_filename(
                    filename=argparse_function_filename,
                    search=["set_cli_args"],
                    emit_func=emit.argparse_function,
                    replacement_node_ir=deepcopy(intermediate_repr),
                    type_wanted=FunctionDef,
                ),
                (argparse_function_filename, True),
            )

    def test__conform_filename_filled(self) -> None:
        """Tests that _conform_filename returns the right result"""

        with TemporaryDirectory() as tempdir:
            argparse_function_filename = os.path.realpath(
                os.path.join(
                    tempdir, "correct_contents{extsep}py".format(extsep=extsep)
                )
            )

            emit.file(
                argparse_func_ast,
                argparse_function_filename,
                mode="wt",
            )

            self.assertTupleEqual(
                _conform_filename(
                    filename=argparse_function_filename,
                    search=["impossibru"],
                    emit_func=emit.argparse_function,
                    replacement_node_ir=deepcopy(intermediate_repr),
                    type_wanted=FunctionDef,
                ),
                (argparse_function_filename, True),
            )

    def test__conform_filename_unchanged(self) -> None:
        """Tests that _conform_filename returns the right result"""

        with TemporaryDirectory() as tempdir:
            argparse_function_filename = os.path.realpath(
                os.path.join(
                    tempdir, "do_not_touch_this{extsep}py".format(extsep=extsep)
                )
            )

            emit.file(argparse_func_ast, argparse_function_filename, mode="wt")
            with patch("sys.stdout", new_callable=StringIO):
                self.assertTupleEqual(
                    _conform_filename(
                        filename=argparse_function_filename,
                        search=["set_cli_args"],
                        emit_func=emit.argparse_function,
                        replacement_node_ir=deepcopy(intermediate_repr),
                        type_wanted=FunctionDef,
                    ),
                    (argparse_function_filename, False),
                )


unittest_main()
