"""
Tests for reeducation
"""
from argparse import Namespace
from ast import ClassDef, FunctionDef
from copy import deepcopy
from functools import partial
from io import StringIO
from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from doctrans import docstring_struct, transformers
from doctrans.conformance import replace_node, _get_name_from_namespace, ground_truth
from doctrans.pure_utils import rpartial
from doctrans.source_transformer import to_code
from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import docstring_structure
from doctrans.tests.mocks.methods import (
    class_with_method_types_ast,
    class_with_method_and_body_types_ast,
    class_with_method_ast,
)
from doctrans.tests.utils_for_tests import unittest_main

"""
# type: Final[bool]
"""
unchanged = True
modified = False


class TestConformance(TestCase):
    """
    Tests must comply. They shall be assimilated.
    """

    def test_ground_truth(self) -> None:
        """ Straight from the ministry. Absolutely. """

        with TemporaryDirectory() as tmpdir:
            self.assertTupleEqual(
                tuple(self.ground_truth_tester(tmpdir=tmpdir,)[0].values()),
                (unchanged, unchanged, unchanged),
            )

    def test_ground_truths(self) -> None:
        """ My truth is being tested. """

        with TemporaryDirectory() as tmpdir:
            tmpdir_join = partial(path.join, tmpdir)

            argparse_functions = [
                (
                    lambda argparse_function: transformers.to_file(
                        argparse_func_ast, argparse_function, mode="wt"
                    )
                    or argparse_function
                )(tmpdir_join("argparse{i}.py".format(i=i)))
                for i in range(10)
            ]

            class_ = tmpdir_join("classes.py")
            transformers.to_file(class_ast, class_, mode="wt")

            function = tmpdir_join("methods.py")
            transformers.to_file(class_with_method_ast, function, mode="wt")

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
                tuple(res.values()),
                (
                    unchanged,
                    unchanged,
                    unchanged,
                    unchanged,
                    unchanged,
                    unchanged,
                    unchanged,
                    unchanged,
                    unchanged,
                    unchanged,
                    unchanged,
                    modified,
                ),
            )

    def test_ground_truth_fails(self) -> None:
        """ Straight from the fake news ministry. """

        with TemporaryDirectory() as tmpdir:
            args = self.ground_truth_tester(tmpdir=tmpdir,)[1]

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

        _docstring_structure = deepcopy(docstring_structure)
        _docstring_structure["returns"]["typ"] = "Tuple[np.ndarray, np.ndarray]"

        with TemporaryDirectory() as tmpdir:
            self.assertTupleEqual(
                tuple(
                    self.ground_truth_tester(
                        tmpdir=tmpdir,
                        _class_ast=transformers.to_class(_docstring_structure),
                    )[0].values()
                ),
                (unchanged, modified, unchanged),
            )

    @staticmethod
    def ground_truth_tester(
        tmpdir,
        _argparse_func_ast=argparse_func_ast,
        _class_ast=class_ast,
        _class_with_method_ast=class_with_method_types_ast,
    ):
        """
        Helper for ground_truth tests

        :param tmpdir: temporary directory
        :type tmpdir: ```str```

        :param _argparse_func_ast: AST node
        :type _argparse_func_ast: ```FunctionDef```

        :param _class_ast: AST node
        :type _class_ast: ```ClassDef```

        :param _class_with_method_ast: AST node
        :type _class_with_method_ast: ```ClassDef```

        :returns: Tuple of strings showing which files changed/unchanged, Args
        :rtype: ```Tuple[Tuple[Literal['unchanged', 'modified'],
                               Literal['unchanged', 'modified'],
                               Literal['unchanged', 'modified']], Namespace]```
        """
        tmpdir_join = partial(path.join, tmpdir)
        argparse_function = tmpdir_join("argparse.py")
        transformers.to_file(_argparse_func_ast, argparse_function, mode="wt")
        class_ = tmpdir_join("classes.py")
        transformers.to_file(_class_ast, class_, mode="wt")
        function = tmpdir_join("methods.py")
        transformers.to_file(_class_with_method_ast, function, mode="wt")

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

    def test_replace_node(self) -> None:
        """ Tests `replace_node` """
        _docstring_structure = deepcopy(docstring_structure)
        same, found = replace_node(
            fun_name="argparse_function",
            from_func=docstring_struct.from_argparse_ast,
            outer_name="set_cli_args",
            inner_name=None,
            outer_node=argparse_func_ast,
            inner_node=None,
            docstring_structure=_docstring_structure,
            typ=FunctionDef,
        )
        self.assertEqual(*map(to_code, (argparse_func_ast, found)))
        self.assertTrue(same)

        same, found = replace_node(
            fun_name="class",
            from_func=docstring_struct.from_class,
            outer_name="ConfigClass",
            inner_name=None,
            outer_node=class_ast,
            inner_node=None,
            docstring_structure=_docstring_structure,
            typ=ClassDef,
        )
        self.assertEqual(*map(to_code, (class_ast, found)))
        self.assertTrue(same)

        function_def = next(
            filter(rpartial(isinstance, FunctionDef), class_with_method_types_ast.body,)
        )
        same, found = replace_node(
            fun_name="function",
            from_func=docstring_struct.from_class_with_method,
            outer_name="C",
            inner_name="method_name",
            outer_node=class_with_method_types_ast,
            inner_node=function_def,
            docstring_structure=_docstring_structure,
            typ=FunctionDef,
        )
        self.assertEqual(*map(to_code, (function_def, found)))
        self.assertTrue(same)

    def test_replace_node_fails(self) -> None:
        """ Tests `replace_node` """
        self.assertRaises(
            NotImplementedError,
            lambda: replace_node(
                fun_name="function",
                from_func=docstring_struct.from_class_with_method,
                outer_name="C",
                inner_name="method_name",
                outer_node=class_with_method_and_body_types_ast,
                inner_node=next(
                    filter(
                        rpartial(isinstance, FunctionDef),
                        class_with_method_types_ast.body,
                    )
                ),
                docstring_structure=deepcopy(docstring_structure),
                typ=FunctionDef,
            ),
        )


unittest_main()
