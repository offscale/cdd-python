""" Tests for ast_utils """
from ast import (
    FunctionDef,
    Module,
    ClassDef,
    Subscript,
    Name,
    arguments,
    arg,
    Constant,
    NameConstant,
    Str,
    Tuple,
    AnnAssign,
    Load,
    Store,
)
from unittest import TestCase
from unittest.mock import patch

from doctrans.ast_utils import (
    to_class_def,
    determine_quoting,
    get_function_type,
    get_value,
    find_in_ast,
)
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.methods import class_with_optional_arg_method_ast
from doctrans.tests.utils_for_tests import run_ast_test, unittest_main


class TestAstUtils(TestCase):
    """ Test class for ast_utils """

    def test_to_class_def(self) -> None:
        """ Test that to_class_def gives the wrapped class back """

        class_def = ClassDef(
            name="", bases=tuple(), keywords=tuple(), decorator_list=[], body=[]
        )
        run_ast_test(self, to_class_def(Module(body=[class_def])), class_def)

    def test_to_named_class_def(self) -> None:
        """ Test that to_class_def gives the wrapped named class back """

        class_def = ClassDef(
            name="foo", bases=tuple(), keywords=tuple(), decorator_list=[], body=[]
        )
        run_ast_test(
            self,
            to_class_def(
                Module(
                    body=[
                        ClassDef(
                            name="bar",
                            bases=tuple(),
                            keywords=tuple(),
                            decorator_list=[],
                            body=[],
                        ),
                        class_def,
                    ]
                ),
                class_name="foo",
            ),
            class_def,
        )

    def test_find_in_ast(self) -> None:
        """ Tests that `find_in_ast` successfully finds nodes in AST """
        run_ast_test(
            self,
            find_in_ast("ConfigClass.dataset_name", class_ast),
            AnnAssign(
                annotation=Name(ctx=Load(), id="str"),
                simple=1,
                target=Name(ctx=Store(), id="dataset_name"),
                value=Constant(kind=None, value="mnist"),
            ),
        )

    def test_find_in_ast_self(self) -> None:
        """ Tests that `find_in_ast` successfully finds itself in AST """
        run_ast_test(self, find_in_ast("ConfigClass", class_ast), class_ast)

    def test_find_in_ast_None(self) -> None:
        """ Tests that `find_in_ast` fails correctly in AST """
        self.assertIsNone(find_in_ast("John Galt", class_ast))

    def test_find_in_ast_no_val(self) -> None:
        """ Tests that `find_in_ast` correctly gives AST node from
         `def class C(object): def method_name(self,dataset_name: str,…)`"""
        run_ast_test(
            self,
            find_in_ast(
                "C.method_name.dataset_name", class_with_optional_arg_method_ast
            ),
            arg(
                annotation=Name(ctx=Load(), id="str"),
                arg="dataset_name",
                type_comment=None,
            ),
        )

    def test_to_class_def_fails(self) -> None:
        """ Test that to_class_def throws the right errors """

        self.assertRaises(NotImplementedError, lambda: to_class_def(None))
        self.assertRaises(NotImplementedError, lambda: to_class_def(""))
        self.assertRaises(NotImplementedError, lambda: to_class_def(FunctionDef()))
        self.assertRaises(TypeError, lambda: to_class_def(Module(body=[])))
        self.assertRaises(
            NotImplementedError,
            lambda: to_class_def(Module(body=[ClassDef(), ClassDef()])),
        )

    def test_determine_quoting_fails(self) -> None:
        """" Tests that determine_quoting fails on unknown input """
        self.assertRaises(
            NotImplementedError,
            lambda: determine_quoting(Subscript(value=Name(id="impossibru"))),
        )
        self.assertRaises(NotImplementedError, lambda: determine_quoting(FunctionDef()))
        self.assertRaises(NotImplementedError, lambda: determine_quoting(ClassDef()))
        self.assertRaises(NotImplementedError, lambda: determine_quoting(Module()))

    def test_get_function_type(self) -> None:
        """ Test get_function_type returns the right type """
        self.assertIsNone(
            get_function_type(
                FunctionDef(
                    args=arguments(
                        args=[
                            arg(
                                annotation=None, arg="something else", type_comment=None
                            )
                        ]
                    )
                )
            )
        )
        self.assertIsNone(get_function_type(FunctionDef(args=arguments(args=[]))))
        self.assertEqual(
            get_function_type(
                FunctionDef(
                    args=arguments(
                        args=[arg(annotation=None, arg="self", type_comment=None)]
                    )
                )
            ),
            "self",
        )
        self.assertEqual(
            get_function_type(
                FunctionDef(
                    args=arguments(
                        args=[arg(annotation=None, arg="cls", type_comment=None)]
                    )
                )
            ),
            "cls",
        )

    def test_set_value(self) -> None:
        """ Tests that `set_value` returns the right type for the right Python version """
        with patch("doctrans.ast_utils.PY3_8", True):
            import doctrans.ast_utils

            self.assertIsInstance(doctrans.ast_utils.set_value(None, None), Constant)

        with patch("doctrans.ast_utils.PY3_8", False):
            import doctrans.ast_utils

            self.assertIsInstance(
                doctrans.ast_utils.set_value(None, None), NameConstant
            )

        with patch("doctrans.ast_utils.PY3_8", True):
            import doctrans.ast_utils

            self.assertIsInstance(doctrans.ast_utils.set_value("foo", None), Constant)

        with patch("doctrans.ast_utils.PY3_8", False):
            import doctrans.ast_utils

            self.assertIsInstance(doctrans.ast_utils.set_value("foo", None), Str)

    def test_get_value(self) -> None:
        """ Tests get_value succeeds """
        val = "foo"
        self.assertEqual(get_value(Str(s=val)), val)
        self.assertEqual(get_value(Constant(value=val)), val)
        self.assertIsInstance(get_value(Tuple()), Tuple)
        self.assertIsInstance(get_value(Tuple()), Tuple)
        self.assertIsInstance(get_value(Name()), Name)

    def test_get_value_fails(self) -> None:
        """ Tests get_value fails properly """
        self.assertRaises(NotImplementedError, lambda: get_value(None))
        self.assertRaises(NotImplementedError, lambda: get_value(""))
        self.assertRaises(NotImplementedError, lambda: get_value(0))
        self.assertRaises(NotImplementedError, lambda: get_value(0.0))
        self.assertRaises(NotImplementedError, lambda: get_value([]))


unittest_main()
