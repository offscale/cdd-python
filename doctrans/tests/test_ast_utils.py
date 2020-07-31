""" Tests for ast_utils """
from ast import FunctionDef, Module, ClassDef, Subscript, Name, arguments, arg, Constant, NameConstant, Str, Tuple
from unittest import TestCase
from unittest.mock import patch

from doctrans.ast_utils import to_class_def, determine_quoting, get_function_type, get_value
from doctrans.tests.utils_for_tests import run_ast_test, unittest_main


class TestAstUtils(TestCase):
    """ Test class for ast_utils """

    def test_to_class_def(self) -> None:
        """ Test that to_class_def gives the wrapped class back """
        class_def = ClassDef(name='', bases=tuple(), keywords=tuple(), decorator_list=[], body=[])
        run_ast_test(self, to_class_def(Module(body=[class_def])), class_def)

    def test_to_class_def_fails(self) -> None:
        """ Test that to_class_def throws the right errors """
        self.assertRaises(NotImplementedError, lambda: to_class_def(None))
        self.assertRaises(NotImplementedError, lambda: to_class_def(''))
        self.assertRaises(NotImplementedError, lambda: to_class_def(FunctionDef()))
        self.assertRaises(TypeError, lambda: to_class_def(Module(body=[])))
        self.assertRaises(NotImplementedError,
                          lambda: to_class_def(Module(body=[
                              ClassDef(), ClassDef()
                          ])))

    def test_determine_quoting_fails(self) -> None:
        """" Tests that determine_quoting fails on unknown input """
        self.assertRaises(NotImplementedError,
                          lambda: determine_quoting(Subscript(value=Name(id='impossibru'))))
        self.assertRaises(NotImplementedError,
                          lambda: determine_quoting(FunctionDef()))
        self.assertRaises(NotImplementedError,
                          lambda: determine_quoting(ClassDef()))
        self.assertRaises(NotImplementedError,
                          lambda: determine_quoting(Module()))

    def test_get_function_type(self) -> None:
        """ Test get_function_type returns the right type """
        self.assertIsNone(
            get_function_type(FunctionDef(args=arguments(args=[arg(annotation=None,
                                                                   arg='something else',
                                                                   type_comment=None)]))))
        self.assertIsNone(
            get_function_type(FunctionDef(args=arguments(args=[]))))
        self.assertEqual(
            get_function_type(
                FunctionDef(args=arguments(args=[arg(annotation=None,
                                                     arg='self',
                                                     type_comment=None)]))),
            'self'
        )
        self.assertEqual(
            get_function_type(
                FunctionDef(args=arguments(args=[arg(annotation=None,
                                                     arg='cls',
                                                     type_comment=None)]))),
            'cls'
        )

    def test_set_value(self):
        """ Tests that `set_value` returns the right type for the right Python version """
        with patch('doctrans.ast_utils.PY3_8', True):
            import doctrans.ast_utils
            self.assertIsInstance(doctrans.ast_utils.set_value(None, None), Constant)

        with patch('doctrans.ast_utils.PY3_8', False):
            import doctrans.ast_utils
            self.assertIsInstance(doctrans.ast_utils.set_value(None, None), NameConstant)

        with patch('doctrans.ast_utils.PY3_8', True):
            import doctrans.ast_utils
            self.assertIsInstance(doctrans.ast_utils.set_value('foo', None), Constant)

        with patch('doctrans.ast_utils.PY3_8', False):
            import doctrans.ast_utils
            self.assertIsInstance(doctrans.ast_utils.set_value('foo', None), Str)

    def test_get_value(self):
        """ Tests get_value succeeds """
        val = 'foo'
        self.assertEqual(get_value(Str(s=val)), val)
        self.assertEqual(get_value(Constant(value=val)), val)
        self.assertIsInstance(get_value(Tuple()), Tuple)
        self.assertIsInstance(get_value(Tuple()), Tuple)
        self.assertIsInstance(get_value(Name()), Name)

    def test_get_value_fails(self):
        """ Tests get_value fails properly """
        self.assertRaises(NotImplementedError, lambda: get_value(None))
        self.assertRaises(NotImplementedError, lambda: get_value(''))
        self.assertRaises(NotImplementedError, lambda: get_value(0))
        self.assertRaises(NotImplementedError, lambda: get_value(.0))
        self.assertRaises(NotImplementedError, lambda: get_value([]))


unittest_main()
