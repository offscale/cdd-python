""" Tests for ast_utils """
from ast import FunctionDef, Module, ClassDef, Subscript, Name
from unittest import TestCase, main as unittest_main

from doctrans.ast_utils import to_class_def, determine_quoting
from doctrans.tests.utils_for_tests import run_ast_test


class TestAstUtils(TestCase):
    """ Test class for ast_utils """

    def test_to_class_def(self) -> None:
        """ Test that to_class_def gives the wrapped class back """
        class_def = ClassDef()
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


if __name__ == '__main__':
    unittest_main()
