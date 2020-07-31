"""
Tests for source_transformer
"""
from _ast import ClassDef
from platform import python_version_tuple
from unittest import TestCase
from unittest.mock import patch

from doctrans.tests.utils_for_tests import unittest_main


class TestSourceTransformer(TestCase):
    """
    Tests for source_transformer
    """

    def test_to_code(self) -> None:
        """
        Tests to_source in Python 3.9 and < 3.9
        """
        class_def = ClassDef(name='Classy', bases=tuple(), decorator_list=[], body=[], keywords=tuple())

        lt_39 = python_version_tuple() < ('3', '9')

        with patch('doctrans.source_transformer.python_version_tuple', lambda: ('3', '9', '0')):
            import doctrans.source_transformer
            self.assertRaises(ImportError, lambda: doctrans.source_transformer.to_code(class_def)) if lt_39 \
                else self.assertEqual(doctrans.source_transformer.to_code(class_def).rstrip('\n'),
                                      'class Classy:')

        with patch('doctrans.source_transformer.python_version_tuple', lambda: ('3', '8', '0')):
            import doctrans.source_transformer
            self.assertEqual(doctrans.source_transformer.to_code(class_def).rstrip('\n'),
                             'class Classy:') if lt_39 \
                else self.assertRaises(ImportError, lambda: doctrans.source_transformer.to_code(class_def))


unittest_main()
