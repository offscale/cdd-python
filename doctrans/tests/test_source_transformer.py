"""
Tests for source_transformer
"""
from ast import ClassDef
from unittest import TestCase
from unittest.mock import patch

from doctrans.pure_utils import PY_GTE_3_9
from doctrans.tests.utils_for_tests import unittest_main


class TestSourceTransformer(TestCase):
    """
    Tests for source_transformer
    """

    def test_to_code(self) -> None:
        """
        Tests to_source in Python 3.9 and < 3.9
        """
        class_def = ClassDef(
            name="Classy",
            bases=tuple(),
            decorator_list=[],
            body=[],
            keywords=tuple(),
            identifier_name=None,
            expr=None,
        )

        with patch("doctrans.source_transformer.version_info", (3, 9, 0)):
            import doctrans.source_transformer

            self.assertEqual(
                doctrans.source_transformer.to_code(class_def).rstrip("\n"),
                "class Classy:",
            ) if PY_GTE_3_9 else self.assertRaises(
                AttributeError, lambda: doctrans.source_transformer.to_code(class_def)
            )

        with patch("doctrans.source_transformer.version_info", (3, 8, 0)):
            import doctrans.source_transformer

            self.assertEqual(
                doctrans.source_transformer.to_code(class_def).rstrip("\n"),
                "class Classy:",
            )


unittest_main()
