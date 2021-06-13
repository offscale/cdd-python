"""
Tests for source_transformer
"""

from ast import ClassDef
from unittest import TestCase
from unittest.mock import patch

from cdd.pure_utils import PY_GTE_3_9
from cdd.tests.utils_for_tests import unittest_main


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

        with patch("cdd.source_transformer.version_info", (3, 9, 0)):
            import cdd.source_transformer

            self.assertEqual(
                cdd.source_transformer.to_code(class_def).rstrip("\n"),
                "class Classy:",
            ) if PY_GTE_3_9 else self.assertRaises(
                AttributeError, lambda: cdd.source_transformer.to_code(class_def)
            )

        with patch("cdd.source_transformer.version_info", (3, 8, 0)):
            import cdd.source_transformer

            self.assertEqual(
                "class Classy:"
                if PY_GTE_3_9
                else cdd.source_transformer.to_code(class_def).rstrip("\n"),
                "class Classy:",
            )


unittest_main()
