""" Tests for doctrans """

from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cdd.doctrans import doctrans
from cdd.tests.mocks.methods import return_ast
from cdd.tests.utils_for_tests import unittest_main


class TestDocTrans(TestCase):
    """Test class for doctrans.py"""

    def test_doctrans(self) -> None:
        """Tests doctrans"""

        with TemporaryDirectory() as temp_dir, patch(
            "cdd.doctrans.DocTrans",
            lambda **kwargs: type(
                "DocTrans", tuple(), {"visit": lambda *args: return_ast}
            )(),
        ):
            filename = path.join(temp_dir, "foo")
            with open(filename, "wt") as f:
                f.write("5*5")
            self.assertIsNone(
                doctrans(
                    filename=filename,
                    no_word_wrap=None,
                    docstring_format="numpydoc",
                    type_annotations=True,
                )
            )


unittest_main()
