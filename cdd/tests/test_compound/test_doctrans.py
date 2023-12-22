""" Tests for doctrans """

from copy import deepcopy
from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cdd.compound.doctrans import doctrans
from cdd.shared.ast_utils import annotate_ancestry
from cdd.shared.source_transformer import to_code
from cdd.tests.mocks.doctrans import function_type_annotated
from cdd.tests.mocks.methods import return_ast
from cdd.tests.utils_for_tests import unittest_main


class TestDocTrans(TestCase):
    """Test class for doctrans.py"""

    def test_doctrans_append(self) -> None:
        """Tests doctrans"""

        with TemporaryDirectory() as temp_dir, patch(
            "cdd.compound.doctrans_utils.DocTrans",
            lambda **kwargs: type(
                "DocTrans", tuple(), {"visit": lambda *args: return_ast}
            )(),
        ):
            filename: str = path.join(temp_dir, "foo")
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

    def test_doctrans_replace(self) -> None:
        """Tests doctrans"""

        with TemporaryDirectory() as temp_dir:
            filename: str = path.join(
                temp_dir, "fun{extsep}py".format(extsep=path.extsep)
            )
            original_node = annotate_ancestry(deepcopy(function_type_annotated))
            with open(filename, "wt") as f:
                f.write(to_code(original_node))
            self.assertIsNone(
                doctrans(
                    filename=filename,
                    no_word_wrap=None,
                    docstring_format="rest",
                    type_annotations=False,
                )
            )
            # with open(filename, "rt") as f:
            #    src = f.read()
            # new_node = ast_parse(src, skip_docstring_remit=True).body[0]
            # run_ast_test(
            #     self, new_node, gold=function_type_in_docstring, skip_black=True
            # )


unittest_main()
