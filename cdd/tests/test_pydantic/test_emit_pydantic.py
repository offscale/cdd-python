"""
Tests for `cdd.emit.pydantic`
"""

from unittest import TestCase

import cdd.pydantic.emit
from cdd.tests.mocks.ir import pydantic_ir
from cdd.tests.mocks.pydantic import pydantic_class_cls_def
from cdd.tests.utils_for_tests import run_ast_test, unittest_main


class TestEmitPyDantic(TestCase):
    """Tests emission"""

    def test_to_pydantic(self) -> None:
        """
        Tests whether `pydantic` produces `pydantic_class_cls_def` given `pydantic_ir`
        """
        run_ast_test(
            self,
            gen_ast=cdd.pydantic.emit.pydantic(pydantic_ir),
            gold=pydantic_class_cls_def,
        )


unittest_main()
