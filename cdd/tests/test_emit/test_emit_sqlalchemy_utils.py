"""
Tests for `cdd.emit.sqlalchemy.utils.sqlalchemy_utils`
"""

from copy import deepcopy
from unittest import TestCase

from cdd.emit.utils.sqlalchemy_utils import ensure_has_primary_key
from cdd.tests.mocks.ir import intermediate_repr_no_default_sql_doc
from cdd.tests.utils_for_tests import unittest_main


class TestEmitSqlAlchemyUtils(TestCase):
    """Tests cdd.emit.sqlalchemy.utils.sqlalchemy_utils"""

    def test_ensure_has_primary_key(self) -> None:
        """
        Tests `cdd.emit.sqlalchemy.utils.sqlalchemy_utils.ensure_has_primary_key`
        """
        res = ensure_has_primary_key(deepcopy(intermediate_repr_no_default_sql_doc))
        self.assertIsNotNone(res)
        self.assertDictEqual(
            res,
            deepcopy(intermediate_repr_no_default_sql_doc),
        )


unittest_main()
