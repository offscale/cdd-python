"""
Tests for `cdd.emit.sqlalchemy.utils.sqlalchemy_utils`
"""

from ast import Call, Load, Name
from collections import OrderedDict
from copy import deepcopy
from unittest import TestCase

from cdd.emit.utils.sqlalchemy_utils import ensure_has_primary_key
from cdd.tests.mocks.ir import (
    intermediate_repr_empty,
    intermediate_repr_no_default_doc,
    intermediate_repr_no_default_sql_doc,
)
from cdd.tests.utils_for_tests import unittest_main


class TestEmitSqlAlchemyUtils(TestCase):
    """Tests cdd.emit.sqlalchemy.utils.sqlalchemy_utils"""

    def test_ensure_has_primary_key(self) -> None:
        """
        Tests `cdd.emit.sqlalchemy.utils.sqlalchemy_utils.ensure_has_primary_key`
        """
        self.assertDictEqual(
            ensure_has_primary_key(deepcopy(intermediate_repr_no_default_sql_doc)),
            intermediate_repr_no_default_sql_doc,
        )

        self.assertDictEqual(
            ensure_has_primary_key(deepcopy(intermediate_repr_no_default_doc)),
            intermediate_repr_no_default_sql_doc,
        )

        ir = deepcopy(intermediate_repr_empty)
        ir["params"] = OrderedDict((("foo", {"doc": "My doc", "typ": "str"}),))
        res = ensure_has_primary_key(deepcopy(ir))
        ir["params"]["id"] = {
            "doc": "[PK]",
            "typ": "int",
            "x_typ": {
                "sql": {
                    "constraints": {
                        "server_default": Call(
                            args=[], func=Name(ctx=Load(), id="Identity"), keywords=[]
                        )
                    }
                }
            },
        }
        self.assertIsInstance(
            res["params"]["id"]["x_typ"]["sql"]["constraints"]["server_default"], Call
        )
        res["params"]["id"]["x_typ"]["sql"]["constraints"]["server_default"] = ir[
            "params"
        ]["id"]["x_typ"]["sql"]["constraints"]["server_default"]
        self.assertDictEqual(res, ir)

    maxDiff = None


unittest_main()
