"""
Tests for `cdd.emit.sqlalchemy.utils.sqlalchemy_utils`
"""
from ast import Call, Load, Name, keyword
from collections import OrderedDict
from copy import deepcopy
from unittest import TestCase

from cdd.ast_utils import set_value
from cdd.emit.utils.sqlalchemy_utils import (
    ensure_has_primary_key,
    param_to_sqlalchemy_column_call,
    update_args_infer_typ_sqlalchemy,
)
from cdd.tests.mocks.ir import (
    intermediate_repr_empty,
    intermediate_repr_no_default_doc,
    intermediate_repr_no_default_sql_doc,
    intermediate_repr_node_pk,
)
from cdd.tests.mocks.sqlalchemy import node_fk_call
from cdd.tests.utils_for_tests import run_ast_test, unittest_main


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

    def test_param_to_sqlalchemy_column_call_when_sql_constraints(self) -> None:
        """Tests that with SQL constraints the SQLalchemy column is correctly generated"""
        run_ast_test(
            self,
            param_to_sqlalchemy_column_call(
                (
                    "foo",
                    {
                        "doc": "",
                        "typ": "str",
                        "x_typ": {"sql": {"constraints": {"index": True}}},
                    },
                ),
                include_name=False,
            ),
            gold=Call(
                func=Name(id="Column", ctx=Load()),
                args=[Name(id="String", ctx=Load())],
                keywords=[keyword(arg="index", value=set_value(True))],
            ),
        )

    def test_param_to_sqlalchemy_column_call_when_foreign_key(self) -> None:
        """Tests that SQLalchemy column with simple foreign key is correctly generated"""
        run_ast_test(
            self,
            param_to_sqlalchemy_column_call(
                (
                    lambda _name: (
                        _name,
                        deepcopy(intermediate_repr_node_pk["params"][_name]),
                    )
                )("primary_element"),
                include_name=True,
            ),
            gold=node_fk_call,
        )

    def test_update_args_infer_typ_sqlalchemy_when_simple_array(self) -> None:
        """Tests that SQLalchemy can infer the typ from a simple array"""
        args = []
        update_args_infer_typ_sqlalchemy(
            {"items": {"type": "string"}, "typ": ""}, args, "", False, {}
        )
        self.assertEqual(len(args), 1)
        run_ast_test(
            self,
            args[0],
            gold=Call(
                func=Name(id="ARRAY", ctx=Load()),
                args=[Name(id="String", ctx=Load())],
                keywords=[],
                expr=None,
                expr_func=None,
            ),
        )

    def test_update_args_infer_typ_sqlalchemy_when_simple_union(self) -> None:
        """Tests that SQLalchemy can infer the typ from a simple Union"""
        args = []
        update_args_infer_typ_sqlalchemy(
            {"typ": "Union[string, Small]"}, args, "", False, {}
        )
        self.assertEqual(len(args), 1)
        run_ast_test(
            self,
            args[0],
            gold=Name(id="Small", ctx=Load()),
        )


unittest_main()
