"""
Tests for `cdd.emit.sqlalchemy.utils.sqlalchemy_utils`
"""

import ast
from ast import (
    Assign,
    Call,
    Import,
    ImportFrom,
    Load,
    Module,
    Name,
    Store,
    alias,
    keyword,
)
from collections import OrderedDict
from copy import deepcopy
from os import mkdir, path
from tempfile import TemporaryDirectory
from unittest import TestCase

from cdd.compound.openapi.utils.emit_utils import (
    ensure_has_primary_key,
    generate_create_from_attr_staticmethod,
    param_to_sqlalchemy_column_call,
    sqlalchemy_class_to_table,
    sqlalchemy_table_to_class,
    update_fk_for_file,
    update_with_imports_from_columns,
)
from cdd.shared.ast_utils import set_value
from cdd.shared.pure_utils import rpartial
from cdd.shared.source_transformer import to_code
from cdd.sqlalchemy.utils.shared_utils import update_args_infer_typ_sqlalchemy
from cdd.tests.mocks.ir import (
    intermediate_repr_empty,
    intermediate_repr_no_default_doc,
    intermediate_repr_no_default_sql_doc,
    intermediate_repr_node_pk,
)
from cdd.tests.mocks.sqlalchemy import (
    config_hybrid_ast,
    config_tbl_with_comments_ast,
    create_from_attr_mock,
    element_pk_fk_ass,
    node_fk_call,
    node_pk_tbl_ass,
    node_pk_tbl_call,
    node_pk_tbl_class,
)
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

    def test_ensure_has_primary_key_from_id(self) -> None:
        """
        Tests `cdd.emit.sqlalchemy.utils.sqlalchemy_utils.ensure_has_primary_key`
        """
        ir = deepcopy(intermediate_repr_empty)
        ir["params"] = OrderedDict(
            (
                ("id", {"doc": "My doc", "typ": "str"}),
                ("not_pk_id", {"doc": "", "typ": "str"}),
            )
        )
        res = ensure_has_primary_key(deepcopy(ir))
        ir["params"]["id"]["doc"] = "[PK] {}".format(ir["params"]["id"]["doc"])
        self.assertDictEqual(res, ir)

    def test_generate_create_from_attr_staticmethod(self):
        """Tests that `generate_create_from_attr` staticmethod is correctly constructed"""
        run_ast_test(
            self,
            generate_create_from_attr_staticmethod(
                OrderedDict(
                    (
                        ("id", {"doc": "My doc", "typ": "str"}),
                        ("not_pk_id", {"doc": "", "typ": "str"}),
                    )
                ),
                cls_name="foo",
                docstring_format="rest",
            ),
            create_from_attr_mock,
            skip_black=True,
        )

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
                keywords=[keyword(arg="index", value=set_value(True), identifier=None)],
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

    def test_update_args_infer_typ_sqlalchemy_when_simple_array_in_typ(self) -> None:
        """Tests that SQLalchemy can infer the typ from a simple array (in `typ`)"""
        args = []
        update_args_infer_typ_sqlalchemy({"typ": "List[str]"}, args, "", False, {})
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

    def test_update_with_imports_from_columns(self) -> None:
        """
        Tests basic `update_with_imports_from_columns` usage

        Confirms that this:
        ```
        primary_element = Column(element, ForeignKey('element.not_the_right_primary_key'))
        ```

        Turns into this:
        ```
        primary_element = Column(Integer, ForeignKey('element.element_id'))
        ```
        """
        with TemporaryDirectory() as tempdir:
            mod_name = "test_update_with_imports_from_columns"
            temp_mod_dir = path.join(tempdir, mod_name)
            mkdir(temp_mod_dir)
            node_filename = path.join(
                temp_mod_dir, "Node{sep}py".format(sep=path.extsep)
            )
            element_filename = path.join(
                temp_mod_dir, "Element{sep}py".format(sep=path.extsep)
            )
            node_pk_with_phase1_fk = deepcopy(node_pk_tbl_class)
            node_pk_with_phase1_fk.body[2] = Assign(
                targets=[Name(id="primary_element", ctx=Store())],
                value=Call(
                    func=Name(id="Column", ctx=Load()),
                    args=[
                        Name(id="Element", ctx=Load()),
                        Call(
                            func=Name(id="ForeignKey", ctx=Load()),
                            args=[set_value("element.not_the_right_primary_key")],
                            keywords=[],
                        ),
                    ],
                    keywords=[],
                ),
                lineno=None,
            )

            with open(node_filename, "wt") as f:
                f.write(to_code(node_pk_with_phase1_fk))

            element_class = sqlalchemy_table_to_class(element_pk_fk_ass)
            element_class.name = "Element"

            with open(element_filename, "wt") as f:
                f.write(to_code(element_class))

            update_with_imports_from_columns(node_filename)

            with open(node_filename, "rt") as f:
                node_filename_str = f.read()
            gen_mod = ast.parse(node_filename_str)

        gen_imports = tuple(
            filter(rpartial(isinstance, (ImportFrom, Import)), gen_mod.body)
        )
        self.assertEqual(len(gen_imports), 1)

        run_ast_test(
            self,
            gen_imports[0],
            ImportFrom(
                module=".".join(("test_update_with_imports_from_columns", "element")),
                names=[
                    alias(
                        "Element",
                        None,
                        identifier=None,
                        identifier_name=None,
                    )
                ],
                level=0,
            ),
        )

    def test_update_fk_for_file(self) -> None:
        """
        Tests basic `update_with_imports_from_columns` usage

        Confirms that this:
        ```
        primary_element = Column(element, ForeignKey('element.not_the_right_primary_key'))
        ```

        Turns into this:
        ```
        primary_element = Column(Integer, ForeignKey('element.element_id'))
        ```
        """
        with TemporaryDirectory() as tempdir:
            mod_name = "test_update_with_imports_from_columns"
            temp_mod_dir = path.join(tempdir, mod_name)
            mkdir(temp_mod_dir)
            node_filename = path.join(
                temp_mod_dir, "node{sep}py".format(sep=path.extsep)
            )
            element_filename = path.join(
                temp_mod_dir, "element{sep}py".format(sep=path.extsep)
            )
            node_pk_with_phase1_fk = deepcopy(node_pk_tbl_class)
            node_pk_with_phase1_fk.body[2] = Assign(
                targets=[Name(id="primary_element", ctx=Store())],
                value=Call(
                    func=Name(id="Column", ctx=Load()),
                    args=[
                        Name(id="element", ctx=Load()),
                        Call(
                            func=Name(id="ForeignKey", ctx=Load()),
                            args=[set_value("element.not_the_right_primary_key")],
                            keywords=[],
                        ),
                    ],
                    keywords=[],
                ),
                lineno=None,
            )
            with open(node_filename, "wt") as f:
                f.write(
                    to_code(
                        Module(
                            body=[
                                ImportFrom(
                                    module=".".join(
                                        (
                                            mod_name,
                                            path.splitext(path.basename(node_filename))[
                                                0
                                            ],
                                        )
                                    ),
                                    names=[
                                        alias(
                                            "element",
                                            None,
                                            identifier=None,
                                            identifier_name=None,
                                        )
                                    ],
                                    level=0,
                                ),
                                node_pk_tbl_class,
                            ],
                            type_ignores=[],
                        )
                    )
                )

            with open(element_filename, "wt") as f:
                f.write(
                    to_code(
                        Module(
                            body=[sqlalchemy_table_to_class(element_pk_fk_ass)],
                            type_ignores=[],
                        )
                    )
                )

            update_fk_for_file(node_filename)

            with open(node_filename, "rt") as f:
                node_filename_str = f.read()
            gen_mod = ast.parse(node_filename_str)

        run_ast_test(
            self,
            gen_mod.body[1],
            gold=node_pk_tbl_class,
        )

    def test_sqlalchemy_table_to_class(self) -> None:
        """Tests that `sqlalchemy_table_to_class` works"""
        run_ast_test(
            self,
            gen_ast=sqlalchemy_table_to_class(deepcopy(node_pk_tbl_ass)),
            gold=node_pk_tbl_class,
        )

    def test_sqlalchemy_class_to_table(self) -> None:
        """Tests that `sqlalchemy_class_to_table` works"""
        run_ast_test(
            self,
            sqlalchemy_class_to_table(
                deepcopy(node_pk_tbl_class), parse_original_whitespace=False
            ),
            gold=node_pk_tbl_call,
        )

    def test_sqlalchemy_hybrid_class_to_table(self) -> None:
        """Tests that `sqlalchemy_class_to_table` works on hybrid class"""
        gold = deepcopy(config_tbl_with_comments_ast)
        gold.targets[0].id = "__table__"
        run_ast_test(
            self,
            sqlalchemy_class_to_table(
                deepcopy(config_hybrid_ast), parse_original_whitespace=False
            ),
            gold=gold,
        )


unittest_main()
