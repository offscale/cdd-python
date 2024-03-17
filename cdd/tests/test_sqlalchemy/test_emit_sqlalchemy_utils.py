"""
Tests for `cdd.emit.sqlalchemy.utils.sqlalchemy_utils`
"""

import ast
import json
from ast import (
    AST,
    Assign,
    Call,
    ClassDef,
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
from functools import partial
from os import mkdir, path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Tuple, Union
from unittest import TestCase
from unittest.mock import patch

import cdd.shared.pure_utils
import cdd.sqlalchemy.utils.emit_utils
from cdd.shared.ast_utils import set_value
from cdd.shared.source_transformer import to_code
from cdd.shared.types import IntermediateRepr
from cdd.sqlalchemy.utils.shared_utils import (
    _handle_union_of_length_2,
    update_args_infer_typ_sqlalchemy,
)
from cdd.tests.mocks.ir import (
    intermediate_repr_empty,
    intermediate_repr_no_default_doc,
    intermediate_repr_no_default_sql_doc,
    intermediate_repr_node_pk,
)
from cdd.tests.mocks.openapi_emit_utils import column_fk, column_fk_gold, id_column
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
        Tests `cdd.emit.sqlalchemy.utils.sqlalchemy_utils.cdd.sqlalchemy.utils.emit_utils.ensure_has_primary_key`
        """
        self.assertDictEqual(
            cdd.sqlalchemy.utils.emit_utils.ensure_has_primary_key(
                deepcopy(intermediate_repr_no_default_sql_doc)
            ),
            intermediate_repr_no_default_sql_doc,
        )

        self.assertDictEqual(
            cdd.sqlalchemy.utils.emit_utils.ensure_has_primary_key(
                deepcopy(intermediate_repr_no_default_doc)
            ),
            intermediate_repr_no_default_sql_doc,
        )

        ir: IntermediateRepr = deepcopy(intermediate_repr_empty)
        ir["params"] = OrderedDict((("foo", {"doc": "My doc", "typ": "str"}),))
        res = cdd.sqlalchemy.utils.emit_utils.ensure_has_primary_key(deepcopy(ir))
        ir["params"]["id"] = {
            "doc": "[PK]",
            "typ": "int",
            "x_typ": {
                "sql": {
                    "constraints": {
                        "server_default": Call(
                            args=[],
                            func=Name(
                                ctx=Load(), id="Identity", lineno=None, col_offset=None
                            ),
                            keywords=[],
                            lineno=None,
                            col_offset=None,
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
        Tests `cdd.emit.sqlalchemy.utils.sqlalchemy_utils.cdd.sqlalchemy.utils.emit_utils.ensure_has_primary_key`
        """
        ir: IntermediateRepr = deepcopy(intermediate_repr_empty)
        ir["params"] = OrderedDict(
            (
                ("id", {"doc": "My doc", "typ": "str"}),
                ("not_pk_id", {"doc": "", "typ": "str"}),
            )
        )
        res = cdd.sqlalchemy.utils.emit_utils.ensure_has_primary_key(deepcopy(ir))
        ir["params"]["id"]["doc"] = "[PK] {}".format(ir["params"]["id"]["doc"])
        self.assertDictEqual(res, ir)

    def test_generate_create_from_attr_staticmethod(self) -> None:
        """Tests that `generate_create_from_attr` staticmethod is correctly constructed"""
        run_ast_test(
            self,
            cdd.sqlalchemy.utils.emit_utils.generate_create_from_attr_staticmethod(
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
            cdd.sqlalchemy.utils.emit_utils.param_to_sqlalchemy_column_calls(
                (
                    "foo",
                    {
                        "doc": "",
                        "typ": "str",
                        "x_typ": {"sql": {"constraints": {"index": True}}},
                    },
                ),
                include_name=False,
            )[0],
            gold=Call(
                func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
                args=[Name(id="String", ctx=Load(), lineno=None, col_offset=None)],
                keywords=[keyword(arg="index", value=set_value(True), identifier=None)],
                lineno=None,
                col_offset=None,
            ),
        )

    def test_param_to_sqlalchemy_column_call_when_foreign_key(self) -> None:
        """Tests that SQLalchemy column with simple foreign key is correctly generated"""
        run_ast_test(
            self,
            cdd.sqlalchemy.utils.emit_utils.param_to_sqlalchemy_column_calls(
                (
                    lambda _name: (
                        _name,
                        deepcopy(intermediate_repr_node_pk["params"][_name]),
                    )
                )("primary_element"),
                include_name=True,
            )[0],
            gold=node_fk_call,
        )

    def test_param_to_sqlalchemy_column_call_for_schema_comment(self) -> None:
        """Tests that SQLalchemy column is generated with schema as comment"""
        run_ast_test(
            self,
            cdd.sqlalchemy.utils.emit_utils.param_to_sqlalchemy_column_calls(
                (
                    "foo",
                    {
                        "doc": "",
                        "typ": "dict",
                        "ir": intermediate_repr_no_default_doc,
                    },
                ),
                include_name=False,
            )[0],
            gold=Call(
                func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
                args=[Name(id="JSON", ctx=Load(), lineno=None, col_offset=None)],
                keywords=[
                    keyword(
                        arg="comment",
                        value=set_value(
                            "[schema={}]".format(
                                json.dumps(intermediate_repr_no_default_doc)
                            )
                        ),
                        identifier=None,
                    )
                ],
                lineno=None,
                col_offset=None,
            ),
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
                func=Name(id="ARRAY", ctx=Load(), lineno=None, col_offset=None),
                args=[Name(id="String", ctx=Load(), lineno=None, col_offset=None)],
                keywords=[],
                expr=None,
                expr_func=None,
                lineno=None,
                col_offset=None,
            ),
        )

    # TODO
    # def test_param_to_sqlalchemy_column_call_for_complex_union(self) -> None:
    #     """Tests that SQLalchemy column is generated from a complex union"""
    #     run_ast_test(
    #         self,
    #         cdd.sqlalchemy.utils.emit_utils.param_to_sqlalchemy_column_calls(
    #             (
    #                 "epoch_bool_or_int",
    #                 {
    #                     "doc": "",
    #                     "typ": 'Union[Literal["epoch"], bool, int]',
    #                 },
    #             ),
    #             include_name=False,
    #         ),
    #         gold=Call(
    #             func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
    #             args=[Name(id="JSON", ctx=Load(), lineno=None, col_offset=None)],
    #             keywords=[
    #                 keyword(
    #                     arg="comment",
    #                     value=set_value(
    #                         "[schema={}]".format(
    #                             json.dumps(intermediate_repr_no_default_doc)
    #                         )
    #                     ),
    #                     identifier=None,
    #                 )
    #             ],
    #             lineno=None,
    #             col_offset=None,
    #         ),
    #     )

    def test_update_args_infer_typ_sqlalchemy_when_simple_array_in_typ(self) -> None:
        """Tests that SQLalchemy can infer the typ from a simple array (in `typ`)"""
        args = []
        update_args_infer_typ_sqlalchemy({"typ": "List[str]"}, args, "", False, {})
        self.assertEqual(len(args), 1)
        run_ast_test(
            self,
            args[0],
            gold=Call(
                func=Name(id="ARRAY", ctx=Load(), lineno=None, col_offset=None),
                args=[Name(id="String", ctx=Load(), lineno=None, col_offset=None)],
                keywords=[],
                expr=None,
                expr_func=None,
                lineno=None,
                col_offset=None,
            ),
        )

    # TODO: Think about what this union should do
    # def test_update_args_infer_typ_sqlalchemy_when_simple_union(self) -> None:
    #     """Tests that SQLalchemy can infer the typ from a simple Union"""
    #     args = []
    #     update_args_infer_typ_sqlalchemy(
    #         {"typ": "Union[string, Small]"}, args, "", False, {}
    #     )
    #     self.assertEqual(len(args), 1)
    #     run_ast_test(
    #         self,
    #         args[0],
    #         gold=Name(id="Small", ctx=Load(), lineno=None, col_offset=None),
    #     )

    def test_update_args_infer_typ_sqlalchemy_calls__handle_union_of_length_2(
        self,
    ) -> None:
        """Tests that `update_args_infer_typ_sqlalchemy` calls `_handle_union_of_length_2`"""
        args = []
        with patch(
            "cdd.sqlalchemy.utils.shared_utils._handle_union_of_length_2", lambda _: 5
        ):
            update_args_infer_typ_sqlalchemy(
                {"typ": "Union[string, Small]"}, args, "", False, {}
            )
        self.assertEqual(len(args), 1)
        self.assertListEqual(args, [5])

    def test_update_args_infer_typ_sqlalchemy_early_exit(self) -> None:
        """Tests that `update_args_infer_typ_sqlalchemy` exits early"""
        _update_args_infer_typ_sqlalchemy: Callable[
            [dict], Tuple[bool, Optional[Union[List[AST], Tuple[AST]]]]
        ] = partial(
            update_args_infer_typ_sqlalchemy,
            args=[],
            name="",
            nullable=True,
            x_typ_sql={},
        )
        self.assertTupleEqual(
            _update_args_infer_typ_sqlalchemy({"typ": None}), (False, None)
        )
        self.assertTupleEqual(
            _update_args_infer_typ_sqlalchemy(
                {"typ": None, "default": cdd.shared.ast_utils.NoneStr},
            ),
            (True, None),
        )

    def test_update_with_imports_from_columns(self) -> None:
        """
        Tests basic `cdd.sqlalchemy.utils.emit_utils.update_with_imports_from_columns` usage

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
            mod_name: str = "test_update_with_imports_from_columns"
            temp_mod_dir: str = path.join(tempdir, mod_name)
            mkdir(temp_mod_dir)
            node_filename: str = path.join(
                temp_mod_dir, "Node{sep}py".format(sep=path.extsep)
            )
            element_filename: str = path.join(
                temp_mod_dir, "Element{sep}py".format(sep=path.extsep)
            )
            node_pk_with_phase1_fk: ClassDef = deepcopy(node_pk_tbl_class)
            node_pk_with_phase1_fk.body[2] = Assign(
                targets=[
                    Name(
                        id="primary_element", ctx=Store(), lineno=None, col_offset=None
                    )
                ],
                value=Call(
                    func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
                    args=[
                        Name(id="Element", ctx=Load(), lineno=None, col_offset=None),
                        Call(
                            func=Name(
                                id="ForeignKey",
                                ctx=Load(),
                                lineno=None,
                                col_offset=None,
                            ),
                            args=[set_value("element.not_the_right_primary_key")],
                            keywords=[],
                            lineno=None,
                            col_offset=None,
                        ),
                    ],
                    keywords=[],
                    lineno=None,
                    col_offset=None,
                ),
                lineno=None,
            )

            with open(node_filename, "wt") as f:
                f.write(to_code(node_pk_with_phase1_fk))

            element_class: ClassDef = (
                cdd.sqlalchemy.utils.emit_utils.sqlalchemy_table_to_class(
                    element_pk_fk_ass
                )
            )
            element_class.name = "Element"

            with open(element_filename, "wt") as f:
                f.write(to_code(element_class))

            cdd.sqlalchemy.utils.emit_utils.update_with_imports_from_columns(
                node_filename
            )

            with open(node_filename, "rt") as f:
                node_filename_str: str = f.read()
            gen_mod: Module = ast.parse(node_filename_str)

        gen_imports = tuple(
            filter(
                cdd.shared.pure_utils.rpartial(isinstance, (ImportFrom, Import)),
                gen_mod.body,
            )
        )  # type: tuple[Union[ImportFrom, Import]]
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
        Tests basic `cdd.sqlalchemy.utils.emit_utils.update_with_imports_from_columns` usage

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
            mod_name: str = "test_update_with_imports_from_columns"
            temp_mod_dir: str = path.join(tempdir, mod_name)
            mkdir(temp_mod_dir)
            node_filename: str = path.join(
                temp_mod_dir, "node{sep}py".format(sep=path.extsep)
            )
            element_filename: str = path.join(
                temp_mod_dir, "element{sep}py".format(sep=path.extsep)
            )
            node_pk_with_phase1_fk: ClassDef = deepcopy(node_pk_tbl_class)
            node_pk_with_phase1_fk.body[2] = Assign(
                targets=[
                    Name(
                        id="primary_element", ctx=Store(), lineno=None, col_offset=None
                    )
                ],
                value=Call(
                    func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
                    args=[
                        Name(id="element", ctx=Load(), lineno=None, col_offset=None),
                        Call(
                            func=Name(
                                id="ForeignKey",
                                ctx=Load(),
                                lineno=None,
                                col_offset=None,
                            ),
                            args=[set_value("element.not_the_right_primary_key")],
                            keywords=[],
                            lineno=None,
                            col_offset=None,
                        ),
                    ],
                    keywords=[],
                    lineno=None,
                    col_offset=None,
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
                            body=[
                                cdd.sqlalchemy.utils.emit_utils.sqlalchemy_table_to_class(
                                    element_pk_fk_ass
                                )
                            ],
                            type_ignores=[],
                        )
                    )
                )

            cdd.sqlalchemy.utils.emit_utils.update_fk_for_file(node_filename)

            with open(node_filename, "rt") as f:
                node_filename_str: str = f.read()
            gen_mod: Module = ast.parse(node_filename_str)

        run_ast_test(
            self,
            gen_mod.body[1],
            gold=node_pk_tbl_class,
        )

    def test_sqlalchemy_table_to_class(self) -> None:
        """Tests that `cdd.sqlalchemy.utils.emit_utils.sqlalchemy_table_to_class` works"""
        run_ast_test(
            self,
            gen_ast=cdd.sqlalchemy.utils.emit_utils.sqlalchemy_table_to_class(
                deepcopy(node_pk_tbl_ass)
            ),
            gold=node_pk_tbl_class,
        )

    def test_sqlalchemy_class_to_table(self) -> None:
        """Tests that `cdd.sqlalchemy.utils.emit_utils.sqlalchemy_class_to_table` works"""
        run_ast_test(
            self,
            cdd.sqlalchemy.utils.emit_utils.sqlalchemy_class_to_table(
                deepcopy(node_pk_tbl_class), parse_original_whitespace=False
            ),
            gold=node_pk_tbl_call,
        )

    def test_sqlalchemy_hybrid_class_to_table(self) -> None:
        """Tests that `cdd.sqlalchemy.utils.emit_utils.sqlalchemy_class_to_table` works on hybrid class"""
        gold = deepcopy(config_tbl_with_comments_ast)
        gold.targets[0].id = "__table__"
        run_ast_test(
            self,
            cdd.sqlalchemy.utils.emit_utils.sqlalchemy_class_to_table(
                deepcopy(config_hybrid_ast), parse_original_whitespace=False
            ),
            gold=gold,
        )

    def test_rewrite_fk(self) -> None:
        """
        Tests whether `rewrite_fk` produces `openapi_dict` given `model_paths` and `routes_paths`
        """
        sqlalchemy_cls = deepcopy(node_pk_tbl_class)
        sqlalchemy_cls.name = "TableName0"
        sqlalchemy_cls.body[0].value = set_value("table_name0")
        sqlalchemy_cls.body[1:] = [
            column_fk,
            id_column,
        ]

        with TemporaryDirectory() as temp_dir:
            mod_dir: str = path.join(temp_dir, "table_name0")
            mkdir(mod_dir)
            init_path: str = path.join(mod_dir, cdd.shared.pure_utils.INIT_FILENAME)
            with open(init_path, "wt") as f:
                f.write(to_code(sqlalchemy_cls))
            with patch(
                "cdd.sqlalchemy.utils.emit_utils.find_module_filepath",
                lambda _, __: init_path,
            ):
                gen_ast = cdd.sqlalchemy.utils.emit_utils.rewrite_fk(
                    {"TableName0": "table_name0"},
                    column_fk,
                )

        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=column_fk_gold,
        )

    def test__handle_union_of_length_2(self) -> None:
        """Tests that `_handle_union_of_length_2` works"""
        run_ast_test(
            self,
            gen_ast=_handle_union_of_length_2("Union[int, float]"),
            gold=Name(
                "Float",
                Load(),
                lineno=None,
                col_offset=None,
            ),
        )


unittest_main()
