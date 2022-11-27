"""
Tests for `cdd.emit.sqlalchemy`
"""

import os
from copy import deepcopy
from platform import system
from unittest import TestCase, skipIf

import cdd.emit.argparse_function
import cdd.emit.class_
import cdd.emit.docstring
import cdd.emit.file
import cdd.emit.function
import cdd.emit.json_schema
import cdd.emit.sqlalchemy
import cdd.parse.argparse_function
from cdd.tests.mocks.ir import intermediate_repr_no_default_sql_doc
from cdd.tests.mocks.sqlalchemy import config_decl_base_ast, config_tbl_ast
from cdd.tests.utils_for_tests import run_ast_test, unittest_main


class TestEmitSqlAlchemy(TestCase):
    """Tests emission"""

    def test_to_sqlalchemy_table(self) -> None:
        """
        Tests that `emit.sqlalchemy_table` with `intermediate_repr_no_default_sql_doc` produces `config_tbl_ast`
        """
        run_ast_test(
            self,
            cdd.emit.sqlalchemy.sqlalchemy_table(
                deepcopy(intermediate_repr_no_default_sql_doc), name="config_tbl"
            ),
            gold=config_tbl_ast,
        )

    @skipIf(
        "GITHUB_ACTIONS" in os.environ and system() in frozenset(("Darwin", "Linux")),
        "GitHub Actions fails this test on macOS & Linux (unable to replicate locally)",
    )
    def test_to_sqlalchemy(self) -> None:
        """
        Tests that `emit.sqlalchemy` with `intermediate_repr_no_default_sql_doc` produces `config_tbl_ast`
        """
        system() in frozenset(("Darwin", "Linux")) and print("test_to_sqlalchemy")

        ir = deepcopy(intermediate_repr_no_default_sql_doc)
        ir["name"] = "Config"
        run_ast_test(
            self,
            cdd.emit.sqlalchemy.sqlalchemy(
                ir,
                # class_name="Config",
                table_name="config_tbl",
            ),
            gold=config_decl_base_ast,
        )


unittest_main()
