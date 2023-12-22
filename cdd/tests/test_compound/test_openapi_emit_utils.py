"""
Tests OpenAPI emit_utils
"""

from copy import deepcopy
from os import mkdir, path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cdd.compound.openapi.utils.emit_utils import rewrite_fk
from cdd.shared.ast_utils import set_value
from cdd.shared.source_transformer import to_code
from cdd.tests.mocks.openapi_emit_utils import column_fk, column_fk_gold, id_column
from cdd.tests.mocks.sqlalchemy import node_pk_tbl_class
from cdd.tests.utils_for_tests import run_ast_test, unittest_main


class TestOpenApiEmitUtils(TestCase):
    """Tests whether `openapi` can construct a `dict`"""

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
            init_path: str = path.join(mod_dir, "__init__{}py".format(path.extsep))
            with open(init_path, "wt") as f:
                f.write(to_code(sqlalchemy_cls))
            with patch(
                "cdd.compound.openapi.utils.emit_utils.find_module_filepath",
                lambda _, __: init_path,
            ):
                gen_ast = rewrite_fk(
                    {"TableName0": "table_name0"},
                    column_fk,
                )

        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=column_fk_gold,
        )


unittest_main()
