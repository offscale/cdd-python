"""
Tests OpenAPI bulk
"""

from functools import partial
from os import path
from os.path import extsep
from tempfile import TemporaryDirectory
from unittest import TestCase

from cdd.compound.openapi.gen_openapi import openapi_bulk
from cdd.shared.pure_utils import INIT_FILENAME
from cdd.tests.mocks.openapi import openapi_dict_with_sql_types
from cdd.tests.mocks.routes import (
    create_route,
    destroy_route,
    read_route,
    route_mock_prelude,
)
from cdd.tests.mocks.sqlalchemy import (
    config_tbl_with_comments_str,
    sqlalchemy_imports_str,
)
from cdd.tests.utils_for_tests import unittest_main


class TestOpenApiBulk(TestCase):
    """Tests whether `openapi` can construct a `dict`"""

    def test_openapi_bulk(self) -> None:
        """
        Tests whether `openapi_bulk` produces `openapi_dict` given `model_paths` and `routes_paths`
        """
        with TemporaryDirectory() as tempdir:
            temp_dir_join = partial(path.join, tempdir)
            open(temp_dir_join(INIT_FILENAME), "a").close()

            models_filename: str = temp_dir_join(
                "models{extsep}py".format(extsep=extsep)
            )
            routes_filename: str = temp_dir_join(
                "routes{extsep}py".format(extsep=extsep)
            )

            with open(models_filename, "wt") as f:
                f.write(
                    "\n".join((sqlalchemy_imports_str, config_tbl_with_comments_str))
                )

            with open(routes_filename, "wt") as f:
                f.write(
                    "\n".join(
                        (route_mock_prelude, create_route, read_route, destroy_route)
                    )
                )

            gen, gold = (
                openapi_bulk(
                    app_name="rest_api",
                    model_paths=(models_filename,),
                    routes_paths=(routes_filename,),
                ),
                openapi_dict_with_sql_types,
            )

            self.assertEqual(
                *map(
                    lambda d: d["components"]["schemas"]["Config"]["description"],
                    (gen, gold),
                )
            )

            self.assertDictEqual(gen, gold)


unittest_main()
