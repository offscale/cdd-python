"""
Tests the FastAPI route parser
"""

from ast import parse
from copy import deepcopy
from unittest import TestCase

import cdd.routes.parse.fastapi
from cdd.tests.mocks.fastapi_routes import (
    fastapi_post_create_config_async_func,
    fastapi_post_create_config_str,
)
from cdd.tests.mocks.openapi import openapi_dict
from cdd.tests.utils_for_tests import run_ast_test, unittest_main


class TestFastApiRoutesParse(TestCase):
    """
    Tests FastAPI route parser
    """

    def test_from_fastapi_post_create_config(self) -> None:
        """
        Tests whether `cdd.routes.parse.fastapi` produces `fastapi_post_create_config_str`
              from `fastapi_post_create_config_async_func`
        """

        # Roundtrip sanity
        run_ast_test(
            self,
            parse(fastapi_post_create_config_str).body[0],
            fastapi_post_create_config_async_func,
        )

        fastapi_func_resp = cdd.routes.parse.fastapi.fastapi(
            fastapi_post_create_config_async_func
        )
        self.assertEqual(fastapi_func_resp[0], "/api/config")

        mock_api_config = deepcopy(openapi_dict["paths"]["/api/config"])
        del mock_api_config["post"]["summary"], mock_api_config["post"]["requestBody"]
        mock_api_config["post"]["responses"].update(
            {
                404: mock_api_config["post"]["responses"].pop("400"),
                201: mock_api_config["post"]["responses"].pop("201"),
            }
        )

        self.assertDictEqual(fastapi_func_resp[1], mock_api_config)


unittest_main()
