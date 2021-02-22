"""
Tests OpenAPI
"""
from unittest import TestCase

from cdd.openapi.emit import openapi
from cdd.openapi.emitter_utils import NameModelRouteIdCrud
from cdd.tests.mocks.json_schema import config_schema
from cdd.tests.mocks.openapi import openapi_dict
from cdd.tests.mocks.routes import route_config
from cdd.tests.utils_for_tests import unittest_main


class TestOpenApi(TestCase):
    """ Tests whether `NameModelRouteIdCrud` can construct a `dict` """

    def test_openapi_emitter(self) -> None:
        """
        Tests whether `openapi.emit` produces `openapi_dict` given `NameModelRouteIdCrud`
        """
        self.assertDictEqual(
            openapi(
                (
                    NameModelRouteIdCrud(
                        name=route_config["name"],
                        model=config_schema,
                        route=route_config["route"],
                        id=route_config["primary_key"],
                        crud="CRD",
                    ),
                )
            ),
            openapi_dict,
        )


unittest_main()
