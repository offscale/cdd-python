"""
Tests route parsing
"""
from unittest import TestCase

from cdd.routes import parse
from cdd.routes.emit import create_util
from cdd.routes.emit_constants import create_helper_variants
from cdd.tests.mocks.openapi import openapi_dict
from cdd.tests.mocks.routes import (
    create_route,
    destroy_route,
    read_route,
    route_config,
    route_mock_prelude,
)
from cdd.tests.utils_for_tests import inspectable_compile, unittest_main


class TestRouteEmit(TestCase):
    """ Tests `routes.parse` """

    route_id_url = "{route_config[route]}/{{{route_config[primary_key]}}}".format(
        route_config=route_config
    )

    def test_create(self) -> None:
        """
        Tests whether `create_route` is produced by `emit.route`
        """
        _create_route = inspectable_compile(route_mock_prelude + create_route).create
        self.assertDictEqual(
            parse.bottle(_create_route),
            openapi_dict["paths"][route_config["route"]]["post"],
        )

    def test_create_util(self) -> None:
        """
        Tests whether `create_util` is produced by `create_util`
        """
        self.assertEqual(
            create_util(name=route_config["name"], route=route_config["route"]),
            create_helper_variants[-1].format(
                name=route_config["name"], route=route_config["route"]
            ),
        )

    def test_read(self) -> None:
        """
        Tests whether `read_route` is produced by `emit.route`
        """
        _read_route = inspectable_compile(route_mock_prelude + read_route).read
        self.assertDictEqual(
            parse.bottle(_read_route), openapi_dict["paths"][self.route_id_url]["get"]
        )

    def test_delete(self) -> None:
        """
        Tests whether `destroy_route` is produced by `emit.route`
        """
        _destroy_route = inspectable_compile(route_mock_prelude + destroy_route).destroy
        self.assertDictEqual(
            parse.bottle(_destroy_route),
            openapi_dict["paths"][self.route_id_url]["delete"],
        )


unittest_main()
