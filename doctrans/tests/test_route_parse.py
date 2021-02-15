"""
Tests route parsing
"""
from unittest import TestCase

from doctrans.routes import parse
from doctrans.tests.mocks.openapi import openapi_dict
from doctrans.tests.mocks.routes import (
    create_route,
    destroy_route,
    read_route,
    route_config,
)
from doctrans.tests.utils_for_tests import inspectable_compile, unittest_main


class TestRouteEmit(TestCase):
    """ Tests `routes.parse` """

    route_id_url = "{route_config[route]}/{{{route_config[primary_key]}}}".format(
        route_config=route_config
    )
    prelude = (
        "f = lambda h: lambda g=None: g\n"
        'rest_api = type("App", tuple(), {"get": f, "post": f, "put": f, "delete": f})\n'
    )

    def test_create(self) -> None:
        """
        Tests whether `create_route` is produced by `emit.route`
        """
        _create_route = inspectable_compile(self.prelude + create_route).create
        self.assertDictEqual(
            parse.bottle(_create_route),
            openapi_dict["paths"][route_config["route"]]["post"],
        )

    def test_read(self) -> None:
        """
        Tests whether `read_route` is produced by `emit.route`
        """
        _read_route = inspectable_compile(self.prelude + read_route).read
        self.assertDictEqual(
            parse.bottle(_read_route), openapi_dict["paths"][self.route_id_url]["get"]
        )

    def test_delete(self) -> None:
        """
        Tests whether `destroy_route` is produced by `emit.route`
        """
        _destroy_route = inspectable_compile(self.prelude + destroy_route).destroy
        self.assertDictEqual(
            parse.bottle(_destroy_route),
            openapi_dict["paths"][self.route_id_url]["delete"],
        )


unittest_main()
